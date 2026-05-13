"""
Agent LLM Router - Central router for LLM service management

The router provides:
- Dynamic provider registration/deregistration
- Multiple routing strategies (priority, round-robin, failover, latency)
- Automatic health monitoring
- Thread-safe operations

Usage:
    from webapp.router import get_router, LLMProviderConfig
    
    router = get_router()
    response = router.chat(messages=[{"role": "user", "content": "Hello!"}])
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type

from .config import (
    LLMProviderType,
    RoutingStrategy,
    LLMProviderConfig,
    ProviderStatus,
    ChatResponse,
)
from .base import LLMAdapter
from .adapters import (
    OllamaAdapter,
    VLLMAdapter,
    TGIAdapter,
    OpenAICompatibleAdapter,
    AnthropicAdapter,
    OpenAIAdapter,
    GoogleAdapter,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Token Usage Tracking
# =============================================================================

@dataclass
class TokenUsageStats:
    """Token usage statistics for a provider."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    request_count: int = 0
    
    def add(self, prompt: int, completion: int) -> None:
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.total_tokens += prompt + completion
        self.request_count += 1
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "request_count": self.request_count,
        }


class TokenUsageTracker:
    """Tracks token usage across all providers."""
    
    def __init__(self):
        self._usage: Dict[str, TokenUsageStats] = {}
        self._lock = threading.Lock()
    
    def record(self, provider: str, model: str, usage: Optional[Dict[str, int]]) -> None:
        """Record token usage for a request."""
        if not usage:
            return
        
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        
        with self._lock:
            key = f"{provider}/{model}"
            if key not in self._usage:
                self._usage[key] = TokenUsageStats()
            self._usage[key].add(prompt_tokens, completion_tokens)
        
        # Log the usage
        total = prompt_tokens + completion_tokens
        logger.info(
            f"🔢 Token Usage [{provider}/{model}]: "
            f"prompt={prompt_tokens}, completion={completion_tokens}, total={total}"
        )
    
    def get_usage(self, provider: Optional[str] = None) -> Dict[str, Any]:
        """Get token usage stats."""
        with self._lock:
            if provider:
                # Filter by provider prefix
                return {
                    k: v.to_dict() for k, v in self._usage.items()
                    if k.startswith(f"{provider}/")
                }
            return {k: v.to_dict() for k, v in self._usage.items()}
    
    def get_totals(self) -> Dict[str, int]:
        """Get total token usage across all providers."""
        with self._lock:
            totals = TokenUsageStats()
            for stats in self._usage.values():
                totals.prompt_tokens += stats.prompt_tokens
                totals.completion_tokens += stats.completion_tokens
                totals.total_tokens += stats.total_tokens
                totals.request_count += stats.request_count
            return totals.to_dict()
    
    def reset(self) -> None:
        """Reset all usage stats."""
        with self._lock:
            self._usage.clear()
        logger.info("Token usage stats reset")


# Global token tracker instance
_token_tracker = TokenUsageTracker()


def get_token_usage() -> Dict[str, Any]:
    """Get current token usage stats."""
    return {
        "by_provider": _token_tracker.get_usage(),
        "totals": _token_tracker.get_totals(),
    }


def reset_token_usage() -> None:
    """Reset token usage stats."""
    _token_tracker.reset()


# =============================================================================
# Adapter Registry
# =============================================================================

ADAPTER_REGISTRY: Dict[LLMProviderType, Type[LLMAdapter]] = {
    k: v for k, v in {
        LLMProviderType.ANTHROPIC: AnthropicAdapter,
        LLMProviderType.OPENAI: OpenAIAdapter,
        LLMProviderType.GOOGLE: GoogleAdapter,
        LLMProviderType.GROQ: OpenAICompatibleAdapter,  # Groq uses OpenAI-compatible API
        LLMProviderType.OLLAMA: OllamaAdapter,
        LLMProviderType.VLLM: VLLMAdapter,
        LLMProviderType.TGI: TGIAdapter,
        LLMProviderType.LMSTUDIO: OpenAICompatibleAdapter,
        LLMProviderType.OPENAI_COMPATIBLE: OpenAICompatibleAdapter,
    }.items() if v is not None
}


def register_adapter(provider_type: LLMProviderType, adapter_class: Type[LLMAdapter]) -> None:
    """Register a custom adapter for a provider type."""
    ADAPTER_REGISTRY[provider_type] = adapter_class
    logger.info(f"Registered adapter {adapter_class.__name__} for {provider_type.value}")


# =============================================================================
# Agent LLM Router
# =============================================================================

class AgentLLMRouter:
    """
    Central router for managing and routing LLM requests to multiple providers.
    
    Allows users to interact with the AI agent regardless of which LLM service
    they're running - whether it's Ollama locally, vLLM in a container, or
    cloud APIs like OpenAI/Anthropic.
    
    Features:
    - Dynamic provider registration/deregistration
    - Automatic failover to available providers
    - Multiple routing strategies
    - Health monitoring
    
    Thread-Safety:
        All operations are thread-safe. Uses locks for registry modifications.
    """
    
    _instance: Optional["AgentLLMRouter"] = None
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        """Singleton pattern for global router instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(
        self,
        routing_strategy: RoutingStrategy = RoutingStrategy.FAILOVER,
        auto_discover: bool = True,
    ):
        if getattr(self, '_initialized', False):
            return
        
        self._routing_strategy = routing_strategy
        
        # Provider registry
        self._providers: Dict[str, LLMProviderConfig] = {}
        self._provider_status: Dict[str, ProviderStatus] = {}
        self._providers_lock = threading.RLock()
        
        # Adapter instances (lazy loaded)
        self._adapters: Dict[LLMProviderType, LLMAdapter] = {}
        self._adapters_lock = threading.Lock()
        
        # Round-robin state
        self._rr_index = 0
        self._rr_lock = threading.Lock()
        
        # Auto-discover providers from environment. Wrap in try/except so a
        # single misconfigured/unavailable provider cannot permanently wedge
        # the singleton in an un-initialized state (which would block every
        # later get_router() call — including credential activation).
        if auto_discover:
            try:
                self._auto_discover_providers()
            except Exception as e:
                logger.error(f"Provider auto-discovery failed: {e}", exc_info=True)

        self._initialized = True
        logger.info(f"AgentLLMRouter initialized with strategy: {routing_strategy.value}")
    
    def _auto_discover_providers(self) -> None:
        """Auto-discover LLM providers from environment variables."""
        
        # Check for Anthropic
        anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
        anthropic_model = os.environ.get("ANTHROPIC_MODEL")
        if anthropic_key and anthropic_model:
            self.register_provider(LLMProviderConfig(
                name="anthropic",
                provider_type=LLMProviderType.ANTHROPIC,
                api_key=anthropic_key,
                model=anthropic_model,
                priority=1,
                supports_tools=True,
                supports_vision=True,
            ))
        elif anthropic_key:
            logger.warning("ANTHROPIC_API_KEY set but ANTHROPIC_MODEL not specified - provider not registered")

        # Check for OpenAI
        openai_key = os.environ.get("OPENAI_API_KEY")
        openai_model = os.environ.get("OPENAI_MODEL")
        if openai_key and openai_model:
            self.register_provider(LLMProviderConfig(
                name="openai",
                provider_type=LLMProviderType.OPENAI,
                api_key=openai_key,
                model=openai_model,
                priority=2,
                supports_tools=True,
                supports_vision=True,
            ))
        elif openai_key:
            logger.warning("OPENAI_API_KEY set but OPENAI_MODEL not specified - provider not registered")

        # Check for Google
        google_key = os.environ.get("GOOGLE_API_KEY")
        google_model = os.environ.get("GOOGLE_MODEL")
        if google_key and google_model:
            self.register_provider(LLMProviderConfig(
                name="google",
                provider_type=LLMProviderType.GOOGLE,
                api_key=google_key,
                model=google_model,
                priority=3,
                supports_tools=True,
                supports_vision=True,
            ))
        elif google_key:
            logger.warning("GOOGLE_API_KEY set but GOOGLE_MODEL not specified - provider not registered")
        
        # Check for Groq
        groq_key = os.environ.get("GROQ_API_KEY")
        groq_model = os.environ.get("GROQ_MODEL")
        if groq_key and groq_model:
            self.register_provider(LLMProviderConfig(
                name="groq",
                provider_type=LLMProviderType.GROQ,
                api_key=groq_key,
                model=groq_model,
                priority=4,
                supports_tools=True,
                supports_vision=False,  # Groq doesn't support vision yet
            ))
        elif groq_key:
            logger.warning("GROQ_API_KEY set but GROQ_MODEL not specified - provider not registered")
        
        llm_url = os.environ.get("LLM_SERVER_URL")
        llm_model = os.environ.get("LLM_MODEL_NAME")
        llm_key = os.environ.get("LLM_API_KEY")
        eip_token = os.environ.get("EIP_ACCESS_TOKEN")

        # EdgeAI built-in OpenAI-compatible endpoint.
        # When the agent runs inside an EdgeAI deployment, the BFF injects
        # EIP_ACCESS_TOKEN (a JWT bearer for the platform's Agent OpenAI proxy)
        # alongside LLM_SERVER_URL and LLM_MODEL_NAME. We auto-register that as
        # the highest-priority provider so users don't have to configure their
        # own cloud API keys. The proxy lives at "{LLM_SERVER_URL}/openai" —
        # append it here if the env var doesn't already include the suffix.
        edgeai_builtin_registered = False
        if eip_token and llm_url:
            base_url = llm_url.rstrip("/")
            if not base_url.endswith("/openai"):
                base_url = f"{base_url}/openai"
            self.register_provider(LLMProviderConfig(
                name="edgeai-builtin",
                provider_type=LLMProviderType.OPENAI_COMPATIBLE,
                url=base_url,
                model=llm_model or "edgeai-default",
                api_key=eip_token,
                priority=1,
                supports_tools=True,
                supports_vision=True,
                metadata={
                    "builtin": True,
                    "managed_by": "edgeai-platform",
                    "description": "EdgeAI built-in OpenAI-compatible endpoint",
                },
            ))
            edgeai_builtin_registered = True
        elif eip_token and not llm_url:
            logger.warning(
                "EIP_ACCESS_TOKEN set but LLM_SERVER_URL not specified - "
                "EdgeAI built-in provider not registered"
            )

        # Generic OpenAI-compatible local LLM (LLM_API_KEY-based).
        # Skipped when the EdgeAI built-in provider has already claimed the
        # same URL — otherwise we'd register two providers pointing at the
        # same endpoint with different auth, which is confusing and racey.
        if llm_url and llm_model and llm_key and not edgeai_builtin_registered:
            self.register_provider(LLMProviderConfig(
                name="local-llm",
                provider_type=LLMProviderType.OPENAI_COMPATIBLE,
                url=llm_url,
                model=llm_model,
                api_key=llm_key,
                priority=5,
                supports_tools=os.environ.get("LLM_SUPPORTS_TOOLS", "true").lower() == "true",
            ))
        elif llm_url and not llm_model and not edgeai_builtin_registered:
            logger.warning("LLM_SERVER_URL set but LLM_MODEL_NAME not specified - provider not registered")
        elif llm_url and not llm_key and not eip_token:
            logger.info("LLM_SERVER_URL set but LLM_API_KEY not provided - local LLM provider not auto-registered")
        
        # Check for Ollama
        ollama_url = os.environ.get("OLLAMA_URL", "http://localhost:11434")
        ollama_model = os.environ.get("OLLAMA_MODEL")
        if ollama_model:
            self.register_provider(LLMProviderConfig(
                name="ollama",
                provider_type=LLMProviderType.OLLAMA,
                url=ollama_url,
                model=ollama_model,
                priority=10,
                supports_tools=True,
            ))
        elif os.environ.get("USE_OLLAMA"):
            logger.warning("USE_OLLAMA set but OLLAMA_MODEL not specified - provider not registered")
        
        # Load from JSON config
        providers_json = os.environ.get("LLM_PROVIDERS")
        if providers_json:
            try:
                providers = json.loads(providers_json)
                for provider_data in providers:
                    config = LLMProviderConfig.from_dict(provider_data)
                    self.register_provider(config)
            except (json.JSONDecodeError, KeyError) as e:
                logger.error(f"Failed to parse LLM_PROVIDERS: {e}")
    
    def _get_adapter(self, provider_type: LLMProviderType) -> LLMAdapter:
        """Get or create an adapter instance for a provider type."""
        with self._adapters_lock:
            if provider_type not in self._adapters:
                adapter_class = ADAPTER_REGISTRY.get(provider_type) or OpenAICompatibleAdapter
                if adapter_class is None:
                    raise RuntimeError(
                        f"No adapter available for provider type '{provider_type.value}'. "
                        f"The adapter module failed to import at startup — check earlier "
                        f"logs for the underlying ImportError (e.g. missing dependency)."
                    )
                self._adapters[provider_type] = adapter_class()
            return self._adapters[provider_type]
    
    # =========================================================================
    # Provider Registry Operations
    # =========================================================================
    
    def register_provider(self, config: LLMProviderConfig) -> bool:
        """
        Register a new LLM provider.
        
        Args:
            config: Provider configuration
            
        Returns:
            True if registered successfully
        """
        with self._providers_lock:
            if config.name in self._providers:
                logger.info(f"Provider '{config.name}' already registered, updating config")
            
            self._providers[config.name] = config
            self._provider_status[config.name] = ProviderStatus(
                name=config.name,
                available=False,
                last_check=0,
            )
            
            logger.info(f"Registered LLM provider: {config.name} ({config.provider_type.value})")

            # Check availability — never let an adapter failure prevent
            # registration. A missing/broken adapter for one provider type
            # must not take down the whole router (and block other providers,
            # including user-imported credentials, from being registered).
            try:
                self._check_provider_availability(config.name)
            except Exception as e:
                logger.error(
                    f"Availability check failed for provider '{config.name}': {e}. "
                    f"Provider is registered but marked unavailable."
                )
                status = self._provider_status.get(config.name)
                if status is not None:
                    status.available = False
                    status.last_check = time.time()
                    status.last_error = str(e)

            return True
    
    def unregister_provider(self, name: str) -> bool:
        """Remove a provider from the router."""
        with self._providers_lock:
            if name in self._providers:
                del self._providers[name]
                del self._provider_status[name]
                logger.info(f"Unregistered LLM provider: {name}")
                return True
            return False
    
    def get_provider(self, name: str) -> Optional[LLMProviderConfig]:
        """Get a provider configuration by name."""
        with self._providers_lock:
            return self._providers.get(name)
    
    def list_providers(self) -> List[Dict[str, Any]]:
        """List all registered providers with their status."""
        with self._providers_lock:
            result = []
            for name, config in self._providers.items():
                status = self._provider_status.get(name)
                result.append({
                    **config.to_dict(),
                    "status": status.to_dict() if status else None,
                })
            return result
    
    # =========================================================================
    # Availability Checking
    # =========================================================================
    
    def _check_provider_availability(self, name: str) -> bool:
        """Check availability of a single provider."""
        with self._providers_lock:
            if name not in self._providers:
                return False
            config = self._providers[name]
        
        adapter = self._get_adapter(config.provider_type)
        available, latency, error = adapter.check_availability(config)
        
        # Don't call list_models on every health check - it's expensive
        # Models are only fetched on-demand via the /llm/models endpoint
        
        with self._providers_lock:
            if name in self._provider_status:
                status = self._provider_status[name]
                status.available = available
                status.last_check = time.time()
                status.latency_ms = latency
                status.last_error = error
                
                if not available:
                    status.error_count += 1
        
        if available:
            logger.debug(f"Provider {name} available (latency: {latency:.1f}ms)")
        else:
            logger.warning(f"Provider {name} unavailable: {error}")
        
        return available
    
    def check_all_providers(self) -> Dict[str, bool]:
        """Check availability of all registered providers."""
        results = {}
        
        with self._providers_lock:
            provider_names = list(self._providers.keys())
        
        for name in provider_names:
            try:
                results[name] = self._check_provider_availability(name)
            except Exception as e:
                logger.error(f"Availability check failed for provider '{name}': {e}")
                results[name] = False
                with self._providers_lock:
                    status = self._provider_status.get(name)
                    if status is not None:
                        status.available = False
                        status.last_check = time.time()
                        status.last_error = str(e)

        return results
    
    # =========================================================================
    # Routing
    # =========================================================================
    
    def _select_provider(self, require_tools: bool = False) -> Optional[LLMProviderConfig]:
        """Select the best available provider based on routing strategy."""
        
        with self._providers_lock:
            # Filter to enabled and available providers
            candidates = []
            for name, config in self._providers.items():
                if not config.enabled:
                    continue
                
                status = self._provider_status.get(name)
                if not status or not status.available:
                    continue
                
                if require_tools and not config.supports_tools:
                    continue
                
                candidates.append((name, config, status))
        
        if not candidates:
            return None
        
        # Apply routing strategy
        if self._routing_strategy == RoutingStrategy.PRIORITY:
            candidates.sort(key=lambda x: x[1].priority)
            return candidates[0][1]
        
        elif self._routing_strategy == RoutingStrategy.ROUND_ROBIN:
            with self._rr_lock:
                idx = self._rr_index % len(candidates)
                self._rr_index += 1
            return candidates[idx][1]
        
        elif self._routing_strategy == RoutingStrategy.LATENCY:
            candidates.sort(key=lambda x: x[2].latency_ms)
            return candidates[0][1]
        
        elif self._routing_strategy == RoutingStrategy.FAILOVER:
            # Use priority order, failover handled in chat()
            candidates.sort(key=lambda x: x[1].priority)
            return candidates[0][1]
        
        # Default to priority
        candidates.sort(key=lambda x: x[1].priority)
        return candidates[0][1]
    
    def _get_all_providers_by_priority(self, require_tools: bool = False) -> List[LLMProviderConfig]:
        """Get all available providers sorted by priority (for failover)."""
        with self._providers_lock:
            candidates = []
            for name, config in self._providers.items():
                if not config.enabled:
                    continue
                
                status = self._provider_status.get(name)
                if not status or not status.available:
                    continue
                
                if require_tools and not config.supports_tools:
                    continue
                
                candidates.append(config)
            
            candidates.sort(key=lambda x: x.priority)
            return candidates
    
    # =========================================================================
    # Chat Interface
    # =========================================================================
    
    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        provider_name: Optional[str] = None,
        **kwargs
    ) -> ChatResponse:
        """
        Send a chat request to an LLM provider.
        
        Automatically routes to the best available provider, with failover
        support if the primary provider fails.
        
        Args:
            messages: List of chat messages
            tools: Optional list of tool schemas for function calling
            provider_name: Optional specific provider to use
            **kwargs: Additional arguments passed to the adapter
            
        Returns:
            ChatResponse with the LLM's response
            
        Raises:
            RuntimeError: If no providers are available
        """
        require_tools = tools is not None and len(tools) > 0
        
        # If specific provider requested
        if provider_name:
            config = self.get_provider(provider_name)
            if not config:
                raise RuntimeError(f"Provider '{provider_name}' not found")
            
            adapter = self._get_adapter(config.provider_type)
            response = adapter.chat(config, messages, tools, **kwargs)
            
            # Track token usage
            _token_tracker.record(config.name, config.model or "unknown", response.usage)
            
            # Update stats
            with self._providers_lock:
                if provider_name in self._provider_status:
                    self._provider_status[provider_name].total_requests += 1
            
            return response
        
        # Failover strategy: try providers in priority order
        if self._routing_strategy == RoutingStrategy.FAILOVER:
            providers = self._get_all_providers_by_priority(require_tools)
            
            if not providers:
                raise RuntimeError("No LLM providers available")
            
            last_error = None
            for config in providers:
                try:
                    adapter = self._get_adapter(config.provider_type)
                    response = adapter.chat(config, messages, tools, **kwargs)
                    
                    # Track token usage
                    _token_tracker.record(config.name, config.model or "unknown", response.usage)
                    
                    # Update stats
                    with self._providers_lock:
                        if config.name in self._provider_status:
                            self._provider_status[config.name].total_requests += 1
                    
                    return response
                    
                except Exception as e:
                    logger.warning(f"Provider {config.name} failed: {e}, trying next...")
                    last_error = e
                    
                    # Mark as unavailable temporarily
                    with self._providers_lock:
                        if config.name in self._provider_status:
                            self._provider_status[config.name].error_count += 1
                            self._provider_status[config.name].last_error = str(e)
                    continue
            
            raise RuntimeError(f"All providers failed. Last error: {last_error}")
        
        # Other strategies: select single provider
        config = self._select_provider(require_tools)
        if not config:
            raise RuntimeError("No LLM providers available")
        
        adapter = self._get_adapter(config.provider_type)
        response = adapter.chat(config, messages, tools, **kwargs)
        
        # Track token usage
        _token_tracker.record(config.name, config.model or "unknown", response.usage)
        
        # Update stats
        with self._providers_lock:
            if config.name in self._provider_status:
                self._provider_status[config.name].total_requests += 1
        
        return response
    
    def chat_stream(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        provider_name: Optional[str] = None,
        **kwargs
    ):
        """
        Send a streaming chat request to an LLM provider.
        
        Returns a generator that yields SSE-style events:
        - {"type": "token", "content": "..."} - Text token
        - {"type": "tool_call", ...} - Tool call data
        - {"type": "done", "response": ChatResponse} - Final response
        - {"type": "error", "error": "..."} - Error occurred
        
        Args:
            messages: List of chat messages
            tools: Optional list of tool schemas for function calling
            provider_name: Optional specific provider to use
            **kwargs: Additional arguments passed to the adapter
            
        Yields:
            Dict events with streaming response data
            
        Raises:
            RuntimeError: If no providers are available
        """
        require_tools = tools is not None and len(tools) > 0
        
        # If specific provider requested
        if provider_name:
            config = self.get_provider(provider_name)
            if not config:
                yield {"type": "error", "error": f"Provider '{provider_name}' not found"}
                return
            
            adapter = self._get_adapter(config.provider_type)
            
            for event in adapter.chat_stream(config, messages, tools, **kwargs):
                # Track usage when done or complete
                event_type = event.get("type")
                if event_type in ("done", "complete"):
                    response = event.get("response") or event.get("full_response")
                    if response and hasattr(response, 'usage'):
                        _token_tracker.record(config.name, config.model or "unknown", response.usage)
                        with self._providers_lock:
                            if provider_name in self._provider_status:
                                self._provider_status[provider_name].total_requests += 1
                yield event
            return
        
        # Select provider
        config = self._select_provider(require_tools)
        if not config:
            yield {"type": "error", "error": "No LLM providers available"}
            return
        
        adapter = self._get_adapter(config.provider_type)
        
        for event in adapter.chat_stream(config, messages, tools, **kwargs):
            # Track usage when done or complete
            event_type = event.get("type")
            if event_type in ("done", "complete"):
                response = event.get("response") or event.get("full_response")
                if response and hasattr(response, 'usage'):
                    _token_tracker.record(config.name, config.model or "unknown", response.usage)
                    with self._providers_lock:
                        if config.name in self._provider_status:
                            self._provider_status[config.name].total_requests += 1
            yield event
    
    # =========================================================================
    # Configuration
    # =========================================================================
    
    def set_routing_strategy(self, strategy: RoutingStrategy) -> None:
        """Change the routing strategy."""
        old = self._routing_strategy
        self._routing_strategy = strategy
        logger.info(f"Routing strategy changed: {old.value} -> {strategy.value}")
    
    def get_active_provider(self) -> Optional[Dict[str, Any]]:
        """Get the currently active (highest priority available) provider."""
        config = self._select_provider()
        if config:
            status = self._provider_status.get(config.name)
            return {
                **config.to_dict(),
                "status": status.to_dict() if status else None,
            }
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Export router state as dictionary."""
        return {
            "routing_strategy": self._routing_strategy.value,
            "providers": self.list_providers(),
            "active_provider": self.get_active_provider(),
        }


# =============================================================================
# Module-level convenience functions
# =============================================================================

def get_router() -> AgentLLMRouter:
    """Get the global LLM router instance."""
    return AgentLLMRouter()


def register_provider(config: LLMProviderConfig) -> bool:
    """Register a provider with the global router."""
    return get_router().register_provider(config)


def chat(
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]] = None,
    **kwargs
) -> ChatResponse:
    """Send a chat request using the global router."""
    return get_router().chat(messages, tools, **kwargs)
