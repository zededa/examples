"""
LLM Router Package - Dynamic LLM Service Routing

This package provides a flexible router for connecting to various LLM backends,
allowing users to bring in any LLM service they want.

Supported Providers:
    - Anthropic (Claude)
    - OpenAI (GPT-4, etc.)
    - Google (Gemini)
    - Ollama (local)
    - vLLM (self-hosted)
    - TGI (Text Generation Inference)
    - LM Studio (local)
    - Any OpenAI-compatible API

Usage:
    from webapp.router import get_router, LLMProviderConfig
    
    router = get_router()
    router.register_provider(LLMProviderConfig(
        name="ollama-local",
        provider_type="ollama",
        url="http://localhost:11434",
        model="llama3.2"
    ))
    
    response = router.chat(messages=[{"role": "user", "content": "Hello!"}])
"""

from .config import (
    LLMProviderType,
    RoutingStrategy,
    LLMProviderConfig,
    ProviderStatus,
    ChatMessage,
    ChatResponse,
    DEFAULT_MODELS,
    detect_provider_type_from_url,
)

from .base import LLMAdapter

from .llm_router import (
    AgentLLMRouter,
    get_router,
    register_provider,
    chat,
    get_token_usage,
    reset_token_usage,
)

from .rate_limit_config import (
    RateLimitConfig,
    get_rate_limit_config,
    is_rate_limit_error,
    is_retryable_error,
    RETRYABLE_STATUS_CODES,
    NON_RETRYABLE_STATUS_CODES,
)

from .resilience import (
    ResilientLLMClient,
    RequestMetrics,
    ConcurrencyLimiter,
    RequestDeduplicator,
    RateLimitErrorResponse,
    RateLimitException,
    make_resilient_request,
    get_concurrency_limiter,
    get_deduplicator,
    get_resilience_stats,
    estimate_tokens,
    estimate_messages_tokens,
    calculate_backoff,
)

__all__ = [
    # Config
    "LLMProviderType",
    "RoutingStrategy", 
    "LLMProviderConfig",
    "ProviderStatus",
    "ChatMessage",
    "ChatResponse",
    "DEFAULT_MODELS",
    # Base
    "LLMAdapter",
    # Router
    "AgentLLMRouter",
    "get_router",
    "register_provider",
    "chat",
    # Token tracking
    "get_token_usage",
    "reset_token_usage",
    # Rate limit config
    "RateLimitConfig",
    "get_rate_limit_config",
    "is_rate_limit_error",
    "is_retryable_error",
    "RETRYABLE_STATUS_CODES",
    "NON_RETRYABLE_STATUS_CODES",
    # Resilience
    "ResilientLLMClient",
    "RequestMetrics",
    "ConcurrencyLimiter",
    "RequestDeduplicator",
    "RateLimitErrorResponse",
    "RateLimitException",
    "make_resilient_request",
    "get_concurrency_limiter",
    "get_deduplicator",
    "get_resilience_stats",
    "estimate_tokens",
    "estimate_messages_tokens",
    "calculate_backoff",
]
