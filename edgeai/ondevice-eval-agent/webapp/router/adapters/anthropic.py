"""
Anthropic Adapter - Claude API support

Supports Claude models via the official Anthropic SDK.
Includes streaming support for real-time token delivery.
Includes production-grade rate limit handling and resilience.
https://www.anthropic.com/
"""

import json
import logging
import os
import threading
import time
from typing import Any, Dict, Generator, List, Optional, Tuple

from ..base import LLMAdapter
from ..config import LLMProviderConfig, ChatResponse
from ..rate_limit_config import (
    get_rate_limit_config,
    is_rate_limit_error,
    is_retryable_error,
    extract_retry_after,
)
from ..resilience import (
    make_resilient_request,
    RateLimitException,
    RateLimitErrorResponse,
    RequestMetrics,
    generate_request_id,
    get_concurrency_limiter,
    calculate_backoff,
)

logger = logging.getLogger(__name__)


def _normalize_messages_for_anthropic(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert OpenAI-style tool messages into Anthropic tool_use/tool_result format."""
    normalized: List[Dict[str, Any]] = []

    for msg in messages:
        role = msg.get("role")

        # Convert assistant tool_calls (OpenAI style) to Anthropic tool_use content blocks
        if role == "assistant" and msg.get("tool_calls"):
            content_blocks: List[Dict[str, Any]] = []

            text_content = msg.get("content") or ""
            if isinstance(text_content, str) and text_content.strip():
                content_blocks.append({"type": "text", "text": text_content})

            for tc in msg.get("tool_calls", []):
                tool_id = tc.get("id") or generate_request_id()
                func = tc.get("function", {})
                tool_name = func.get("name") or "tool"
                args_raw = func.get("arguments") or "{}"
                try:
                    tool_input = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
                except json.JSONDecodeError:
                    tool_input = {}
                content_blocks.append({
                    "type": "tool_use",
                    "id": tool_id,
                    "name": tool_name,
                    "input": tool_input,
                })

            normalized.append({"role": "assistant", "content": content_blocks})
            continue

        # Convert tool results (OpenAI style role="tool") to Anthropic tool_result blocks
        if role == "tool":
            tool_call_id = msg.get("tool_call_id") or msg.get("id") or generate_request_id()
            raw_content = msg.get("content", "")
            # Anthropic expects tool_result content to be a list of text blocks
            if isinstance(raw_content, list):
                content_blocks = []
                for item in raw_content:
                    if isinstance(item, dict) and "text" in item:
                        content_blocks.append({"type": "text", "text": item.get("text", "")})
                    else:
                        content_blocks.append({"type": "text", "text": str(item)})
            else:
                content_blocks = [{"type": "text", "text": str(raw_content)}]

            normalized.append({
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": tool_call_id,
                    "content": content_blocks,
                }]
            })
            continue

        # Leave other messages as-is
        normalized.append(msg)

    return normalized


def _convert_tools_to_anthropic_format(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert tools to Anthropic format.
    
    Handles both native Anthropic format and OpenAI function-calling format.
    OpenAI format: {"type": "function", "function": {"name": ..., "description": ..., "parameters": ...}}
    Anthropic format: {"name": ..., "description": ..., "input_schema": ...}
    """
    anthropic_tools = []
    for tool in tools:
        # Check if tool is in OpenAI function-calling format
        if tool.get("type") == "function" and "function" in tool:
            func = tool["function"]
            anthropic_tools.append({
                "name": func.get("name"),
                "description": func.get("description", ""),
                "input_schema": func.get("parameters", {})
            })
        else:
            # Already in Anthropic-compatible format
            anthropic_tools.append({
                "name": tool.get("name"),
                "description": tool.get("description", ""),
                "input_schema": tool.get("input_schema", {})
            })
    return anthropic_tools


class AnthropicAdapter(LLMAdapter):
    """Adapter for Anthropic Claude API."""

    # Class-level cache for models list (protected by _models_cache_lock)
    _models_cache: List[str] = []
    _models_cache_time: float = 0
    _models_cache_ttl: float = 300  # 5 minutes
    _models_cache_lock: threading.Lock = threading.Lock()

    def __init__(self):
        super().__init__()
        self._client = None
    
    def _get_client(self, config: LLMProviderConfig):
        """Get or create Anthropic client."""
        try:
            import anthropic
            api_key = config.api_key or os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                return None
            return anthropic.Anthropic(api_key=api_key)
        except ImportError:
            return None
    
    def check_availability(self, config: LLMProviderConfig) -> Tuple[bool, float, Optional[str]]:
        client = self._get_client(config)
        if not client:
            return False, 0.0, "Anthropic SDK not installed or API key not set"

        # Make a lightweight API call to verify key validity and connectivity.
        try:
            start = time.time()
            client.models.list(limit=1)
            latency = (time.time() - start) * 1000
            return True, latency, None
        except Exception as e:
            return False, 0.0, f"Anthropic API check failed: {e}"
    
    def list_models(self, config: LLMProviderConfig) -> List[str]:
        """Fetch available models from Anthropic API with caching."""
        with AnthropicAdapter._models_cache_lock:
            # Return cached models if still valid
            if AnthropicAdapter._models_cache and (time.time() - AnthropicAdapter._models_cache_time) < AnthropicAdapter._models_cache_ttl:
                return list(AnthropicAdapter._models_cache)

        client = self._get_client(config)
        if not client:
            logger.warning("Anthropic client not available for listing models")
            with AnthropicAdapter._models_cache_lock:
                return list(AnthropicAdapter._models_cache)

        try:
            # Fetch models from API
            page = client.models.list(limit=100)
            models = [model.id for model in page.data]

            # Update cache
            with AnthropicAdapter._models_cache_lock:
                AnthropicAdapter._models_cache = models
                AnthropicAdapter._models_cache_time = time.time()

            logger.debug(f"Fetched {len(models)} models from Anthropic API")
            return models
        except Exception as e:
            logger.error(f"Failed to fetch Anthropic models: {e}")
            with AnthropicAdapter._models_cache_lock:
                return list(AnthropicAdapter._models_cache)
    
    def chat(
        self,
        config: LLMProviderConfig,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> ChatResponse:
        """
        Send a chat request to Anthropic with automatic retry and rate limit handling.
        
        Features:
        - Automatic retry with exponential backoff on 429/5xx errors
        - Concurrency limiting to prevent request storms
        - Structured error responses for rate limits
        - Comprehensive logging for observability
        """
        client = self._get_client(config)
        if not client:
            raise RuntimeError("Anthropic client not available")
        
        # Extract system message if present
        system_msg = None
        chat_messages = []
        for msg in messages:
            if msg.get("role") == "system":
                system_msg = msg.get("content", "")
            else:
                chat_messages.append(msg)

        # Normalize OpenAI-style tool messages for Anthropic
        chat_messages = _normalize_messages_for_anthropic(chat_messages)
        
        if not config.model:
            raise ValueError("No model specified in Anthropic config")
        
        request_params: Dict[str, Any] = {
            "model": config.model,
            "max_tokens": config.max_tokens,
            "messages": chat_messages,
        }
        
        if system_msg:
            request_params["system"] = system_msg
        
        if tools and config.supports_tools:
            request_params["tools"] = _convert_tools_to_anthropic_format(tools)

        # Layer 3 of the overflow pipeline: server-side context compaction.
        # Returns {} when OVERFLOW_ENABLED or OVERFLOW_ANTHROPIC_COMPACTION_ENABLED
        # are false, so this is a safe merge in either case.
        try:
            from agents.context.anthropic_compaction import build_kwargs as _compaction_kwargs
            request_params.update(_compaction_kwargs())
        except Exception as _exc:  # pragma: no cover - defensive
            logger.debug("anthropic compaction kwargs unavailable: %s", _exc)

        # Use resilient request wrapper for automatic retry and rate limit handling
        rate_config = get_rate_limit_config()
        request_id = generate_request_id()
        limiter = get_concurrency_limiter()
        
        # Log request start
        logger.info(
            f"🚀 Anthropic request start | id={request_id} | model={config.model}",
            extra={
                "event": "anthropic_request_start",
                "request_id": request_id,
                "model": config.model,
                "provider": config.name,
            }
        )
        
        # Acquire concurrency slot
        if not limiter.acquire(timeout=rate_config.request_timeout):
            raise TimeoutError(
                f"Timed out waiting for concurrency slot after {rate_config.request_timeout}s"
            )
        
        start_time = time.time()
        last_error = None
        retry_count = 0
        
        try:
            for attempt in range(1, rate_config.max_retries + 1):
                try:
                    response = client.messages.create(**request_params)
                    
                    # Success - extract content and tool calls
                    content = ""
                    tool_calls = []
                    
                    for block in response.content:
                        if hasattr(block, 'text'):
                            content += block.text
                        elif hasattr(block, 'type') and block.type == 'tool_use':
                            tool_calls.append({
                                "id": block.id,
                                "name": block.name,
                                "arguments": json.dumps(block.input) if isinstance(block.input, dict) else block.input,
                            })
                    
                    duration_ms = (time.time() - start_time) * 1000
                    logger.info(
                        f"✅ Anthropic request success | id={request_id} | "
                        f"duration={duration_ms:.0f}ms | retries={retry_count}",
                        extra={
                            "event": "anthropic_request_success",
                            "request_id": request_id,
                            "duration_ms": duration_ms,
                            "retry_count": retry_count,
                            "input_tokens": response.usage.input_tokens,
                            "output_tokens": response.usage.output_tokens,
                        }
                    )
                    
                    return ChatResponse(
                        content=content,
                        provider=config.name,
                        model=response.model,
                        tool_calls=tool_calls if tool_calls else None,
                        usage={
                            "prompt_tokens": response.usage.input_tokens,
                            "completion_tokens": response.usage.output_tokens,
                        },
                        finish_reason=response.stop_reason
                    )
                    
                except Exception as e:
                    last_error = e
                    error_str = str(e)
                    
                    # Check if this error is retryable
                    if not is_retryable_error(e):
                        logger.error(
                            f"❌ Anthropic non-retryable error | id={request_id} | error={error_str}",
                            extra={
                                "event": "anthropic_non_retryable_error",
                                "request_id": request_id,
                                "error": error_str,
                            }
                        )
                        raise
                    
                    # Check if we have retries left
                    if attempt >= rate_config.max_retries:
                        break
                    
                    # Calculate backoff
                    retry_after_hint = extract_retry_after(e)
                    backoff = calculate_backoff(attempt, rate_config, retry_after_hint)
                    retry_count = attempt
                    
                    if is_rate_limit_error(e):
                        logger.warning(
                            f"⏳ Anthropic rate limited | id={request_id} | "
                            f"attempt={attempt}/{rate_config.max_retries} | backoff={backoff:.2f}s",
                            extra={
                                "event": "anthropic_rate_limited",
                                "request_id": request_id,
                                "attempt": attempt,
                                "backoff_seconds": backoff,
                                "retry_after_hint": retry_after_hint,
                            }
                        )
                    else:
                        logger.warning(
                            f"🔄 Anthropic retry | id={request_id} | "
                            f"attempt={attempt}/{rate_config.max_retries} | backoff={backoff:.2f}s | error={error_str[:100]}",
                            extra={
                                "event": "anthropic_retry",
                                "request_id": request_id,
                                "attempt": attempt,
                                "backoff_seconds": backoff,
                                "error": error_str,
                            }
                        )
                    
                    # Wait before retry
                    time.sleep(backoff)
            
            # All retries exhausted
            duration_ms = (time.time() - start_time) * 1000
            
            if last_error and is_rate_limit_error(last_error):
                retry_after = extract_retry_after(last_error)
                logger.error(
                    f"❌ Anthropic rate limit exhausted | id={request_id} | "
                    f"retries={rate_config.max_retries} | duration={duration_ms:.0f}ms",
                    extra={
                        "event": "anthropic_rate_limit_exhausted",
                        "request_id": request_id,
                        "retry_count": rate_config.max_retries,
                        "duration_ms": duration_ms,
                    }
                )
                
                raise RateLimitException(
                    RateLimitErrorResponse(
                        error="RATE_LIMITED",
                        retry_after=retry_after,
                        action="failed",
                        provider=config.name,
                        model=config.model,
                        message=str(last_error),
                    )
                )
            
            logger.error(
                f"❌ Anthropic request failed | id={request_id} | "
                f"retries={rate_config.max_retries} | error={str(last_error) if last_error else 'Unknown'}",
                extra={
                    "event": "anthropic_request_failed",
                    "request_id": request_id,
                    "retry_count": rate_config.max_retries,
                    "error": str(last_error) if last_error else "Unknown",
                }
            )
            
            if last_error:
                raise last_error
            raise RuntimeError("Request failed after all retries")
            
        finally:
            limiter.release()
    
    def supports_streaming(self) -> bool:
        """Anthropic SDK supports streaming."""
        return True
    
    def chat_stream(
        self,
        config: LLMProviderConfig,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Stream a chat response from Anthropic Claude.
        
        Yields events:
        - {"type": "token", "content": "..."} - Text tokens
        - {"type": "tool_call", "id": ..., "name": ..., "arguments": ...}
        - {"type": "done", "response": ChatResponse}
        - {"type": "error", "error": "..."}
        """
        client = self._get_client(config)
        if not client:
            yield {"type": "error", "error": "Anthropic client not available"}
            return
        
        # Extract system message if present
        system_msg = None
        chat_messages = []
        for msg in messages:
            if msg.get("role") == "system":
                system_msg = msg.get("content", "")
            else:
                chat_messages.append(msg)

        # Normalize OpenAI-style tool messages for Anthropic
        chat_messages = _normalize_messages_for_anthropic(chat_messages)
        
        if not config.model:
            raise ValueError("No model specified in Anthropic config")
        
        request_params: Dict[str, Any] = {
            "model": config.model,
            "max_tokens": config.max_tokens,
            "messages": chat_messages,
        }
        
        if system_msg:
            request_params["system"] = system_msg
        
        if tools and config.supports_tools:
            request_params["tools"] = _convert_tools_to_anthropic_format(tools)

        # Layer 3 of the overflow pipeline: server-side context compaction.
        # Same merge as the non-streaming chat() path, so long streaming
        # turns also get the compact-2026-01-12 behavior. Returns {} when
        # the feature is disabled.
        try:
            from agents.context.anthropic_compaction import build_kwargs as _compaction_kwargs
            request_params.update(_compaction_kwargs())
        except Exception as _exc:  # pragma: no cover - defensive
            logger.debug("anthropic streaming compaction kwargs unavailable: %s", _exc)

        try:
            # Accumulators
            full_content = ""
            tool_calls = []
            current_tool_call = None
            model_name = config.model
            input_tokens = 0
            output_tokens = 0
            finish_reason = None

            with client.messages.stream(**request_params) as stream:
                for event in stream:
                    # Message start - contains model info
                    if hasattr(event, 'type') and event.type == 'message_start':
                        if hasattr(event, 'message'):
                            model_name = getattr(event.message, 'model', model_name)
                            if hasattr(event.message, 'usage'):
                                input_tokens = getattr(event.message.usage, 'input_tokens', 0)
                    
                    # Content block start (text or tool_use)
                    elif hasattr(event, 'type') and event.type == 'content_block_start':
                        if hasattr(event, 'content_block'):
                            block = event.content_block
                            if hasattr(block, 'type') and block.type == 'tool_use':
                                current_tool_call = {
                                    "id": getattr(block, 'id', ''),
                                    "name": getattr(block, 'name', ''),
                                    "arguments": ""
                                }
                    
                    # Content block delta (streaming content)
                    elif hasattr(event, 'type') and event.type == 'content_block_delta':
                        if hasattr(event, 'delta'):
                            delta = event.delta
                            # Text delta
                            if hasattr(delta, 'text'):
                                full_content += delta.text
                                yield {"type": "token", "content": delta.text}
                            # Tool input delta (JSON being streamed)
                            elif hasattr(delta, 'partial_json') and current_tool_call:
                                current_tool_call["arguments"] += delta.partial_json
                    
                    # Content block stop — emit the finalized tool_call
                    # immediately so the orchestrator can start executing
                    # the tool while the stream is still live. Holding them
                    # until the stream finished (the old behavior) made
                    # tool_start / tool_end show up in the UI "after the
                    # fact" instead of in real time.
                    elif hasattr(event, 'type') and event.type == 'content_block_stop':
                        if current_tool_call:
                            tool_calls.append(current_tool_call)
                            yield {
                                "type": "tool_call",
                                "id": current_tool_call["id"],
                                "name": current_tool_call["name"],
                                "arguments": current_tool_call["arguments"],
                            }
                            current_tool_call = None

                    # Message delta (contains finish reason and output token count)
                    elif hasattr(event, 'type') and event.type == 'message_delta':
                        if hasattr(event, 'delta'):
                            finish_reason = getattr(event.delta, 'stop_reason', None)
                        if hasattr(event, 'usage'):
                            output_tokens = getattr(event.usage, 'output_tokens', 0)
            
            # Final response
            response = ChatResponse(
                content=full_content,
                provider=config.name,
                model=model_name,
                tool_calls=tool_calls if tool_calls else None,
                usage={
                    "prompt_tokens": input_tokens,
                    "completion_tokens": output_tokens,
                },
                finish_reason=finish_reason
            )
            yield {"type": "done", "response": response}
            
        except Exception as e:
            logger.error(f"Anthropic streaming error: {e}")
            yield {"type": "error", "error": str(e)}
