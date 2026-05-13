"""
OpenAI Adapter - GPT models support

Supports GPT-4, GPT-4o, and other OpenAI models via the official SDK.
Includes streaming support for real-time token delivery.
Includes production-grade rate limit handling and resilience.
https://platform.openai.com/
"""

import json
import logging
import os
import time
from typing import Any, Dict, Generator, List, Optional, Tuple

from ..base import LLMAdapter
from ..config import LLMProviderConfig, ChatResponse
from ..rate_limit_config import (
    get_rate_limit_config,
    is_retryable_error,
    is_rate_limit_error,
    extract_retry_after,
)
from ..resilience import (
    calculate_backoff,
    get_concurrency_limiter,
    generate_request_id,
    RateLimitException,
    RateLimitErrorResponse,
)

logger = logging.getLogger(__name__)


def _normalize_usage(usage) -> Optional[Dict[str, Any]]:
    """
    Normalize usage data to consistent keys across all adapters.
    
    Standard keys:
    - prompt_tokens: Number of input tokens
    - completion_tokens: Number of output tokens  
    - total_tokens: Sum of prompt + completion
    """
    if not usage:
        return None
    
    prompt_tokens = getattr(usage, 'prompt_tokens', 0) or 0
    completion_tokens = getattr(usage, 'completion_tokens', 0) or 0
    
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


class OpenAIAdapter(LLMAdapter):
    """Adapter for OpenAI API."""
    
    # Class-level cache for models list
    _models_cache: List[str] = []
    _models_cache_time: float = 0
    _models_cache_ttl: float = 300  # 5 minutes
    
    def _get_client(self, config: LLMProviderConfig):
        """Get or create OpenAI client."""
        try:
            from openai import OpenAI
            api_key = config.api_key or os.environ.get("OPENAI_API_KEY")
            if not api_key:
                return None
            return OpenAI(api_key=api_key)
        except ImportError:
            return None
    
    def check_availability(self, config: LLMProviderConfig) -> Tuple[bool, float, Optional[str]]:
        client = self._get_client(config)
        if not client:
            return False, 0.0, "OpenAI SDK not installed or API key not set"
        
        # Just verify the API key format - don't make API calls on health check
        api_key = config.api_key or os.environ.get("OPENAI_API_KEY")
        if api_key and api_key.startswith("sk-"):
            return True, 0.0, None
        return False, 0.0, "Invalid API key format"
    
    def list_models(self, config: LLMProviderConfig) -> List[str]:
        """Fetch available models from OpenAI API with caching."""
        # Return cached models if still valid
        if OpenAIAdapter._models_cache and (time.time() - OpenAIAdapter._models_cache_time) < OpenAIAdapter._models_cache_ttl:
            return OpenAIAdapter._models_cache
        
        client = self._get_client(config)
        if not client:
            return OpenAIAdapter._models_cache  # Return stale cache if available
        
        try:
            models = client.models.list()
            model_list = [m.id for m in models.data if "gpt" in m.id.lower()]
            
            # Update cache
            OpenAIAdapter._models_cache = model_list
            OpenAIAdapter._models_cache_time = time.time()
            
            return model_list
        except Exception as e:
            logger.error(f"OpenAI list_models error: {e}")
            return OpenAIAdapter._models_cache  # Return stale cache on error
    
    def chat(
        self,
        config: LLMProviderConfig,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> ChatResponse:
        """
        Send a chat request to OpenAI with automatic retry and rate limit handling.
        
        Features:
        - Automatic retry with exponential backoff on 429/5xx errors
        - Concurrency limiting to prevent request storms
        - Structured error responses for rate limits (RateLimitException)
        - Comprehensive logging for observability
        """
        client = self._get_client(config)
        if not client:
            raise RuntimeError("OpenAI client not available")
        
        if not config.model:
            raise ValueError("No model specified in OpenAI config")
        
        request_params: Dict[str, Any] = {
            "model": config.model,
            "messages": messages,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
        }
        
        if tools and config.supports_tools:
            request_params["tools"] = self._convert_tools_to_openai_format(tools)
        
        # Resilience: retry with exponential backoff
        rate_config = get_rate_limit_config()
        limiter = get_concurrency_limiter()
        request_id = generate_request_id()
        
        # Log request start
        logger.info(
            f"🚀 OpenAI request start | id={request_id} | model={config.model}",
            extra={
                "event": "openai_request_start",
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
        last_error: Optional[Exception] = None
        retry_count = 0
        
        try:
            for attempt in range(1, rate_config.max_retries + 1):
                try:
                    response = client.chat.completions.create(**request_params)
                    choice = response.choices[0]
                    message = choice.message
                    
                    tool_calls = None
                    if message.tool_calls:
                        tool_calls = [
                            {
                                "id": tc.id,
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            }
                            for tc in message.tool_calls
                        ]
                    
                    # Success
                    duration_ms = (time.time() - start_time) * 1000
                    usage = _normalize_usage(response.usage)
                    
                    logger.info(
                        f"✅ OpenAI request success | id={request_id} | "
                        f"duration={duration_ms:.0f}ms | retries={retry_count}",
                        extra={
                            "event": "openai_request_success",
                            "request_id": request_id,
                            "duration_ms": duration_ms,
                            "retry_count": retry_count,
                            "prompt_tokens": usage.get("prompt_tokens") if usage else None,
                            "completion_tokens": usage.get("completion_tokens") if usage else None,
                        }
                    )
                    
                    return ChatResponse(
                        content=message.content or "",
                        provider=config.name,
                        model=response.model,
                        tool_calls=tool_calls,
                        usage=usage,
                        finish_reason=choice.finish_reason
                    )
                    
                except Exception as e:
                    last_error = e
                    error_str = str(e)
                    
                    # Check if this error is retryable
                    if not is_retryable_error(e):
                        logger.error(
                            f"❌ OpenAI non-retryable error | id={request_id} | error={error_str}",
                            extra={
                                "event": "openai_non_retryable_error",
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
                            f"⏳ OpenAI rate limited | id={request_id} | "
                            f"attempt={attempt}/{rate_config.max_retries} | backoff={backoff:.2f}s",
                            extra={
                                "event": "openai_rate_limited",
                                "request_id": request_id,
                                "attempt": attempt,
                                "backoff_seconds": backoff,
                                "retry_after_hint": retry_after_hint,
                            }
                        )
                    else:
                        logger.warning(
                            f"🔄 OpenAI retry | id={request_id} | "
                            f"attempt={attempt}/{rate_config.max_retries} | backoff={backoff:.2f}s | error={error_str[:100]}",
                            extra={
                                "event": "openai_retry",
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
                    f"❌ OpenAI rate limit exhausted | id={request_id} | "
                    f"retries={rate_config.max_retries} | duration={duration_ms:.0f}ms",
                    extra={
                        "event": "openai_rate_limit_exhausted",
                        "request_id": request_id,
                        "retry_count": rate_config.max_retries,
                        "duration_ms": duration_ms,
                    }
                )
                
                # Raise structured rate limit exception like Anthropic
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
                f"❌ OpenAI request failed | id={request_id} | "
                f"retries={rate_config.max_retries} | error={str(last_error) if last_error else 'Unknown'}",
                extra={
                    "event": "openai_request_failed",
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
        """OpenAI SDK supports streaming."""
        return True
    
    def chat_stream(
        self,
        config: LLMProviderConfig,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Stream a chat response from OpenAI.
        
        Includes concurrency limiting around the entire stream lifecycle.
        
        Yields events:
        - {"type": "token", "content": "..."} - Text tokens
        - {"type": "tool_call", "id": ..., "name": ..., "arguments": ...}
        - {"type": "done", "response": ChatResponse}
        - {"type": "error", "error": "..."}
        """
        client = self._get_client(config)
        if not client:
            yield {"type": "error", "error": "OpenAI client not available"}
            return
        
        if not config.model:
            yield {"type": "error", "error": "No model specified in OpenAI config"}
            return
        
        request_params: Dict[str, Any] = {
            "model": config.model,
            "messages": messages,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "stream": True,
            "stream_options": {"include_usage": True},  # Get usage in final chunk
        }
        
        if tools and config.supports_tools:
            request_params["tools"] = self._convert_tools_to_openai_format(tools)
        
        # Acquire concurrency slot for the entire stream lifecycle
        rate_config = get_rate_limit_config()
        limiter = get_concurrency_limiter()
        request_id = generate_request_id()
        
        logger.info(
            f"🚀 OpenAI streaming request | id={request_id} | model={config.model}",
            extra={
                "event": "openai_stream_start",
                "request_id": request_id,
                "model": config.model,
                "provider": config.name,
            }
        )
        
        if not limiter.acquire(timeout=rate_config.request_timeout):
            yield {"type": "error", "error": f"Timed out waiting for concurrency slot after {rate_config.request_timeout}s"}
            return
        
        start_time = time.time()
        
        try:
            stream = client.chat.completions.create(**request_params)
            
            # Accumulators
            full_content = ""
            tool_calls_accum: Dict[int, Dict[str, Any]] = {}  # index -> {id, name, arguments}
            model_name = config.model
            finish_reason = None
            usage = None
            
            for chunk in stream:
                # Check for usage in final chunk
                if hasattr(chunk, 'usage') and chunk.usage:
                    usage = _normalize_usage(chunk.usage)
                
                if not chunk.choices:
                    continue
                
                choice = chunk.choices[0]
                delta = choice.delta
                
                # Track finish reason
                if choice.finish_reason:
                    finish_reason = choice.finish_reason
                
                # Track model
                if hasattr(chunk, 'model') and chunk.model:
                    model_name = chunk.model
                
                # Content tokens
                if delta.content:
                    full_content += delta.content
                    yield {"type": "token", "content": delta.content}
                
                # Tool calls
                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        idx = tc.index
                        if idx not in tool_calls_accum:
                            tool_calls_accum[idx] = {
                                "id": tc.id or "",
                                "name": tc.function.name if tc.function and tc.function.name else "",
                                "arguments": ""
                            }
                        
                        # Accumulate tool call data
                        if tc.id:
                            tool_calls_accum[idx]["id"] = tc.id
                        if tc.function:
                            if tc.function.name:
                                tool_calls_accum[idx]["name"] = tc.function.name
                            if tc.function.arguments:
                                tool_calls_accum[idx]["arguments"] += tc.function.arguments
            
            # Build final tool calls list
            final_tool_calls = None
            if tool_calls_accum:
                final_tool_calls = [
                    tool_calls_accum[idx]
                    for idx in sorted(tool_calls_accum.keys())
                ]
                # Yield tool calls
                for tc in final_tool_calls:
                    yield {
                        "type": "tool_call",
                        "id": tc["id"],
                        "name": tc["name"],
                        "arguments": tc["arguments"]
                    }
            
            # Log success
            duration_ms = (time.time() - start_time) * 1000
            logger.info(
                f"✅ OpenAI stream complete | id={request_id} | duration={duration_ms:.0f}ms",
                extra={
                    "event": "openai_stream_success",
                    "request_id": request_id,
                    "duration_ms": duration_ms,
                    "content_length": len(full_content),
                    "tool_calls_count": len(final_tool_calls) if final_tool_calls else 0,
                }
            )
            
            # Final response
            final_response = ChatResponse(
                content=full_content,
                provider=config.name,
                model=model_name,
                tool_calls=final_tool_calls,
                usage=usage,
                finish_reason=finish_reason
            )
            yield {"type": "done", "response": final_response}
            
        except Exception as e:
            logger.error(
                f"❌ OpenAI streaming error | id={request_id} | error={e}",
                extra={
                    "event": "openai_stream_error",
                    "request_id": request_id,
                    "error": str(e),
                }
            )
            yield {"type": "error", "error": str(e)}
        finally:
            # Always release the concurrency slot
            limiter.release()
