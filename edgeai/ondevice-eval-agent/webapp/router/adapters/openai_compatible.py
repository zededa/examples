"""
OpenAI-Compatible Adapter - Generic adapter for OpenAI API compatible servers

Works with LM Studio, LocalAI, Groq, and any other OpenAI-compatible API.
Supports both streaming and non-streaming responses.
Includes production-grade rate limit handling and resilience.
"""

import json
import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Generator, List, Optional, Tuple

from ..base import LLMAdapter
from ..config import LLMProviderConfig, ChatResponse, LLMProviderType
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


# =============================================================================
# Error Response for HTTP Details
# =============================================================================

@dataclass
class HTTPErrorDetails:
    """
    Captures HTTP error details for better error handling.
    
    Preserves status code, headers (especially Retry-After), and body
    for upstream error handling.
    """
    status_code: int
    headers: Dict[str, str]
    body: str
    retry_after: Optional[float] = None
    
    @classmethod
    def from_response(cls, response: Any) -> "HTTPErrorDetails":
        """Extract error details from an HTTP response object."""
        headers = dict(response.headers) if hasattr(response, "headers") else {}
        retry_after = None
        
        # Extract Retry-After header (can be seconds or HTTP date)
        if "Retry-After" in headers:
            try:
                retry_after = float(headers["Retry-After"])
            except (ValueError, TypeError):
                # Could be an HTTP date, default to reasonable backoff
                retry_after = 60.0
        
        return cls(
            status_code=response.status_code,
            headers=headers,
            body=response.text if hasattr(response, "text") else str(response),
            retry_after=retry_after,
        )

# Cloud provider base URLs
CLOUD_PROVIDER_URLS = {
    LLMProviderType.GROQ: "https://api.groq.com/openai/v1",
}


def _normalize_usage(usage_data: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Normalize usage data to consistent keys across all adapters.
    
    Standard keys:
    - prompt_tokens: Number of input tokens
    - completion_tokens: Number of output tokens  
    - total_tokens: Sum of prompt + completion (optional)
    
    Handles variations like 'input_tokens' vs 'prompt_tokens'.
    """
    if not usage_data:
        return None
    
    prompt_tokens = usage_data.get("prompt_tokens") or usage_data.get("input_tokens", 0)
    completion_tokens = usage_data.get("completion_tokens") or usage_data.get("output_tokens", 0)
    
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


class OpenAICompatibleAdapter(LLMAdapter):
    """Adapter for any OpenAI-compatible API (LM Studio, LocalAI, Groq, etc.)."""
    
    DEFAULT_URL = "http://localhost:1234"
    
    # Class-level cache for models list (keyed by base URL, protected by lock)
    _models_cache: Dict[str, List[str]] = {}
    _models_cache_time: Dict[str, float] = {}
    _models_cache_ttl: float = 300  # 5 minutes
    _models_cache_lock: threading.Lock = threading.Lock()
    
    def _get_base_url(self, config: LLMProviderConfig) -> str:
        """Get the base URL for the provider, handling cloud providers specially."""
        # Check if this is a cloud provider with a known URL
        if config.provider_type in CLOUD_PROVIDER_URLS:
            return CLOUD_PROVIDER_URLS[config.provider_type]
        
        # Otherwise use the configured URL or default
        return self._normalize_url(config.url or self.DEFAULT_URL)
    
    def _normalize_url(self, url: str) -> str:
        """Ensure URL has /v1 suffix when no path is present.

        Only appends /v1 to bare host:port URLs (e.g. http://localhost:1234).
        If the URL already contains a path (e.g. /api or /v2), it is left as-is
        so non-standard servers are not broken.
        """
        url = url.rstrip('/')
        if url.endswith('/v1'):
            return url
        # Parse the path; only add /v1 if the URL has no meaningful path
        from urllib.parse import urlparse
        parsed = urlparse(url)
        if not parsed.path or parsed.path == '/':
            url = f"{url}/v1"
        return url
    
    def check_availability(self, config: LLMProviderConfig) -> Tuple[bool, float, Optional[str]]:
        """Check if provider is available without making expensive API calls."""
        url = self._get_base_url(config)
        
        # For cloud providers (Groq), just verify API key is present
        if config.provider_type in CLOUD_PROVIDER_URLS:
            if config.api_key:
                return True, 0.0, None
            return False, 0.0, "API key not set"
        
        # For local servers, do a quick connectivity check
        try:
            start = time.time()
            headers = {}
            if config.api_key:
                headers["Authorization"] = f"Bearer {config.api_key}"
            
            # Just check if server responds, don't fetch full model list
            response = self._get_session().get(f"{url}/models", headers=headers, timeout=2)
            latency = (time.time() - start) * 1000
            
            if response.status_code == 200:
                return True, latency, None
            return False, latency, f"Status code: {response.status_code}"
        except Exception as e:
            return False, 0.0, str(e)
    
    def list_models(self, config: LLMProviderConfig) -> List[str]:
        """Fetch available models with caching."""
        url = self._get_base_url(config)
        cache_key = url

        # Return cached models if still valid
        with OpenAICompatibleAdapter._models_cache_lock:
            if cache_key in OpenAICompatibleAdapter._models_cache:
                cache_time = OpenAICompatibleAdapter._models_cache_time.get(cache_key, 0)
                if (time.time() - cache_time) < OpenAICompatibleAdapter._models_cache_ttl:
                    return list(OpenAICompatibleAdapter._models_cache[cache_key])

        try:
            headers = {}
            if config.api_key:
                headers["Authorization"] = f"Bearer {config.api_key}"

            logger.debug(f"Fetching models from {url}/models (api_key present: {bool(config.api_key)})")
            response = self._get_session().get(f"{url}/models", headers=headers, timeout=10)

            if response.status_code == 200:
                data = response.json()
                model_list = [m.get("id", "") for m in data.get("data", [])]

                # Update cache
                with OpenAICompatibleAdapter._models_cache_lock:
                    OpenAICompatibleAdapter._models_cache[cache_key] = model_list
                    OpenAICompatibleAdapter._models_cache_time[cache_key] = time.time()

                return model_list

            # Raise an error for non-200 responses so it's reported to the user
            error_text = response.text[:200] if response.text else "Unknown error"
            raise RuntimeError(f"HTTP {response.status_code}: {error_text}")

        except Exception as e:
            logger.error(f"OpenAI-compatible list_models error: {e}")
            raise  # Re-raise to let caller handle and report the error
    
    def chat(
        self,
        config: LLMProviderConfig,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> ChatResponse:
        """
        Send a chat request with automatic retry and rate limit handling.
        
        Features:
        - Automatic retry with exponential backoff on 429/5xx errors
        - Concurrency limiting to prevent request storms
        - Structured error responses for rate limits (RateLimitException)
        - Comprehensive logging for observability
        - HTTP error details preserved for Retry-After header
        """
        url = self._get_base_url(config)
        
        if not config.model:
            raise ValueError("No model specified in OpenAI-compatible config. Set the model via environment variable or configuration.")
        
        headers = {"Content-Type": "application/json"}
        if config.api_key:
            headers["Authorization"] = f"Bearer {config.api_key}"
        
        payload = {
            "model": config.model,
            "messages": messages,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
        }
        
        if tools and config.supports_tools:
            payload["tools"] = self._convert_tools_to_openai_format(tools)
        
        # Resilience: retry with exponential backoff
        rate_config = get_rate_limit_config()
        limiter = get_concurrency_limiter()
        request_id = generate_request_id()
        
        # Log request start
        logger.info(
            f"OpenAI-compatible request start | id={request_id} | model={config.model} | messages={len(messages)}",
            extra={
                "event": "openai_compatible_request_start",
                "request_id": request_id,
                "model": config.model,
                "provider": config.name,
                "message_count": len(messages),
            }
        )
        # Log message details at DEBUG level only
        if logger.isEnabledFor(logging.DEBUG):
            for i, msg in enumerate(messages):
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                preview = content[:50] + "..." if len(str(content)) > 50 else content
                logger.debug(f"Message {i}: role={role} preview={preview}")
        
        # Acquire concurrency slot
        if not limiter.acquire(timeout=rate_config.request_timeout):
            raise TimeoutError(
                f"Timed out waiting for concurrency slot after {rate_config.request_timeout}s"
            )
        
        start_time = time.time()
        last_error: Optional[Exception] = None
        last_http_details: Optional[HTTPErrorDetails] = None
        retry_count = 0
        
        try:
            for attempt in range(1, rate_config.max_retries + 1):
                try:
                    response = self._get_session().post(
                        f"{url}/chat/completions",
                        headers=headers,
                        json=payload,
                        timeout=config.timeout
                    )
                    
                    # Handle HTTP errors with preserved details
                    if response.status_code != 200:
                        last_http_details = HTTPErrorDetails.from_response(response)
                        error_msg = f"HTTP {response.status_code}: {response.text[:200]}"
                        
                        # Create exception with HTTP details for error classification
                        http_error = RuntimeError(error_msg)
                        http_error.status_code = response.status_code  # type: ignore
                        http_error.response = response  # type: ignore
                        raise http_error
                    
                    data = response.json()
                    # Guard against empty choices list from API
                    choices = data.get("choices", [])
                    if not choices:
                        raise RuntimeError("OpenAI-compatible API returned empty choices list")
                    choice = choices[0]
                    message = choice.get("message", {})
                    
                    tool_calls = None
                    if "tool_calls" in message:
                        tool_calls = [
                            {
                                "id": tc.get("id"),
                                "name": tc.get("function", {}).get("name"),
                                "arguments": tc.get("function", {}).get("arguments"),
                            }
                            for tc in message["tool_calls"]
                        ]
                    
                    # Success
                    duration_ms = (time.time() - start_time) * 1000
                    usage = _normalize_usage(data.get("usage"))
                    
                    logger.info(
                        f"OpenAI-compatible request success | id={request_id} | "
                        f"duration={duration_ms:.0f}ms | retries={retry_count}",
                        extra={
                            "event": "openai_compatible_request_success",
                            "request_id": request_id,
                            "duration_ms": duration_ms,
                            "retry_count": retry_count,
                            "prompt_tokens": usage.get("prompt_tokens") if usage else None,
                            "completion_tokens": usage.get("completion_tokens") if usage else None,
                        }
                    )
                    
                    return ChatResponse(
                        content=message.get("content", "") or "",
                        provider=config.name,
                        model=data.get("model", config.model or "unknown"),
                        tool_calls=tool_calls,
                        usage=usage,
                        finish_reason=choice.get("finish_reason")
                    )
                    
                except Exception as e:
                    last_error = e
                    error_str = str(e)
                    
                    # Check if this error is retryable
                    if not is_retryable_error(e):
                        logger.error(
                            f"OpenAI-compatible non-retryable error | id={request_id} | error={error_str}",
                            extra={
                                "event": "openai_compatible_non_retryable_error",
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
                    # Also check HTTP details for Retry-After
                    if last_http_details and last_http_details.retry_after:
                        retry_after_hint = last_http_details.retry_after
                    
                    backoff = calculate_backoff(attempt, rate_config, retry_after_hint)
                    retry_count = attempt
                    
                    if is_rate_limit_error(e):
                        logger.warning(
                            f"OpenAI-compatible rate limited | id={request_id} | "
                            f"attempt={attempt}/{rate_config.max_retries} | backoff={backoff:.2f}s",
                            extra={
                                "event": "openai_compatible_rate_limited",
                                "request_id": request_id,
                                "attempt": attempt,
                                "backoff_seconds": backoff,
                                "retry_after_hint": retry_after_hint,
                                "http_status": getattr(e, "status_code", None),
                            }
                        )
                    else:
                        logger.warning(
                            f"OpenAI-compatible retry | id={request_id} | "
                            f"attempt={attempt}/{rate_config.max_retries} | backoff={backoff:.2f}s",
                            extra={
                                "event": "openai_compatible_retry",
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
                # Extract retry_after from HTTP details or error
                retry_after = None
                if last_http_details:
                    retry_after = last_http_details.retry_after
                if not retry_after:
                    retry_after = extract_retry_after(last_error)
                
                logger.error(
                    f"OpenAI-compatible rate limit exhausted | id={request_id} | "
                    f"retries={rate_config.max_retries} | duration={duration_ms:.0f}ms",
                    extra={
                        "event": "openai_compatible_rate_limit_exhausted",
                        "request_id": request_id,
                        "retry_count": rate_config.max_retries,
                        "duration_ms": duration_ms,
                        "http_status": last_http_details.status_code if last_http_details else None,
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
                f"OpenAI-compatible request failed | id={request_id} | "
                f"retries={rate_config.max_retries} | error={str(last_error) if last_error else 'Unknown'}",
                extra={
                    "event": "openai_compatible_request_failed",
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
    
    def chat_stream(
        self,
        config: LLMProviderConfig,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Send a streaming chat request using Server-Sent Events (SSE).
        
        Works with LM Studio, Groq, and other OpenAI-compatible APIs that support streaming.
        Includes concurrency limiting around the entire stream lifecycle.
        
        Yields events:
        - {"type": "token", "content": "..."} - Text token/chunk
        - {"type": "tool_call", ...} - Tool call chunk
        - {"type": "done", "response": ChatResponse} - Final response
        - {"type": "error", "error": "..."} - Error occurred
        """
        url = self._get_base_url(config)
        
        if not config.model:
            yield {"type": "error", "error": "No model specified in OpenAI-compatible config. Set the model via environment variable or configuration."}
            return
        
        headers = {"Content-Type": "application/json"}
        if config.api_key:
            headers["Authorization"] = f"Bearer {config.api_key}"
        
        payload = {
            "model": config.model,
            "messages": messages,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "stream": True,  # Enable streaming
        }
        
        if tools and config.supports_tools:
            payload["tools"] = self._convert_tools_to_openai_format(tools)
        
        # Acquire concurrency slot for the entire stream lifecycle
        rate_config = get_rate_limit_config()
        limiter = get_concurrency_limiter()
        request_id = generate_request_id()
        
        logger.info(
            f"OpenAI-compatible streaming request | id={request_id} | model={config.model} | messages={len(messages)}",
            extra={
                "event": "openai_compatible_stream_start",
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
            response = self._get_session().post(
                f"{url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=config.timeout,
                stream=True  # Enable response streaming
            )
            
            if response.status_code != 200:
                error_details = HTTPErrorDetails.from_response(response)
                
                # Check if it's a rate limit error
                if response.status_code == 429:
                    logger.warning(
                        f"OpenAI-compatible stream rate limited | id={request_id} | "
                        f"retry_after={error_details.retry_after}",
                        extra={
                            "event": "openai_compatible_stream_rate_limited",
                            "request_id": request_id,
                            "http_status": response.status_code,
                            "retry_after": error_details.retry_after,
                        }
                    )
                    yield {
                        "type": "error",
                        "error": f"Rate limited (HTTP 429)",
                        "retry_after": error_details.retry_after,
                        "status_code": response.status_code,
                    }
                else:
                    yield {
                        "type": "error",
                        "error": f"HTTP {response.status_code}: {error_details.body[:200]}",
                        "status_code": response.status_code,
                    }
                return
            
            # Accumulators for building final response
            full_content = ""
            tool_calls_acc: Dict[int, Dict[str, Any]] = {}  # index -> tool call data
            model_name = config.model or "unknown"
            finish_reason = None
            usage = None
            
            # Parse SSE stream
            for line in response.iter_lines(decode_unicode=True):
                if not line:
                    continue
                
                # Handle SSE format: "data: {...}"
                if line.startswith("data: "):
                    data_str = line[6:]  # Remove "data: " prefix
                    
                    # Check for stream end
                    if data_str.strip() == "[DONE]":
                        break
                    
                    try:
                        data = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    # Mid-stream error event. The EdgeAI Agent proxy emits
                    # rate-limit errors as an SSE `data:` payload after the
                    # initial HTTP 200, since rate limiting only kicks in
                    # once token-bucket bookkeeping completes. We surface
                    # this as a structured error event so the UI can show
                    # the retry-after window instead of a generic failure.
                    if isinstance(data, dict) and "error" in data:
                        err = data["error"]
                        err_obj = err if isinstance(err, dict) else {"message": str(err)}
                        err_type = (err_obj.get("type") or err_obj.get("code") or "").lower()
                        retry_after_raw = err_obj.get("retry_after")
                        try:
                            retry_after = float(retry_after_raw) if retry_after_raw is not None else None
                        except (TypeError, ValueError):
                            retry_after = None

                        if err_type == "rate_limit_exceeded" or "rate_limit" in err_type:
                            logger.warning(
                                f"OpenAI-compatible stream rate_limit_exceeded | id={request_id} | "
                                f"retry_after={retry_after}",
                                extra={
                                    "event": "openai_compatible_stream_rate_limited",
                                    "request_id": request_id,
                                    "http_status": 429,
                                    "retry_after": retry_after,
                                },
                            )
                            yield {
                                "type": "error",
                                "error": err_obj.get("message", "Rate limit exceeded"),
                                "retry_after": retry_after,
                                "status_code": 429,
                                "error_code": "rate_limit_exceeded",
                            }
                            return

                        yield {
                            "type": "error",
                            "error": err_obj.get("message", "Stream error"),
                            "retry_after": retry_after,
                        }
                        return

                    # Extract model name
                    if "model" in data:
                        model_name = data["model"]
                    
                    # Extract usage (sometimes sent in final chunk)
                    if "usage" in data:
                        usage = _normalize_usage(data["usage"])
                    
                    # Process choices
                    choices = data.get("choices", [])
                    for choice in choices:
                        delta = choice.get("delta", {})
                        
                        # Check finish reason
                        if choice.get("finish_reason"):
                            finish_reason = choice["finish_reason"]
                        
                        # Handle content delta (text token)
                        if "content" in delta and delta["content"]:
                            content = delta["content"]
                            full_content += content
                            yield {"type": "token", "content": content}
                        
                        # Handle tool calls
                        if "tool_calls" in delta:
                            for tc in delta["tool_calls"]:
                                idx = tc.get("index", 0)
                                
                                # Initialize tool call entry
                                if idx not in tool_calls_acc:
                                    tool_calls_acc[idx] = {
                                        "id": tc.get("id", ""),
                                        "name": "",
                                        "arguments": ""
                                    }
                                
                                # Accumulate tool call data
                                if tc.get("id"):
                                    tool_calls_acc[idx]["id"] = tc["id"]
                                
                                func = tc.get("function", {})
                                if func.get("name"):
                                    tool_calls_acc[idx]["name"] = func["name"]
                                if func.get("arguments"):
                                    tool_calls_acc[idx]["arguments"] += func["arguments"]
            
            # Build final tool calls list
            final_tool_calls = None
            if tool_calls_acc:
                final_tool_calls = [
                    tool_calls_acc[idx] for idx in sorted(tool_calls_acc.keys())
                ]
                # Emit tool call events
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
                f"OpenAI-compatible stream complete | id={request_id} | duration={duration_ms:.0f}ms",
                extra={
                    "event": "openai_compatible_stream_success",
                    "request_id": request_id,
                    "duration_ms": duration_ms,
                    "content_length": len(full_content),
                    "tool_calls_count": len(final_tool_calls) if final_tool_calls else 0,
                }
            )
            
            # Emit final done event with complete response
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
                f"OpenAI-compatible streaming error | id={request_id} | error={e}",
                extra={
                    "event": "openai_compatible_stream_error",
                    "request_id": request_id,
                    "error": str(e),
                }
            )
            yield {"type": "error", "error": str(e)}
        finally:
            # Always release the concurrency slot
            limiter.release()
    
    def supports_streaming(self) -> bool:
        """OpenAI-compatible APIs support streaming."""
        return True
