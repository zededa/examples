"""
TGI Adapter - Hugging Face Text Generation Inference

TGI provides optimized inference for text generation models.
https://github.com/huggingface/text-generation-inference
"""

import json
import logging
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
    RequestMetrics,
    _request_logger,
)

logger = logging.getLogger(__name__)


class TGIAdapter(LLMAdapter):
    """Adapter for Hugging Face Text Generation Inference server."""
    
    DEFAULT_URL = "http://localhost:8080"
    
    def check_availability(self, config: LLMProviderConfig) -> Tuple[bool, float, Optional[str]]:
        try:
            url = config.url or self.DEFAULT_URL
            start = time.time()
            response = self._get_session().get(f"{url}/health", timeout=5)
            latency = (time.time() - start) * 1000
            
            if response.status_code == 200:
                return True, latency, None
            return False, latency, f"Status code: {response.status_code}"
        except Exception as e:
            return False, 0.0, str(e)
    
    def list_models(self, config: LLMProviderConfig) -> List[str]:
        try:
            url = config.url or self.DEFAULT_URL
            response = self._get_session().get(f"{url}/info", timeout=10)
            if response.status_code == 200:
                data = response.json()
                model_id = data.get("model_id", "unknown")
                return [model_id]
            return []
        except Exception as e:
            logger.error(f"TGI list_models error: {e}")
            return []
    
    def chat(
        self,
        config: LLMProviderConfig,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> ChatResponse:
        url = config.url or self.DEFAULT_URL
        
        if not config.model:
            raise ValueError("No model specified in TGI config. Set the model via environment variable or configuration.")
        
        headers = {"Content-Type": "application/json"}
        if config.api_key:
            headers["Authorization"] = f"Bearer {config.api_key}"
        
        payload = {
            "model": config.model or "tgi",
            "messages": messages,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "stream": False,
        }
        
        if tools and config.supports_tools:
            payload["tools"] = self._convert_tools_to_openai_format(tools)
        
        # Resilience: retry with exponential backoff
        rate_config = get_rate_limit_config()
        limiter = get_concurrency_limiter()
        request_id = generate_request_id()
        metrics = RequestMetrics(
            request_id=request_id,
            provider="tgi",
            model=config.model or "tgi",
            start_time=time.time(),
        )
        
        _request_logger.log_request_start(metrics)
        
        if not limiter.acquire(timeout=rate_config.request_timeout):
            raise TimeoutError(f"Timed out waiting for concurrency slot after {rate_config.request_timeout}s")
        
        try:
            last_error: Optional[Exception] = None
            
            for attempt in range(1, rate_config.max_retries + 1):
                try:
                    result = self._try_chat_request(url, headers, payload, config, messages)
                    
                    # Success
                    metrics.end_time = time.time()
                    metrics.final_status = "success"
                    metrics.retry_count = attempt - 1
                    if result.usage:
                        metrics.actual_tokens = {
                            "input": result.usage.get("prompt_tokens", 0),
                            "output": result.usage.get("completion_tokens", 0),
                        }
                    _request_logger.log_request_success(metrics)
                    return result
                    
                except Exception as e:
                    last_error = e
                    error_str = str(e)
                    
                    if not is_retryable_error(e):
                        metrics.end_time = time.time()
                        metrics.final_status = "failed"
                        metrics.error_message = error_str
                        _request_logger.log_request_failure(metrics)
                        raise
                    
                    if is_rate_limit_error(e):
                        retry_after = extract_retry_after(e)
                        _request_logger.log_rate_limited(metrics, retry_after)
                    
                    if attempt >= rate_config.max_retries:
                        break
                    
                    retry_after_hint = extract_retry_after(e)
                    backoff = calculate_backoff(attempt, rate_config, retry_after_hint)
                    metrics.backoff_durations.append(backoff)
                    
                    _request_logger.log_retry_attempt(metrics, attempt, backoff, error_str)
                    time.sleep(backoff)
            
            # All retries exhausted
            metrics.end_time = time.time()
            metrics.retry_count = rate_config.max_retries - 1
            metrics.final_status = "rate_limited" if is_rate_limit_error(last_error) else "failed"
            metrics.error_message = str(last_error) if last_error else "Unknown error"
            _request_logger.log_request_failure(metrics)
            
            if last_error:
                raise last_error
            raise RuntimeError("Request failed after all retries")
        finally:
            limiter.release()
    
    def _try_chat_request(
        self,
        url: str,
        headers: Dict[str, str],
        payload: Dict[str, Any],
        config: LLMProviderConfig,
        messages: List[Dict[str, Any]],
    ) -> ChatResponse:
        """Try chat completions endpoint first (TGI v2), then fall back to generate (TGI v1)."""
        # Try chat completions endpoint first (TGI v2)
        chat_endpoint_not_found = False
        
        try:
            response = self._get_session().post(
                f"{url}/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=config.timeout
            )
            
            if response.status_code == 200:
                data = response.json()
                # Guard against empty choices list from API
                choices = data.get("choices", [])
                if not choices:
                    raise RuntimeError("TGI returned empty choices list")
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
                
                return ChatResponse(
                    content=message.get("content", "") or "",
                    provider=config.name,
                    model=config.model or "tgi",
                    tool_calls=tool_calls,
                    usage=data.get("usage"),
                    finish_reason=choice.get("finish_reason")
                )
            elif response.status_code in (404, 405):
                # Endpoint not found/not supported - fall back to /generate
                chat_endpoint_not_found = True
            else:
                # Other errors (401, 403, 400, 500, etc.) should not fall back
                raise RuntimeError(
                    f"TGI chat completions error: {response.status_code} - {response.text}"
                )
        except RuntimeError:
            # Re-raise RuntimeError (our own errors) without falling back
            raise
        except Exception as e:
            # Connection errors or other transport issues - fall back to /generate
            logger.debug(f"TGI chat completions endpoint failed with {type(e).__name__}: {e}, falling back to /generate")
            chat_endpoint_not_found = True
        
        if not chat_endpoint_not_found:
            # Should not reach here, but guard against it
            raise RuntimeError("Unexpected state in _try_chat_request")
        
        # Fall back to generate endpoint (TGI v1)
        prompt = self._messages_to_prompt(messages)
        gen_payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": config.max_tokens,
                "temperature": config.temperature,
            }
        }
        
        response = self._get_session().post(
            f"{url}/generate",
            headers=headers,
            json=gen_payload,
            timeout=config.timeout
        )
        
        if response.status_code != 200:
            raise RuntimeError(f"TGI error: {response.status_code} - {response.text}")
        
        data = response.json()
        
        return ChatResponse(
            content=data.get("generated_text", ""),
            provider=config.name,
            model=config.model or "tgi",
        )
    
    def _messages_to_prompt(self, messages: List[Dict[str, Any]]) -> str:
        """Convert chat messages to a prompt string for TGI v1."""
        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
        prompt_parts.append("Assistant:")
        return "\n".join(prompt_parts)

    def supports_streaming(self) -> bool:
        """TGI supports streaming."""
        return True

    def chat_stream(
        self,
        config: LLMProviderConfig,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Stream a chat response from TGI.
        
        TGI supports OpenAI-compatible streaming via SSE.
        
        Yields events:
        - {"type": "token", "content": "..."} - Text tokens
        - {"type": "tool_call", "id": ..., "name": ..., "arguments": ...}
        - {"type": "done", "response": ChatResponse}
        - {"type": "error", "error": "..."}
        """
        url = config.url or self.DEFAULT_URL
        
        if not config.model:
            yield {"type": "error", "error": "No model specified in TGI config. Set the model via environment variable or configuration."}
            return
        
        headers = {"Content-Type": "application/json"}
        if config.api_key:
            headers["Authorization"] = f"Bearer {config.api_key}"
        
        payload = {
            "model": config.model or "tgi",
            "messages": messages,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "stream": True,
        }
        
        if tools and config.supports_tools:
            payload["tools"] = self._convert_tools_to_openai_format(tools)
        
        # Acquire concurrency slot
        rate_config = get_rate_limit_config()
        limiter = get_concurrency_limiter()
        request_id = generate_request_id()
        
        logger.info(
            f"TGI streaming request | id={request_id} | model={config.model}",
            extra={
                "event": "tgi_stream_start",
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
                f"{url}/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=config.timeout,
                stream=True
            )
            
            if response.status_code != 200:
                yield {"type": "error", "error": f"TGI error: {response.status_code} - {response.text}"}
                return
            
            # Accumulators
            full_content = ""
            tool_calls_acc: Dict[int, Dict[str, Any]] = {}
            model_name = config.model or "tgi"
            finish_reason = None
            usage = None
            
            # Parse SSE stream
            for line in response.iter_lines(decode_unicode=True):
                if not line:
                    continue
                
                if line.startswith("data: "):
                    data_str = line[6:]
                    
                    if data_str.strip() == "[DONE]":
                        break
                    
                    try:
                        data = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue
                    
                    if "model" in data:
                        model_name = data["model"]
                    
                    if "usage" in data:
                        usage = data["usage"]
                    
                    choices = data.get("choices", [])
                    for choice in choices:
                        delta = choice.get("delta", {})
                        
                        if choice.get("finish_reason"):
                            finish_reason = choice["finish_reason"]
                        
                        if "content" in delta and delta["content"]:
                            content = delta["content"]
                            full_content += content
                            yield {"type": "token", "content": content}
                        
                        if "tool_calls" in delta:
                            for tc in delta["tool_calls"]:
                                idx = tc.get("index", 0)
                                
                                if idx not in tool_calls_acc:
                                    tool_calls_acc[idx] = {
                                        "id": tc.get("id", ""),
                                        "name": "",
                                        "arguments": ""
                                    }
                                
                                if tc.get("id"):
                                    tool_calls_acc[idx]["id"] = tc["id"]
                                
                                func = tc.get("function", {})
                                if func.get("name"):
                                    tool_calls_acc[idx]["name"] = func["name"]
                                if func.get("arguments"):
                                    tool_calls_acc[idx]["arguments"] += func["arguments"]
            
            # Build final tool calls
            final_tool_calls = None
            if tool_calls_acc:
                final_tool_calls = [
                    tool_calls_acc[idx] for idx in sorted(tool_calls_acc.keys())
                ]
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
                f"TGI stream complete | id={request_id} | duration={duration_ms:.0f}ms",
                extra={
                    "event": "tgi_stream_success",
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
                f"TGI streaming error | id={request_id} | error={e}",
                extra={
                    "event": "tgi_stream_error",
                    "request_id": request_id,
                    "error": str(e),
                }
            )
            yield {"type": "error", "error": str(e)}
        finally:
            limiter.release()
