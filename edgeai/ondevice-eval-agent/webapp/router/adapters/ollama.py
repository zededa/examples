"""
Ollama Adapter - Local LLM server support

Ollama allows running open-source LLMs locally with a simple API.
Includes streaming support for real-time token delivery.
https://ollama.ai/
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


class OllamaAdapter(LLMAdapter):
    """Adapter for Ollama local LLM server."""
    
    DEFAULT_URL = "http://localhost:11434"
    
    def check_availability(self, config: LLMProviderConfig) -> Tuple[bool, float, Optional[str]]:
        try:
            url = config.url or self.DEFAULT_URL
            start = time.time()
            response = self._get_session().get(f"{url}/api/tags", timeout=5)
            latency = (time.time() - start) * 1000
            
            if response.status_code == 200:
                return True, latency, None
            return False, latency, f"Status code: {response.status_code}"
        except Exception as e:
            return False, 0.0, str(e)
    
    def list_models(self, config: LLMProviderConfig) -> List[str]:
        try:
            url = config.url or self.DEFAULT_URL
            response = self._get_session().get(f"{url}/api/tags", timeout=10)
            if response.status_code == 200:
                data = response.json()
                return [m.get("name", "") for m in data.get("models", [])]
            return []
        except Exception as e:
            logger.error(f"Ollama list_models error: {e}")
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
            raise ValueError("No model specified in Ollama config. Set the model via environment variable or configuration.")
        
        payload = {
            "model": config.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": config.temperature,
                "num_predict": config.max_tokens,
            }
        }
        
        # Ollama supports tools in newer versions
        if tools and config.supports_tools:
            payload["tools"] = self._convert_tools_to_ollama_format(tools)
        
        # Resilience: retry with exponential backoff
        rate_config = get_rate_limit_config()
        limiter = get_concurrency_limiter()
        request_id = generate_request_id()
        metrics = RequestMetrics(
            request_id=request_id,
            provider="ollama",
            model=config.model or "unknown",
            start_time=time.time(),
        )
        
        _request_logger.log_request_start(metrics)
        
        acquired = False
        try:
            if not limiter.acquire(timeout=rate_config.request_timeout):
                raise TimeoutError(f"Timed out waiting for concurrency slot after {rate_config.request_timeout}s")
            acquired = True
            
            last_error: Optional[Exception] = None
            
            for attempt in range(1, rate_config.max_retries + 1):
                try:
                    response = self._get_session().post(
                        f"{url}/api/chat",
                        json=payload,
                        timeout=config.timeout
                    )
                    
                    if response.status_code != 200:
                        raise RuntimeError(f"Ollama error: {response.status_code} - {response.text}")
                    
                    data = response.json()
                    message = data.get("message", {})
                    
                    # Handle tool calls
                    tool_calls = None
                    if "tool_calls" in message:
                        tool_calls = message["tool_calls"]
                    
                    # Success
                    metrics.end_time = time.time()
                    metrics.final_status = "success"
                    metrics.retry_count = attempt - 1
                    metrics.actual_tokens = {
                        "input": data.get("prompt_eval_count", 0),
                        "output": data.get("eval_count", 0),
                    }
                    _request_logger.log_request_success(metrics)
                    
                    return ChatResponse(
                        content=message.get("content", ""),
                        provider=config.name,
                        model=config.model or "unknown",
                        tool_calls=tool_calls,
                        usage={
                            "prompt_tokens": data.get("prompt_eval_count", 0),
                            "completion_tokens": data.get("eval_count", 0),
                        },
                        finish_reason=data.get("done_reason", "stop")
                    )
                    
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
                    # Release concurrency slot during backoff sleep so other
                    # requests can proceed while we wait.
                    limiter.release()
                    acquired = False
                    try:
                        time.sleep(backoff)
                    finally:
                        if not limiter.acquire(timeout=rate_config.request_timeout):
                            raise TimeoutError(
                                f"Timed out re-acquiring concurrency slot after backoff"
                            )
                        acquired = True

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
            if acquired:
                limiter.release()
    
    def _convert_tools_to_ollama_format(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert tool schemas to Ollama format."""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.get("name"),
                    "description": tool.get("description"),
                    "parameters": tool.get("input_schema", tool.get("parameters", {}))
                }
            }
            for tool in tools
        ]
    
    def supports_streaming(self) -> bool:
        """Ollama supports streaming by default."""
        return True
    
    def chat_stream(
        self,
        config: LLMProviderConfig,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Stream a chat response from Ollama.
        
        Ollama returns newline-delimited JSON objects.
        
        Yields events:
        - {"type": "token", "content": "..."} - Text tokens
        - {"type": "tool_call", "id": ..., "name": ..., "arguments": ...}
        - {"type": "done", "response": ChatResponse}
        - {"type": "error", "error": "..."}
        """
        url = config.url or self.DEFAULT_URL
        
        if not config.model:
            yield {"type": "error", "error": "No model specified in Ollama config. Set the model via environment variable or configuration."}
            return
        
        payload = {
            "model": config.model,
            "messages": messages,
            "stream": True,  # Enable streaming
            "options": {
                "temperature": config.temperature,
                "num_predict": config.max_tokens,
            }
        }
        
        # Ollama supports tools in newer versions
        if tools and config.supports_tools:
            payload["tools"] = self._convert_tools_to_ollama_format(tools)
        
        # Enforce concurrency limits consistent with chat()
        rate_config = get_rate_limit_config()
        limiter = get_concurrency_limiter()
        
        if not limiter.acquire(timeout=rate_config.request_timeout):
            yield {"type": "error", "error": f"Timed out waiting for concurrency slot after {rate_config.request_timeout}s"}
            return
        
        try:
            response = self._get_session().post(
                f"{url}/api/chat",
                json=payload,
                timeout=config.timeout,
                stream=True
            )
            
            try:
                if response.status_code != 200:
                    yield {"type": "error", "error": f"Ollama error: {response.status_code} - {response.text}"}
                    return
                
                # Accumulators
                full_content = ""
                tool_calls = None
                prompt_tokens = 0
                completion_tokens = 0
                finish_reason = "stop"
                
                for line in response.iter_lines():
                    if not line:
                        continue
                    
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    
                    message = data.get("message", {})
                    
                    # Content tokens
                    content = message.get("content", "")
                    if content:
                        full_content += content
                        yield {"type": "token", "content": content}
                    
                    # Tool calls (appear in final message)
                    if "tool_calls" in message:
                        tool_calls = message["tool_calls"]
                    
                    # Check if done
                    if data.get("done", False):
                        prompt_tokens = data.get("prompt_eval_count", 0)
                        completion_tokens = data.get("eval_count", 0)
                        finish_reason = data.get("done_reason", "stop")
                
                # Yield tool calls
                if tool_calls:
                    for tc in tool_calls:
                        func = tc.get("function", {})
                        yield {
                            "type": "tool_call",
                            "id": tc.get("id", ""),
                            "name": func.get("name", ""),
                            "arguments": json.dumps(func.get("arguments", {}))
                        }
                
                # Final response
                response_obj = ChatResponse(
                    content=full_content,
                    provider=config.name,
                    model=config.model or "unknown",
                    tool_calls=tool_calls,
                    usage={
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                    },
                    finish_reason=finish_reason
                )
                yield {"type": "done", "response": response_obj}
            finally:
                response.close()
            
        except Exception as e:
            logger.error(f"Ollama streaming error: {e}")
            yield {"type": "error", "error": str(e)}
        finally:
            limiter.release()
