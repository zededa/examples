"""
Google Adapter - Gemini API support with tool calling

Supports Gemini models via the Google Generative AI SDK.
Includes streaming support for real-time token delivery.
Includes production-grade rate limit handling and resilience.
https://ai.google.dev/
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


def _normalize_usage(usage_data: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Normalize usage data to consistent keys across all adapters.
    
    Standard keys:
    - prompt_tokens: Number of input tokens
    - completion_tokens: Number of output tokens  
    - total_tokens: Sum of prompt + completion
    """
    if not usage_data:
        return None
    
    prompt_tokens = usage_data.get("prompt_tokens", 0) or 0
    completion_tokens = usage_data.get("completion_tokens", 0) or 0
    
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


class GoogleAdapter(LLMAdapter):
    """Adapter for Google Gemini API with tool calling support."""
    
    # Class-level cache for models list
    _models_cache: List[str] = []
    _models_cache_time: float = 0
    _models_cache_ttl: float = 300  # 5 minutes
    
    def _get_client(self, config: LLMProviderConfig):
        """Get Google genai client."""
        try:
            from google import genai
            api_key = config.api_key or os.environ.get("GOOGLE_API_KEY")
            if not api_key:
                return None
            return genai.Client(api_key=api_key)
        except ImportError:
            # Try legacy import
            try:
                import google.generativeai as genai
                api_key = config.api_key or os.environ.get("GOOGLE_API_KEY")
                if not api_key:
                    return None
                genai.configure(api_key=api_key)
                return genai
            except ImportError:
                return None
        except Exception as e:
            logger.error(f"Google client error: {e}")
            return None
    
    def check_availability(self, config: LLMProviderConfig) -> Tuple[bool, float, Optional[str]]:
        api_key = config.api_key or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            return False, 0.0, "Google API key not set"
        
        # Just verify the API key is present - don't make API calls on health check
        # API key format validation is minimal since Google keys vary
        if api_key and len(api_key) > 10:
            return True, 0.0, None
        return False, 0.0, "Invalid API key format"
    
    def list_models(self, config: LLMProviderConfig) -> List[str]:
        """Fetch available models from Google API with caching."""
        # Return cached models if still valid
        if GoogleAdapter._models_cache and (time.time() - GoogleAdapter._models_cache_time) < GoogleAdapter._models_cache_ttl:
            return GoogleAdapter._models_cache
        
        try:
            from google import genai
            api_key = config.api_key or os.environ.get("GOOGLE_API_KEY")
            client = genai.Client(api_key=api_key)
            models = client.models.list()
            model_list = [m.name for m in models if hasattr(m, 'name')]
            
            # Update cache
            GoogleAdapter._models_cache = model_list
            GoogleAdapter._models_cache_time = time.time()
            
            return model_list
        except ImportError:
            # Try legacy API
            try:
                import google.generativeai as genai
                api_key = config.api_key or os.environ.get("GOOGLE_API_KEY")
                if api_key:
                    genai.configure(api_key=api_key)
                models = genai.list_models()
                model_list = [m.name for m in models if "generateContent" in getattr(m, 'supported_generation_methods', [])]
                
                # Update cache
                GoogleAdapter._models_cache = model_list
                GoogleAdapter._models_cache_time = time.time()
                
                return model_list
            except Exception as e:
                logger.error(f"Google list_models error (legacy): {e}")
                return GoogleAdapter._models_cache
        except Exception as e:
            logger.error(f"Google list_models error: {e}")
            return GoogleAdapter._models_cache
    
    def _sanitize_schema_for_google(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize JSON schema for Google API compatibility."""
        if not isinstance(schema, dict):
            return schema
        
        result = {}
        for key, value in schema.items():
            if key == "type":
                # Google doesn't accept ["string", "null"] - extract non-null type
                if isinstance(value, list):
                    non_null_types = [t for t in value if t != "null"]
                    value = non_null_types[0] if non_null_types else "string"
                # Convert to uppercase for Google format
                if isinstance(value, str):
                    value = value.upper()
                result[key] = value
            elif key == "enum":
                # Filter out None values from enum
                if isinstance(value, list):
                    result[key] = [v for v in value if v is not None]
            elif key == "properties":
                # Recursively sanitize properties
                result[key] = {k: self._sanitize_schema_for_google(v) for k, v in value.items()}
            elif key == "items":
                # Recursively sanitize array items
                result[key] = self._sanitize_schema_for_google(value)
            else:
                result[key] = value
        
        return result
    
    def _convert_tools_to_google_format(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert OpenAI-style tools to Google function declarations format."""
        function_declarations = []
        
        for tool in tools:
            if tool.get("type") == "function":
                func = tool.get("function", {})
                params = func.get("parameters", {"type": "object", "properties": {}})
                # Sanitize the schema for Google API compatibility
                sanitized_params = self._sanitize_schema_for_google(params)
                function_declarations.append({
                    "name": func.get("name", ""),
                    "description": func.get("description", ""),
                    "parameters": sanitized_params,
                })
        
        return function_declarations
    
    def chat(
        self,
        config: LLMProviderConfig,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> ChatResponse:
        """
        Send a chat request to Google Gemini with automatic retry and rate limit handling.
        
        Features:
        - Automatic retry with exponential backoff on 429/5xx errors
        - Concurrency limiting to prevent request storms
        - Structured error responses for rate limits (RateLimitException)
        - Comprehensive logging for observability
        """
        api_key = config.api_key or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("Google API key not available")
        
        if not config.model:
            raise ValueError("No model specified in Google config. Set the model via GOOGLE_MODEL environment variable or configuration.")
        model_name = config.model
        
        # Resilience: retry with exponential backoff
        rate_config = get_rate_limit_config()
        limiter = get_concurrency_limiter()
        request_id = generate_request_id()
        
        # Log request start
        logger.info(
            f"🚀 Google request start | id={request_id} | model={model_name}",
            extra={
                "event": "google_request_start",
                "request_id": request_id,
                "model": model_name,
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
                    # Try new google.genai API first
                    try:
                        result = self._chat_new_api(api_key, model_name, config, messages, tools)
                    except ImportError:
                        logger.info("New google.genai not available, falling back to legacy API")
                        result = self._chat_legacy_api(api_key, model_name, config, messages, tools)
                    
                    # Success - normalize usage
                    duration_ms = (time.time() - start_time) * 1000
                    result.usage = _normalize_usage(result.usage)
                    
                    logger.info(
                        f"✅ Google request success | id={request_id} | "
                        f"duration={duration_ms:.0f}ms | retries={retry_count}",
                        extra={
                            "event": "google_request_success",
                            "request_id": request_id,
                            "duration_ms": duration_ms,
                            "retry_count": retry_count,
                            "prompt_tokens": result.usage.get("prompt_tokens") if result.usage else None,
                            "completion_tokens": result.usage.get("completion_tokens") if result.usage else None,
                        }
                    )
                    return result
                    
                except Exception as e:
                    last_error = e
                    error_str = str(e)
                    
                    # Check if this error is retryable
                    if not is_retryable_error(e):
                        logger.error(
                            f"❌ Google non-retryable error | id={request_id} | error={error_str}",
                            extra={
                                "event": "google_non_retryable_error",
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
                            f"⏳ Google rate limited | id={request_id} | "
                            f"attempt={attempt}/{rate_config.max_retries} | backoff={backoff:.2f}s",
                            extra={
                                "event": "google_rate_limited",
                                "request_id": request_id,
                                "attempt": attempt,
                                "backoff_seconds": backoff,
                                "retry_after_hint": retry_after_hint,
                            }
                        )
                    else:
                        logger.warning(
                            f"🔄 Google retry | id={request_id} | "
                            f"attempt={attempt}/{rate_config.max_retries} | backoff={backoff:.2f}s | error={error_str[:100]}",
                            extra={
                                "event": "google_retry",
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
                    f"❌ Google rate limit exhausted | id={request_id} | "
                    f"retries={rate_config.max_retries} | duration={duration_ms:.0f}ms",
                    extra={
                        "event": "google_rate_limit_exhausted",
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
                        model=model_name,
                        message=str(last_error),
                    )
                )
            
            logger.error(
                f"❌ Google request failed | id={request_id} | "
                f"retries={rate_config.max_retries} | error={str(last_error) if last_error else 'Unknown'}",
                extra={
                    "event": "google_request_failed",
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
    
    def _chat_new_api(
        self,
        api_key: str,
        model_name: str,
        config: LLMProviderConfig,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> ChatResponse:
        """Chat using the new google.genai API with tool support."""
        from google import genai
        from google.genai import types
        
        client = genai.Client(api_key=api_key)
        
        # Build contents from messages
        contents = []
        system_instruction = None
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "system":
                # Store system instruction separately
                system_instruction = content
            elif role == "user":
                contents.append(types.Content(
                    role="user",
                    parts=[types.Part(text=content)]
                ))
            elif role == "assistant":
                contents.append(types.Content(
                    role="model",
                    parts=[types.Part(text=content)]
                ))
            elif role == "tool":
                # Tool result - add as user message with function response
                tool_call_id = msg.get("tool_call_id", "")
                func_name = tool_call_id.split("_")[0] if "_" in tool_call_id else "function"
                contents.append(types.Content(
                    role="user",
                    parts=[types.Part(
                        function_response=types.FunctionResponse(
                            name=func_name,
                            response={"result": content}
                        )
                    )]
                ))
        
        # Build config
        gen_config = types.GenerateContentConfig(
            max_output_tokens=config.max_tokens,
            temperature=config.temperature,
        )
        
        # Add system instruction if present
        if system_instruction:
            gen_config.system_instruction = system_instruction
        
        # Add tools if present and supported
        if tools and config.supports_tools:
            function_declarations = self._convert_tools_to_google_format(tools)
            if function_declarations:
                gen_config.tools = [types.Tool(function_declarations=function_declarations)]
        
        # Generate response
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=contents,
                config=gen_config,
            )
        except Exception as e:
            logger.error(f"Google generate_content error: {e}")
            raise RuntimeError(f"Google Gemini error: {e}")
        
        # Extract response content and tool calls
        content = ""
        tool_calls = None
        
        if response.candidates and len(response.candidates) > 0:
            candidate = response.candidates[0]
            
            if candidate.content and candidate.content.parts:
                for part in candidate.content.parts:
                    # Check for function call
                    if hasattr(part, 'function_call') and part.function_call:
                        fc = part.function_call
                        if tool_calls is None:
                            tool_calls = []
                        
                        # Convert args to JSON string
                        args_dict = dict(fc.args) if fc.args else {}
                        tool_calls.append({
                            "id": f"{fc.name}_{int(time.time() * 1000)}",
                            "name": fc.name,
                            "arguments": json.dumps(args_dict),
                        })
                    
                    # Check for text content
                    elif hasattr(part, 'text') and part.text:
                        content += part.text
            
            # Check finish reason for issues
            finish_reason = getattr(candidate, 'finish_reason', None)
            if finish_reason:
                # Handle various finish reasons
                reason_value = finish_reason.value if hasattr(finish_reason, 'value') else finish_reason
                if reason_value not in [1, 2, 'STOP', 'MAX_TOKENS', 'TOOL_USE']:
                    reason_names = {
                        3: "SAFETY", 4: "RECITATION", 5: "OTHER",
                        'SAFETY': "SAFETY", 'RECITATION': "RECITATION",
                    }
                    reason_msg = reason_names.get(reason_value, str(reason_value))
                    if not content and not tool_calls:
                        content = f"[Response blocked: {reason_msg}]"
                        logger.warning(f"Google Gemini response blocked: {reason_msg}")
        
        # Get usage if available
        usage = None
        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            usage = {
                "prompt_tokens": getattr(response.usage_metadata, 'prompt_token_count', 0),
                "completion_tokens": getattr(response.usage_metadata, 'candidates_token_count', 0),
            }
        
        return ChatResponse(
            content=content,
            provider=config.name,
            model=model_name,
            tool_calls=tool_calls,
            usage=usage,
            finish_reason="tool_calls" if tool_calls else "stop",
        )
    
    def _chat_legacy_api(
        self,
        api_key: str,
        model_name: str,
        config: LLMProviderConfig,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> ChatResponse:
        """Chat using the legacy google.generativeai API."""
        import google.generativeai as genai
        
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        
        # Convert messages to Gemini format
        gemini_messages = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "system":
                gemini_messages.append({"role": "user", "parts": [f"System instruction: {content}"]})
                gemini_messages.append({"role": "model", "parts": ["Understood. I will follow these instructions."]})
            elif role == "user":
                gemini_messages.append({"role": "user", "parts": [content]})
            elif role == "assistant":
                gemini_messages.append({"role": "model", "parts": [content]})
        
        generation_config = {
            "max_output_tokens": config.max_tokens,
            "temperature": config.temperature,
        }
        
        # Generate without tools (legacy API tool support is limited)
        response = model.generate_content(
            gemini_messages,
            generation_config=generation_config,
        )
        
        # Extract content
        content = ""
        try:
            content = response.text or ""
        except Exception:
            if response.candidates and len(response.candidates) > 0:
                candidate = response.candidates[0]
                if candidate.content and candidate.content.parts:
                    for part in candidate.content.parts:
                        if hasattr(part, 'text') and part.text:
                            content += part.text
        
        return ChatResponse(
            content=content,
            provider=config.name,
            model=model_name,
            usage={
                "prompt_tokens": response.usage_metadata.prompt_token_count if hasattr(response, 'usage_metadata') else 0,
                "completion_tokens": response.usage_metadata.candidates_token_count if hasattr(response, 'usage_metadata') else 0,
            } if hasattr(response, 'usage_metadata') else None,
        )
    
    def supports_streaming(self) -> bool:
        """Google Gemini API supports streaming."""
        return True
    
    def chat_stream(
        self,
        config: LLMProviderConfig,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Stream a chat response from Google Gemini.
        
        Includes concurrency limiting around the entire stream lifecycle.
        
        Yields events:
        - {"type": "token", "content": "..."} - Text tokens
        - {"type": "tool_call", "id": ..., "name": ..., "arguments": ...}
        - {"type": "done", "response": ChatResponse}
        - {"type": "error", "error": "..."}
        """
        api_key = config.api_key or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            yield {"type": "error", "error": "Google API key not available"}
            return
        
        if not config.model:
            raise ValueError("No model specified in Google config. Set the model via GOOGLE_MODEL environment variable or configuration.")
        model_name = config.model
        
        # Acquire concurrency slot for the entire stream lifecycle
        rate_config = get_rate_limit_config()
        limiter = get_concurrency_limiter()
        request_id = generate_request_id()
        
        logger.info(
            f"🚀 Google streaming request | id={request_id} | model={model_name}",
            extra={
                "event": "google_stream_start",
                "request_id": request_id,
                "model": model_name,
                "provider": config.name,
            }
        )
        
        if not limiter.acquire(timeout=rate_config.request_timeout):
            yield {"type": "error", "error": f"Timed out waiting for concurrency slot after {rate_config.request_timeout}s"}
            return
        
        start_time = time.time()
        
        try:
            # Try new google.genai API first
            try:
                yield from self._chat_stream_new_api(api_key, model_name, config, messages, tools, request_id, start_time)
            except ImportError:
                logger.info("New google.genai not available, falling back to legacy streaming")
                yield from self._chat_stream_legacy_api(api_key, model_name, config, messages, tools, request_id, start_time)
        finally:
            # Always release the concurrency slot
            limiter.release()
    
    def _chat_stream_new_api(
        self,
        api_key: str,
        model_name: str,
        config: LLMProviderConfig,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        request_id: str = "",
        start_time: float = 0,
    ) -> Generator[Dict[str, Any], None, None]:
        """Stream using the new google.genai API."""
        from google import genai
        from google.genai import types
        
        client = genai.Client(api_key=api_key)
        
        # Build contents from messages
        contents = []
        system_instruction = None
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "system":
                system_instruction = content
            elif role == "user":
                contents.append(types.Content(
                    role="user",
                    parts=[types.Part(text=content)]
                ))
            elif role == "assistant":
                contents.append(types.Content(
                    role="model",
                    parts=[types.Part(text=content)]
                ))
            elif role == "tool":
                tool_call_id = msg.get("tool_call_id", "")
                func_name = tool_call_id.split("_")[0] if "_" in tool_call_id else "function"
                contents.append(types.Content(
                    role="user",
                    parts=[types.Part(
                        function_response=types.FunctionResponse(
                            name=func_name,
                            response={"result": content}
                        )
                    )]
                ))
        
        # Build config
        gen_config = types.GenerateContentConfig(
            max_output_tokens=config.max_tokens,
            temperature=config.temperature,
        )
        
        if system_instruction:
            gen_config.system_instruction = system_instruction
        
        if tools and config.supports_tools:
            function_declarations = self._convert_tools_to_google_format(tools)
            if function_declarations:
                gen_config.tools = [types.Tool(function_declarations=function_declarations)]
        
        try:
            # Accumulators
            full_content = ""
            tool_calls = []
            prompt_tokens = 0
            completion_tokens = 0
            
            # Stream the response
            stream = client.models.generate_content_stream(
                model=model_name,
                contents=contents,
                config=gen_config,
            )
            
            for chunk in stream:
                # Extract usage if available
                if hasattr(chunk, 'usage_metadata') and chunk.usage_metadata:
                    prompt_tokens = getattr(chunk.usage_metadata, 'prompt_token_count', prompt_tokens)
                    completion_tokens = getattr(chunk.usage_metadata, 'candidates_token_count', completion_tokens)
                
                if not chunk.candidates:
                    continue
                
                candidate = chunk.candidates[0]
                
                if candidate.content and candidate.content.parts:
                    for part in candidate.content.parts:
                        # Function call
                        if hasattr(part, 'function_call') and part.function_call:
                            fc = part.function_call
                            args_dict = dict(fc.args) if fc.args else {}
                            tc = {
                                "id": f"{fc.name}_{int(time.time() * 1000)}",
                                "name": fc.name,
                                "arguments": json.dumps(args_dict),
                            }
                            tool_calls.append(tc)
                        
                        # Text content
                        elif hasattr(part, 'text') and part.text:
                            full_content += part.text
                            yield {"type": "token", "content": part.text}
            
            # Yield tool calls
            for tc in tool_calls:
                yield {
                    "type": "tool_call",
                    "id": tc["id"],
                    "name": tc["name"],
                    "arguments": tc["arguments"]
                }
            
            # Log success
            duration_ms = (time.time() - start_time) * 1000 if start_time else 0
            logger.info(
                f"✅ Google stream complete | id={request_id} | duration={duration_ms:.0f}ms",
                extra={
                    "event": "google_stream_success",
                    "request_id": request_id,
                    "duration_ms": duration_ms,
                    "content_length": len(full_content),
                    "tool_calls_count": len(tool_calls),
                }
            )
            
            # Final response with normalized usage
            usage = _normalize_usage({
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            })
            
            final_response = ChatResponse(
                content=full_content,
                provider=config.name,
                model=model_name,
                tool_calls=tool_calls if tool_calls else None,
                usage=usage,
                finish_reason="tool_calls" if tool_calls else "stop",
            )
            yield {"type": "done", "response": final_response}
            
        except Exception as e:
            logger.error(
                f"❌ Google streaming error | id={request_id} | error={e}",
                extra={
                    "event": "google_stream_error",
                    "request_id": request_id,
                    "error": str(e),
                }
            )
            yield {"type": "error", "error": str(e)}
    
    def _chat_stream_legacy_api(
        self,
        api_key: str,
        model_name: str,
        config: LLMProviderConfig,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        request_id: str = "",
        start_time: float = 0,
    ) -> Generator[Dict[str, Any], None, None]:
        """Stream using the legacy google.generativeai API."""
        try:
            import google.generativeai as genai
            
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(model_name)
            
            # Convert messages to Gemini format
            gemini_messages = []
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                
                if role == "system":
                    gemini_messages.append({"role": "user", "parts": [f"System instruction: {content}"]})
                    gemini_messages.append({"role": "model", "parts": ["Understood. I will follow these instructions."]})
                elif role == "user":
                    gemini_messages.append({"role": "user", "parts": [content]})
                elif role == "assistant":
                    gemini_messages.append({"role": "model", "parts": [content]})
            
            generation_config = {
                "max_output_tokens": config.max_tokens,
                "temperature": config.temperature,
            }
            
            # Accumulators
            full_content = ""
            prompt_tokens = 0
            completion_tokens = 0
            
            # Stream the response
            response = model.generate_content(
                gemini_messages,
                generation_config=generation_config,
                stream=True,
            )
            
            for chunk in response:
                # Extract text from chunk
                try:
                    text = chunk.text
                    if text:
                        full_content += text
                        yield {"type": "token", "content": text}
                except Exception:
                    # Some chunks may not have text
                    pass
                
                # Extract usage if available
                if hasattr(chunk, 'usage_metadata'):
                    prompt_tokens = getattr(chunk.usage_metadata, 'prompt_token_count', prompt_tokens)
                    completion_tokens = getattr(chunk.usage_metadata, 'candidates_token_count', completion_tokens)
            
            # Log success
            duration_ms = (time.time() - start_time) * 1000 if start_time else 0
            logger.info(
                f"✅ Google legacy stream complete | id={request_id} | duration={duration_ms:.0f}ms",
                extra={
                    "event": "google_legacy_stream_success",
                    "request_id": request_id,
                    "duration_ms": duration_ms,
                    "content_length": len(full_content),
                }
            )
            
            # Final response with normalized usage
            usage = _normalize_usage({
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            })
            
            final_response = ChatResponse(
                content=full_content,
                provider=config.name,
                model=model_name,
                usage=usage,
            )
            yield {"type": "done", "response": final_response}
            
        except Exception as e:
            logger.error(
                f"❌ Google legacy streaming error | id={request_id} | error={e}",
                extra={
                    "event": "google_legacy_stream_error",
                    "request_id": request_id,
                    "error": str(e),
                }
            )
            yield {"type": "error", "error": str(e)}
