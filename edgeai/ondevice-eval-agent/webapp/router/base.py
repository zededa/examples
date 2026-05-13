"""
LLM Adapter Base Class

Abstract base class that all LLM provider adapters must implement.
Provides common functionality like session management with retries.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Generator, List, Optional, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .config import LLMProviderConfig, ChatResponse
from .rate_limit_config import get_rate_limit_config, RETRYABLE_STATUS_CODES


class LLMAdapter(ABC):
    """
    Base class for LLM provider adapters.
    
    All adapters must implement:
    - check_availability(): Test if the provider is reachable
    - list_models(): Get available models
    - chat(): Send a chat completion request
    
    Optional:
    - chat_stream(): Send a streaming chat completion request
    """
    
    def __init__(self):
        self._session: Optional[requests.Session] = None
    
    def _get_session(self) -> requests.Session:
        """Get or create a requests session with retry logic."""
        if self._session is None:
            self._session = requests.Session()
            # Use centralized rate limit configuration - never hardcode limits
            rate_config = get_rate_limit_config()
            # Disable HTTP-level retries for POST to avoid multiplying
            # retries with the explicit retry/backoff loops in each adapter.
            # Only idempotent methods are retried at the transport layer.
            retry_strategy = Retry(
                total=rate_config.max_retries,
                backoff_factor=rate_config.backoff_base,
                status_forcelist=list(RETRYABLE_STATUS_CODES),
                allowed_methods=["GET", "PUT", "DELETE", "HEAD", "OPTIONS"],
                respect_retry_after_header=True,
            )
            adapter = HTTPAdapter(max_retries=retry_strategy)
            self._session.mount('http://', adapter)
            self._session.mount('https://', adapter)
        return self._session
    
    @abstractmethod
    def check_availability(self, config: LLMProviderConfig) -> Tuple[bool, float, Optional[str]]:
        """
        Check if the provider is available.
        
        Args:
            config: Provider configuration
            
        Returns:
            Tuple of (available, latency_ms, error_message)
        """
        pass
    
    @abstractmethod
    def list_models(self, config: LLMProviderConfig) -> List[str]:
        """
        List available models from this provider.
        
        Args:
            config: Provider configuration
            
        Returns:
            List of model names/IDs
        """
        pass
    
    @abstractmethod
    def chat(
        self,
        config: LLMProviderConfig,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> ChatResponse:
        """
        Send a chat request to the LLM.
        
        Args:
            config: Provider configuration
            messages: List of chat messages [{"role": "user", "content": "..."}]
            tools: Optional list of tool schemas for function calling
            **kwargs: Additional provider-specific arguments
            
        Returns:
            ChatResponse with the LLM's response
            
        Raises:
            RuntimeError: If the request fails
        """
        pass
    
    def chat_stream(
        self,
        config: LLMProviderConfig,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Send a streaming chat request to the LLM.
        
        Yields SSE-style events with the following types:
        - {"type": "token", "content": "..."} - Text token (only for true streaming)
        - {"type": "tool_call", "id": "...", "name": "...", "arguments": "..."} - Tool call
        - {"type": "done", "response": ChatResponse} - Final response (streaming complete)
        - {"type": "complete", "response": str} - Non-streaming atomic response
        - {"type": "error", "error": "..."} - Error occurred
        
        Default implementation returns atomic response (no simulated streaming).
        Override in subclasses that support true streaming.
        
        Args:
            config: Provider configuration
            messages: List of chat messages
            tools: Optional list of tool schemas
            **kwargs: Additional arguments
            
        Yields:
            Dict events with streaming response data
        """
        # Default: return atomic response (no simulated streaming)
        # Subclasses that support true streaming should override this method
        try:
            response = self.chat(config, messages, tools, **kwargs)
            # Return complete response atomically - no token-by-token emission
            yield {
                "type": "complete",
                "response": response.content,
                "streaming": False,
                "full_response": response.to_dict()
            }
        except Exception as e:
            yield {"type": "error", "error": str(e)}
    
    def supports_streaming(self) -> bool:
        """Check if this adapter supports true streaming."""
        return False
    
    def _convert_tools_to_openai_format(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Convert tool schemas to OpenAI function calling format.
        
        This is the most common format, used by OpenAI, vLLM, TGI, and others.
        Handles both raw schemas and already-converted OpenAI format.
        """
        converted = []
        for tool in tools:
            # Check if already in OpenAI format
            if tool.get("type") == "function" and "function" in tool:
                converted.append(tool)
            else:
                # Convert from raw schema format
                converted.append({
                    "type": "function",
                    "function": {
                        "name": tool.get("name"),
                        "description": tool.get("description"),
                        "parameters": tool.get("input_schema", tool.get("parameters", {}))
                    }
                })
        return converted
