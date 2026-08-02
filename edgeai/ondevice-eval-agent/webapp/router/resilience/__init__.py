"""
Resilient LLM client primitives.

Package layout:
    metrics.py      - RequestMetrics + RequestLogger + generate_request_id
    estimation.py   - token heuristics (shared with the overflow pipeline in PR 3)
    __init__.py     - concurrency limiter, request deduplication, backoff
                      calculation, make_resilient_request wrapper, stats,
                      and the ResilientLLMClient facade

Provides:
    1. Automatic retry with exponential backoff (2^attempt + jitter, max 30s)
    2. Concurrency limiting via semaphore
    3. Request deduplication for burst prevention
    4. Token estimation and prompt protection
    5. Structured error responses for rate limits
    6. Comprehensive observability logging

Usage:
    from router.resilience import ResilientLLMClient, RequestMetrics

    client = ResilientLLMClient(anthropic_client)
    response = client.chat(messages, model="claude-sonnet-4-6")

Thread Safety:
    All operations are thread-safe via semaphores and locks.
"""

import asyncio
import hashlib
import logging
import random
import threading
import time
import json
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Generic
from functools import wraps
import traceback

from ..rate_limit_config import (
    get_rate_limit_config,
    is_rate_limit_error,
    is_retryable_error,
    extract_retry_after,
    RateLimitConfig,
    RETRYABLE_STATUS_CODES,
)

from .metrics import (
    RequestMetrics,
    generate_request_id,
    RequestLogger,
    _request_logger,
)
from .estimation import (
    estimate_tokens,
    estimate_messages_tokens,
)

logger = logging.getLogger(__name__)

T = TypeVar('T')


# ============================================================================
# Concurrency Limiter
# ============================================================================

class ConcurrencyLimiter:
    """
    Limits concurrent LLM requests using a semaphore.
    
    Prevents request storms by allowing at most N concurrent requests.
    Additional requests wait in a queue.
    """
    
    def __init__(self, max_concurrent: int = 2):
        self._semaphore = threading.Semaphore(max_concurrent)
        self._max_concurrent = max_concurrent
        self._active_count = 0
        self._waiting_count = 0
        self._lock = threading.Lock()
        self._stats = {
            "total_acquired": 0,
            "total_waited": 0,
            "max_wait_time": 0.0,
        }
    
    def acquire(self, timeout: Optional[float] = None) -> bool:
        """
        Acquire a slot for making a request.
        
        Args:
            timeout: Maximum time to wait (None for infinite)
            
        Returns:
            True if acquired, False if timeout
        """
        with self._lock:
            self._waiting_count += 1
        
        start = time.time()
        acquired = self._semaphore.acquire(timeout=timeout)
        elapsed = time.time() - start
        
        with self._lock:
            self._waiting_count -= 1
            if acquired:
                self._active_count += 1
                self._stats["total_acquired"] += 1
                if elapsed > 0.01:  # Only count meaningful waits
                    self._stats["total_waited"] += 1
                    self._stats["max_wait_time"] = max(self._stats["max_wait_time"], elapsed)
        
        if acquired and elapsed > 0.1:
            logger.debug(f"Concurrency slot acquired after {elapsed:.2f}s wait")
        
        return acquired
    
    def release(self):
        """Release a slot after request completion."""
        with self._lock:
            self._active_count = max(0, self._active_count - 1)
        self._semaphore.release()
    
    @property
    def active_requests(self) -> int:
        """Number of currently active requests."""
        with self._lock:
            return self._active_count
    
    @property
    def waiting_requests(self) -> int:
        """Number of requests waiting for a slot."""
        with self._lock:
            return self._waiting_count
    
    def get_stats(self) -> Dict[str, Any]:
        """Get concurrency limiter statistics."""
        with self._lock:
            return {
                "max_concurrent": self._max_concurrent,
                "active_requests": self._active_count,
                "waiting_requests": self._waiting_count,
                **self._stats,
            }
    
    def __enter__(self):
        self.acquire()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
        return False


# Global concurrency limiter (lazy-initialized)
_concurrency_limiter: Optional[ConcurrencyLimiter] = None
_limiter_lock = threading.Lock()


def get_concurrency_limiter() -> ConcurrencyLimiter:
    """Get the global concurrency limiter."""
    global _concurrency_limiter
    if _concurrency_limiter is None:
        with _limiter_lock:
            if _concurrency_limiter is None:
                config = get_rate_limit_config()
                _concurrency_limiter = ConcurrencyLimiter(config.max_concurrency)
                logger.info(f"Initialized concurrency limiter with max_concurrent={config.max_concurrency}")
    return _concurrency_limiter


# ============================================================================
# Request Deduplication
# ============================================================================

class RequestDeduplicator:
    """
    Prevents duplicate requests within a time window.
    
    Uses LRU cache with TTL to detect and deduplicate identical prompts
    that fire repeatedly (e.g., from rapid user clicks or agent loops).
    """
    
    def __init__(self, window_seconds: float = 5.0, max_size: int = 100):
        self._cache: OrderedDict[str, Tuple[float, Any]] = OrderedDict()
        self._window = window_seconds
        self._max_size = max_size
        self._lock = threading.Lock()
        self._stats = {
            "total_requests": 0,
            "deduplicated": 0,
        }
    
    def _compute_hash(self, messages: List[Dict[str, Any]], model: str) -> str:
        """Compute a hash for request deduplication."""
        # Create a deterministic string representation
        key_data = json.dumps({
            "messages": messages,
            "model": model,
        }, sort_keys=True)
        return hashlib.sha256(key_data.encode()).hexdigest()
    
    def _cleanup_expired(self):
        """Remove expired entries from cache."""
        now = time.time()
        expired = []
        for key, (timestamp, _) in self._cache.items():
            if now - timestamp > self._window:
                expired.append(key)
            else:
                break  # OrderedDict is ordered by insertion time
        
        for key in expired:
            del self._cache[key]
    
    def check_duplicate(
        self, 
        messages: List[Dict[str, Any]], 
        model: str
    ) -> Tuple[bool, Optional[Any], str]:
        """
        Check if this request is a duplicate.
        
        Returns:
            Tuple of (is_duplicate, cached_response, request_hash)
        """
        request_hash = self._compute_hash(messages, model)
        
        with self._lock:
            self._stats["total_requests"] += 1
            self._cleanup_expired()
            
            if request_hash in self._cache:
                timestamp, response = self._cache[request_hash]
                if time.time() - timestamp <= self._window:
                    self._stats["deduplicated"] += 1
                    # Move to end (most recently used)
                    self._cache.move_to_end(request_hash)
                    return True, response, request_hash
        
        return False, None, request_hash
    
    def cache_response(self, request_hash: str, response: Any):
        """Cache a successful response for deduplication."""
        with self._lock:
            # Enforce max size
            while len(self._cache) >= self._max_size:
                self._cache.popitem(last=False)
            
            self._cache[request_hash] = (time.time(), response)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get deduplication statistics."""
        with self._lock:
            return {
                "cache_size": len(self._cache),
                "window_seconds": self._window,
                **self._stats,
                "dedup_rate": (
                    self._stats["deduplicated"] / self._stats["total_requests"]
                    if self._stats["total_requests"] > 0 else 0
                ),
            }


# Global deduplicator
_deduplicator: Optional[RequestDeduplicator] = None
_dedup_lock = threading.Lock()


def get_deduplicator() -> RequestDeduplicator:
    """Get the global request deduplicator."""
    global _deduplicator
    if _deduplicator is None:
        with _dedup_lock:
            if _deduplicator is None:
                config = get_rate_limit_config()
                _deduplicator = RequestDeduplicator(config.dedup_window_seconds)
                logger.info(f"Initialized request deduplicator with window={config.dedup_window_seconds}s")
    return _deduplicator


# ============================================================================
# Rate Limit Error Response
# ============================================================================

@dataclass
class RateLimitErrorResponse:
    """Structured error response for rate limit scenarios."""
    error: str = "RATE_LIMITED"
    retry_after: Optional[float] = None
    action: str = "retrying"  # retrying, queued, failed
    provider: Optional[str] = None
    model: Optional[str] = None
    message: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "error": self.error,
            "retry_after": self.retry_after,
            "action": self.action,
            "provider": self.provider,
            "model": self.model,
            "message": self.message,
        }


# ============================================================================
# Exponential Backoff Calculator
# ============================================================================

def calculate_backoff(
    attempt: int,
    config: Optional[RateLimitConfig] = None,
    retry_after_hint: Optional[float] = None
) -> float:
    """
    Calculate backoff duration using exponential backoff with jitter.
    
    Formula: min(base^attempt + jitter, max_backoff)
    
    If retry_after_hint is provided (from API response), use it as a floor.
    
    Args:
        attempt: The current retry attempt number (1-indexed)
        config: Rate limit configuration
        retry_after_hint: Optional hint from API response
        
    Returns:
        Backoff duration in seconds
    """
    if config is None:
        config = get_rate_limit_config()
    
    # Exponential backoff: 2^attempt
    backoff = config.backoff_base ** attempt
    
    # Add jitter: random value between 0 and jitter * backoff
    jitter = random.uniform(0, config.backoff_jitter * backoff)
    backoff += jitter
    
    # Respect retry_after hint if provided
    if retry_after_hint is not None:
        backoff = max(backoff, retry_after_hint)
    
    # Cap at maximum
    backoff = min(backoff, config.backoff_max)
    
    return backoff


# ============================================================================
# Resilient Request Wrapper
# ============================================================================

def with_retry(
    func: Callable[..., T],
    provider: str,
    model: str,
    config: Optional[RateLimitConfig] = None,
) -> Callable[..., T]:
    """
    Decorator/wrapper that adds retry logic to an LLM API call.
    
    Args:
        func: The function to wrap (should make the actual API call)
        provider: Provider name for logging
        model: Model name for logging
        config: Rate limit configuration
        
    Returns:
        Wrapped function with retry logic
    """
    if config is None:
        config = get_rate_limit_config()
    
    @wraps(func)
    def wrapper(*args, **kwargs) -> T:
        metrics = RequestMetrics(
            request_id=generate_request_id(),
            provider=provider,
            model=model,
            start_time=time.time(),
        )
        
        # Estimate tokens if messages are provided
        messages = kwargs.get('messages', args[0] if args else [])
        if messages:
            metrics.token_estimate = estimate_messages_tokens(messages, model)
        
        _request_logger.log_request_start(metrics)
        
        last_error: Optional[Exception] = None
        
        for attempt in range(1, config.max_retries + 1):
            try:
                result = func(*args, **kwargs)
                
                # Success!
                metrics.end_time = time.time()
                metrics.final_status = "success"
                metrics.retry_count = attempt - 1
                
                # Extract token usage if available
                if hasattr(result, 'usage'):
                    metrics.actual_tokens = {
                        "input": getattr(result.usage, 'input_tokens', 0),
                        "output": getattr(result.usage, 'output_tokens', 0),
                    }
                
                _request_logger.log_request_success(metrics)
                return result
                
            except Exception as e:
                last_error = e
                error_str = str(e)
                
                # Check if this error is retryable
                if not is_retryable_error(e):
                    metrics.end_time = time.time()
                    metrics.final_status = "failed"
                    metrics.error_message = error_str
                    _request_logger.log_request_failure(metrics)
                    raise
                
                # Check if rate limited specifically
                if is_rate_limit_error(e):
                    retry_after = extract_retry_after(e)
                    _request_logger.log_rate_limited(metrics, retry_after)
                
                # Check if we have retries left
                if attempt >= config.max_retries:
                    break
                
                # Calculate backoff
                retry_after_hint = extract_retry_after(e)
                backoff = calculate_backoff(attempt, config, retry_after_hint)
                metrics.backoff_durations.append(backoff)
                
                _request_logger.log_retry_attempt(metrics, attempt, backoff, error_str)
                
                # Wait before retry
                time.sleep(backoff)
        
        # All retries exhausted
        metrics.end_time = time.time()
        metrics.retry_count = config.max_retries
        metrics.final_status = "rate_limited" if is_rate_limit_error(last_error) else "failed"
        metrics.error_message = str(last_error) if last_error else "Unknown error"
        
        _request_logger.log_request_failure(metrics)
        
        if last_error:
            raise last_error
        raise RuntimeError("Request failed after all retries")
    
    return wrapper


def make_resilient_request(
    request_func: Callable[..., T],
    messages: List[Dict[str, Any]],
    provider: str,
    model: str,
    config: Optional[RateLimitConfig] = None,
    enable_dedup: bool = True,
    **kwargs
) -> T:
    """
    Execute an LLM request with full resilience features.
    
    Features:
    1. Concurrency limiting (prevents request storms)
    2. Request deduplication (prevents duplicate prompts)
    3. Token estimation and validation
    4. Automatic retry with exponential backoff
    5. Comprehensive logging
    
    Args:
        request_func: Function that makes the actual API call
        messages: Chat messages
        provider: Provider name
        model: Model name
        config: Rate limit configuration
        enable_dedup: Whether to check for duplicates
        **kwargs: Additional arguments passed to request_func
        
    Returns:
        API response
        
    Raises:
        Various API-specific exceptions on non-retryable errors
        RuntimeError if all retries exhausted
    """
    if config is None:
        config = get_rate_limit_config()
    
    request_id = generate_request_id()
    metrics = RequestMetrics(
        request_id=request_id,
        provider=provider,
        model=model,
        start_time=time.time(),
    )
    
    # Token estimation and validation
    token_estimate = estimate_messages_tokens(messages, model)
    metrics.token_estimate = token_estimate
    
    if token_estimate > config.max_prompt_tokens:
        if config.auto_truncate_prompts:
            # Truncate by removing older messages (keep system and recent)
            _request_logger.log_prompt_truncated(
                token_estimate,
                config.max_prompt_tokens,
                config.max_prompt_tokens
            )
            # Simple truncation: keep first (system) and last few messages
            if len(messages) > 3:
                messages = [messages[0]] + messages[-2:]
                token_estimate = estimate_messages_tokens(messages, model)
                metrics.token_estimate = token_estimate
        else:
            raise ValueError(
                f"Prompt exceeds maximum token limit "
                f"({token_estimate} > {config.max_prompt_tokens})"
            )
    
    # Deduplication check
    if enable_dedup and config.enable_deduplication:
        deduplicator = get_deduplicator()
        is_dup, cached_response, request_hash = deduplicator.check_duplicate(messages, model)
        
        if is_dup and cached_response is not None:
            metrics.was_deduplicated = True
            metrics.end_time = time.time()
            metrics.final_status = "deduplicated"
            _request_logger.log_deduplication(request_hash, metrics)
            return cached_response
    else:
        request_hash = None
    
    # Get prompt preview for logging
    prompt_preview = ""
    for msg in messages:
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str):
                prompt_preview = content
                break
    
    _request_logger.log_request_start(metrics, prompt_preview)
    
    # Acquire concurrency slot
    limiter = get_concurrency_limiter()
    
    if not limiter.acquire(timeout=config.request_timeout):
        metrics.end_time = time.time()
        metrics.final_status = "timeout"
        metrics.error_message = "Timed out waiting for concurrency slot"
        _request_logger.log_request_failure(metrics)
        raise TimeoutError(
            f"Timed out waiting for concurrency slot after {config.request_timeout}s"
        )
    
    try:
        # Execute with retry logic
        last_error: Optional[Exception] = None
        
        for attempt in range(1, config.max_retries + 1):
            try:
                result = request_func(messages=messages, **kwargs)
                
                # Success!
                metrics.end_time = time.time()
                metrics.final_status = "success"
                metrics.retry_count = attempt - 1
                
                # Extract token usage if available
                if hasattr(result, 'usage'):
                    metrics.actual_tokens = {
                        "input": getattr(result.usage, 'input_tokens', 
                                        getattr(result.usage, 'prompt_tokens', 0)),
                        "output": getattr(result.usage, 'output_tokens',
                                         getattr(result.usage, 'completion_tokens', 0)),
                    }
                
                # Cache for deduplication
                if request_hash and config.enable_deduplication:
                    get_deduplicator().cache_response(request_hash, result)
                
                _request_logger.log_request_success(metrics)
                return result
                
            except Exception as e:
                last_error = e
                error_str = str(e)
                
                # Check if this error is retryable
                if not is_retryable_error(e):
                    metrics.end_time = time.time()
                    metrics.final_status = "failed"
                    metrics.error_message = error_str
                    _request_logger.log_request_failure(metrics)
                    raise
                
                # Check if rate limited
                if is_rate_limit_error(e):
                    retry_after = extract_retry_after(e)
                    _request_logger.log_rate_limited(metrics, retry_after)
                
                # Check if we have retries left
                if attempt >= config.max_retries:
                    break
                
                # Calculate backoff
                retry_after_hint = extract_retry_after(e)
                backoff = calculate_backoff(attempt, config, retry_after_hint)
                metrics.backoff_durations.append(backoff)
                
                _request_logger.log_retry_attempt(metrics, attempt, backoff, error_str)
                
                # Wait before retry
                time.sleep(backoff)
        
        # All retries exhausted
        metrics.end_time = time.time()
        metrics.retry_count = config.max_retries
        
        if last_error and is_rate_limit_error(last_error):
            metrics.final_status = "rate_limited"
            retry_after = extract_retry_after(last_error)
            metrics.error_message = str(last_error)
            _request_logger.log_request_failure(metrics)
            
            # Return structured error for rate limits
            raise RateLimitException(
                RateLimitErrorResponse(
                    error="RATE_LIMITED",
                    retry_after=retry_after,
                    action="failed",
                    provider=provider,
                    model=model,
                    message=str(last_error),
                )
            )
        else:
            metrics.final_status = "failed"
            metrics.error_message = str(last_error) if last_error else "Unknown error"
            _request_logger.log_request_failure(metrics)
            
            if last_error:
                raise last_error
            raise RuntimeError("Request failed after all retries")
    
    finally:
        limiter.release()


class RateLimitException(Exception):
    """Exception raised when rate limits are exhausted."""
    
    def __init__(self, error_response: RateLimitErrorResponse):
        self.error_response = error_response
        super().__init__(error_response.message or "Rate limit exceeded")
    
    def to_dict(self) -> Dict[str, Any]:
        return self.error_response.to_dict()


# ============================================================================
# Statistics and Health
# ============================================================================

def get_resilience_stats() -> Dict[str, Any]:
    """Get statistics about rate limit handling and resilience."""
    return {
        "concurrency": get_concurrency_limiter().get_stats(),
        "deduplication": get_deduplicator().get_stats(),
        "config": get_rate_limit_config().to_dict(),
    }


def reset_resilience_stats():
    """Reset all resilience statistics (for testing)."""
    global _concurrency_limiter, _deduplicator
    with _limiter_lock:
        _concurrency_limiter = None
    with _dedup_lock:
        _deduplicator = None


# ============================================================================
# Resilient LLM Client Wrapper
# ============================================================================

class ResilientLLMClient:
    """
    A wrapper class that adds resilience features to any LLM client.
    
    This class wraps an existing LLM client (Anthropic, OpenAI, etc.) and adds:
    - Automatic retry with exponential backoff
    - Concurrency limiting
    - Request deduplication
    - Token estimation and protection
    - Structured logging
    
    Usage:
        import anthropic
        from router.resilience import ResilientLLMClient
        
        raw_client = anthropic.Anthropic(api_key="...")
        client = ResilientLLMClient(
            raw_client,
            provider="anthropic",
            model="claude-3-opus"
        )
        
        # Now use with automatic resilience
        response = client.messages_create(
            messages=[{"role": "user", "content": "Hello!"}],
            max_tokens=1024
        )
    """
    
    def __init__(
        self,
        client: Any,
        provider: str,
        model: str,
        config: Optional[RateLimitConfig] = None,
        enable_dedup: bool = True,
    ):
        """
        Initialize a resilient LLM client wrapper.
        
        Args:
            client: The underlying LLM client (anthropic.Anthropic, openai.OpenAI, etc.)
            provider: Provider name for logging
            model: Default model name
            config: Rate limit configuration (uses global if None)
            enable_dedup: Whether to enable request deduplication
        """
        self._client = client
        self._provider = provider
        self._model = model
        self._config = config or get_rate_limit_config()
        self._enable_dedup = enable_dedup
        self._request_count = 0
        self._error_count = 0
        self._lock = threading.Lock()
    
    @property
    def provider(self) -> str:
        return self._provider
    
    @property
    def model(self) -> str:
        return self._model
    
    @property
    def stats(self) -> Dict[str, Any]:
        """Get client statistics."""
        with self._lock:
            return {
                "provider": self._provider,
                "model": self._model,
                "request_count": self._request_count,
                "error_count": self._error_count,
            }
    
    def messages_create(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Any:
        """
        Create a chat completion with full resilience features.
        
        This is the main entry point for Anthropic-style APIs.
        
        Args:
            messages: List of chat messages
            model: Model override (uses default if None)
            max_tokens: Max output tokens (uses config default if None)
            **kwargs: Additional arguments passed to the client
            
        Returns:
            API response
            
        Raises:
            RateLimitException: If rate limits exhausted
            Various client-specific exceptions for non-retryable errors
        """
        model = model or self._model
        max_tokens = max_tokens or self._config.max_output_tokens
        
        with self._lock:
            self._request_count += 1
        
        try:
            return make_resilient_request(
                request_func=self._make_anthropic_request,
                messages=messages,
                provider=self._provider,
                model=model,
                config=self._config,
                enable_dedup=self._enable_dedup,
                model_param=model,
                max_tokens=max_tokens,
                **kwargs
            )
        except Exception as e:
            with self._lock:
                self._error_count += 1
            raise
    
    def _make_anthropic_request(
        self,
        messages: List[Dict[str, Any]],
        model_param: str,
        max_tokens: int,
        **kwargs
    ) -> Any:
        """Make the actual Anthropic API call."""
        return self._client.messages.create(
            model=model_param,
            max_tokens=max_tokens,
            messages=messages,
            **kwargs
        )
    
    def chat_completions_create(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Any:
        """
        Create a chat completion with full resilience features.
        
        This is the main entry point for OpenAI-style APIs.
        
        Args:
            messages: List of chat messages
            model: Model override (uses default if None)
            max_tokens: Max output tokens (uses config default if None)
            **kwargs: Additional arguments passed to the client
            
        Returns:
            API response
        """
        model = model or self._model
        max_tokens = max_tokens or self._config.max_output_tokens
        
        with self._lock:
            self._request_count += 1
        
        try:
            return make_resilient_request(
                request_func=self._make_openai_request,
                messages=messages,
                provider=self._provider,
                model=model,
                config=self._config,
                enable_dedup=self._enable_dedup,
                model_param=model,
                max_tokens=max_tokens,
                **kwargs
            )
        except Exception as e:
            with self._lock:
                self._error_count += 1
            raise
    
    def _make_openai_request(
        self,
        messages: List[Dict[str, Any]],
        model_param: str,
        max_tokens: int,
        **kwargs
    ) -> Any:
        """Make the actual OpenAI API call."""
        return self._client.chat.completions.create(
            model=model_param,
            max_tokens=max_tokens,
            messages=messages,
            **kwargs
        )
