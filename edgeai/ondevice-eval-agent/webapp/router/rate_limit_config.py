"""
Rate Limit Configuration - Centralized configuration for LLM request resilience.

All rate limiting, retry, and concurrency settings are configurable via
environment variables or direct configuration.

This module provides:
- Retry configuration with exponential backoff
- Concurrency limits (max in-flight requests)
- Token/prompt protection limits
- Fallback behavior settings

NEVER hardcode limits - all values come from this configuration.
"""

import os
import re
import threading
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


@dataclass
class RateLimitConfig:
    """
    Configuration for rate limit handling and request resilience.
    
    All values can be overridden via environment variables.
    """
    
    # Retry settings
    max_retries: int = field(default_factory=lambda: int(os.environ.get('LLM_MAX_RETRIES', '5')))
    backoff_base: float = field(default_factory=lambda: float(os.environ.get('LLM_BACKOFF_BASE', '2.0')))
    backoff_max: float = field(default_factory=lambda: float(os.environ.get('LLM_BACKOFF_MAX', '30.0')))
    backoff_jitter: float = field(default_factory=lambda: float(os.environ.get('LLM_BACKOFF_JITTER', '0.5')))
    
    # Concurrency settings
    max_concurrency: int = field(default_factory=lambda: int(os.environ.get('LLM_MAX_CONCURRENCY', '2')))
    request_queue_size: int = field(default_factory=lambda: int(os.environ.get('LLM_REQUEST_QUEUE_SIZE', '100')))
    request_timeout: float = field(default_factory=lambda: float(os.environ.get('LLM_REQUEST_TIMEOUT', '120.0')))
    
    # Token protection settings
    max_prompt_tokens: int = field(default_factory=lambda: int(os.environ.get('LLM_MAX_PROMPT_TOKENS', '100000')))
    max_output_tokens: int = field(default_factory=lambda: int(os.environ.get('LLM_MAX_OUTPUT_TOKENS', '4096')))
    auto_truncate_prompts: bool = field(default_factory=lambda: os.environ.get('LLM_AUTO_TRUNCATE', 'true').lower() == 'true')
    
    # Deduplication settings
    enable_deduplication: bool = field(default_factory=lambda: os.environ.get('LLM_ENABLE_DEDUP', 'true').lower() == 'true')
    dedup_window_seconds: float = field(default_factory=lambda: float(os.environ.get('LLM_DEDUP_WINDOW', '5.0')))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for logging/serialization."""
        return {
            "max_retries": self.max_retries,
            "backoff_base": self.backoff_base,
            "backoff_max": self.backoff_max,
            "backoff_jitter": self.backoff_jitter,
            "max_concurrency": self.max_concurrency,
            "request_queue_size": self.request_queue_size,
            "request_timeout": self.request_timeout,
            "max_prompt_tokens": self.max_prompt_tokens,
            "max_output_tokens": self.max_output_tokens,
            "auto_truncate_prompts": self.auto_truncate_prompts,
            "enable_deduplication": self.enable_deduplication,
            "dedup_window_seconds": self.dedup_window_seconds,
        }
    
    def log_config(self):
        """Log current configuration for observability."""
        logger.info(
            "Rate limit configuration loaded",
            extra={"rate_limit_config": self.to_dict()}
        )


# HTTP status codes that should trigger retry
RETRYABLE_STATUS_CODES = {
    429,  # Rate limited
    500,  # Internal server error
    502,  # Bad gateway
    503,  # Service unavailable
    504,  # Gateway timeout
}

# HTTP status codes that should NEVER be retried
NON_RETRYABLE_STATUS_CODES = {
    400,  # Bad request
    401,  # Unauthorized
    403,  # Forbidden
    404,  # Not found
    422,  # Unprocessable entity
}

# Error types that indicate rate limiting
RATE_LIMIT_ERROR_PATTERNS = [
    "rate limit",
    "rate_limit",
    "too many requests",
    "quota exceeded",
    "requests per minute",
    "rpm",
    "tokens per minute",
    "tpm",
]


def _extract_status_code(error: Exception) -> Optional[int]:
    """Best-effort extraction of an HTTP status code from provider SDK errors."""
    for attr in ("status_code", "status", "http_status", "code"):
        value = getattr(error, attr, None)
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.isdigit():
            return int(value)
    response = getattr(error, "response", None)
    if response is not None:
        for attr in ("status_code", "status"):
            value = getattr(response, attr, None)
            if isinstance(value, int):
                return value
    return None


def is_rate_limit_error(error: Exception) -> bool:
    """Check if an exception is a rate limit error."""
    if error is None:
        return False
    status_code = _extract_status_code(error)
    if status_code == 429:
        return True
    class_name = error.__class__.__name__.lower()
    if "ratelimit" in class_name or "rate_limit" in class_name:
        return True
    error_str = str(error).lower()
    for pattern in RATE_LIMIT_ERROR_PATTERNS:
        if pattern in ("rpm", "tpm"):
            if re.search(rf"\b{pattern}\b", error_str):
                return True
        elif pattern in error_str:
            return True
    return False


def is_retryable_error(error: Exception) -> bool:
    """
    Determine if an error should be retried.
    
    Retryable:
    - Rate limit errors (429)
    - Transient server errors (5xx)
    - Connection errors
    - Timeout errors
    
    Non-retryable:
    - Authentication errors (401, 403)
    - Invalid request errors (400, 422)
    - Not found errors (404)
    """
    error_str = str(error).lower()
    status_code = _extract_status_code(error)
    if status_code is not None:
        if status_code in NON_RETRYABLE_STATUS_CODES:
            return False
        if status_code in RETRYABLE_STATUS_CODES:
            return True
    
    # Check for non-retryable patterns first
    non_retryable_patterns = [
        "authentication",
        "unauthorized",
        "forbidden",
        "invalid api key",
        "invalid_api_key",
        "api key invalid",
        "invalid request",
        "invalid_request_error",
        "malformed",
        "not found",
    ]
    
    if any(pattern in error_str for pattern in non_retryable_patterns):
        return False
    
    # Check for HTTP status codes in error message
    for code in NON_RETRYABLE_STATUS_CODES:
        if f"status {code}" in error_str or f"error {code}" in error_str:
            return False
    
    # Check for retryable patterns
    retryable_patterns = [
        "rate limit",
        "rate_limit",
        "too many requests",
        "429",
        "500",
        "502",
        "503",
        "504",
        "timeout",
        "timed out",
        "connection",
        "temporary",
        "overloaded",
        "capacity",
    ]
    
    return any(pattern in error_str for pattern in retryable_patterns)


def extract_retry_after(error: Exception) -> Optional[float]:
    """
    Extract retry-after value from error if present.
    
    Anthropic and other providers often include retry-after hints.
    """
    error_str = str(error)
    
    # Try to find retry_after in the error message
    
    # Pattern: retry_after=X or retry-after: X or "wait X seconds"
    patterns = [
        r'retry[_-]?after[:\s=]+(\d+(?:\.\d+)?)',
        r'wait\s+(\d+(?:\.\d+)?)\s*seconds?',
        r'try again in\s+(\d+(?:\.\d+)?)\s*seconds?',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, error_str, re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except (ValueError, IndexError):
                continue
    
    return None


# Global configuration instance
_config: Optional[RateLimitConfig] = None
_config_lock = threading.Lock()


def get_rate_limit_config() -> RateLimitConfig:
    """Get the global rate limit configuration (lazy-loaded, thread-safe singleton)."""
    global _config
    if _config is None:
        with _config_lock:
            # Double-check after acquiring lock
            if _config is None:
                _config = RateLimitConfig()
                _config.log_config()
    return _config


def reset_config():
    """Reset configuration to reload from environment (for testing)."""
    global _config
    with _config_lock:
        _config = None
