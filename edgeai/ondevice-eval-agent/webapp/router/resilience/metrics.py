"""
Per-request metrics and structured logging for the resilient LLM client.

RequestMetrics captures the shape of a single LLM call (ids, durations,
retry count, token counts, final status). RequestLogger wraps a standard
logger and emits structured events with consistent field names so
observability consumers (Langfuse, log scrapers) can correlate.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..rate_limit_config import get_rate_limit_config

logger = logging.getLogger(__name__)


@dataclass
class RequestMetrics:
    """Metrics for a single LLM request for observability."""
    request_id: str
    provider: str
    model: str
    start_time: float
    end_time: Optional[float] = None
    token_estimate: Optional[int] = None
    actual_tokens: Optional[Dict[str, int]] = None
    retry_count: int = 0
    backoff_durations: List[float] = field(default_factory=list)
    final_status: str = "pending"  # pending, success, rate_limited, failed
    error_message: Optional[str] = None
    was_deduplicated: bool = False

    @property
    def duration_ms(self) -> Optional[float]:
        if self.end_time and self.start_time:
            return (self.end_time - self.start_time) * 1000
        return None

    def to_log_dict(self) -> Dict[str, Any]:
        """Convert to dictionary suitable for structured logging."""
        return {
            "request_id": self.request_id,
            "provider": self.provider,
            "model": self.model,
            "duration_ms": self.duration_ms,
            "token_estimate": self.token_estimate,
            "actual_tokens": self.actual_tokens,
            "retry_count": self.retry_count,
            "backoff_durations": self.backoff_durations,
            "total_backoff_seconds": sum(self.backoff_durations),
            "final_status": self.final_status,
            "error_message": self.error_message,
            "was_deduplicated": self.was_deduplicated,
        }


def generate_request_id() -> str:
    """Generate a unique request ID for tracking."""
    return f"llm-{uuid.uuid4().hex[:12]}"


class RequestLogger:
    """Structured logger for LLM request lifecycle events."""

    def __init__(self, logger_instance: logging.Logger):
        self._logger = logger_instance

    def log_request_start(self, metrics: RequestMetrics, prompt_preview: str = ""):
        self._logger.info(
            f"🚀 LLM request start | id={metrics.request_id} | "
            f"provider={metrics.provider} | model={metrics.model} | "
            f"token_estimate={metrics.token_estimate}",
            extra={
                "event": "llm_request_start",
                "request_id": metrics.request_id,
                "provider": metrics.provider,
                "model": metrics.model,
                "token_estimate": metrics.token_estimate,
                "prompt_preview": prompt_preview[:100] if prompt_preview else "",
            }
        )

    def log_retry_attempt(self, metrics: RequestMetrics, attempt: int, backoff: float, error: str):
        self._logger.warning(
            f"🔄 LLM retry | id={metrics.request_id} | "
            f"attempt={attempt}/{get_rate_limit_config().max_retries} | "
            f"backoff={backoff:.2f}s | error={error[:100]}",
            extra={
                "event": "llm_retry_attempt",
                "request_id": metrics.request_id,
                "attempt": attempt,
                "max_retries": get_rate_limit_config().max_retries,
                "backoff_seconds": backoff,
                "error": error,
            }
        )

    def log_rate_limited(self, metrics: RequestMetrics, retry_after: Optional[float] = None):
        self._logger.warning(
            f"⏳ LLM rate limited | id={metrics.request_id} | "
            f"provider={metrics.provider} | retry_after={retry_after}s",
            extra={
                "event": "llm_rate_limited",
                "request_id": metrics.request_id,
                "provider": metrics.provider,
                "model": metrics.model,
                "retry_after": retry_after,
            }
        )

    def log_request_success(self, metrics: RequestMetrics):
        self._logger.info(
            f"✅ LLM request success | id={metrics.request_id} | "
            f"duration={metrics.duration_ms:.0f}ms | retries={metrics.retry_count} | "
            f"tokens={metrics.actual_tokens}",
            extra={
                "event": "llm_request_success",
                **metrics.to_log_dict(),
            }
        )

    def log_request_failure(self, metrics: RequestMetrics):
        self._logger.error(
            f"❌ LLM request failed | id={metrics.request_id} | "
            f"status={metrics.final_status} | retries={metrics.retry_count} | "
            f"error={metrics.error_message}",
            extra={
                "event": "llm_request_failure",
                **metrics.to_log_dict(),
            }
        )

    def log_prompt_truncated(self, original_tokens: int, truncated_tokens: int, max_tokens: int):
        self._logger.warning(
            f"✂️ Prompt truncated | original={original_tokens} | "
            f"truncated_to={truncated_tokens} | max={max_tokens}",
            extra={
                "event": "prompt_truncated",
                "original_tokens": original_tokens,
                "truncated_tokens": truncated_tokens,
                "max_tokens": max_tokens,
            }
        )

    def log_deduplication(self, prompt_hash: str, metrics: RequestMetrics):
        self._logger.info(
            f"🔁 Request deduplicated | id={metrics.request_id} | hash={prompt_hash[:16]}",
            extra={
                "event": "request_deduplicated",
                "request_id": metrics.request_id,
                "prompt_hash": prompt_hash,
            }
        )


_request_logger = RequestLogger(logger)


__all__ = [
    "RequestMetrics",
    "generate_request_id",
    "RequestLogger",
    "_request_logger",
]
