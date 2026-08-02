"""
Tests for webapp/router/resilience.py and webapp/router/rate_limit_config.py.
"""

import time
import threading
from unittest.mock import MagicMock

import pytest

from router.rate_limit_config import (
    RateLimitConfig,
    get_rate_limit_config,
    is_rate_limit_error,
    is_retryable_error,
    extract_retry_after,
)
from router.resilience import (
    ConcurrencyLimiter,
    RequestDeduplicator,
    calculate_backoff,
    estimate_tokens,
)


# ============================================================================
# RateLimitConfig defaults and env overrides
# ============================================================================

class TestRateLimitConfigDefaults:

    def test_defaults(self, reset_rate_limit_config, clean_env):
        cfg = RateLimitConfig()
        assert cfg.max_retries == 5
        assert cfg.backoff_base == pytest.approx(2.0)
        assert cfg.backoff_max == pytest.approx(30.0)
        assert cfg.backoff_jitter == pytest.approx(0.5)

    def test_env_override_max_retries(self, reset_rate_limit_config, monkeypatch):
        monkeypatch.setenv("LLM_MAX_RETRIES", "10")
        cfg = RateLimitConfig()
        assert cfg.max_retries == 10


# ============================================================================
# is_rate_limit_error
# ============================================================================

class TestIsRateLimitError:

    def test_status_code_429(self, reset_rate_limit_config):
        exc = Exception("too many requests")
        exc.status_code = 429  # type: ignore[attr-defined]
        assert is_rate_limit_error(exc) is True

    def test_rate_limit_in_message(self, reset_rate_limit_config):
        exc = Exception("Request failed: rate limit exceeded")
        assert is_rate_limit_error(exc) is True

    def test_normal_exception(self, reset_rate_limit_config):
        exc = ValueError("bad value")
        assert is_rate_limit_error(exc) is False


# ============================================================================
# is_retryable_error
# ============================================================================

class TestIsRetryableError:

    def test_status_code_500_retryable(self, reset_rate_limit_config):
        exc = RuntimeError("server error")
        exc.status_code = 500  # type: ignore[attr-defined]
        assert is_retryable_error(exc) is True

    def test_status_code_401_not_retryable(self, reset_rate_limit_config):
        exc = RuntimeError("unauthorized")
        exc.status_code = 401  # type: ignore[attr-defined]
        assert is_retryable_error(exc) is False


# ============================================================================
# extract_retry_after
# ============================================================================

class TestExtractRetryAfter:

    def test_returns_float_when_present(self, reset_rate_limit_config):
        exc = Exception("Please retry_after=30 seconds")
        result = extract_retry_after(exc)
        assert result == pytest.approx(30.0)

    def test_returns_none_when_absent(self, reset_rate_limit_config):
        exc = Exception("generic error")
        assert extract_retry_after(exc) is None


# ============================================================================
# ConcurrencyLimiter
# ============================================================================

class TestConcurrencyLimiter:

    def test_acquire_release_cycle(self):
        limiter = ConcurrencyLimiter(max_concurrent=2)
        assert limiter.acquire(timeout=1) is True
        assert limiter.active_requests == 1
        limiter.release()
        assert limiter.active_requests == 0

    def test_max_concurrent_enforced(self):
        limiter = ConcurrencyLimiter(max_concurrent=1)
        assert limiter.acquire(timeout=1) is True
        # Second acquire should fail immediately with tiny timeout
        assert limiter.acquire(timeout=0.1) is False
        limiter.release()

    def test_context_manager(self):
        limiter = ConcurrencyLimiter(max_concurrent=2)
        with limiter:
            assert limiter.active_requests == 1
        assert limiter.active_requests == 0

    def test_stats_tracking(self):
        limiter = ConcurrencyLimiter(max_concurrent=5)
        limiter.acquire(timeout=1)
        limiter.release()
        stats = limiter.get_stats()
        assert stats["total_acquired"] == 1
        assert stats["max_concurrent"] == 5


# ============================================================================
# RequestDeduplicator
# ============================================================================

class TestRequestDeduplicator:

    def test_first_request_not_duplicate(self):
        dedup = RequestDeduplicator(window_seconds=5.0)
        is_dup, cached, _ = dedup.check_duplicate(
            [{"role": "user", "content": "hello"}], "model-a"
        )
        assert is_dup is False
        assert cached is None

    def test_identical_within_window_is_duplicate(self):
        dedup = RequestDeduplicator(window_seconds=60.0)
        msgs = [{"role": "user", "content": "hello"}]
        _, _, req_hash = dedup.check_duplicate(msgs, "model-a")
        # Cache a response
        dedup.cache_response(req_hash, "cached-result")
        is_dup, cached, _ = dedup.check_duplicate(msgs, "model-a")
        assert is_dup is True
        assert cached == "cached-result"

    def test_different_messages_not_duplicate(self):
        dedup = RequestDeduplicator(window_seconds=60.0)
        _, _, h1 = dedup.check_duplicate([{"role": "user", "content": "a"}], "m")
        dedup.cache_response(h1, "r1")
        is_dup, _, _ = dedup.check_duplicate([{"role": "user", "content": "b"}], "m")
        assert is_dup is False

    def test_expired_entry_not_duplicate(self):
        dedup = RequestDeduplicator(window_seconds=0.0)  # immediate expiry
        msgs = [{"role": "user", "content": "hello"}]
        _, _, req_hash = dedup.check_duplicate(msgs, "m")
        dedup.cache_response(req_hash, "old")
        # Even the smallest pause exceeds a 0-second window
        time.sleep(0.01)
        is_dup, _, _ = dedup.check_duplicate(msgs, "m")
        assert is_dup is False

    def test_cache_response_stores_and_returns(self):
        dedup = RequestDeduplicator(window_seconds=60.0)
        msgs = [{"role": "user", "content": "test"}]
        _, _, req_hash = dedup.check_duplicate(msgs, "m")
        dedup.cache_response(req_hash, {"answer": 42})
        is_dup, cached, _ = dedup.check_duplicate(msgs, "m")
        assert is_dup is True
        assert cached == {"answer": 42}


# ============================================================================
# calculate_backoff
# ============================================================================

class TestCalculateBackoff:

    def test_exponential_growth(self, reset_rate_limit_config, clean_env):
        cfg = RateLimitConfig()
        # Zero jitter for deterministic comparison
        cfg.backoff_jitter = 0.0
        b1 = calculate_backoff(1, config=cfg)
        b2 = calculate_backoff(2, config=cfg)
        assert b2 > b1

    def test_respects_max(self, reset_rate_limit_config, clean_env):
        cfg = RateLimitConfig()
        cfg.backoff_jitter = 0.0
        cfg.backoff_max = 5.0
        b = calculate_backoff(100, config=cfg)
        assert b <= 5.0

    def test_retry_after_hint_used_as_floor(self, reset_rate_limit_config, clean_env):
        cfg = RateLimitConfig()
        cfg.backoff_jitter = 0.0
        cfg.backoff_max = 999.0
        hint = 60.0
        b = calculate_backoff(1, config=cfg, retry_after_hint=hint)
        assert b >= hint


# ============================================================================
# estimate_tokens
# ============================================================================

class TestEstimateTokens:

    def test_empty_string_returns_zero(self):
        assert estimate_tokens("") == 0

    def test_non_empty_returns_positive(self):
        result = estimate_tokens("This is a test sentence with several words.")
        assert result > 0
        assert isinstance(result, int)
