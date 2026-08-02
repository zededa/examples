"""
LLM Inference Client.

Lightweight client for interfacing with LLM serving backends (vLLM, llama.cpp)
over the OpenAI-compatible API. Handles service discovery via environment
variables, health checking, model listing, inference, and performance metrics.

Both vLLM and llama.cpp expose OpenAI-compatible endpoints:
    - GET  /v1/models
    - POST /v1/chat/completions
    - POST /v1/completions
    - GET  /metrics  (Prometheus, vLLM only)

Service discovery mirrors the Triton pattern. URLs are resolved in order:
    OPENAI_API_BASE_URLS -> base URL injected by the on-device Helm chart
                            (plural; may be comma-separated; may carry a
                            trailing ``/v1`` path — stripped automatically)
    LLM_SERVER_URL       -> legacy single-URL fallback
    default              -> http://localhost:8000

    LLM_SERVER_TYPE      -> "vllm" or "llamacpp" (affects metrics parsing)
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from typing import Any, Dict, Final, List, Optional

import requests
from openai import OpenAI

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

_ENV_LLM_SERVER_URL: Final[str] = "LLM_SERVER_URL"
_ENV_OPENAI_API_BASE_URLS: Final[str] = "OPENAI_API_BASE_URLS"
_ENV_LLM_SERVER_TYPE: Final[str] = "LLM_SERVER_TYPE"

_DEFAULT_LLM_SERVER_URL: Final[str] = "http://localhost:8000"
_DEFAULT_TIMEOUT: Final[int] = 120


def _resolve_llm_base_url(explicit: Optional[str]) -> str:
    """
    Resolve the LLM server base URL from (in order): explicit arg →
    ``OPENAI_API_BASE_URLS`` → ``LLM_SERVER_URL`` → localhost default.

    ``OPENAI_API_BASE_URLS`` is injected by the on-device Helm chart in the
    OpenWebUI convention: it may be a single URL or a comma-separated list,
    and it commonly carries a trailing ``/v1`` path. The first entry is used
    and any trailing ``/v1`` is stripped so callers can unconditionally
    append OpenAI-style paths (e.g. ``/v1/models``, ``/v1/chat/completions``)
    without doubling the prefix.
    """
    raw = (
        explicit
        or os.environ.get(_ENV_OPENAI_API_BASE_URLS)
        or os.environ.get(_ENV_LLM_SERVER_URL)
        or _DEFAULT_LLM_SERVER_URL
    )
    first = raw.split(",")[0].strip().rstrip("/")
    if first.endswith("/v1"):
        first = first[:-3]
    return first


class LLMServerType(str, Enum):
    """Supported LLM serving backends."""
    VLLM = "vllm"
    LLAMACPP = "llamacpp"
    UNKNOWN = "unknown"


# =============================================================================
# Data Classes
# =============================================================================

@dataclass(frozen=True)
class LLMModelInfo:
    """Information about a served LLM model."""
    id: str
    created: Optional[int] = None
    owned_by: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMPerformanceMetrics:
    """Performance metrics for an LLM inference request."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    time_to_first_token_ms: Optional[float] = None
    total_time_ms: float = 0.0
    tokens_per_second: float = 0.0


@dataclass
class LLMServerMetrics:
    """Server-level metrics scraped from the Prometheus endpoint."""
    raw: Dict[str, float] = field(default_factory=dict)
    avg_generation_throughput_tps: Optional[float] = None
    avg_prompt_throughput_tps: Optional[float] = None
    running_requests: Optional[int] = None
    waiting_requests: Optional[int] = None
    gpu_cache_usage_pct: Optional[float] = None


# =============================================================================
# LLM Client
# =============================================================================

class LLMInferenceClient:
    """
    Client for LLM serving backends (vLLM, llama.cpp) over the
    OpenAI-compatible REST API.

    Thread Safety:
        The OpenAI SDK client is thread-safe. This class is safe for
        concurrent use from multiple threads.
    """

    __slots__ = ("_base_url", "_server_type", "_openai", "_timeout")

    def __init__(
        self,
        base_url: Optional[str] = None,
        server_type: Optional[str] = None,
        timeout: int = _DEFAULT_TIMEOUT,
    ) -> None:
        self._base_url = _resolve_llm_base_url(base_url)

        raw_type = (
            server_type
            or os.environ.get(_ENV_LLM_SERVER_TYPE, "")
        ).lower().strip()
        try:
            self._server_type = LLMServerType(raw_type)
        except ValueError:
            self._server_type = LLMServerType.UNKNOWN

        self._timeout = timeout

        self._openai = OpenAI(
            base_url=f"{self._base_url}/v1",
            api_key="not-needed",
            timeout=float(timeout),
        )

        logger.info(
            "LLMInferenceClient initialised: base_url=%s, server_type=%s",
            self._base_url,
            self._server_type.value,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def base_url(self) -> str:
        return self._base_url

    @property
    def server_type(self) -> LLMServerType:
        return self._server_type

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    def is_healthy(self) -> bool:
        """Return True if the LLM server is reachable."""
        try:
            resp = requests.get(
                f"{self._base_url}/v1/models", timeout=10
            )
            return resp.status_code == 200
        except Exception as exc:
            logger.debug("LLM health check failed: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Model Listing
    # ------------------------------------------------------------------

    def list_models(self) -> List[LLMModelInfo]:
        """List models served by the LLM backend."""
        try:
            response = self._openai.models.list()
            models: List[LLMModelInfo] = []
            for m in response.data:
                models.append(LLMModelInfo(
                    id=m.id,
                    created=getattr(m, "created", None),
                    owned_by=getattr(m, "owned_by", None),
                    raw=m.model_dump() if hasattr(m, "model_dump") else {},
                ))
            return models
        except Exception as exc:
            logger.error("Failed to list LLM models: %s", exc)
            raise

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def chat_completion(
        self,
        model: str,
        messages: List[Dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.7,
        stream: bool = False,
    ) -> Dict[str, Any]:
        """
        Send a chat completion request and return the result with timing.

        Returns a dict with keys: response, usage, performance.
        """
        t_start = time.perf_counter()

        completion = self._openai.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=False,
        )

        total_time = (time.perf_counter() - t_start) * 1000.0  # ms

        usage = completion.usage
        prompt_tokens = usage.prompt_tokens if usage else 0
        completion_tokens = usage.completion_tokens if usage else 0
        total_tokens = usage.total_tokens if usage else 0

        tokens_per_sec = (
            (completion_tokens / (total_time / 1000.0))
            if total_time > 0 and completion_tokens > 0
            else 0.0
        )

        response_text = ""
        if completion.choices:
            response_text = completion.choices[0].message.content or ""

        return {
            "response": response_text,
            "model": completion.model,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            },
            "performance": {
                "total_time_ms": round(total_time, 3),
                "tokens_per_second": round(tokens_per_sec, 2),
            },
            "finish_reason": (
                completion.choices[0].finish_reason
                if completion.choices
                else None
            ),
        }

    def chat_completion_streaming(
        self,
        model: str,
        messages: List[Dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> Dict[str, Any]:
        """
        Send a streaming chat completion to measure time-to-first-token.

        Returns the same dict shape as ``chat_completion()`` with an
        additional ``performance.time_to_first_token_ms`` field.  Token
        usage is best-effort — vLLM returns it via ``stream_options``
        while llama.cpp may not.
        """
        t_start = time.perf_counter()
        t_first_token: Optional[float] = None
        response_parts: List[str] = []
        prompt_tokens = 0
        completion_tokens = 0
        finish_reason: Optional[str] = None
        model_id: Optional[str] = None

        # Try with stream_options first (vLLM ≥0.4 supports this).
        # Fall back gracefully if the backend rejects the extra kwarg.
        stream_kwargs: Dict[str, Any] = dict(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
        )
        try:
            stream = self._openai.chat.completions.create(
                **stream_kwargs,
                stream_options={"include_usage": True},
            )
        except Exception:
            # Backend doesn't support stream_options — retry without.
            stream = self._openai.chat.completions.create(**stream_kwargs)

        for chunk in stream:
            # Record TTFT on the first chunk that carries content.
            if chunk.choices:
                delta_content = chunk.choices[0].delta.content
                if t_first_token is None and delta_content:
                    t_first_token = time.perf_counter()
                if delta_content:
                    response_parts.append(delta_content)
                if chunk.choices[0].finish_reason:
                    finish_reason = chunk.choices[0].finish_reason
            if chunk.model:
                model_id = chunk.model
            # vLLM sends usage in the final chunk when stream_options is set.
            if hasattr(chunk, "usage") and chunk.usage:
                prompt_tokens = chunk.usage.prompt_tokens or 0
                completion_tokens = chunk.usage.completion_tokens or 0

        total_time = (time.perf_counter() - t_start) * 1000.0
        ttft_ms = (
            (t_first_token - t_start) * 1000.0
            if t_first_token is not None
            else None
        )
        total_tokens = prompt_tokens + completion_tokens
        tokens_per_sec = (
            (completion_tokens / (total_time / 1000.0))
            if total_time > 0 and completion_tokens > 0
            else 0.0
        )

        return {
            "response": "".join(response_parts),
            "model": model_id or model,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            },
            "performance": {
                "total_time_ms": round(total_time, 3),
                "tokens_per_second": round(tokens_per_sec, 2),
                "time_to_first_token_ms": (
                    round(ttft_ms, 3) if ttft_ms is not None else None
                ),
            },
            "finish_reason": finish_reason,
        }

    def text_completion(
        self,
        model: str,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> Dict[str, Any]:
        """
        Send a text completion request (non-chat) and return the result
        with timing.
        """
        t_start = time.perf_counter()

        completion = self._openai.completions.create(
            model=model,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        total_time = (time.perf_counter() - t_start) * 1000.0

        usage = completion.usage
        prompt_tokens = usage.prompt_tokens if usage else 0
        completion_tokens = usage.completion_tokens if usage else 0
        total_tokens = usage.total_tokens if usage else 0

        tokens_per_sec = (
            (completion_tokens / (total_time / 1000.0))
            if total_time > 0 and completion_tokens > 0
            else 0.0
        )

        response_text = ""
        if completion.choices:
            response_text = completion.choices[0].text or ""

        return {
            "response": response_text,
            "model": completion.model,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            },
            "performance": {
                "total_time_ms": round(total_time, 3),
                "tokens_per_second": round(tokens_per_sec, 2),
            },
            "finish_reason": (
                completion.choices[0].finish_reason
                if completion.choices
                else None
            ),
        }

    # ------------------------------------------------------------------
    # Server-Level Metrics (Prometheus)
    # ------------------------------------------------------------------

    def get_server_metrics(self) -> Optional[LLMServerMetrics]:
        """
        Scrape Prometheus metrics from the LLM server.

        vLLM exposes metrics at GET /metrics. llama.cpp does not have a
        standard metrics endpoint, so this returns None for llama.cpp.
        """
        try:
            resp = requests.get(
                f"{self._base_url}/metrics", timeout=10
            )
            if resp.status_code != 200:
                logger.debug("Metrics endpoint returned %d", resp.status_code)
                return None

            return self._parse_prometheus_metrics(resp.text)
        except Exception as exc:
            logger.debug("Failed to fetch LLM server metrics: %s", exc)
            return None

    @staticmethod
    def _parse_prometheus_metrics(text: str) -> LLMServerMetrics:
        """Parse Prometheus text format into LLMServerMetrics."""
        raw: Dict[str, float] = {}

        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    raw[parts[0]] = float(parts[1])
                except ValueError:
                    continue

        # vLLM-specific gauge names
        metrics = LLMServerMetrics(raw=raw)
        metrics.avg_generation_throughput_tps = raw.get(
            "vllm:avg_generation_throughput_toks_per_s"
        )
        metrics.avg_prompt_throughput_tps = raw.get(
            "vllm:avg_prompt_throughput_toks_per_s"
        )
        metrics.running_requests = (
            int(raw["vllm:num_requests_running"])
            if "vllm:num_requests_running" in raw
            else None
        )
        metrics.waiting_requests = (
            int(raw["vllm:num_requests_waiting"])
            if "vllm:num_requests_waiting" in raw
            else None
        )
        metrics.gpu_cache_usage_pct = raw.get("vllm:gpu_cache_usage_perc")

        return metrics


# =============================================================================
# Singleton accessor (mirrors get_client() in mcp/base.py)
# =============================================================================

@lru_cache(maxsize=1)
def get_llm_client() -> LLMInferenceClient:
    """Get or create the shared LLMInferenceClient singleton."""
    return LLMInferenceClient()
