"""
Centralized configuration.

Groups environment-driven settings into small dataclasses so components
can accept a typed config object instead of reading os.environ directly.

Usage:
    from config import load_settings
    settings = load_settings()
    if settings.langfuse.enabled:
        ...
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Set


def _env_bool(name: str, default: bool = False) -> bool:
    val = os.environ.get(name, "").strip().lower()
    if not val:
        return default
    return val in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


@dataclass
class AppSettings:
    """Flask app shell settings (previously hard-coded in app.py)."""
    title: str = field(default_factory=lambda: os.environ.get("APP_TITLE", "OnDevice Eval Agent"))
    description: str = field(default_factory=lambda: os.environ.get(
        "APP_DESCRIPTION",
        "ZEDEDA's ML Model Inference and Evaluation Interface Agent",
    ))
    logo_url: str = field(default_factory=lambda: os.environ.get("LOGO_URL", ""))
    primary_color: str = field(default_factory=lambda: os.environ.get("PRIMARY_COLOR", "#3498db"))
    upload_folder: str = field(default_factory=lambda: os.environ.get("UPLOAD_FOLDER", "/tmp/uploads/"))
    max_content_mb: int = field(default_factory=lambda: _env_int("MAX_CONTENT_MB", 16))
    allowed_extensions: Set[str] = field(default_factory=lambda: set(
        os.environ.get("ALLOWED_EXTENSIONS", "png,jpg,jpeg,gif,bmp,webp").split(",")
    ))
    max_log_entries: int = field(default_factory=lambda: _env_int("MAX_LOG_ENTRIES", 100))
    debug: bool = field(default_factory=lambda: _env_bool("FLASK_DEBUG", False))


@dataclass
class LangfuseSettings:
    """
    Langfuse Cloud tracing.

    Targets https://cloud.langfuse.com by default. Self-hosted Langfuse
    is out of scope for this deployment; override `host` only if the
    account is hosted elsewhere.

    All fields are optional; when `enabled=False` (the default) the
    TracingService is an inert shell with zero hot-path overhead.
    """
    enabled: bool = field(default_factory=lambda: _env_bool("LANGFUSE_ENABLED", False))
    public_key: str | None = field(default_factory=lambda: os.environ.get("LANGFUSE_PUBLIC_KEY") or None)
    secret_key: str | None = field(default_factory=lambda: os.environ.get("LANGFUSE_SECRET_KEY") or None)
    host: str = field(default_factory=lambda: os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com"))
    flush_on_response: bool = field(default_factory=lambda: _env_bool("LANGFUSE_FLUSH_ON_RESPONSE", True))
    # Extra tag appended to every trace so multi-deployment views can slice by box.
    deployment_tag: str | None = field(default_factory=lambda: os.environ.get("LANGFUSE_DEPLOYMENT_TAG") or None)


@dataclass
class OverflowSettings:
    """
    4-layer context overflow protection thresholds.

    All thresholds are token counts. Estimated via langchain-core's
    count_tokens_approximately (char-based heuristic, same complexity as
    len(s)/~4). Defaults are tuned for 1M-context Claude — other providers
    hit their own ceilings well before Layer 4 ever fires.

    Layers:
      1. Conversation summarization at `conversation_trigger_tokens`
      2. Tool-result summarization when total > `tool_context_threshold_tokens`
         AND an individual tool result > `tool_result_threshold_tokens`
      3. Anthropic server-side compaction at `anthropic_compaction_tokens`
      4. Hard trim ceiling at `hard_ceiling_tokens`
    """
    enabled: bool = field(default_factory=lambda: _env_bool("OVERFLOW_ENABLED", True))

    # Layer 1
    conversation_trigger_tokens: int = field(
        default_factory=lambda: _env_int("OVERFLOW_CONVERSATION_TRIGGER_TOKENS", 400_000)
    )
    keep_messages: int = field(default_factory=lambda: _env_int("OVERFLOW_KEEP_MESSAGES", 40))

    # Layer 2
    tool_context_threshold_tokens: int = field(
        default_factory=lambda: _env_int("OVERFLOW_TOOL_CONTEXT_THRESHOLD_TOKENS", 600_000)
    )
    tool_result_threshold_tokens: int = field(
        default_factory=lambda: _env_int("OVERFLOW_TOOL_RESULT_THRESHOLD_TOKENS", 10_000)
    )
    tool_summary_max_tokens: int = field(
        default_factory=lambda: _env_int("OVERFLOW_TOOL_SUMMARY_MAX_TOKENS", 500)
    )

    # Layer 3
    anthropic_compaction_enabled: bool = field(
        default_factory=lambda: _env_bool("OVERFLOW_ANTHROPIC_COMPACTION_ENABLED", True)
    )
    anthropic_compaction_tokens: int = field(
        default_factory=lambda: _env_int("OVERFLOW_ANTHROPIC_COMPACTION_TOKENS", 800_000)
    )

    # Layer 4
    hard_ceiling_tokens: int = field(
        default_factory=lambda: _env_int("OVERFLOW_HARD_CEILING_TOKENS", 900_000)
    )

    # Summarization model: leave None to use the active provider's default.
    summary_model: str | None = field(
        default_factory=lambda: os.environ.get("OVERFLOW_SUMMARY_MODEL") or None
    )


@dataclass
class ToolsSettings:
    """
    Tool-dispatch behavior.

    Parallel execution: when a single assistant turn emits multiple
    tool_calls, fan them out across a ThreadPoolExecutor instead of
    running them serially. The single biggest real-world speedup for
    multi-tool turns. Set TOOLS_PARALLEL_EXECUTION=false for the old
    serial path.
    """
    parallel_execution: bool = field(default_factory=lambda: _env_bool("TOOLS_PARALLEL_EXECUTION", True))
    max_parallel_tools: int = field(default_factory=lambda: _env_int("TOOLS_MAX_PARALLEL", 8))


@dataclass
class DeploymentSettings:
    """
    Per-Helm-release deployment behavior.

    The business-logic image is reused across many single-model Helm
    releases. These settings drive the features that make the agent
    *aware* of its deployment: auto-baseline, scheduled sanity evals,
    Prometheus metrics, drift detection.

    Storage root is shared with session storage; the deployment DB
    lives at `{SESSION_STORAGE_ROOT}/deployment/deployment.db`.
    """
    # Master switch. When off, bootstrap is a no-op and /metrics returns 404.
    enabled: bool = field(default_factory=lambda: _env_bool("DEPLOYMENT_ENABLED", True))

    # Auto-baseline on first boot
    auto_baseline: bool = field(default_factory=lambda: _env_bool("DEPLOYMENT_AUTO_BASELINE", True))
    baseline_iterations: int = field(default_factory=lambda: _env_int("DEPLOYMENT_BASELINE_ITERATIONS", 20))
    baseline_warmup: int = field(default_factory=lambda: _env_int("DEPLOYMENT_BASELINE_WARMUP", 3))
    # Seconds to wait for Triton to become ready before giving up (retried later).
    triton_ready_timeout_s: int = field(default_factory=lambda: _env_int("DEPLOYMENT_TRITON_READY_TIMEOUT_S", 120))
    # Optional real-image path mounted via ConfigMap/PVC. When absent we
    # generate a synthetic RGB image of the model's expected input shape.
    sample_image_path: str | None = field(default_factory=lambda: os.environ.get("DEPLOYMENT_SAMPLE_IMAGE_PATH") or None)

    # Scheduled sanity eval
    sanity_enabled: bool = field(default_factory=lambda: _env_bool("DEPLOYMENT_SANITY_ENABLED", True))
    sanity_interval_s: int = field(default_factory=lambda: _env_int("DEPLOYMENT_SANITY_INTERVAL_S", 600))
    sanity_iterations: int = field(default_factory=lambda: _env_int("DEPLOYMENT_SANITY_ITERATIONS", 5))
    # p95 ratio (current / baseline) above which a drift_event is recorded.
    drift_alert_threshold: float = field(default_factory=lambda: _env_float("DEPLOYMENT_DRIFT_ALERT_THRESHOLD", 1.30))

    # Deployment identity — set by Helm, surfaced in metrics/traces.
    model_name: str | None = field(default_factory=lambda: os.environ.get("MODEL_NAME") or None)
    mlflow_run_id: str | None = field(default_factory=lambda: os.environ.get("MLFLOW_RUN_ID") or None)
    deployment_id: str | None = field(default_factory=lambda: os.environ.get("DEPLOYMENT_ID") or None)


@dataclass
class Settings:
    """Top-level settings container."""
    app: AppSettings = field(default_factory=AppSettings)
    langfuse: LangfuseSettings = field(default_factory=LangfuseSettings)
    overflow: OverflowSettings = field(default_factory=OverflowSettings)
    tools: ToolsSettings = field(default_factory=ToolsSettings)
    deployment: DeploymentSettings = field(default_factory=DeploymentSettings)


_settings: Settings | None = None


def load_settings(reload: bool = False) -> Settings:
    """Load (or reload) process-wide settings from the environment."""
    global _settings
    if _settings is None or reload:
        _settings = Settings()
    return _settings


def get_settings() -> Settings:
    """Return the cached settings, loading on first access."""
    return load_settings()


__all__ = [
    "AppSettings",
    "LangfuseSettings",
    "OverflowSettings",
    "ToolsSettings",
    "DeploymentSettings",
    "Settings",
    "load_settings",
    "get_settings",
]
