"""
Prometheus metrics for the eval-agent sidecar.

These are the per-deployment signals a central Prometheus should scrape
from every Helm release. Combined across the fleet they answer:

  - Which edge boxes are seeing drift?
  - What's p95 latency by model_name across the fleet?
  - Which models are currently ready on which hosts?

Every gauge carries `model_name` and `mlflow_run_id` labels so fleet
dashboards can slice by either. A separate `ondevice_eval_deployment_info`
constant-1 gauge with extra labels (`deployment_id`, `version`) acts as
a relabeling anchor à la `kube_pod_info`.

The module stays silent when `prometheus_client` is not installed — the
Flask `/metrics` endpoint just 404s in that case, which is fine.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# Lazy import of prometheus_client
# =============================================================================

try:
    from prometheus_client import CollectorRegistry, Gauge, generate_latest, CONTENT_TYPE_LATEST
    _PROM_AVAILABLE = True
except Exception:  # pragma: no cover — missing dep path
    _PROM_AVAILABLE = False
    CollectorRegistry = None  # type: ignore
    Gauge = None  # type: ignore
    generate_latest = None  # type: ignore
    CONTENT_TYPE_LATEST = "text/plain; version=0.0.4; charset=utf-8"


# =============================================================================
# Registry + metric handles
# =============================================================================

_registry = None
_metrics: Dict[str, Any] = {}
_labels: Dict[str, str] = {"model_name": "unknown", "mlflow_run_id": ""}


def _init_registry() -> None:
    """Create a dedicated registry so we don't clobber the default one."""
    global _registry, _metrics
    if not _PROM_AVAILABLE or _registry is not None:
        return
    _registry = CollectorRegistry()
    label_names = ("model_name", "mlflow_run_id")
    _metrics = {
        "info": Gauge(
            "ondevice_eval_deployment_info",
            "Per-deployment identity; constant 1 with labels for joining across metrics.",
            ("model_name", "mlflow_run_id", "deployment_id"),
            registry=_registry,
        ),
        "model_ready": Gauge(
            "ondevice_eval_model_ready",
            "1 when the single loaded model reports READY from Triton, else 0.",
            label_names, registry=_registry,
        ),
        "inference_p95_ms": Gauge(
            "ondevice_eval_inference_p95_ms",
            "p95 latency (ms) of the most recent sanity eval run.",
            label_names, registry=_registry,
        ),
        "inference_mean_ms": Gauge(
            "ondevice_eval_inference_mean_ms",
            "Mean latency (ms) of the most recent sanity eval run.",
            label_names, registry=_registry,
        ),
        "baseline_p95_ms": Gauge(
            "ondevice_eval_baseline_p95_ms",
            "p95 latency (ms) captured at first-boot baseline.",
            label_names, registry=_registry,
        ),
        "drift_score": Gauge(
            "ondevice_eval_drift_score",
            "Ratio of current p95 to baseline p95. >1.0 means slower than baseline.",
            label_names, registry=_registry,
        ),
        "gpu_util_pct": Gauge(
            "ondevice_eval_gpu_util_pct",
            "Mean GPU utilization (%) sampled during the most recent eval run.",
            label_names, registry=_registry,
        ),
        "junction_temp_c": Gauge(
            "ondevice_eval_junction_temp_c",
            "Mean Jetson junction temperature (°C) during the most recent eval run.",
            label_names, registry=_registry,
        ),
        "total_power_w": Gauge(
            "ondevice_eval_total_power_w",
            "Mean total board power (W) during the most recent eval run.",
            label_names, registry=_registry,
        ),
        "eval_accuracy": Gauge(
            "ondevice_eval_accuracy",
            "Accuracy from the most recent labeled eval run. Unset when no dataset is configured.",
            label_names, registry=_registry,
        ),
        "last_eval_ts": Gauge(
            "ondevice_eval_last_eval_timestamp_seconds",
            "Unix timestamp of the most recent sanity eval run.",
            label_names, registry=_registry,
        ),
    }


def set_identity(model_name: Optional[str], mlflow_run_id: Optional[str], deployment_id: Optional[str]) -> None:
    """
    Set labels applied to every subsequent gauge update.

    Called once at bootstrap after Triton has reported the loaded model.
    The `info` gauge is set to 1 so central dashboards can `group by`
    model_name/runId/deployment_id across scrapes.
    """
    if not _PROM_AVAILABLE:
        return
    _init_registry()
    global _labels
    _labels = {
        "model_name": model_name or "unknown",
        "mlflow_run_id": mlflow_run_id or "",
    }
    try:
        _metrics["info"].labels(
            model_name=_labels["model_name"],
            mlflow_run_id=_labels["mlflow_run_id"],
            deployment_id=deployment_id or "",
        ).set(1)
    except Exception as e:
        logger.debug("set_identity info gauge failed: %s", e)


def set_model_ready(ready: bool) -> None:
    if not _PROM_AVAILABLE:
        return
    _init_registry()
    try:
        _metrics["model_ready"].labels(**_labels).set(1 if ready else 0)
    except Exception:
        pass


def record_baseline(baseline: Any) -> None:
    """Update the `baseline_*` gauges from a Baseline DTO."""
    if not _PROM_AVAILABLE or baseline is None:
        return
    _init_registry()
    try:
        if baseline.inference_p95_ms is not None:
            _metrics["baseline_p95_ms"].labels(**_labels).set(baseline.inference_p95_ms)
    except Exception as e:
        logger.debug("record_baseline failed: %s", e)


def record_run(run: Any, *, baseline_p95_ms: Optional[float] = None) -> None:
    """
    Update the `current_*` + drift gauges from a ProfileResult-shaped run.

    `run` may be a deployment.runner.ProfileResult or a store.Run — the
    attribute access is the same for the fields we care about.
    """
    if not _PROM_AVAILABLE or run is None:
        return
    _init_registry()
    try:
        if run.inference_p95_ms is not None:
            _metrics["inference_p95_ms"].labels(**_labels).set(run.inference_p95_ms)
        if run.inference_mean_ms is not None:
            _metrics["inference_mean_ms"].labels(**_labels).set(run.inference_mean_ms)
        if run.gpu_util_mean is not None:
            _metrics["gpu_util_pct"].labels(**_labels).set(run.gpu_util_mean)
        if run.junction_temp_mean is not None:
            _metrics["junction_temp_c"].labels(**_labels).set(run.junction_temp_mean)
        if run.total_power_mean_w is not None:
            _metrics["total_power_w"].labels(**_labels).set(run.total_power_mean_w)
        if getattr(run, "accuracy", None) is not None:
            _metrics["eval_accuracy"].labels(**_labels).set(run.accuracy)
        _metrics["last_eval_ts"].labels(**_labels).set(time.time())
        if baseline_p95_ms and run.inference_p95_ms:
            _metrics["drift_score"].labels(**_labels).set(
                run.inference_p95_ms / baseline_p95_ms
            )
    except Exception as e:
        logger.debug("record_run failed: %s", e)


# =============================================================================
# Scrape helper (used by the Flask /metrics handler)
# =============================================================================

def render() -> bytes:
    """Serialize the registry in Prometheus text format."""
    if not _PROM_AVAILABLE or _registry is None:
        return b""
    return generate_latest(_registry)


def content_type() -> str:
    return CONTENT_TYPE_LATEST


def available() -> bool:
    return _PROM_AVAILABLE and _registry is not None
