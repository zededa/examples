"""
Deployment health report — the single JSON object the `get_deployment_health`
tool and (optionally) future HTTP endpoints return.

Assembles:

  - Model identity   (name, mlflow_run_id, deployment_id, ready)
  - Baseline         (p95 + hardware at first-boot)
  - Current          (most recent sanity run p95, drift score, last run ts)
  - Hardware         (fresh one-shot Jetson snapshot)
  - Drift history    (last N drift events)
  - Alerts           (human-readable flags: drift, not-ready, etc.)

Kept deliberately denormalized so an LLM can read it end-to-end in one
tool call without chasing cross-references.
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def build_health_report() -> Dict[str, Any]:
    """Return a dict-of-dicts with the full deployment health snapshot."""
    from config import get_settings
    from deployment.store import get_store

    settings = get_settings().deployment
    report: Dict[str, Any] = {
        "generated_at": time.time(),
        "enabled": settings.enabled,
        "model": {
            "name": settings.model_name,
            "mlflow_run_id": settings.mlflow_run_id,
            "deployment_id": settings.deployment_id,
            "ready": None,
            "server_type": None,
            "server_url": None,
        },
        "baseline": None,
        "current": None,
        "hardware": None,
        "drift_events": [],
        "alerts": [],
        "config": {
            "auto_baseline": settings.auto_baseline,
            "sanity_enabled": settings.sanity_enabled,
            "sanity_interval_s": settings.sanity_interval_s,
            "sanity_iterations": settings.sanity_iterations,
            "drift_alert_threshold": settings.drift_alert_threshold,
            "baseline_iterations": settings.baseline_iterations,
        },
    }

    if not settings.enabled:
        report["alerts"].append("deployment features disabled (DEPLOYMENT_ENABLED=false)")
        return report

    # Model identity + readiness — fresh check, cheap.
    model_name = _probe_model_identity(report)

    # Hardware — one-shot sample, best-effort.
    _probe_hardware(report)

    # Baseline + runs from store.
    store = get_store()
    if store is None:
        report["alerts"].append("deployment store unavailable — baseline + history missing")
        return report

    baseline = store.get_active_baseline()
    if baseline is not None:
        report["baseline"] = _baseline_to_dict(baseline)
    else:
        report["alerts"].append("no baseline captured yet — first-boot profiling still pending")

    latest_sanity = store.get_latest_run(kind="sanity")
    if latest_sanity is None:
        # Fall back to the baseline-kind run so "current" isn't empty right
        # after bootstrap but before the first sanity tick.
        latest_sanity = store.get_latest_run(kind="baseline")
    if latest_sanity is not None:
        current = _run_to_dict(latest_sanity)
        if (
            baseline
            and baseline.inference_p95_ms
            and latest_sanity.inference_p95_ms
        ):
            drift = latest_sanity.inference_p95_ms / baseline.inference_p95_ms
            current["drift_score"] = round(drift, 3)
            if drift >= settings.drift_alert_threshold:
                report["alerts"].append(
                    f"p95 drift {drift:.2f}x above baseline threshold "
                    f"({settings.drift_alert_threshold:.2f})"
                )
        report["current"] = current

    # Drift history — last 5 events is plenty for a health call.
    drift_rows = store.list_drift_events(limit=5)
    report["drift_events"] = [
        {
            "created_at": d.created_at,
            "drift_score": round(d.drift_score, 3),
            "baseline_p95_ms": d.baseline_p95_ms,
            "current_p95_ms": d.current_p95_ms,
        }
        for d in drift_rows
    ]

    if report["model"]["ready"] is False:
        report["alerts"].append(f"model {model_name or '<unknown>'} is not READY on Triton")

    return report


# =============================================================================
# Probes
# =============================================================================

def _probe_model_identity(report: Dict[str, Any]) -> Optional[str]:
    """Populate `report['model']` with a fresh Triton probe."""
    from tools.base import get_client
    from deployment.runner import discover_model_name

    client = get_client()
    model_name = None
    try:
        model_name = discover_model_name(client, env_hint=report["model"]["name"])
        report["model"]["name"] = model_name or report["model"]["name"]
        if model_name:
            report["model"]["ready"] = bool(client.check_model_ready(model_name))
        else:
            report["model"]["ready"] = False
    except Exception as e:
        logger.debug("model probe failed: %s", e)
        report["model"]["ready"] = False
    try:
        report["model"]["server_type"] = client.detect_server_type()
    except Exception:
        pass
    try:
        report["model"]["server_url"] = client.server_url
    except Exception:
        pass
    return model_name


def _probe_hardware(report: Dict[str, Any]) -> None:
    try:
        from eval.hardware_metrics import read_snapshot
        snap = read_snapshot()
        report["hardware"] = {
            "gpu_util_pct": snap.gpu_util_pct,
            "cpu_temp_c": snap.cpu_temp_c,
            "junction_temp_c": snap.junction_temp_c,
            "total_power_w": snap.total_power_w,
            "vdd_gpu_soc_w": snap.vdd_gpu_soc_w,
            "vdd_cpu_cv_w": snap.vdd_cpu_cv_w,
            "timestamp": snap.timestamp,
        }
    except Exception as e:
        logger.debug("hardware probe failed: %s", e)
        report["hardware"] = None


# =============================================================================
# Serializers
# =============================================================================

def _baseline_to_dict(b: Any) -> Dict[str, Any]:
    return {
        "id": b.id,
        "created_at": b.created_at,
        "model_name": b.model_name,
        "mlflow_run_id": b.mlflow_run_id,
        "model_type": b.model_type,
        "iterations": b.iterations,
        "inference_mean_ms": b.inference_mean_ms,
        "inference_p50_ms": b.inference_p50_ms,
        "inference_p95_ms": b.inference_p95_ms,
        "inference_p99_ms": b.inference_p99_ms,
        "gpu_util_mean": b.gpu_util_mean,
        "junction_temp_mean": b.junction_temp_mean,
        "total_power_mean_w": b.total_power_mean_w,
        "accuracy": b.accuracy,
    }


def _run_to_dict(r: Any) -> Dict[str, Any]:
    return {
        "id": r.id,
        "created_at": r.created_at,
        "kind": r.kind,
        "iterations": r.iterations,
        "inference_mean_ms": r.inference_mean_ms,
        "inference_p95_ms": r.inference_p95_ms,
        "gpu_util_mean": r.gpu_util_mean,
        "junction_temp_mean": r.junction_temp_mean,
        "total_power_mean_w": r.total_power_mean_w,
        "accuracy": r.accuracy,
        "success": r.success,
        "error": r.error,
    }
