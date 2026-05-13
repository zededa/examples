"""
Deployment bootstrap — the one entry point called from app.py.

`start()` fires a daemon thread that:

  1. Waits for Triton to report the single loaded model as READY.
  2. Sets Prometheus `deployment_info` / `model_ready` labels so /metrics
     returns useful values even before the first baseline.
  3. If no active baseline exists yet for this (model_name, mlflow_run_id)
     tuple, runs the baseline profile and persists it.
  4. Starts the sanity-eval scheduler.

Everything is in a background thread so Flask comes up immediately and
a not-yet-ready Triton never blocks the pod from serving `/health`.
"""

from __future__ import annotations

import logging
import threading
from typing import Optional

logger = logging.getLogger(__name__)

_started = False
_lock = threading.Lock()


def start() -> None:
    """Kick off bootstrap. Idempotent — safe to call more than once."""
    global _started
    with _lock:
        if _started:
            return
        _started = True

    try:
        from config import get_settings
        settings = get_settings().deployment
    except Exception as e:
        logger.warning("deployment bootstrap skipped (settings load failed): %s", e)
        return

    if not settings.enabled:
        logger.info("Deployment bootstrap disabled (DEPLOYMENT_ENABLED=false)")
        return

    t = threading.Thread(
        target=_run,
        name="deployment-bootstrap",
        daemon=True,
    )
    t.start()


def _run() -> None:
    try:
        from config import get_settings
        from tools.base import get_client
        from deployment.runner import (
            discover_model_name, wait_for_model_ready, run_profile, detect_model_type,
        )
        from deployment.store import get_store
        from deployment import metrics as prom
        from deployment import scheduler

        settings = get_settings().deployment
        client = get_client()

        # 1. Discover model name.
        model_name = discover_model_name(client, env_hint=settings.model_name)
        if not model_name:
            logger.warning(
                "deployment bootstrap: no model reported by Triton yet (MODEL_NAME=%r); "
                "retrying once after Triton readiness",
                settings.model_name,
            )

        # 2. Wait for readiness. Use the env hint if discovery came up empty.
        poll_name = model_name or settings.model_name
        if not poll_name:
            logger.error("deployment bootstrap: no model name to poll; giving up")
            return

        ready = wait_for_model_ready(client, poll_name, settings.triton_ready_timeout_s)
        if not ready:
            # Set labels anyway so /metrics reports model_ready=0
            prom.set_identity(poll_name, settings.mlflow_run_id, settings.deployment_id)
            prom.set_model_ready(False)
            logger.error("deployment bootstrap: model %s never became ready", poll_name)
            return

        # One more discover in case Triton was still loading at step 1.
        model_name = discover_model_name(client, env_hint=poll_name) or poll_name

        # 3. Publish identity + readiness.
        prom.set_identity(model_name, settings.mlflow_run_id, settings.deployment_id)
        prom.set_model_ready(True)

        store = get_store()
        if store is None:
            logger.warning("deployment bootstrap: store unavailable — metrics will not persist")
            return

        # 4. Auto-baseline unless one already exists.
        if settings.auto_baseline and not store.has_baseline_for(model_name, settings.mlflow_run_id):
            logger.info(
                "Capturing first-boot baseline for model=%s mlflow_run_id=%s (%d iter)",
                model_name, settings.mlflow_run_id, settings.baseline_iterations,
            )
            model_type = detect_model_type(client, model_name)
            result = run_profile(
                model_name=model_name,
                iterations=settings.baseline_iterations,
                warmup=settings.baseline_warmup,
                sample_image_path=settings.sample_image_path,
                sample_hardware=True,
            )
            if not result.success:
                logger.error("baseline run failed: %s", result.error)
            else:
                store.save_baseline(
                    model_name=result.model_name,
                    mlflow_run_id=settings.mlflow_run_id,
                    model_type=model_type,
                    iterations=result.iterations,
                    inference_mean_ms=result.inference_mean_ms,
                    inference_p50_ms=result.inference_p50_ms,
                    inference_p95_ms=result.inference_p95_ms,
                    inference_p99_ms=result.inference_p99_ms,
                    gpu_util_mean=result.gpu_util_mean,
                    junction_temp_mean=result.junction_temp_mean,
                    total_power_mean_w=result.total_power_mean_w,
                    accuracy=result.accuracy,
                    metadata={
                        "deployment_id": settings.deployment_id,
                        "sample_source": result.details.get("sample_source"),
                    },
                )
                # Also write a baseline-kind run so time-series views include it.
                store.save_run(
                    kind="baseline",
                    model_name=result.model_name,
                    iterations=result.iterations,
                    inference_mean_ms=result.inference_mean_ms,
                    inference_p95_ms=result.inference_p95_ms,
                    gpu_util_mean=result.gpu_util_mean,
                    junction_temp_mean=result.junction_temp_mean,
                    total_power_mean_w=result.total_power_mean_w,
                    accuracy=result.accuracy,
                    success=True,
                    details=result.details,
                )
                logger.info(
                    "Baseline saved: p95=%.2fms mean=%.2fms (%d iter)",
                    result.inference_p95_ms or 0.0,
                    result.inference_mean_ms or 0.0,
                    result.iterations,
                )

        # Publish latest baseline to Prometheus regardless of freshness.
        prom.record_baseline(store.get_active_baseline())

        # 5. Start the sanity scheduler.
        if settings.sanity_enabled:
            scheduler.start(
                model_name=model_name,
                interval_s=settings.sanity_interval_s,
                iterations=settings.sanity_iterations,
                drift_threshold=settings.drift_alert_threshold,
                sample_image_path=settings.sample_image_path,
            )
    except Exception as e:
        logger.exception("deployment bootstrap crashed: %s", e)
