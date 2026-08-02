"""
Background sanity-eval scheduler.

One daemon thread per process. Every `sanity_interval_s`:

  1. Run a short profile (fewer iterations than the baseline).
  2. Persist it as a `run` row.
  3. If current p95 / baseline p95 > `drift_alert_threshold`, record a
     `drift` event and log a warning.
  4. Update Prometheus gauges.

Cheap by design — a 5-iter profile every 10 min is negligible compute
even on a Jetson Nano, yet catches thermal throttling, model decay, and
competing-workload slowdowns that batch benchmarks miss.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional

logger = logging.getLogger(__name__)


class SanityEvalScheduler:
    """Daemon-thread loop. Idempotent `start` / `stop`."""

    def __init__(self, *, model_name: str, interval_s: int, iterations: int,
                 drift_threshold: float, sample_image_path: Optional[str] = None) -> None:
        self._model_name = model_name
        self._interval_s = max(30, int(interval_s))
        self._iterations = max(1, int(iterations))
        self._drift_threshold = max(1.0, float(drift_threshold))
        self._sample_image_path = sample_image_path
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # --- lifecycle -----------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="deployment-sanity", daemon=True,
        )
        self._thread.start()
        logger.info(
            "Sanity-eval scheduler started (model=%s interval=%ds iterations=%d drift=%.2f)",
            self._model_name, self._interval_s, self._iterations, self._drift_threshold,
        )

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    # --- loop ----------------------------------------------------------------

    def _run(self) -> None:
        # Brief jitter so multiple pods don't scrape-collide at fleet scale.
        self._stop.wait(5.0)
        while not self._stop.is_set():
            try:
                self._tick()
            except Exception as e:
                # A single bad iteration should never stop the thread.
                logger.exception("sanity-eval tick failed: %s", e)
            self._stop.wait(self._interval_s)

    def _tick(self) -> None:
        # Lazy imports keep the scheduler module side-effect free at import time.
        from deployment.runner import run_profile
        from deployment.store import get_store
        from deployment import metrics as prom

        store = get_store()
        if store is None:
            # Store disabled — skip silently; runner would be pointless.
            return

        result = run_profile(
            model_name=self._model_name,
            iterations=self._iterations,
            warmup=1,
            sample_image_path=self._sample_image_path,
            sample_hardware=True,
        )

        run_id = store.save_run(
            kind="sanity",
            model_name=result.model_name,
            iterations=result.iterations,
            inference_mean_ms=result.inference_mean_ms,
            inference_p95_ms=result.inference_p95_ms,
            gpu_util_mean=result.gpu_util_mean,
            junction_temp_mean=result.junction_temp_mean,
            total_power_mean_w=result.total_power_mean_w,
            accuracy=result.accuracy,
            success=result.success,
            error=result.error,
            details=result.details,
        )

        baseline = store.get_active_baseline()
        baseline_p95 = baseline.inference_p95_ms if baseline else None

        # Drift detection: only meaningful once we have both sides.
        if (
            result.success
            and baseline_p95
            and result.inference_p95_ms
            and result.inference_p95_ms / baseline_p95 >= self._drift_threshold
        ):
            score = result.inference_p95_ms / baseline_p95
            store.save_drift(
                drift_score=score,
                baseline_p95_ms=baseline_p95,
                current_p95_ms=result.inference_p95_ms,
                run_id=run_id,
            )
            logger.warning(
                "Drift detected: current p95=%.2fms vs baseline %.2fms (%.2fx) model=%s",
                result.inference_p95_ms, baseline_p95, score, result.model_name,
            )

        prom.record_run(result, baseline_p95_ms=baseline_p95)


# Module-level singleton so `start` is idempotent across re-imports.
_scheduler: Optional[SanityEvalScheduler] = None
_lock = threading.Lock()


def start(*, model_name: str, interval_s: int, iterations: int,
          drift_threshold: float, sample_image_path: Optional[str] = None) -> None:
    global _scheduler
    with _lock:
        if _scheduler is not None:
            return
        _scheduler = SanityEvalScheduler(
            model_name=model_name,
            interval_s=interval_s,
            iterations=iterations,
            drift_threshold=drift_threshold,
            sample_image_path=sample_image_path,
        )
        _scheduler.start()


def stop() -> None:
    global _scheduler
    with _lock:
        if _scheduler is not None:
            _scheduler.stop()
            _scheduler = None
