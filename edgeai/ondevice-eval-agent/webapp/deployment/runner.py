"""
Baseline + sanity-eval runner.

Runs N iterations of inference against the loaded model (discovered via
Triton), captures a hardware sample at each step, and returns a
structured `ProfileResult`. Used by:

  - `bootstrap.start()` — the first-boot baseline
  - `scheduler.py`      — the recurring sanity eval

The sample image is either a real one pointed at by
`DEPLOYMENT_SAMPLE_IMAGE_PATH` (ConfigMap/PVC-mounted) or a synthetic
RGB JPEG generated in-memory. Synthetic images are fine for
*latency/thermal* baselining because the preprocess + infer + postprocess
cost is the same; they are not valid for accuracy scoring, which is why
`accuracy` stays None in that path.
"""

from __future__ import annotations

import io
import logging
import math
import os
import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# Result DTO
# =============================================================================

@dataclass
class ProfileResult:
    model_name: str
    iterations: int
    success: bool
    error: Optional[str] = None

    # Latency (ms)
    inference_mean_ms: Optional[float] = None
    inference_p50_ms: Optional[float] = None
    inference_p95_ms: Optional[float] = None
    inference_p99_ms: Optional[float] = None

    # Hardware aggregates (may be None when running off-Jetson)
    gpu_util_mean: Optional[float] = None
    junction_temp_mean: Optional[float] = None
    total_power_mean_w: Optional[float] = None

    # Accuracy (None unless a labeled dataset was used; not wired yet).
    accuracy: Optional[float] = None

    # Full detail payload for the `run.details_json` column.
    details: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Sample image
# =============================================================================

def _load_sample_bytes(path_override: Optional[str]) -> Tuple[bytes, str]:
    """
    Return `(bytes, source)` for the sample image.

    `source` is a short tag so the `run.details` payload records where
    the image came from ("path" or "synthetic").
    """
    if path_override and os.path.exists(path_override):
        with open(path_override, "rb") as f:
            return f.read(), "path"

    # Synthetic: deterministic low-noise 224x224 RGB JPEG. Triton's
    # preprocessor will resize to whatever the model actually wants.
    from PIL import Image  # lazy import; Pillow is already a runtime dep
    img = Image.new("RGB", (224, 224), color=(128, 128, 128))
    # Add a faint gradient so the image isn't pathologically uniform —
    # some models have preprocessing quirks on constant images.
    pixels = img.load()
    for x in range(0, 224, 8):
        for y in range(0, 224, 8):
            pixels[x, y] = ((x * 255) // 224, (y * 255) // 224, 128)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue(), "synthetic"


# =============================================================================
# Stats
# =============================================================================

def _percentile(sorted_vals: List[float], p: float) -> float:
    """Nearest-rank percentile; mirrors inference_latency.py's convention."""
    n = len(sorted_vals)
    idx = min(math.ceil(p * n) - 1, n - 1)
    return float(sorted_vals[max(idx, 0)])


def _latency_stats(values: List[float]) -> Dict[str, float]:
    if not values:
        return {}
    sv = sorted(values)
    out: Dict[str, float] = {
        "count": float(len(values)),
        "min": round(sv[0], 3),
        "max": round(sv[-1], 3),
        "mean": round(statistics.mean(values), 3),
        "median": round(statistics.median(values), 3),
    }
    if len(values) >= 2:
        out["stdev"] = round(statistics.stdev(values), 3)
    if len(values) >= 5:
        out["p50"] = round(_percentile(sv, 0.50), 3)
        out["p95"] = round(_percentile(sv, 0.95), 3)
        out["p99"] = round(_percentile(sv, 0.99), 3)
    else:
        out["p50"] = out["median"]
        out["p95"] = out["max"]
        out["p99"] = out["max"]
    return out


# =============================================================================
# Triton readiness
# =============================================================================

def wait_for_model_ready(client, model_name: str, timeout_s: int) -> bool:
    """
    Poll until the model is READY or we time out. Called before the
    auto-baseline; gives Triton time to finish loading after pod start.
    """
    deadline = time.time() + max(1, timeout_s)
    last_err: Optional[str] = None
    while time.time() < deadline:
        try:
            if client.check_model_ready(model_name):
                return True
        except Exception as e:
            last_err = str(e)
        time.sleep(2.0)
    logger.warning(
        "Model %s not ready within %ds (last err: %s)",
        model_name, timeout_s, last_err,
    )
    return False


def discover_model_name(client, env_hint: Optional[str] = None) -> Optional[str]:
    """
    Single-model deployment: pick the one model Triton reports as READY.

    When multiple show up (shouldn't happen in the Helm chart shape) we
    prefer the env-provided hint; otherwise the first.
    """
    try:
        models = client.get_available_models() or []
    except Exception as e:
        logger.warning("Could not list models from Triton: %s", e)
        return env_hint or None
    if not models:
        return env_hint or None
    if env_hint and env_hint in models:
        return env_hint
    return models[0]


# =============================================================================
# Core run
# =============================================================================

def run_profile(
    *,
    model_name: str,
    iterations: int,
    warmup: int = 0,
    sample_image_path: Optional[str] = None,
    sample_hardware: bool = True,
) -> ProfileResult:
    """
    Execute `iterations` timed inferences against `model_name`.

    Uses the same inference path as the `get_inference_latency` tool
    (preprocess → gRPC infer → postprocess) so the baseline is
    apples-to-apples comparable with ad-hoc measurements.

    `sample_hardware=True` runs a BackgroundSampler for Jetson GPU/power
    stats — it degrades gracefully to all-None fields on non-Jetson hosts.
    """
    from tools.base import get_client
    client = get_client()

    try:
        sample_bytes, sample_source = _load_sample_bytes(sample_image_path)
    except Exception as e:
        return ProfileResult(
            model_name=model_name, iterations=0, success=False,
            error=f"sample image load failed: {e}",
        )

    # Warmup — discarded
    for _ in range(max(0, warmup)):
        try:
            _infer_once(client, model_name, sample_bytes)
        except Exception as e:
            logger.debug("warmup iteration failed: %s", e)

    sampler = None
    if sample_hardware:
        try:
            from eval.hardware_metrics import BackgroundSampler
            sampler = BackgroundSampler(interval_ms=250)
            sampler.start()
        except Exception as e:
            logger.debug("hardware sampler unavailable: %s", e)
            sampler = None

    latencies_ms: List[float] = []
    server_compute_ms: List[float] = []
    failed = 0
    try:
        for i in range(max(1, iterations)):
            try:
                total_ms, server_ms = _infer_once(client, model_name, sample_bytes)
                latencies_ms.append(total_ms)
                if server_ms is not None:
                    server_compute_ms.append(server_ms)
            except Exception as e:
                failed += 1
                logger.debug("iteration %d failed: %s", i + 1, e)
    finally:
        if sampler is not None:
            try:
                sampler.stop()
            except Exception:
                pass

    if not latencies_ms:
        return ProfileResult(
            model_name=model_name, iterations=0, success=False,
            error=f"all {iterations} iterations failed",
        )

    lat_stats = _latency_stats(latencies_ms)
    hw_summary: Dict[str, Any] = {}
    gpu_mean = junc_mean = power_mean = None
    if sampler is not None:
        try:
            from eval.hardware_metrics import aggregate_snapshots
            hw_summary = aggregate_snapshots(sampler.get_samples())
            gpu_mean = (hw_summary.get("gpu_util_pct") or {}).get("mean")
            junc_mean = (hw_summary.get("junction_temp_c") or {}).get("mean")
            power_mean = (hw_summary.get("total_power_w") or {}).get("mean")
        except Exception as e:
            logger.debug("hardware aggregation failed: %s", e)

    details: Dict[str, Any] = {
        "sample_source": sample_source,
        "warmup": warmup,
        "failed_iterations": failed,
        "latency_stats_ms": lat_stats,
    }
    if server_compute_ms:
        details["server_compute_ms"] = _latency_stats(server_compute_ms)
    if hw_summary:
        details["hardware"] = hw_summary

    return ProfileResult(
        model_name=model_name,
        iterations=len(latencies_ms),
        success=True,
        inference_mean_ms=lat_stats.get("mean"),
        inference_p50_ms=lat_stats.get("p50"),
        inference_p95_ms=lat_stats.get("p95"),
        inference_p99_ms=lat_stats.get("p99"),
        gpu_util_mean=gpu_mean,
        junction_temp_mean=junc_mean,
        total_power_mean_w=power_mean,
        details=details,
    )


def _infer_once(
    client, model_name: str, sample_bytes: bytes,
) -> Tuple[float, Optional[float]]:
    """
    Run one preprocess → infer → postprocess cycle.

    Returns `(total_ms, server_compute_ms_or_None)`.
    """
    t0 = time.perf_counter()
    arr = client.preprocess_image_bytes(sample_bytes, model_name=model_name)
    if arr is None:
        raise RuntimeError("preprocess returned None")
    response = client.send_inference_request(arr, model_name, measure_latency=True)
    if response is None:
        raise RuntimeError("inference returned None")
    # Exercise the postprocess path so the run mirrors real traffic.
    try:
        client.process_prediction(response, model_name)
    except Exception:
        # Some models need class_names etc.; ignore for pure-latency runs.
        pass
    total_ms = (time.perf_counter() - t0) * 1000.0
    server_ms = None
    lat = response.get("latency") if isinstance(response, dict) else None
    if isinstance(lat, (int, float)):
        server_ms = float(lat) * 1000.0
    return total_ms, server_ms


# =============================================================================
# Model type probe
# =============================================================================

def detect_model_type(client, model_name: str) -> Optional[str]:
    """Best-effort model-type label for the baseline row. None on failure."""
    try:
        from tools.catalog.model_type import infer_model_type_from_shapes
        input_spec = client.get_model_input_spec(model_name)
        output_specs = client.get_all_output_specs(model_name)
        return infer_model_type_from_shapes(input_spec, output_specs).get("type")
    except Exception as e:
        logger.debug("model type detection failed: %s", e)
        return None
