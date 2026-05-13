"""
Inference Latency Tool

Measures and reports inference latency for deployed models.
Supports single-shot and multi-iteration benchmarking with
detailed timing breakdowns (preprocessing, inference, postprocessing).

When the Triton metrics endpoint (localhost:8002/metrics) is available,
server-side timing is obtained from Prometheus counters, providing an
accurate breakdown of queue time, compute-input, compute-infer, and
compute-output durations that are not affected by client-side overhead.
"""

import logging
import math
import os
import time
import statistics
from typing import Dict, Any, List, Optional

from tools.base import ok, error_response, get_client
from tools.registry import register_tool
from sessions.registry import SESSION_STORAGE_ROOT

logger = logging.getLogger(__name__)

# Limits
MAX_ITERATIONS = 100
DEFAULT_ITERATIONS = 1
DEFAULT_WARMUP = 0


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def _fetch_model_metrics(client, model_name: str) -> Optional[Dict[str, float]]:
    """
    Fetch Triton server-side latency metrics for *model_name*.

    Returns a dict with cumulative counters (in ms) and request_count,
    or None if the metrics endpoint is unavailable.
    """
    try:
        return client.get_model_metrics(model_name)
    except Exception as e:
        logger.debug(f"Could not fetch metrics for {model_name}: {e}")
        return None


def _compute_metrics_delta(
    before: Optional[Dict[str, float]],
    after: Optional[Dict[str, float]],
) -> Optional[Dict[str, float]]:
    """
    Compute the per-request delta between two metrics snapshots.

    Both *before* and *after* contain cumulative counters.  We take
    the difference and divide durations by the request-count delta
    to get per-request averages.
    """
    if not before or not after:
        return None

    count_before = before.get("request_count", 0)
    count_after = after.get("request_count", 0)
    n = count_after - count_before
    if n <= 0:
        return None

    delta: Dict[str, float] = {}
    for key in ("request_duration_ms", "queue_ms", "compute_input_ms",
                "compute_infer_ms", "compute_output_ms"):
        val_before = before.get(key)
        val_after = after.get(key)
        if val_before is not None and val_after is not None:
            delta[key] = round((val_after - val_before) / n, 3)

    delta["request_count_delta"] = n
    return delta


# ---------------------------------------------------------------------------
# Core timing
# ---------------------------------------------------------------------------

def _run_single_inference_timed(
    client,
    image_path: str,
    file_bytes: bytes,
    model_name: str,
) -> Dict[str, Any]:
    """
    Run a single inference and return granular timing breakdown in milliseconds.

    When the Triton metrics endpoint is reachable, server-side counters
    (queue, compute-input, compute-infer, compute-output) are captured
    around the request and included in the returned dict.

    Returns dict with keys:
        model_check_ms, preprocess_ms, inference_ms, postprocess_ms,
        total_ms, and optionally server_latency_ms plus
        server_metrics (dict of per-request server-side durations).
    """
    timings: Dict[str, Any] = {}

    total_start = time.perf_counter()

    # 1) Model readiness check
    t0 = time.perf_counter()
    model_ready = client.check_model_ready(model_name)
    timings["model_check_ms"] = (time.perf_counter() - t0) * 1000.0

    if not model_ready:
        raise RuntimeError(f"Model {model_name} is not ready")

    # 2) Preprocessing
    t0 = time.perf_counter()
    image_array = client.preprocess_image_bytes(file_bytes, model_name=model_name)
    timings["preprocess_ms"] = (time.perf_counter() - t0) * 1000.0

    if image_array is None:
        raise RuntimeError("Failed to preprocess image")

    # 3) Snapshot metrics BEFORE inference
    metrics_before = _fetch_model_metrics(client, model_name)

    # 4) Inference (gRPC round-trip)
    t0 = time.perf_counter()
    response = client.send_inference_request(
        image_array, model_name, measure_latency=True
    )
    timings["inference_ms"] = (time.perf_counter() - t0) * 1000.0

    if response is None:
        raise RuntimeError("Inference request failed - no response from server")

    # 5) Snapshot metrics AFTER inference
    metrics_after = _fetch_model_metrics(client, model_name)

    # Capture client-reported latency if available (gRPC round-trip)
    if "latency" in response:
        timings["server_latency_ms"] = response["latency"] * 1000.0

    # Compute server-side metrics delta
    metrics_delta = _compute_metrics_delta(metrics_before, metrics_after)
    if metrics_delta:
        timings["server_metrics"] = metrics_delta

    # 6) Post-processing (prediction decode)
    t0 = time.perf_counter()
    prediction = client.process_prediction(response, model_name)
    timings["postprocess_ms"] = (time.perf_counter() - t0) * 1000.0

    timings["total_ms"] = (time.perf_counter() - total_start) * 1000.0

    return timings


def _compute_stats(values: List[float]) -> Dict[str, float]:
    """Compute summary statistics for a list of float values."""
    if not values:
        return {}
    n = len(values)
    result: Dict[str, float] = {
        "count": n,
        "min": round(min(values), 3),
        "max": round(max(values), 3),
        "mean": round(statistics.mean(values), 3),
    }
    if n >= 2:
        result["stdev"] = round(statistics.stdev(values), 3)
        result["median"] = round(statistics.median(values), 3)
    else:
        result["stdev"] = 0.0
        result["median"] = result["mean"]

    # Percentiles (nearest-rank method)
    sorted_vals = sorted(values)
    if n >= 5:
        for label, p in (("p90", 0.9), ("p95", 0.95), ("p99", 0.99)):
            idx = min(math.ceil(p * n) - 1, n - 1)
            result[label] = round(sorted_vals[idx], 3)

    return result


def get_inference_latency(
    model_name: str,
    image_path: str,
    iterations: int = DEFAULT_ITERATIONS,
    warmup_iterations: int = DEFAULT_WARMUP,
) -> Dict[str, Any]:
    """
    Measure inference latency for a model with detailed timing breakdown.

    Runs one or more inference iterations and returns per-phase timing
    (preprocessing, gRPC inference, postprocessing) and aggregate statistics.
    When the Triton metrics endpoint is available, server-side timing
    (queue, compute-input, compute-infer, compute-output) is included
    for each iteration.

    Warmup iterations are executed but excluded from reported statistics.

    Args:
        model_name: Name of the deployed model.
        image_path: Path to the image file to use for measurement.
        iterations: Number of timed iterations to run (1-100, default 1).
        warmup_iterations: Number of warmup iterations before measurement (default 0).

    Returns:
        Latency measurements with per-phase breakdown and statistics.
    """
    try:
        # --- Input validation ---------------------------------------------------
        if not model_name:
            return error_response(
                ValueError("model_name is required"),
                operation="get_inference_latency",
            )

        if not image_path:
            return error_response(
                ValueError("image_path is required"),
                operation="get_inference_latency",
            )

        # Security: prevent path traversal
        real_path = os.path.realpath(image_path)
        real_storage_root = os.path.realpath(SESSION_STORAGE_ROOT)
        if not real_path.startswith(real_storage_root + os.sep) and real_path != real_storage_root:
            return error_response(
                ValueError("Invalid file path - access denied"),
                operation="get_inference_latency",
            )

        if not os.path.exists(real_path):
            return error_response(
                FileNotFoundError(f"Image not found: {image_path}"),
                operation="get_inference_latency",
            )

        iterations = max(1, min(int(iterations), MAX_ITERATIONS))
        warmup_iterations = max(0, min(int(warmup_iterations), 10))

        # --- Read image bytes once ------------------------------------------------
        with open(real_path, "rb") as f:
            file_bytes = f.read()

        client = get_client()

        # --- Warmup ---------------------------------------------------------------
        for i in range(warmup_iterations):
            try:
                _run_single_inference_timed(client, image_path, file_bytes, model_name)
                logger.debug(f"Warmup iteration {i + 1}/{warmup_iterations} complete")
            except RuntimeError as e:
                return error_response(
                    e,
                    operation="get_inference_latency",
                    phase="warmup",
                    iteration=i + 1,
                )

        # --- Timed iterations -----------------------------------------------------
        all_timings: List[Dict[str, Any]] = []
        for i in range(iterations):
            try:
                t = _run_single_inference_timed(client, image_path, file_bytes, model_name)
                all_timings.append(t)
                logger.debug(
                    f"Iteration {i + 1}/{iterations}: "
                    f"inference={t['inference_ms']:.2f}ms  total={t['total_ms']:.2f}ms"
                )
            except RuntimeError as e:
                return error_response(
                    e,
                    operation="get_inference_latency",
                    phase="measurement",
                    iteration=i + 1,
                    completed_iterations=len(all_timings),
                )

        # --- Build response -------------------------------------------------------
        # Collect per-phase value lists
        phase_keys = [
            "model_check_ms",
            "preprocess_ms",
            "inference_ms",
            "postprocess_ms",
            "total_ms",
        ]
        has_server_latency = any("server_latency_ms" in t for t in all_timings)
        if has_server_latency:
            phase_keys.append("server_latency_ms")

        phase_values: Dict[str, List[float]] = {k: [] for k in phase_keys}
        for t in all_timings:
            for k in phase_keys:
                if k in t:
                    phase_values[k].append(t[k])

        # Collect server-side metrics if available
        has_server_metrics = any("server_metrics" in t for t in all_timings)
        server_metrics_summary: Optional[Dict[str, Any]] = None
        if has_server_metrics:
            sm_keys = ("queue_ms", "compute_input_ms", "compute_infer_ms",
                       "compute_output_ms", "request_duration_ms")
            sm_values: Dict[str, List[float]] = {k: [] for k in sm_keys}
            for t in all_timings:
                sm = t.get("server_metrics")
                if sm:
                    for k in sm_keys:
                        if k in sm:
                            sm_values[k].append(sm[k])
            server_metrics_summary = {}
            for k in sm_keys:
                if sm_values[k]:
                    server_metrics_summary[k] = _compute_stats(sm_values[k])

        # Single-iteration shortcut
        if iterations == 1:
            single: Dict[str, Any] = {}
            for k, v in all_timings[0].items():
                if k == "server_metrics":
                    single[k] = v
                elif isinstance(v, float):
                    single[k] = round(v, 3)
                else:
                    single[k] = v

            # Build summary with server-side metrics if available
            sm = all_timings[0].get("server_metrics")
            summary_parts = [
                f"Inference latency for {model_name}: "
                f"{single.get('inference_ms', 0):.1f}ms gRPC round-trip, "
                f"{single.get('total_ms', 0):.1f}ms total"
            ]
            if sm:
                summary_parts.append(
                    f" | Server-side: "
                    f"queue={sm.get('queue_ms', 0):.1f}ms, "
                    f"compute={sm.get('compute_infer_ms', 0):.1f}ms"
                )
            summary = "".join(summary_parts)

            response_data: Dict[str, Any] = {
                "model_name": model_name,
                "image_path": image_path,
                "iterations": 1,
                "warmup_iterations": warmup_iterations,
                "latency": single,
                "unit": "milliseconds",
                "protocol": "gRPC",
            }
            if server_metrics_summary:
                response_data["server_metrics"] = server_metrics_summary

            return ok(data=response_data, message=summary)

        # Multi-iteration: compute statistics per phase
        phase_stats: Dict[str, Dict[str, float]] = {}
        for k in phase_keys:
            if phase_values[k]:
                phase_stats[k] = _compute_stats(phase_values[k])

        # Per-iteration raw data (truncated if too many)
        raw_iterations = all_timings if iterations <= 20 else None
        raw_note = (
            None
            if iterations <= 20
            else f"Raw per-iteration data omitted ({iterations} iterations). Statistics are provided instead."
        )

        inf_stats = phase_stats.get("inference_ms", {})
        total_stats = phase_stats.get("total_ms", {})
        summary_parts = [
            f"Latency for {model_name} over {iterations} iterations (gRPC): "
            f"inference mean={inf_stats.get('mean', 0):.1f}ms "
            f"(min={inf_stats.get('min', 0):.1f}, max={inf_stats.get('max', 0):.1f}), "
            f"total mean={total_stats.get('mean', 0):.1f}ms"
        ]
        if server_metrics_summary:
            ci = server_metrics_summary.get("compute_infer_ms", {})
            if ci:
                summary_parts.append(
                    f" | Server compute mean={ci.get('mean', 0):.1f}ms"
                )
        summary = "".join(summary_parts)

        response_data = {
            "model_name": model_name,
            "image_path": image_path,
            "iterations": iterations,
            "warmup_iterations": warmup_iterations,
            "statistics": phase_stats,
            "unit": "milliseconds",
            "protocol": "gRPC",
        }
        if server_metrics_summary:
            response_data["server_metrics"] = server_metrics_summary
        if raw_iterations is not None:
            response_data["per_iteration"] = [
                {k: (round(v, 3) if isinstance(v, float) else v)
                 for k, v in t.items()}
                for t in raw_iterations
            ]
        if raw_note:
            response_data["note"] = raw_note

        return ok(data=response_data, message=summary)

    except Exception as e:
        logger.error(f"Error measuring inference latency: {e}", exc_info=True)
        return error_response(
            e,
            operation="get_inference_latency",
            model_name=model_name,
            image_path=image_path,
        )


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------

register_tool(
    name="get_inference_latency",
    func=get_inference_latency,
    description=(
        "Measure inference latency for a deployed model with a detailed per-phase "
        "timing breakdown (model check, preprocessing, gRPC inference, postprocessing). "
        "When the Triton metrics endpoint is available, includes accurate server-side "
        "timing (queue wait, compute-input, compute-infer, compute-output). "
        "Supports multiple iterations for statistical analysis (mean, median, p90, p95, p99) "
        "and optional warmup iterations to exclude cold-start effects. "
        "Use this tool when the user asks about model speed, latency, performance, or throughput."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "model_name": {
                "type": "string",
                "description": "Name of the deployed model to benchmark",
            },
            "image_path": {
                "type": "string",
                "description": "Path to an uploaded image file to use for the measurement",
            },
            "iterations": {
                "type": "integer",
                "default": 1,
                "minimum": 1,
                "maximum": 100,
                "description": (
                    "Number of timed inference iterations (default 1). "
                    "Use higher values (e.g. 10-50) for reliable statistics."
                ),
            },
            "warmup_iterations": {
                "type": "integer",
                "default": 0,
                "minimum": 0,
                "maximum": 10,
                "description": (
                    "Number of warmup iterations before measurement (default 0). "
                    "Warmup results are discarded, useful to exclude cold-start latency."
                ),
            },
        },
        "required": ["model_name", "image_path"],
    },
)
