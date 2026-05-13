"""
LLM Benchmark Tool

Runs a throughput/latency benchmark against an LLM serving backend.
Measures time-to-first-token (TTFT), tokens/sec, and per-prompt latency
with optional Jetson hardware metrics (GPU utilization, temperature, power).
"""

import logging
import statistics
from typing import Any, Dict, List, Optional

from tools.base import ok, error_response
from tools.registry import register_tool

logger = logging.getLogger(__name__)

DEFAULT_PROMPTS = [
    "Explain quantum computing in simple terms.",
    "Write a short function in Python that reverses a linked list.",
    "What are the key differences between TCP and UDP?",
    "Summarize the main ideas behind transformer neural networks.",
    "Describe the process of photosynthesis step by step.",
]

MAX_PROMPTS = 20
MAX_ITERATIONS = 5
MAX_MAX_TOKENS = 1024


def _get_llm_client():
    from client.llm_client import get_llm_client
    return get_llm_client()


def _compute_stats(values: List[float]) -> Dict[str, float]:
    """Compute summary statistics for a list of floats."""
    if not values:
        return {}
    result: Dict[str, float] = {
        "count": len(values),
        "min": round(min(values), 3),
        "max": round(max(values), 3),
        "mean": round(statistics.mean(values), 3),
    }
    if len(values) >= 2:
        result["stdev"] = round(statistics.stdev(values), 3)
        result["median"] = round(statistics.median(values), 3)
    return result


def _run_benchmark_core(
    model_name: str,
    prompts: List[str],
    iterations: int,
    max_tokens: int,
    temperature: float,
    measure_hardware: bool,
    sample_interval_ms: int,
) -> Dict[str, Any]:
    """
    Core benchmark logic, separated for reuse by llm_compare_models.

    Returns the raw result dict (not wrapped in ok/error_response).
    """
    client = _get_llm_client()

    if not client.is_healthy():
        raise ConnectionError(
            f"LLM server at {client.base_url} is not reachable"
        )

    # Resolve model name
    if not model_name:
        models = client.list_models()
        if not models:
            raise ValueError("No LLM models available on the server")
        model_name = models[0].id

    # Start hardware sampling
    sampler = None
    if measure_hardware:
        try:
            from eval.hardware_metrics import BackgroundSampler
            sampler = BackgroundSampler(interval_ms=sample_interval_ms)
            sampler.start()
        except Exception as e:
            logger.warning("Hardware metrics unavailable: %s", e)

    # Run benchmark
    latency_values: List[float] = []
    ttft_values: List[float] = []
    tps_values: List[float] = []
    completion_tok_values: List[int] = []
    prompt_tok_values: List[int] = []
    per_prompt: List[Dict[str, Any]] = []

    try:
        for prompt_idx, prompt in enumerate(prompts):
            for iteration in range(iterations):
                messages = [{"role": "user", "content": prompt}]
                try:
                    resp = client.chat_completion_streaming(
                        model=model_name,
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                    )

                    perf = resp["performance"]
                    usage = resp["usage"]

                    latency_values.append(perf["total_time_ms"])
                    tps_values.append(perf["tokens_per_second"])
                    completion_tok_values.append(usage["completion_tokens"])
                    prompt_tok_values.append(usage["prompt_tokens"])

                    if perf.get("time_to_first_token_ms") is not None:
                        ttft_values.append(perf["time_to_first_token_ms"])

                    per_prompt.append({
                        "prompt_index": prompt_idx,
                        "iteration": iteration + 1,
                        "latency_ms": perf["total_time_ms"],
                        "ttft_ms": perf.get("time_to_first_token_ms"),
                        "tokens_per_second": perf["tokens_per_second"],
                        "completion_tokens": usage["completion_tokens"],
                    })

                except Exception as e:
                    logger.warning(
                        "Benchmark prompt %d iter %d failed: %s",
                        prompt_idx, iteration + 1, e,
                    )
                    per_prompt.append({
                        "prompt_index": prompt_idx,
                        "iteration": iteration + 1,
                        "error": str(e),
                    })
    finally:
        if sampler:
            sampler.stop()

    if not latency_values:
        raise RuntimeError("All benchmark iterations failed")

    # Aggregate
    aggregate: Dict[str, Any] = {
        "tokens_per_second": _compute_stats(tps_values),
        "latency_ms": _compute_stats(latency_values),
    }
    if ttft_values:
        aggregate["ttft_ms"] = _compute_stats(ttft_values)
    if prompt_tok_values:
        aggregate["avg_prompt_tokens"] = round(statistics.mean(prompt_tok_values), 1)
    if completion_tok_values:
        aggregate["avg_completion_tokens"] = round(
            statistics.mean(completion_tok_values), 1
        )

    result: Dict[str, Any] = {
        "model_name": model_name,
        "server_url": client.base_url,
        "total_prompts": len(prompts),
        "iterations_per_prompt": iterations,
        "max_tokens": max_tokens,
        "successful_runs": len(latency_values),
        "total_runs": len(prompts) * iterations,
        "aggregate": aggregate,
    }

    # Include per-prompt details only if manageable
    if len(per_prompt) <= 20:
        result["per_prompt"] = per_prompt

    # Hardware metrics
    if sampler:
        try:
            from eval.hardware_metrics import aggregate_snapshots
            samples = sampler.get_samples()
            if samples:
                result["hardware"] = aggregate_snapshots(samples)
        except Exception as e:
            logger.warning("Failed to aggregate hardware metrics: %s", e)

    # Server metrics (Prometheus)
    try:
        server_metrics = client.get_server_metrics()
        if server_metrics and server_metrics.avg_generation_throughput_tps is not None:
            result["server_metrics"] = {
                "avg_generation_throughput_tps": round(
                    server_metrics.avg_generation_throughput_tps, 2
                ),
                "gpu_cache_usage_pct": (
                    round(server_metrics.gpu_cache_usage_pct, 4)
                    if server_metrics.gpu_cache_usage_pct is not None
                    else None
                ),
            }
    except Exception:
        pass

    # Summary message
    mean_tps = round(statistics.mean(tps_values), 2)
    mean_lat = round(statistics.mean(latency_values), 1)
    ttft_msg = ""
    if ttft_values:
        mean_ttft = round(statistics.mean(ttft_values), 1)
        ttft_msg = f", TTFT {mean_ttft}ms"
    result["message"] = (
        f"{model_name}: {mean_tps} tok/s, {mean_lat}ms latency{ttft_msg} "
        f"({len(latency_values)}/{len(prompts) * iterations} runs)"
    )

    return result


def llm_run_benchmark(
    model_name: str = "",
    prompts: Optional[List[str]] = None,
    iterations: int = 1,
    max_tokens: int = 256,
    temperature: float = 0.0,
    measure_hardware: bool = True,
    sample_interval_ms: int = 500,
    session_id: str = "",
) -> Dict[str, Any]:
    """
    Run an LLM throughput/latency benchmark with TTFT measurement.

    Sends each prompt to the model, measures per-request latency,
    time-to-first-token, and tokens/sec.  Optionally collects Jetson
    hardware metrics (GPU utilization, temperature, power draw) during
    the benchmark run.

    Args:
        model_name: Model to benchmark. If empty, uses the first available.
        prompts: List of prompt strings. If empty, uses 5 default prompts.
        iterations: Times to repeat each prompt (1-5, default 1).
        max_tokens: Max tokens per generation (1-1024, default 256).
        temperature: Sampling temperature (default 0.0 for determinism).
        measure_hardware: Collect Jetson hardware metrics (default True).
        sample_interval_ms: Hardware sampling interval in ms (default 500).
        session_id: If provided, saves results to session storage.

    Returns:
        Benchmark results with aggregate stats and optional hardware metrics.
    """
    try:
        # Sanitize inputs
        if prompts is None or not prompts:
            prompts = DEFAULT_PROMPTS
        prompts = prompts[:MAX_PROMPTS]
        iterations = max(1, min(int(iterations), MAX_ITERATIONS))
        max_tokens = max(1, min(int(max_tokens), MAX_MAX_TOKENS))
        sample_interval_ms = max(100, min(int(sample_interval_ms), 5000))

        result = _run_benchmark_core(
            model_name=model_name,
            prompts=prompts,
            iterations=iterations,
            max_tokens=max_tokens,
            temperature=temperature,
            measure_hardware=measure_hardware,
            sample_interval_ms=sample_interval_ms,
        )

        # Persist if session provided
        if session_id:
            try:
                from eval.result_store import save_result
                filename = save_result(session_id, "benchmark", result)
                result["saved_as"] = filename
            except Exception as e:
                logger.warning("Failed to save benchmark result: %s", e)

        return ok(**result)

    except Exception as e:
        logger.error("LLM benchmark failed: %s", e, exc_info=True)
        return error_response(e, operation="llm_run_benchmark")


register_tool(
    name="llm_run_benchmark",
    func=llm_run_benchmark,
    description=(
        "Run an LLM throughput and latency benchmark with time-to-first-token "
        "(TTFT) measurement. Sends prompts to the model and measures tokens/sec, "
        "latency, and TTFT per request. Optionally collects Jetson hardware "
        "metrics (GPU utilization, temperature, power draw) during the run. "
        "Use this to measure how fast an LLM generates text on this edge device."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "model_name": {
                "type": "string",
                "description": (
                    "Name/ID of the LLM model to benchmark. "
                    "If empty, uses the first available model."
                ),
            },
            "prompts": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "List of prompts to benchmark with. "
                    "If empty, uses 5 default diverse prompts."
                ),
            },
            "iterations": {
                "type": "integer",
                "default": 1,
                "minimum": 1,
                "maximum": 5,
                "description": "Times to repeat each prompt (default 1).",
            },
            "max_tokens": {
                "type": "integer",
                "default": 256,
                "minimum": 1,
                "maximum": 1024,
                "description": "Maximum tokens to generate per prompt.",
            },
            "temperature": {
                "type": "number",
                "default": 0.0,
                "description": "Sampling temperature (0.0 for deterministic).",
            },
            "measure_hardware": {
                "type": "boolean",
                "default": True,
                "description": "Collect Jetson hardware metrics during benchmark.",
            },
            "session_id": {
                "type": "string",
                "description": "Session ID for persisting results.",
            },
        },
        "required": [],
    },
)
