"""
LLM Performance Tool

Retrieves performance metrics for LLM serving backends (vLLM, llama.cpp).
Supports both Prometheus server-side metrics (vLLM) and inference-based
measurement (send a standard prompt, measure tokens/sec).
"""

import logging
import statistics
from typing import Any, Dict, List, Optional

from tools.base import ok, error_response
from tools.registry import register_tool

logger = logging.getLogger(__name__)

DEFAULT_BENCH_PROMPT = "Explain the theory of relativity in simple terms."
DEFAULT_BENCH_ITERATIONS = 3
MAX_BENCH_ITERATIONS = 20


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


def llm_get_performance(
    model_name: str = "",
    iterations: int = DEFAULT_BENCH_ITERATIONS,
    prompt: str = DEFAULT_BENCH_PROMPT,
    max_tokens: int = 128,
) -> Dict[str, Any]:
    """
    Get performance metrics for an LLM model.

    Strategy:
        1. Fetch Prometheus server-side metrics if available (vLLM).
        2. Run inference-based benchmarking: send a prompt N times,
           measure tokens/sec, latency, and token counts.

    Args:
        model_name: Model to benchmark. If empty, uses the first available model.
        iterations: Number of benchmark iterations (1-20, default 3).
        prompt: Prompt to use for inference-based benchmarking.
        max_tokens: Max tokens to generate per iteration (default 128).

    Returns:
        Performance metrics including tokens/sec, latency, and optional
        server-side Prometheus metrics.
    """
    try:
        client = _get_llm_client()

        if not client.is_healthy():
            return error_response(
                ConnectionError(
                    f"LLM server at {client.base_url} is not reachable"
                ),
                operation="llm_get_performance",
            )

        # Resolve model name if not provided
        if not model_name:
            models = client.list_models()
            if not models:
                return error_response(
                    ValueError("No LLM models available on the server"),
                    operation="llm_get_performance",
                )
            model_name = models[0].id

        iterations = max(1, min(int(iterations), MAX_BENCH_ITERATIONS))
        max_tokens = max(1, min(int(max_tokens), 2048))

        result: Dict[str, Any] = {
            "model_name": model_name,
            "server_url": client.base_url,
            "server_type": client.server_type.value,
        }

        # ------------------------------------------------------------------
        # 1. Server-side Prometheus metrics (vLLM)
        # ------------------------------------------------------------------
        server_metrics = client.get_server_metrics()
        if server_metrics is not None:
            sm: Dict[str, Any] = {}
            if server_metrics.avg_generation_throughput_tps is not None:
                sm["avg_generation_throughput_tps"] = round(
                    server_metrics.avg_generation_throughput_tps, 2
                )
            if server_metrics.avg_prompt_throughput_tps is not None:
                sm["avg_prompt_throughput_tps"] = round(
                    server_metrics.avg_prompt_throughput_tps, 2
                )
            if server_metrics.running_requests is not None:
                sm["running_requests"] = server_metrics.running_requests
            if server_metrics.waiting_requests is not None:
                sm["waiting_requests"] = server_metrics.waiting_requests
            if server_metrics.gpu_cache_usage_pct is not None:
                sm["gpu_cache_usage_pct"] = round(
                    server_metrics.gpu_cache_usage_pct, 4
                )
            if sm:
                result["server_metrics"] = sm

        # ------------------------------------------------------------------
        # 2. Inference-based benchmarking
        # ------------------------------------------------------------------
        messages = [{"role": "user", "content": prompt}]

        tps_values: List[float] = []
        latency_values: List[float] = []
        prompt_tok_values: List[int] = []
        completion_tok_values: List[int] = []
        per_iteration: List[Dict[str, Any]] = []

        for i in range(iterations):
            try:
                resp = client.chat_completion(
                    model=model_name,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=0.0,
                )

                perf = resp["performance"]
                usage = resp["usage"]

                tps_values.append(perf["tokens_per_second"])
                latency_values.append(perf["total_time_ms"])
                prompt_tok_values.append(usage["prompt_tokens"])
                completion_tok_values.append(usage["completion_tokens"])

                per_iteration.append({
                    "iteration": i + 1,
                    "tokens_per_second": perf["tokens_per_second"],
                    "total_time_ms": perf["total_time_ms"],
                    "prompt_tokens": usage["prompt_tokens"],
                    "completion_tokens": usage["completion_tokens"],
                })
            except Exception as e:
                logger.warning("Benchmark iteration %d failed: %s", i + 1, e)
                per_iteration.append({
                    "iteration": i + 1,
                    "error": str(e),
                })

        if not tps_values:
            return error_response(
                RuntimeError("All benchmark iterations failed"),
                operation="llm_get_performance",
                model_name=model_name,
            )

        bench: Dict[str, Any] = {
            "prompt_used": prompt,
            "max_tokens": max_tokens,
            "iterations": iterations,
            "successful_iterations": len(tps_values),
            "tokens_per_second": _compute_stats(tps_values),
            "latency_ms": _compute_stats(latency_values),
            "avg_prompt_tokens": round(statistics.mean(prompt_tok_values), 1),
            "avg_completion_tokens": round(
                statistics.mean(completion_tok_values), 1
            ),
        }

        if iterations <= 10:
            bench["per_iteration"] = per_iteration

        result["benchmark"] = bench

        # Summary message
        mean_tps = round(statistics.mean(tps_values), 2)
        mean_lat = round(statistics.mean(latency_values), 1)
        summary = (
            f"{model_name}: {mean_tps} tokens/sec "
            f"(mean latency {mean_lat}ms, {len(tps_values)}/{iterations} iterations)"
        )
        if server_metrics and server_metrics.avg_generation_throughput_tps is not None:
            summary += (
                f" | Server throughput: "
                f"{server_metrics.avg_generation_throughput_tps:.1f} tok/s"
            )

        result["message"] = summary

        return ok(**result)

    except Exception as e:
        logger.error(
            "Error measuring LLM performance: %s", e, exc_info=True
        )
        return error_response(e, operation="llm_get_performance")


register_tool(
    name="llm_get_performance",
    func=llm_get_performance,
    description=(
        "Measure LLM performance metrics including tokens per second, latency, "
        "and throughput. Runs inference-based benchmarking by sending a prompt "
        "multiple times and measuring generation speed. For vLLM servers, also "
        "fetches server-side Prometheus metrics (generation throughput, prompt "
        "throughput, running/waiting requests, GPU cache usage). "
        "Use this when the user asks about LLM speed, tokens per second, "
        "throughput, or performance benchmarks."
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
            "iterations": {
                "type": "integer",
                "default": 3,
                "minimum": 1,
                "maximum": 20,
                "description": (
                    "Number of benchmark iterations (default 3). "
                    "More iterations give more reliable statistics."
                ),
            },
            "prompt": {
                "type": "string",
                "default": DEFAULT_BENCH_PROMPT,
                "description": "Prompt to use for benchmarking.",
            },
            "max_tokens": {
                "type": "integer",
                "default": 128,
                "minimum": 1,
                "maximum": 2048,
                "description": "Maximum tokens to generate per iteration.",
            },
        },
        "required": [],
    },
)
