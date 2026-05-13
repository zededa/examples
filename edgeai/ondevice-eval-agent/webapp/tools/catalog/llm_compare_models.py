"""
LLM Compare Models Tool

Runs the same benchmark or evaluation on two models and returns a
side-by-side comparison with deltas and winner per metric.

Models are run sequentially — on a single-GPU edge device, concurrent
LLM inference would cause OOM or severe thrashing.
"""

import logging
from typing import Any, Dict, List, Optional

from tools.base import ok, error_response
from tools.registry import register_tool

logger = logging.getLogger(__name__)


def _compute_comparison(
    a_stats: Dict[str, Any],
    b_stats: Dict[str, Any],
    higher_is_better: bool = True,
) -> Dict[str, Any]:
    """Compare two stat dicts and determine the winner."""
    a_mean = a_stats.get("mean", 0)
    b_mean = b_stats.get("mean", 0)

    if a_mean == 0 and b_mean == 0:
        return {"delta": 0, "pct_change": 0, "winner": "tie"}

    delta = round(b_mean - a_mean, 3)
    pct_change = round((delta / a_mean) * 100, 1) if a_mean != 0 else 0

    if higher_is_better:
        winner = "model_b" if b_mean > a_mean else ("model_a" if a_mean > b_mean else "tie")
    else:
        winner = "model_b" if b_mean < a_mean else ("model_a" if a_mean < b_mean else "tie")

    return {
        "model_a_mean": a_mean,
        "model_b_mean": b_mean,
        "delta": delta,
        "pct_change": pct_change,
        "winner": winner,
    }


def llm_compare_models(
    model_a: str,
    model_b: str,
    mode: str = "benchmark",
    dataset: str = "",
    prompts: Optional[List[str]] = None,
    iterations: int = 1,
    max_tokens: int = 256,
    temperature: float = 0.0,
    session_id: str = "",
) -> Dict[str, Any]:
    """
    Compare two LLM models side-by-side on benchmark or evaluation tasks.

    Runs the same workload on both models sequentially, then computes
    deltas and determines a winner per metric.

    Args:
        model_a: First model name/ID.
        model_b: Second model name/ID.
        mode: Comparison mode — ``benchmark``, ``eval``, or ``both``.
        dataset: Dataset name (required for ``eval`` and ``both`` modes).
        prompts: Prompts for benchmark mode (uses defaults if empty).
        iterations: Iterations per prompt for benchmark (default 1).
        max_tokens: Max tokens per generation (default 256).
        temperature: Sampling temperature (default 0.0).
        session_id: If provided, saves results to session storage.

    Returns:
        Side-by-side comparison with per-metric winners.
    """
    try:
        if not model_a or not model_b:
            return error_response(
                ValueError("Both model_a and model_b must be specified"),
                operation="llm_compare_models",
            )

        if mode not in ("benchmark", "eval", "both"):
            return error_response(
                ValueError(f"Invalid mode '{mode}'. Must be benchmark, eval, or both"),
                operation="llm_compare_models",
            )

        if mode in ("eval", "both") and not dataset:
            return error_response(
                ValueError("Dataset is required for eval and both modes"),
                operation="llm_compare_models",
            )

        result: Dict[str, Any] = {
            "model_a": {"model_name": model_a},
            "model_b": {"model_name": model_b},
            "mode": mode,
            "comparison": {},
        }

        # Run benchmark comparison
        if mode in ("benchmark", "both"):
            from tools.catalog.llm_run_benchmark import _run_benchmark_core

            logger.info("Benchmarking model_a: %s", model_a)
            a_bench = _run_benchmark_core(
                model_name=model_a,
                prompts=prompts or [],
                iterations=iterations,
                max_tokens=max_tokens,
                temperature=temperature,
                measure_hardware=True,
                sample_interval_ms=500,
            )

            logger.info("Benchmarking model_b: %s", model_b)
            b_bench = _run_benchmark_core(
                model_name=model_b,
                prompts=prompts or [],
                iterations=iterations,
                max_tokens=max_tokens,
                temperature=temperature,
                measure_hardware=True,
                sample_interval_ms=500,
            )

            result["model_a"]["benchmark"] = a_bench
            result["model_b"]["benchmark"] = b_bench

            # Compare benchmark metrics
            bench_comparison: Dict[str, Any] = {}
            a_agg = a_bench.get("aggregate", {})
            b_agg = b_bench.get("aggregate", {})

            if "tokens_per_second" in a_agg and "tokens_per_second" in b_agg:
                bench_comparison["tokens_per_second"] = _compute_comparison(
                    a_agg["tokens_per_second"], b_agg["tokens_per_second"],
                    higher_is_better=True,
                )
            if "latency_ms" in a_agg and "latency_ms" in b_agg:
                bench_comparison["latency_ms"] = _compute_comparison(
                    a_agg["latency_ms"], b_agg["latency_ms"],
                    higher_is_better=False,
                )
            if "ttft_ms" in a_agg and "ttft_ms" in b_agg:
                bench_comparison["ttft_ms"] = _compute_comparison(
                    a_agg["ttft_ms"], b_agg["ttft_ms"],
                    higher_is_better=False,
                )

            result["comparison"]["benchmark"] = bench_comparison

        # Run evaluation comparison
        if mode in ("eval", "both"):
            from tools.catalog.llm_evaluate import _run_evaluate_core

            logger.info("Evaluating model_a: %s on %s", model_a, dataset)
            a_eval = _run_evaluate_core(
                dataset_name=dataset,
                model_name=model_a,
                max_tokens=max_tokens,
                temperature=temperature,
                system_prompt="",
                max_items=50,
            )
            # Remove internal field
            a_eval.pop("_full_per_item", None)

            logger.info("Evaluating model_b: %s on %s", model_b, dataset)
            b_eval = _run_evaluate_core(
                dataset_name=dataset,
                model_name=model_b,
                max_tokens=max_tokens,
                temperature=temperature,
                system_prompt="",
                max_items=50,
            )
            b_eval.pop("_full_per_item", None)

            result["model_a"]["eval"] = a_eval
            result["model_b"]["eval"] = b_eval

            # Compare eval metrics
            eval_comparison: Dict[str, Any] = {
                "accuracy": {
                    "model_a": a_eval.get("accuracy", 0),
                    "model_b": b_eval.get("accuracy", 0),
                    "delta": round(
                        b_eval.get("accuracy", 0) - a_eval.get("accuracy", 0), 4
                    ),
                    "winner": (
                        "model_b"
                        if b_eval.get("accuracy", 0) > a_eval.get("accuracy", 0)
                        else (
                            "model_a"
                            if a_eval.get("accuracy", 0) > b_eval.get("accuracy", 0)
                            else "tie"
                        )
                    ),
                },
            }

            # Per-category accuracy comparison
            all_categories = set(
                list(a_eval.get("by_category", {}).keys())
                + list(b_eval.get("by_category", {}).keys())
            )
            if all_categories:
                cat_comparison: Dict[str, Any] = {}
                for cat in sorted(all_categories):
                    a_cat = a_eval.get("by_category", {}).get(cat, {})
                    b_cat = b_eval.get("by_category", {}).get(cat, {})
                    a_acc = a_cat.get("accuracy", 0)
                    b_acc = b_cat.get("accuracy", 0)
                    cat_comparison[cat] = {
                        "model_a": a_acc,
                        "model_b": b_acc,
                        "delta": round(b_acc - a_acc, 4),
                        "winner": (
                            "model_b" if b_acc > a_acc
                            else ("model_a" if a_acc > b_acc else "tie")
                        ),
                    }
                eval_comparison["by_category"] = cat_comparison

            result["comparison"]["eval"] = eval_comparison

        # Build summary message
        summary_parts = [f"Comparison: {model_a} vs {model_b}"]
        comp = result.get("comparison", {})

        if "benchmark" in comp:
            tps = comp["benchmark"].get("tokens_per_second", {})
            if tps:
                summary_parts.append(
                    f"Throughput: {tps.get('model_a_mean', '?')} vs "
                    f"{tps.get('model_b_mean', '?')} tok/s "
                    f"({tps.get('pct_change', 0):+.1f}%, winner: {tps.get('winner', '?')})"
                )

        if "eval" in comp:
            acc = comp["eval"].get("accuracy", {})
            if acc:
                a_pct = f"{acc.get('model_a', 0):.1%}"
                b_pct = f"{acc.get('model_b', 0):.1%}"
                summary_parts.append(
                    f"Accuracy: {a_pct} vs {b_pct} "
                    f"(winner: {acc.get('winner', '?')})"
                )

        result["message"] = " | ".join(summary_parts)

        # Persist
        if session_id:
            try:
                from eval.result_store import save_result
                filename = save_result(session_id, "comparison", result)
                result["saved_as"] = filename
            except Exception as e:
                logger.warning("Failed to save comparison result: %s", e)

        return ok(**result)

    except Exception as e:
        logger.error("LLM comparison failed: %s", e, exc_info=True)
        return error_response(e, operation="llm_compare_models")


register_tool(
    name="llm_compare_models",
    func=llm_compare_models,
    description=(
        "Compare two LLM models side-by-side. Runs the same benchmark or "
        "evaluation on both models and returns a comparison with deltas and "
        "winner per metric (throughput, latency, TTFT, accuracy). "
        "Models are run sequentially (edge device — no concurrent LLM inference). "
        "Use this to compare different models or quantization levels. "
        "Requires: model_a, model_b. Optional: mode (benchmark/eval/both), "
        "dataset (required for eval), prompts, iterations."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "model_a": {
                "type": "string",
                "description": "First model name/ID.",
            },
            "model_b": {
                "type": "string",
                "description": "Second model name/ID.",
            },
            "mode": {
                "type": "string",
                "enum": ["benchmark", "eval", "both"],
                "default": "benchmark",
                "description": "Comparison mode: benchmark, eval, or both.",
            },
            "dataset": {
                "type": "string",
                "enum": ["general_knowledge", "mmlu_subset", "gsm8k_subset"],
                "description": "Dataset for eval mode (required for eval/both).",
            },
            "prompts": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Prompts for benchmark mode.",
            },
            "iterations": {
                "type": "integer",
                "default": 1,
                "minimum": 1,
                "maximum": 5,
                "description": "Iterations per prompt for benchmark.",
            },
            "max_tokens": {
                "type": "integer",
                "default": 256,
                "description": "Maximum tokens per generation.",
            },
            "temperature": {
                "type": "number",
                "default": 0.0,
                "description": "Sampling temperature.",
            },
            "session_id": {
                "type": "string",
                "description": "Session ID for persisting results.",
            },
        },
        "required": ["model_a", "model_b"],
    },
)
