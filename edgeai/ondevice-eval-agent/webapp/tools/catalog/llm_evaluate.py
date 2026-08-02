"""
LLM Evaluate Tool

Evaluates an LLM on a built-in dataset by sending each prompt, scoring
the response against the expected answer, and computing accuracy metrics
broken down by category.
"""

import logging
import statistics
import time
from typing import Any, Dict, List, Optional

from tools.base import ok, error_response
from tools.registry import register_tool

logger = logging.getLogger(__name__)

MAX_ITEMS = 100
MAX_RESPONSE_ITEMS = 20  # Limit per-item details in MCP response


def _get_llm_client():
    from client.llm_client import get_llm_client
    return get_llm_client()


def _run_evaluate_core(
    dataset_name: str,
    model_name: str,
    max_tokens: int,
    temperature: float,
    system_prompt: str,
    max_items: int,
) -> Dict[str, Any]:
    """
    Core evaluation logic, separated for reuse by llm_compare_models.

    Returns the raw result dict (not wrapped in ok/error_response).
    """
    from eval.dataset_loader import load_dataset
    from eval.scoring import score_response

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

    # Load dataset
    items = load_dataset(dataset_name)
    items = items[:max_items]

    # Run evaluation
    per_item: List[Dict[str, Any]] = []
    correct_count = 0
    total_count = 0
    latency_values: List[float] = []
    by_category: Dict[str, Dict[str, int]] = {}

    for i, item in enumerate(items):
        prompt = item["prompt"]
        expected = item["expected"]
        score_type = item.get("score_type", "contains")
        category = item.get("category", "unknown")

        # Initialize category tracking
        if category not in by_category:
            by_category[category] = {"correct": 0, "total": 0}

        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            t_start = time.perf_counter()
            resp = client.chat_completion(
                model=model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            latency_ms = (time.perf_counter() - t_start) * 1000.0

            response_text = resp.get("response", "")

            # Score the response
            score_result = score_response(response_text, expected, score_type)
            is_correct = score_result["correct"]

            if is_correct:
                correct_count += 1
                by_category[category]["correct"] += 1
            by_category[category]["total"] += 1
            total_count += 1
            latency_values.append(latency_ms)

            per_item.append({
                "index": i,
                "category": category,
                "correct": is_correct,
                "score": score_result["score"],
                "method": score_result["method"],
                "latency_ms": round(latency_ms, 1),
                "prompt": prompt[:100] + ("..." if len(prompt) > 100 else ""),
                "expected": expected[:50],
                "response": response_text[:200] + ("..." if len(response_text) > 200 else ""),
                "detail": score_result.get("detail", ""),
            })

        except Exception as e:
            logger.warning("Eval item %d failed: %s", i, e)
            by_category[category]["total"] += 1
            total_count += 1
            per_item.append({
                "index": i,
                "category": category,
                "correct": False,
                "score": 0.0,
                "error": str(e),
            })

    if total_count == 0:
        raise RuntimeError("All evaluation items failed")

    # Compute accuracy
    accuracy = correct_count / total_count
    category_accuracy = {}
    for cat, counts in sorted(by_category.items()):
        cat_total = counts["total"]
        cat_correct = counts["correct"]
        category_accuracy[cat] = {
            "correct": cat_correct,
            "total": cat_total,
            "accuracy": round(cat_correct / cat_total, 4) if cat_total > 0 else 0.0,
        }

    result: Dict[str, Any] = {
        "model_name": model_name,
        "dataset": dataset_name,
        "total_items": total_count,
        "correct": correct_count,
        "accuracy": round(accuracy, 4),
        "by_category": category_accuracy,
    }

    if latency_values:
        result["latency_ms"] = {
            "mean": round(statistics.mean(latency_values), 1),
            "min": round(min(latency_values), 1),
            "max": round(max(latency_values), 1),
        }
        if len(latency_values) >= 2:
            result["latency_ms"]["median"] = round(
                statistics.median(latency_values), 1
            )

    # Truncate per-item for MCP response
    result["per_item"] = per_item[:MAX_RESPONSE_ITEMS]
    if len(per_item) > MAX_RESPONSE_ITEMS:
        result["per_item_truncated"] = True
        result["total_per_item"] = len(per_item)

    # Full per_item stored internally for persistence
    result["_full_per_item"] = per_item

    # Summary message
    cat_summary = ", ".join(
        f"{cat}: {info['accuracy']:.0%}"
        for cat, info in sorted(category_accuracy.items())
    )
    result["message"] = (
        f"{model_name} on {dataset_name}: {accuracy:.1%} overall "
        f"({correct_count}/{total_count}). {cat_summary}"
    )

    return result


def llm_evaluate(
    dataset: str,
    model_name: str = "",
    max_tokens: int = 128,
    temperature: float = 0.0,
    system_prompt: str = "",
    max_items: int = 50,
    session_id: str = "",
) -> Dict[str, Any]:
    """
    Evaluate an LLM on a built-in dataset.

    Sends each prompt to the model, scores the response against the
    expected answer, and returns accuracy metrics broken down by category.

    Args:
        dataset: Dataset name (general_knowledge, mmlu_subset, gsm8k_subset).
        model_name: Model to evaluate. If empty, uses the first available.
        max_tokens: Max tokens per response (default 128).
        temperature: Sampling temperature (0.0 for deterministic).
        system_prompt: Optional system prompt to prepend.
        max_items: Max dataset items to evaluate (1-100, default 50).
        session_id: If provided, saves results to session storage.

    Returns:
        Evaluation results with accuracy, per-category breakdown, and per-item details.
    """
    try:
        max_items = max(1, min(int(max_items), MAX_ITEMS))
        max_tokens = max(1, min(int(max_tokens), 2048))

        result = _run_evaluate_core(
            dataset_name=dataset,
            model_name=model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            system_prompt=system_prompt,
            max_items=max_items,
        )

        # Save full results (including all per-item) if session provided
        if session_id:
            try:
                from eval.result_store import save_result
                # Use full per-item for persistence, not truncated
                save_data = dict(result)
                save_data["per_item"] = save_data.pop("_full_per_item", result["per_item"])
                filename = save_result(session_id, "eval", save_data)
                result["saved_as"] = filename
            except Exception as e:
                logger.warning("Failed to save eval result: %s", e)

        # Remove internal-only field from MCP response
        result.pop("_full_per_item", None)

        return ok(**result)

    except Exception as e:
        logger.error("LLM evaluation failed: %s", e, exc_info=True)
        return error_response(e, operation="llm_evaluate")


register_tool(
    name="llm_evaluate",
    func=llm_evaluate,
    description=(
        "Evaluate an LLM on a built-in dataset. Sends each prompt to the model, "
        "scores the response against the expected answer, and returns accuracy "
        "metrics broken down by category. "
        "Available datasets: general_knowledge (60 items: geography/science/history), "
        "mmlu_subset (80 items: stem/medicine/law/ethics), "
        "gsm8k_subset (50 math word problems). "
        "Use this to measure how accurate an LLM is on standardized tasks."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "dataset": {
                "type": "string",
                "enum": ["general_knowledge", "mmlu_subset", "gsm8k_subset"],
                "description": "Name of the evaluation dataset to use.",
            },
            "model_name": {
                "type": "string",
                "description": (
                    "Name/ID of the LLM model to evaluate. "
                    "If empty, uses the first available model."
                ),
            },
            "max_tokens": {
                "type": "integer",
                "default": 128,
                "minimum": 1,
                "maximum": 2048,
                "description": "Maximum tokens per response.",
            },
            "temperature": {
                "type": "number",
                "default": 0.0,
                "description": "Sampling temperature (0.0 for deterministic).",
            },
            "system_prompt": {
                "type": "string",
                "description": (
                    "Optional system prompt. For math tasks, consider: "
                    "'Always end your answer with the final number.'"
                ),
            },
            "max_items": {
                "type": "integer",
                "default": 50,
                "minimum": 1,
                "maximum": 100,
                "description": "Maximum number of dataset items to evaluate.",
            },
            "session_id": {
                "type": "string",
                "description": "Session ID for persisting results.",
            },
        },
        "required": ["dataset"],
    },
)
