"""
Compare Models Tool

Side-by-side comparison of two models covering inputs, outputs,
readiness, and inferred model type.
"""

import logging
from typing import Any, Dict

from tools.base import ok, error_response, get_client
from tools.registry import register_tool

logger = logging.getLogger(__name__)


def _model_summary(client: Any, model_name: str) -> Dict[str, Any]:
    """Build a summary dict for a single model."""
    summary: Dict[str, Any] = {"model_name": model_name}

    try:
        summary["ready"] = client.check_model_ready(model_name)
    except Exception:
        summary["ready"] = None

    try:
        input_spec = client.get_model_input_spec(model_name)
        if hasattr(input_spec, "to_dict"):
            input_spec = input_spec.to_dict()
        summary["input_spec"] = input_spec
    except Exception as e:
        summary["input_spec"] = {"error": str(e)}

    try:
        output_spec = client.get_model_output_spec(model_name)
        if hasattr(output_spec, "to_dict"):
            output_spec = output_spec.to_dict()
        summary["output_spec"] = output_spec
    except Exception as e:
        summary["output_spec"] = {"error": str(e)}

    try:
        all_outputs = client.get_all_output_specs(model_name)
        summary["output_count"] = len(all_outputs)
    except Exception:
        summary["output_count"] = None

    try:
        summary["input_shape"] = client.get_model_input_shape(model_name)
    except Exception:
        summary["input_shape"] = None

    return summary


def compare_models(model_a: str, model_b: str) -> Dict[str, Any]:
    """
    Compare two models side-by-side.

    Returns input specs, output specs, readiness, and input shapes
    for both models, plus a ``differences`` section highlighting
    key discrepancies.

    Args:
        model_a: Name of the first model.
        model_b: Name of the second model.

    Returns:
        Dict with per-model summaries and a differences section.
    """
    try:
        client = get_client()

        summary_a = _model_summary(client, model_a)
        summary_b = _model_summary(client, model_b)

        # Build a human-readable differences section
        diffs = []

        if summary_a.get("ready") != summary_b.get("ready"):
            diffs.append(
                f"Readiness: {model_a}={'ready' if summary_a.get('ready') else 'not ready'}, "
                f"{model_b}={'ready' if summary_b.get('ready') else 'not ready'}"
            )

        shape_a = summary_a.get("input_shape")
        shape_b = summary_b.get("input_shape")
        if shape_a != shape_b:
            diffs.append(f"Input shapes differ: {model_a}={shape_a}, {model_b}={shape_b}")

        out_count_a = summary_a.get("output_count")
        out_count_b = summary_b.get("output_count")
        if out_count_a != out_count_b:
            diffs.append(
                f"Output tensor count: {model_a}={out_count_a}, {model_b}={out_count_b}"
            )

        if not diffs:
            diffs.append("No significant differences detected in the inspected fields.")

        return ok(
            model_a=summary_a,
            model_b=summary_b,
            differences=diffs,
            message=f"Comparison of '{model_a}' vs '{model_b}' complete.",
        )
    except Exception as e:
        logger.error(f"Error comparing models {model_a} and {model_b}: {e}")
        return error_response(
            e,
            operation="compare_models",
            model_a=model_a,
            model_b=model_b,
        )


# Register the tool
register_tool(
    name="compare_models",
    func=compare_models,
    description=(
        "Compare two models side-by-side. Returns input specs, output specs, "
        "readiness, and input shapes for both models with a differences summary. "
        "Useful when choosing between models or debugging deployment issues."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "model_a": {
                "type": "string",
                "description": "Name of the first model",
            },
            "model_b": {
                "type": "string",
                "description": "Name of the second model",
            },
        },
        "required": ["model_a", "model_b"],
    },
)
