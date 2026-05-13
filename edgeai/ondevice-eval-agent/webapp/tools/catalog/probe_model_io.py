"""
Probe Model IO Tool

Auto-probes a model's input/output behaviour by running synthetic
inference and analysing the raw output tensors.  Useful for models
the agent has never seen before.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from tools.base import ok, error_response, get_client
from tools.registry import register_tool

logger = logging.getLogger(__name__)

# Re-use the Triton-dtype -> numpy-dtype mapping from the client layer
_TRITON_TO_NP: Dict[str, "np.dtype"] = {
    "BOOL": np.dtype("bool"),
    "UINT8": np.dtype("uint8"),
    "UINT16": np.dtype("uint16"),
    "UINT32": np.dtype("uint32"),
    "UINT64": np.dtype("uint64"),
    "INT8": np.dtype("int8"),
    "INT16": np.dtype("int16"),
    "INT32": np.dtype("int32"),
    "INT64": np.dtype("int64"),
    "FP16": np.dtype("float16"),
    "FP32": np.dtype("float32"),
    "FP64": np.dtype("float64"),
    "BYTES": np.dtype("object"),
}


# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------

def _resolve_shape(shape: List[int]) -> List[int]:
    """Replace dynamic dims (-1) with sensible defaults."""
    resolved = []
    for i, dim in enumerate(shape):
        if dim == -1:
            # First dim is typically batch
            resolved.append(1 if i == 0 else 224)
        else:
            resolved.append(dim)
    return resolved


def _generate_synthetic(
    name: str,
    shape: List[int],
    dtype_str: str,
) -> tuple["np.ndarray", str]:
    """Return (numpy_array, strategy_description) for one model input."""
    resolved = _resolve_shape(shape)
    np_dtype = _TRITON_TO_NP.get(dtype_str, np.dtype("float32"))
    name_lower = name.lower()

    # Image-like: 4-D with a small channel dim
    if len(resolved) == 4:
        b, d1, d2, d3 = resolved
        is_nchw = d1 in (1, 3, 4) and d2 > 4 and d3 > 4
        is_nhwc = d3 in (1, 3, 4) and d1 > 4 and d2 > 4
        if is_nchw or is_nhwc:
            if np.issubdtype(np_dtype, np.integer):
                arr = np.random.randint(0, 256, size=resolved, dtype=np_dtype)
                return arr, "random_pixels_int"
            arr = np.random.rand(*resolved).astype(np_dtype)
            return arr, "random_pixels_float_0_1"

    # Mask inputs (typically int64 ones)
    if "mask" in name_lower:
        arr = np.ones(resolved, dtype=np_dtype)
        return arr, "ones_mask"

    # Token-ID inputs (int32/int64, 2-D)
    if np.issubdtype(np_dtype, np.integer) and len(resolved) == 2:
        arr = np.random.randint(0, 30000, size=resolved, dtype=np_dtype)
        return arr, "random_token_ids"

    # Generic float
    if np.issubdtype(np_dtype, np.floating):
        arr = np.random.randn(*resolved).astype(np_dtype)
        return arr, "random_normal"

    # Generic integer
    if np.issubdtype(np_dtype, np.integer):
        info = np.iinfo(np_dtype)
        lo = max(info.min, 0)
        hi = min(info.max, 255) + 1
        arr = np.random.randint(lo, hi, size=resolved, dtype=np_dtype)
        return arr, "random_int"

    # Fallback
    arr = np.zeros(resolved, dtype=np_dtype)
    return arr, "zeros_fallback"


# ---------------------------------------------------------------------------
# Output analysis
# ---------------------------------------------------------------------------

def _analyse_output(data: "np.ndarray") -> Dict[str, Any]:
    """Compute summary statistics for a single output tensor."""
    flat = data.flatten().astype(np.float64)
    stats: Dict[str, Any] = {
        "shape": list(data.shape),
        "dtype": str(data.dtype),
        "size": int(data.size),
    }

    if flat.size == 0:
        stats["empty"] = True
        return stats

    stats.update({
        "min": float(np.min(flat)),
        "max": float(np.max(flat)),
        "mean": float(np.mean(flat)),
        "std": float(np.std(flat)),
        "nonzero_fraction": float(np.count_nonzero(flat) / flat.size),
    })

    # Value-range classification
    all_positive = float(np.min(flat)) >= 0.0
    bounded_0_1 = all_positive and float(np.max(flat)) <= 1.01

    if bounded_0_1:
        # Check if it looks like softmax (sums to ~1 along last dim)
        try:
            last_dim_sums = data.astype(np.float64).sum(axis=-1).flatten()
            close_to_one = np.allclose(last_dim_sums, 1.0, atol=0.05)
        except Exception:
            close_to_one = False
        if close_to_one:
            stats["value_category"] = "probabilities"
            stats["looks_like_softmax"] = True
        else:
            stats["value_category"] = "normalized_0_1"
            stats["looks_like_softmax"] = False
    elif all_positive and float(np.max(flat)) <= 1000:
        stats["value_category"] = "positive_values"
    elif np.issubdtype(data.dtype, np.integer):
        unique = int(min(len(np.unique(flat[:10000])), 10000))
        stats["value_category"] = "indices"
        stats["unique_count_sample"] = unique
    else:
        stats["value_category"] = "logits"

    # Histogram (10 bins)
    try:
        counts, edges = np.histogram(flat, bins=10)
        stats["histogram"] = {
            "counts": counts.tolist(),
            "edges": [round(float(e), 4) for e in edges.tolist()],
        }
    except Exception:
        pass

    return stats


# ---------------------------------------------------------------------------
# LLM interpretation helper
# ---------------------------------------------------------------------------

def _llm_interpret(
    model_name: str,
    input_profiles: List[Dict[str, Any]],
    output_profiles: List[Dict[str, Any]],
    heuristic: Dict[str, Any],
) -> Optional[str]:
    """Ask the LLM router to explain the IO profile."""
    try:
        from router import get_router
        import json
        router = get_router()
        active = router.get_active_provider()
        if not active or not active.get("status", {}).get("available", False):
            return None

        system = (
            "You are an ML model analysis expert.  Given a model's input/output "
            "tensor profiles (shapes, dtypes, value statistics) and heuristic "
            "analysis, explain:\n"
            "1. What kind of model this is (classification, detection, etc.)\n"
            "2. What each output tensor likely represents\n"
            "3. How to post-process each output for practical use\n"
            "4. What kind of real input data the model expects\n"
            "Be concise (3-4 paragraphs max)."
        )

        user = (
            f"Model name: {model_name}\n\n"
            f"Inputs:\n{json.dumps(input_profiles, indent=2)}\n\n"
            f"Outputs:\n{json.dumps(output_profiles, indent=2, default=str)}\n\n"
            f"Heuristic analysis:\n{json.dumps(heuristic, indent=2)}"
        )

        response = router.chat(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            tools=None,
        )
        if response and response.content:
            return response.content
    except Exception as e:
        logger.warning(f"LLM interpretation failed: {e}")
    return None


# ---------------------------------------------------------------------------
# Tool function
# ---------------------------------------------------------------------------

def probe_model_io(model_name: str) -> Dict[str, Any]:
    """
    Auto-probe a model's input/output behaviour.

    Fetches metadata, generates synthetic test data matching each
    input tensor, runs inference, and analyses the raw outputs to
    determine what the model does and how to process its results.

    Args:
        model_name: Name of the model to probe.

    Returns:
        Comprehensive IO profile with input specs, output statistics,
        heuristic model-type analysis, and optional LLM interpretation.
    """
    try:
        client = get_client()

        # ------------------------------------------------------------------
        # 1. Fetch metadata
        # ------------------------------------------------------------------
        metadata = client.get_model_metadata(model_name, use_cache=False)
        if not metadata:
            return error_response(
                ValueError(f"Cannot fetch metadata for '{model_name}'"),
                operation="probe_model_io",
                model_name=model_name,
                hint="Ensure the model is loaded (even partially) on the server.",
            )

        raw_inputs = metadata.get("inputs", [])
        raw_outputs = metadata.get("outputs", [])

        if not raw_inputs:
            return error_response(
                ValueError("Model metadata lists no inputs"),
                operation="probe_model_io",
                model_name=model_name,
            )

        # ------------------------------------------------------------------
        # 2. Generate synthetic inputs
        # ------------------------------------------------------------------
        input_profiles: List[Dict[str, Any]] = []
        inference_inputs: List[tuple] = []  # (name, np_array, triton_dtype)

        for inp in raw_inputs:
            name = inp["name"]
            shape = inp["shape"]
            dtype = inp["datatype"]
            data, strategy = _generate_synthetic(name, shape, dtype)
            inference_inputs.append((name, data, dtype))
            input_profiles.append({
                "name": name,
                "original_shape": shape,
                "resolved_shape": list(data.shape),
                "dtype": dtype,
                "synthetic_strategy": strategy,
            })

        # ------------------------------------------------------------------
        # 3. Run inference
        # ------------------------------------------------------------------
        inference_succeeded = False
        inference_error = None
        output_profiles: List[Dict[str, Any]] = []

        try:
            result = client.send_raw_inference(model_name, inference_inputs)
            inference_succeeded = True

            for out in result.get("outputs", []):
                data = out["data"]  # numpy array
                stats = _analyse_output(data)
                stats["name"] = out["name"]
                stats["triton_dtype"] = out["datatype"]
                output_profiles.append(stats)
        except Exception as e:
            inference_error = str(e)
            logger.warning(
                f"Synthetic inference failed for {model_name}: {e}"
            )
            # Still build output profiles from metadata alone
            for out in raw_outputs:
                output_profiles.append({
                    "name": out["name"],
                    "shape": out["shape"],
                    "dtype": out["datatype"],
                    "note": "statistics unavailable (inference failed)",
                })

        # ------------------------------------------------------------------
        # 4. Heuristic model-type analysis (reuse existing logic)
        # ------------------------------------------------------------------
        try:
            from tools.catalog.model_type import infer_model_type_from_shapes
            input_spec = client.get_model_input_spec(model_name)
            output_specs = client.get_all_output_specs(model_name)
            heuristic = infer_model_type_from_shapes(input_spec, output_specs)
        except Exception:
            heuristic = {"type": "unknown", "confidence": "low",
                         "reasoning": "Heuristic analysis unavailable"}

        # ------------------------------------------------------------------
        # 5. LLM interpretation (optional)
        # ------------------------------------------------------------------
        # Strip numpy arrays before passing to LLM
        serialisable_outputs = []
        for p in output_profiles:
            clean = {k: v for k, v in p.items()
                     if not isinstance(v, np.ndarray)}
            serialisable_outputs.append(clean)

        llm_text = _llm_interpret(
            model_name, input_profiles, serialisable_outputs, heuristic,
        )

        # ------------------------------------------------------------------
        # 6. Return
        # ------------------------------------------------------------------
        warnings = []
        if not inference_succeeded:
            warnings.append(
                f"Synthetic inference failed: {inference_error}. "
                "Output statistics are unavailable; only metadata is shown."
            )
        if heuristic.get("confidence") == "low":
            warnings.append(
                "Model type confidence is low.  Run real inference or "
                "use web_search / search_model_info to learn more."
            )

        return ok(
            warnings=warnings or None,
            model_name=model_name,
            inputs=input_profiles,
            outputs=serialisable_outputs,
            inference_succeeded=inference_succeeded,
            inference_error=inference_error,
            heuristic_analysis=heuristic,
            llm_interpretation=llm_text,
            analysis_source="heuristic_and_llm" if llm_text else "heuristic_only",
            message=(
                f"IO profile for '{model_name}': "
                f"{len(raw_inputs)} input(s), {len(raw_outputs)} output(s). "
                f"Inferred type: {heuristic.get('type', 'unknown')} "
                f"({heuristic.get('confidence', 'unknown')} confidence)."
            ),
        )

    except Exception as e:
        logger.error(f"Error probing model IO for {model_name}: {e}",
                     exc_info=True)
        return error_response(e, operation="probe_model_io",
                              model_name=model_name)


# ---------------------------------------------------------------------------
# Register
# ---------------------------------------------------------------------------

register_tool(
    name="probe_model_io",
    func=probe_model_io,
    description=(
        "Auto-probe an unknown model's input/output behaviour.  Generates "
        "synthetic test data, runs inference, and analyses raw output tensors "
        "(shape, statistics, value ranges) to determine what the model does "
        "and how to interpret its results.  Returns a comprehensive IO "
        "profile with optional LLM-generated interpretation."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "model_name": {
                "type": "string",
                "description": "Name of the model to probe",
            },
        },
        "required": ["model_name"],
    },
)
