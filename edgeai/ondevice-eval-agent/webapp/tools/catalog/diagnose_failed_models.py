"""
Diagnose Failed Models Tool

Scans the Triton model repository for models that failed to load,
categorises the errors, and optionally uses an LLM to generate
human-readable diagnoses with fix suggestions.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from tools.base import ok, error_response, get_client
from tools.registry import register_tool

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Error categorisation
# ---------------------------------------------------------------------------

_ERROR_PATTERNS: List[tuple[str, list[str]]] = [
    ("config_error", ["config", "pbtxt", "configuration", "parameter",
                       "parse", "invalid argument"]),
    ("shape_error", ["shape", "dimension", "mismatch", "reshape",
                      "size of", "incompatible"]),
    ("missing_files", ["not found", "missing", "no file", "does not exist",
                        "no such file", "failed to open"]),
    ("unsupported_ops", ["unsupported", "operator", "op_type", "opset",
                          "not implemented"]),
    ("backend_error", ["backend", "runtime", "onnxruntime", "tensorrt",
                        "onnx runtime", "libtorch"]),
    ("memory_error", ["memory", "oom", "allocation", "out of memory",
                       "cuda error"]),
    ("version_error", ["version", "version_policy"]),
]


def _categorise_error(reason: str) -> str:
    """Map a Triton error reason string to a category."""
    reason_lower = reason.lower()
    for category, keywords in _ERROR_PATTERNS:
        if any(kw in reason_lower for kw in keywords):
            return category
    return "unknown"


def _quick_fix_hint(category: str) -> str:
    """Return a short hint for the error category."""
    hints = {
        "config_error": (
            "The config.pbtxt has syntax or semantic errors.  Use "
            "fix_model_config to auto-generate a corrected config."
        ),
        "shape_error": (
            "Input or output tensor shapes in the config don't match "
            "the actual model.  Use fix_model_config with auto_fix=True."
        ),
        "missing_files": (
            "Model file(s) are missing from the model repository.  "
            "Check that the storage-initializer and model-copier ran "
            "successfully."
        ),
        "unsupported_ops": (
            "The model contains operators not supported by the backend.  "
            "Consider converting the model or switching to a compatible "
            "backend (e.g. onnxruntime instead of tensorrt)."
        ),
        "backend_error": (
            "The inference backend reported an internal error.  Check "
            "Triton server logs for details.  Try reloading the model "
            "or switching the backend/platform in the config."
        ),
        "memory_error": (
            "The server ran out of memory.  Try reducing max_batch_size, "
            "using a smaller model, or freeing GPU memory."
        ),
        "version_error": (
            "Version policy in config.pbtxt may be misconfigured.  "
            "Ensure the model version directory exists (e.g. 1/)."
        ),
        "unknown": (
            "The error does not match a known pattern.  Check raw_error "
            "for details and consult Triton server logs."
        ),
    }
    return hints.get(category, hints["unknown"])


# ---------------------------------------------------------------------------
# LLM diagnosis helper
# ---------------------------------------------------------------------------

def _llm_diagnose(
    diagnoses: List[Dict[str, Any]],
) -> Optional[str]:
    """Use the LLM router to generate a human-readable diagnosis."""
    try:
        from router import get_router
        import json

        router = get_router()
        active = router.get_active_provider()
        if not active or not active.get("status", {}).get("available", False):
            return None

        system = (
            "You are an NVIDIA Triton Inference Server expert.  "
            "Given model loading failures with their error messages "
            "and categories, provide a concise diagnosis for each "
            "model: (1) plain-English root cause, (2) concrete fix steps.  "
            "Be actionable and specific.  2-3 sentences per model max."
        )

        user = (
            "The following models failed to load on Triton:\n\n"
            + "\n\n".join(
                f"**{d['model_name']}** (category: {d['error_category']})\n"
                f"Error: {d['raw_error']}\n"
                f"Metadata available: {d['metadata_available']}\n"
                f"Config available: {d['config_available']}"
                for d in diagnoses
            )
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
        logger.warning(f"LLM diagnosis failed: {e}")
    return None


# ---------------------------------------------------------------------------
# Tool function
# ---------------------------------------------------------------------------

def diagnose_failed_models() -> Dict[str, Any]:
    """
    Scan the model repository for models that failed to load.

    Returns a structured report with error categorisation, diagnostic
    information, and optional LLM-generated fix suggestions.
    """
    try:
        client = get_client()

        # 1. Get full repository index
        try:
            index = client.get_repository_index()
        except Exception as e:
            return error_response(
                e,
                operation="diagnose_failed_models",
                hint=(
                    "Repository index is Triton-specific.  This server "
                    "may be OpenVINO or the endpoint may be unavailable."
                ),
            )

        if not index:
            return ok(
                failed_models=[],
                total_models=0,
                failed_count=0,
                ready_count=0,
                message="No models found in the repository.",
            )

        # 2. Separate healthy from failed
        ready_models = []
        failed_entries = []
        for entry in index:
            state = (entry.get("state") or "").upper()
            if state == "READY" or state == "":
                ready_models.append(entry["name"])
            else:
                failed_entries.append(entry)

        if not failed_entries:
            return ok(
                failed_models=[],
                total_models=len(index),
                failed_count=0,
                ready_count=len(ready_models),
                ready_models=ready_models,
                message="All models are healthy and READY.",
            )

        # 3. Diagnose each failed model
        diagnoses: List[Dict[str, Any]] = []
        category_counts: Dict[str, int] = {}

        for entry in failed_entries:
            model_name = entry["name"]
            state = entry.get("state", "UNKNOWN")
            reason = entry.get("reason", "No reason provided")
            category = _categorise_error(reason)
            category_counts[category] = category_counts.get(category, 0) + 1

            # Try to fetch metadata / config (may succeed even if UNAVAILABLE)
            metadata = None
            config = None
            try:
                metadata = client.get_model_metadata(model_name,
                                                     use_cache=False)
            except Exception:
                pass
            try:
                config = client.get_model_config(model_name, use_cache=False)
            except Exception:
                pass

            diagnoses.append({
                "model_name": model_name,
                "state": state,
                "raw_error": reason,
                "error_category": category,
                "fix_hint": _quick_fix_hint(category),
                "metadata_available": metadata is not None,
                "config_available": config is not None,
                "metadata": metadata,
                "config": config,
            })

        # 4. Optional LLM diagnosis
        llm_text = _llm_diagnose(diagnoses)

        return ok(
            failed_models=diagnoses,
            total_models=len(index),
            failed_count=len(diagnoses),
            ready_count=len(ready_models),
            ready_models=ready_models,
            error_categories=category_counts,
            llm_diagnosis=llm_text,
            message=(
                f"{len(diagnoses)} model(s) have loading issues "
                f"out of {len(index)} total."
            ),
        )

    except Exception as e:
        logger.error(f"Error diagnosing failed models: {e}", exc_info=True)
        return error_response(e, operation="diagnose_failed_models")


# ---------------------------------------------------------------------------
# Register
# ---------------------------------------------------------------------------

register_tool(
    name="diagnose_failed_models",
    func=diagnose_failed_models,
    description=(
        "Scan the Triton model repository for models that failed to load.  "
        "Returns a structured report with error categorisation, root cause "
        "analysis, and suggested fixes for each failed model.  "
        "No arguments needed."
    ),
    input_schema={
        "type": "object",
        "properties": {},
        "required": [],
    },
)
