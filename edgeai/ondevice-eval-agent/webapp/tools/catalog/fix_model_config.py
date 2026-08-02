"""
Fix Model Config Tool

Generates a corrected config.pbtxt (as JSON) for a model and reloads
it on the Triton server via the ``load_model()`` gRPC API.  Can
auto-derive correct tensor specifications from model metadata or
accept explicit overrides.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional

from tools.base import ok, error_response, get_client
from tools.registry import register_tool

logger = logging.getLogger(__name__)

# Triton metadata dtype (e.g. "FP32") -> config protobuf dtype (e.g. "TYPE_FP32")
_DTYPE_TO_CONFIG: Dict[str, str] = {
    "BOOL": "TYPE_BOOL",
    "UINT8": "TYPE_UINT8", "UINT16": "TYPE_UINT16",
    "UINT32": "TYPE_UINT32", "UINT64": "TYPE_UINT64",
    "INT8": "TYPE_INT8", "INT16": "TYPE_INT16",
    "INT32": "TYPE_INT32", "INT64": "TYPE_INT64",
    "FP16": "TYPE_FP16", "FP32": "TYPE_FP32", "FP64": "TYPE_FP64",
    "BYTES": "TYPE_STRING", "BF16": "TYPE_BF16",
}


def _to_config_dtype(dtype_str: str) -> str:
    """Convert any dtype string to config.pbtxt ``TYPE_*`` format."""
    if dtype_str.startswith("TYPE_"):
        return dtype_str
    return _DTYPE_TO_CONFIG.get(dtype_str, f"TYPE_{dtype_str}")


def _strip_batch_dim(shape: List[int], max_batch_size: int) -> List[int]:
    """
    Return dims appropriate for config.pbtxt.

    When ``max_batch_size > 0`` the leading batch dimension must be
    omitted from the ``dims`` field.  When ``max_batch_size == 0``
    the full shape (including batch) is kept.
    """
    if max_batch_size > 0 and len(shape) > 1:
        return shape[1:]
    return shape


def _build_config(
    model_name: str,
    metadata: Optional[Dict[str, Any]],
    existing_config: Optional[Dict[str, Any]],
    *,
    max_batch_size: Optional[int],
    input_overrides: Optional[List[Dict[str, Any]]],
    output_overrides: Optional[List[Dict[str, Any]]],
    platform: Optional[str],
    backend: Optional[str],
) -> Dict[str, Any]:
    """Build a corrected model config dict from available information."""

    config: Dict[str, Any] = {"name": model_name}

    # --- Platform / backend ---
    if platform:
        config["platform"] = platform
    elif existing_config and existing_config.get("platform"):
        config["platform"] = existing_config["platform"]
    elif metadata:
        plat = metadata.get("platform", "")
        if plat:
            config["platform"] = plat
        else:
            config["platform"] = "onnxruntime_onnx"
    else:
        config["platform"] = "onnxruntime_onnx"

    if backend:
        config["backend"] = backend
    elif existing_config and existing_config.get("backend"):
        config["backend"] = existing_config["backend"]

    # --- max_batch_size ---
    if max_batch_size is not None:
        effective_batch = max_batch_size
    elif existing_config and "max_batch_size" in existing_config:
        effective_batch = existing_config["max_batch_size"]
    elif metadata:
        # Heuristic: if first input dim is -1 (dynamic), allow batching
        inputs = metadata.get("inputs", [])
        if inputs and inputs[0].get("shape", [None])[0] == -1:
            effective_batch = 1
        else:
            effective_batch = 0
    else:
        effective_batch = 0

    config["max_batch_size"] = effective_batch

    # --- Inputs ---
    if input_overrides:
        config["input"] = [
            {
                "name": o["name"],
                "data_type": _to_config_dtype(o.get("data_type", "FP32")),
                "dims": o["dims"],
            }
            for o in input_overrides
        ]
    elif metadata and metadata.get("inputs"):
        config["input"] = [
            {
                "name": inp["name"],
                "data_type": _to_config_dtype(inp["datatype"]),
                "dims": _strip_batch_dim(
                    [d if d != -1 else -1 for d in inp["shape"]],
                    effective_batch,
                ),
            }
            for inp in metadata["inputs"]
        ]
    elif existing_config and existing_config.get("input"):
        config["input"] = existing_config["input"]

    # --- Outputs ---
    if output_overrides:
        config["output"] = [
            {
                "name": o["name"],
                "data_type": _to_config_dtype(o.get("data_type", "FP32")),
                "dims": o["dims"],
            }
            for o in output_overrides
        ]
    elif metadata and metadata.get("outputs"):
        config["output"] = [
            {
                "name": out["name"],
                "data_type": _to_config_dtype(out["datatype"]),
                "dims": _strip_batch_dim(
                    [d if d != -1 else -1 for d in out["shape"]],
                    effective_batch,
                ),
            }
            for out in metadata["outputs"]
        ]
    elif existing_config and existing_config.get("output"):
        config["output"] = existing_config["output"]

    return config


def _render_pbtxt(config: Dict[str, Any]) -> str:
    """Render config dict as a human-readable pbtxt-style string."""
    try:
        from tools.catalog.model_config import _to_pbtxt_like
        return _to_pbtxt_like(config)
    except Exception:
        return json.dumps(config, indent=2)


# ---------------------------------------------------------------------------
# Tool function
# ---------------------------------------------------------------------------

def fix_model_config(
    model_name: str,
    max_batch_size: Optional[int] = None,
    input_overrides: Optional[List[Dict[str, Any]]] = None,
    output_overrides: Optional[List[Dict[str, Any]]] = None,
    platform: Optional[str] = None,
    backend: Optional[str] = None,
    auto_fix: bool = True,
) -> Dict[str, Any]:
    """
    Fix a model's configuration and reload it on the inference server.

    With ``auto_fix=True`` (default), derives the correct config from
    model metadata.  Explicit overrides take precedence over auto-
    detection.  The corrected config is sent to Triton via the
    ``load_model()`` gRPC API (requires ``--model-control-mode=poll``
    or ``explicit``).

    Args:
        model_name: Name of the model to fix.
        max_batch_size: Override max_batch_size (0 = no batching).
        input_overrides: List of ``{"name", "data_type", "dims"}`` dicts.
        output_overrides: Same format for outputs.
        platform: Model platform, e.g. ``"onnxruntime_onnx"``.
        backend: Triton backend name, e.g. ``"onnxruntime"``.
        auto_fix: If True, auto-derive config from model metadata.

    Returns:
        Corrected config, reload status, and new model state.
    """
    try:
        client = get_client()
        warnings: List[str] = []

        # ---------------------------------------------------------------
        # 1. Gather existing state
        # ---------------------------------------------------------------
        original_state = "UNKNOWN"
        original_error = None
        try:
            index = client.get_repository_index()
            for entry in index:
                if entry["name"] == model_name:
                    original_state = entry.get("state", "UNKNOWN")
                    original_error = entry.get("reason", "")
                    break
        except Exception:
            warnings.append("Could not query repository index.")

        existing_config = None
        metadata = None
        try:
            existing_config = client.get_model_config(model_name,
                                                      use_cache=False)
        except Exception:
            pass
        try:
            metadata = client.get_model_metadata(model_name, use_cache=False)
        except Exception:
            pass

        if not auto_fix and not input_overrides and not output_overrides:
            return error_response(
                ValueError(
                    "auto_fix is False but no overrides provided.  "
                    "Either set auto_fix=True or provide input_overrides "
                    "and/or output_overrides."
                ),
                operation="fix_model_config",
                model_name=model_name,
            )

        if auto_fix and not metadata and not existing_config:
            warnings.append(
                "Neither metadata nor existing config is available.  "
                "Generating a minimal config; Triton's auto-complete "
                "(strict-model-config=false) will attempt to fill gaps."
            )

        # ---------------------------------------------------------------
        # 2. Build corrected config
        # ---------------------------------------------------------------
        config = _build_config(
            model_name,
            metadata if auto_fix else None,
            existing_config,
            max_batch_size=max_batch_size,
            input_overrides=input_overrides,
            output_overrides=output_overrides,
            platform=platform,
            backend=backend,
        )

        config_json = json.dumps(config)
        config_pbtxt = _render_pbtxt(config)

        # ---------------------------------------------------------------
        # 3. Reload via gRPC
        # ---------------------------------------------------------------
        reload_succeeded = False
        reload_error = None
        model_control_blocked = False

        try:
            client.load_model(model_name, config=config_json)
            reload_succeeded = True
        except Exception as e:
            reload_error = str(e)
            err_lower = reload_error.lower()
            if "model control" in err_lower or "not allowed" in err_lower:
                model_control_blocked = True
                warnings.append(
                    "Triton's model control mode does not allow API-driven "
                    "load.  The corrected config is returned below — apply "
                    "it manually to config.pbtxt and restart Triton, or "
                    "start Triton with --model-control-mode=explicit."
                )
            else:
                warnings.append(f"load_model failed: {reload_error}")

        # ---------------------------------------------------------------
        # 4. Wait and verify
        # ---------------------------------------------------------------
        new_state = original_state
        new_metadata = None
        new_error = None

        if reload_succeeded:
            for _ in range(5):
                time.sleep(1)
                if client.check_model_ready(model_name):
                    new_state = "READY"
                    break
            else:
                # Check index for updated error
                try:
                    idx = client.get_repository_index()
                    for entry in idx:
                        if entry["name"] == model_name:
                            new_state = entry.get("state", "UNKNOWN")
                            new_error = entry.get("reason", "")
                            break
                except Exception:
                    pass
                if new_state != "READY":
                    warnings.append(
                        "Model did not become READY within 5 seconds.  "
                        "It may still be loading."
                    )

            if new_state == "READY":
                try:
                    new_metadata = client.get_model_metadata(
                        model_name, use_cache=False,
                    )
                except Exception:
                    pass

        # ---------------------------------------------------------------
        # 5. Return
        # ---------------------------------------------------------------
        return ok(
            warnings=warnings or None,
            model_name=model_name,
            action="fix_and_reload",
            previous_state=original_state,
            previous_error=original_error,
            corrected_config=config,
            corrected_config_json=config_json,
            corrected_config_pbtxt=config_pbtxt,
            reload_succeeded=reload_succeeded and not model_control_blocked,
            model_control_blocked=model_control_blocked,
            new_state=new_state,
            new_metadata=new_metadata,
            new_error=new_error,
            message=(
                f"Model '{model_name}' reloaded with corrected config — "
                f"state is now {new_state}."
                if new_state == "READY"
                else f"Corrected config generated for '{model_name}'.  "
                     f"Current state: {new_state}."
            ),
        )

    except Exception as e:
        logger.error(f"Error fixing model config for {model_name}: {e}",
                     exc_info=True)
        return error_response(e, operation="fix_model_config",
                              model_name=model_name)


# ---------------------------------------------------------------------------
# Register
# ---------------------------------------------------------------------------

register_tool(
    name="fix_model_config",
    func=fix_model_config,
    description=(
        "Fix a model's config.pbtxt and reload it on the Triton server.  "
        "Auto-derives correct tensor specs from model metadata by default, "
        "or accepts explicit overrides.  Returns the corrected config "
        "and reload status.  If the server does not support API-driven "
        "load, the corrected config is still returned for manual application."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "model_name": {
                "type": "string",
                "description": "Name of the model to fix and reload",
            },
            "max_batch_size": {
                "type": "integer",
                "description": (
                    "Override max_batch_size (0 = no dynamic batching)"
                ),
            },
            "input_overrides": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "data_type": {"type": "string"},
                        "dims": {
                            "type": "array",
                            "items": {"type": "integer"},
                        },
                    },
                    "required": ["name", "dims"],
                },
                "description": "Override input tensor definitions",
            },
            "output_overrides": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "data_type": {"type": "string"},
                        "dims": {
                            "type": "array",
                            "items": {"type": "integer"},
                        },
                    },
                    "required": ["name", "dims"],
                },
                "description": "Override output tensor definitions",
            },
            "platform": {
                "type": "string",
                "description": (
                    "Model platform (e.g. 'onnxruntime_onnx', "
                    "'tensorrt_plan')"
                ),
            },
            "backend": {
                "type": "string",
                "description": (
                    "Triton backend name (e.g. 'onnxruntime', "
                    "'tensorrt', 'python')"
                ),
            },
            "auto_fix": {
                "type": "boolean",
                "default": True,
                "description": (
                    "If true (default), auto-generate correct config "
                    "from model metadata"
                ),
            },
        },
        "required": ["model_name"],
    },
)
