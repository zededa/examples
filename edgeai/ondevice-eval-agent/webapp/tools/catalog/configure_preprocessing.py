"""
Configure Preprocessing Tool

View or modify the image preprocessing settings used before inference
(normalization mode, target size, data format).
"""

import logging
from typing import Any, Dict, List, Optional

from tools.base import ok, error_response, get_client
from tools.registry import register_tool

logger = logging.getLogger(__name__)

# Human-readable descriptions for normalization modes
_NORMALIZATION_HELP: Dict[str, str] = {
    "imagenet": "ImageNet mean/std normalisation (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]). Standard for most pretrained vision models.",
    "yolo": "Scale pixels to [0, 1] by dividing by 255. Common for YOLO-family detection models.",
    "raw": "No normalisation — pixel values stay in [0, 255]. Useful for models that normalise internally.",
}


def configure_preprocessing(
    normalization: Optional[str] = None,
    target_height: Optional[int] = None,
    target_width: Optional[int] = None,
    data_format: Optional[str] = None,
) -> Dict[str, Any]:
    """
    View or update image preprocessing configuration.

    Called **without** arguments it returns the current settings.
    Pass one or more arguments to update specific settings.

    Args:
        normalization: Normalisation mode — ``"imagenet"``, ``"yolo"``, or ``"raw"``.
        target_height: Target image height in pixels (e.g. 640).
        target_width: Target image width in pixels (e.g. 640).
        data_format: Tensor layout — ``"NCHW"`` or ``"NHWC"``.

    Returns:
        Dict with the current (possibly updated) configuration.
    """
    try:
        client = get_client()
        current_config = client.preprocessing_config

        # Determine if any updates were requested
        updates: Dict[str, Any] = {}
        warnings: List[str] = []

        if normalization is not None:
            allowed = list(_NORMALIZATION_HELP.keys())
            if normalization not in allowed:
                return error_response(
                    ValueError(
                        f"Invalid normalization '{normalization}'. Must be one of: {allowed}"
                    ),
                    operation="configure_preprocessing",
                )
            updates["normalization"] = normalization

        if target_height is not None or target_width is not None:
            h = target_height or current_config.get("target_size", (224, 224))[0]
            w = target_width or current_config.get("target_size", (224, 224))[1]
            if h <= 0 or w <= 0:
                return error_response(
                    ValueError("target_height and target_width must be positive integers"),
                    operation="configure_preprocessing",
                )
            updates["target_size"] = (h, w)

        if data_format is not None:
            allowed_formats = ["NCHW", "NHWC"]
            if data_format not in allowed_formats:
                return error_response(
                    ValueError(
                        f"Invalid data_format '{data_format}'. Must be one of: {allowed_formats}"
                    ),
                    operation="configure_preprocessing",
                )
            updates["data_format"] = data_format

        updated = bool(updates)
        if updated:
            client.set_preprocessing_config(updates)
            logger.info(f"Preprocessing config updated: {updates}")

        # Re-read after potential update
        new_config = client.preprocessing_config

        return ok(
            updated=updated,
            config=new_config,
            normalization_options={
                mode: desc for mode, desc in _NORMALIZATION_HELP.items()
            },
            warnings=warnings,
            message=(
                "Preprocessing configuration updated successfully."
                if updated
                else "Current preprocessing configuration (no changes requested)."
            ),
        )
    except Exception as e:
        logger.error(f"Error configuring preprocessing: {e}")
        return error_response(e, operation="configure_preprocessing")


# Register the tool
register_tool(
    name="configure_preprocessing",
    func=configure_preprocessing,
    description=(
        "View or modify image preprocessing settings used before inference. "
        "Supports normalization mode (imagenet / yolo / raw), target image size, "
        "and data format (NCHW / NHWC). Call without arguments to view current settings."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "normalization": {
                "type": "string",
                "enum": ["imagenet", "yolo", "raw"],
                "description": "Normalisation mode: 'imagenet' (mean/std), 'yolo' (0-1 scaling), or 'raw' (no normalisation)",
            },
            "target_height": {
                "type": "integer",
                "description": "Target image height in pixels (e.g. 640)",
            },
            "target_width": {
                "type": "integer",
                "description": "Target image width in pixels (e.g. 640)",
            },
            "data_format": {
                "type": "string",
                "enum": ["NCHW", "NHWC"],
                "description": "Tensor layout: 'NCHW' (channels first) or 'NHWC' (channels last)",
            },
        },
        "required": [],
    },
)
