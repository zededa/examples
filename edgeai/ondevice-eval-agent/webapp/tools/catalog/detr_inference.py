"""
DETR Inference Tool

Runs inference on DETR (DEtection TRansformer) models which require
special dual-input preprocessing (pixel_values + pixel_mask) and
transformer-based postprocessing.
"""

import base64
import logging
import os
from typing import Any, Dict, Optional

import cv2
import numpy as np

from tools.base import ok, error_response, get_client
from tools.registry import register_tool
from sessions.registry import SESSION_STORAGE_ROOT

logger = logging.getLogger(__name__)


def run_detr_inference(
    model_name: str,
    image_path: str,
    confidence_threshold: float = 0.7,
) -> Dict[str, Any]:
    """
    Run inference on a DETR model.

    DETR models require dual inputs (``pixel_values`` and ``pixel_mask``)
    and produce ``logits`` + ``pred_boxes`` outputs.  The standard
    ``run_inference`` tool does not support this pipeline — use this
    tool for any DETR-family model.

    Args:
        model_name: Name of the deployed DETR model.
        image_path: Path to the image file (from session storage).
        confidence_threshold: Minimum detection confidence (0.0–1.0, default 0.7).

    Returns:
        Dict with detections, annotated image, and timing breakdown.
    """
    try:
        # --- Validate inputs ---
        if not model_name:
            return error_response(
                ValueError("model_name is required"),
                operation="run_detr_inference",
            )
        if not image_path:
            return error_response(
                ValueError("image_path is required"),
                operation="run_detr_inference",
            )

        # Security: prevent path traversal
        real_path = os.path.realpath(image_path)
        real_storage_root = os.path.realpath(SESSION_STORAGE_ROOT)
        if not real_path.startswith(real_storage_root + os.sep) and real_path != real_storage_root:
            return error_response(
                ValueError("Invalid file path — access denied"),
                operation="run_detr_inference",
            )
        if not os.path.exists(real_path):
            return error_response(
                FileNotFoundError(f"Image not found: {image_path}"),
                operation="run_detr_inference",
            )
        if not 0.0 <= confidence_threshold <= 1.0:
            return error_response(
                ValueError("confidence_threshold must be between 0.0 and 1.0"),
                operation="run_detr_inference",
            )

        # --- Read image bytes ---
        with open(real_path, "rb") as f:
            image_bytes = f.read()

        # --- Import the DETR processing module ---
        from processing.detr import run_detr_inference as _detr_infer

        client = get_client()
        result = _detr_infer(
            server_url=client.server_url,
            model_name=model_name,
            image_bytes=image_bytes,
            threshold=confidence_threshold,
        )

        if not result.get("success", False):
            return error_response(
                RuntimeError(result.get("error", "DETR inference failed")),
                operation="run_detr_inference",
                model_name=model_name,
            )

        # --- Build annotated image ---
        annotated_b64: Optional[str] = None
        result_image_path: Optional[str] = None
        try:
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is not None:
                for det in result.get("detections", []):
                    box = det["box"]
                    label = f"{det['label']} {det['score']:.0%}"
                    color = (0, 255, 0)
                    cv2.rectangle(
                        img,
                        (box["xmin"], box["ymin"]),
                        (box["xmax"], box["ymax"]),
                        color,
                        2,
                    )
                    cv2.putText(
                        img, label,
                        (box["xmin"], max(box["ymin"] - 8, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
                    )
                _, buf = cv2.imencode(".png", img)
                annotated_b64 = base64.b64encode(buf.tobytes()).decode("utf-8")

                # Save to session storage
                session_dir = os.path.dirname(image_path)
                result_image_path = os.path.join(
                    session_dir, f"result_{model_name}_detr.png"
                )
                with open(result_image_path, "wb") as fp:
                    fp.write(buf.tobytes())
                logger.info(f"Saved DETR result image to {result_image_path}")
        except Exception as vis_err:
            logger.warning(f"Failed to create DETR visualisation: {vis_err}")

        return ok(
            model_name=model_name,
            processing_type="detr",
            detections=result["detections"],
            detection_count=result["detection_count"],
            confidence_threshold=confidence_threshold,
            original_size=result.get("original_size"),
            latency=result.get("timing"),
            result_image_base64=annotated_b64,
            result_image_path=result_image_path,
            has_visualization=annotated_b64 is not None,
            summary=(
                f"DETR detected {result['detection_count']} object(s) "
                f"above {confidence_threshold:.0%} confidence."
            ),
            message="DETR inference completed successfully.",
        )
    except Exception as e:
        logger.error(f"Error running DETR inference: {e}", exc_info=True)
        return error_response(
            e,
            operation="run_detr_inference",
            model_name=model_name,
            image_path=image_path,
        )


# Register the tool
register_tool(
    name="run_detr_inference",
    func=run_detr_inference,
    description=(
        "Run inference on a DETR (DEtection TRansformer) model. "
        "DETR models require special dual-input preprocessing (pixel_values + pixel_mask) "
        "that the standard run_inference tool does not support. "
        "Returns detected objects with bounding boxes, confidence scores, COCO class labels, "
        "and an annotated visualization image."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "model_name": {
                "type": "string",
                "description": "Name of the deployed DETR model",
            },
            "image_path": {
                "type": "string",
                "description": "Path to the uploaded image file",
            },
            "confidence_threshold": {
                "type": "number",
                "default": 0.7,
                "description": "Minimum detection confidence (0.0–1.0)",
            },
        },
        "required": ["model_name", "image_path"],
    },
)
