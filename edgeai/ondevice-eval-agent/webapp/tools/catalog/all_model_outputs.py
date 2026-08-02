"""
Get All Model Outputs Tool

Returns specifications for every output tensor of a model,
critical for multi-output architectures like YOLOv8 or DETR.
"""

import logging
from typing import Dict, Any

from tools.base import ok, error_response, get_client
from tools.registry import register_tool

logger = logging.getLogger(__name__)


def get_all_model_outputs(model_name: str) -> Dict[str, Any]:
    """
    Get specifications for ALL output tensors of a model.

    Unlike ``get_model_metadata`` which returns only the primary output,
    this tool returns every output tensor — essential for multi-output
    models such as YOLOv8 (boxes + scores + classes) or DETR
    (logits + pred_boxes).

    Args:
        model_name: Name of the model to inspect.

    Returns:
        Dict with list of output specs and count.
    """
    try:
        client = get_client()
        all_outputs = client.get_all_output_specs(model_name)

        # Normalise each spec to a plain dict
        output_list = []
        for spec in all_outputs:
            if hasattr(spec, "to_dict"):
                output_list.append(spec.to_dict())
            elif isinstance(spec, dict):
                output_list.append(spec)
            else:
                output_list.append({"raw": str(spec)})

        return ok(
            model_name=model_name,
            outputs=output_list,
            count=len(output_list),
            message=(
                f"Model '{model_name}' has {len(output_list)} output tensor(s)."
            ),
        )
    except Exception as e:
        logger.error(f"Error getting all output specs for {model_name}: {e}")
        return error_response(
            e, operation="get_all_model_outputs", model_name=model_name
        )


# Register the tool
register_tool(
    name="get_all_model_outputs",
    func=get_all_model_outputs,
    description=(
        "Get specifications for ALL output tensors of a model. "
        "Essential for multi-output models like YOLOv8 or DETR that produce "
        "multiple tensors (e.g., boxes, scores, classes). Returns name, shape, "
        "datatype, and num_classes for every output."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "model_name": {
                "type": "string",
                "description": "Name of the model to inspect",
            }
        },
        "required": ["model_name"],
    },
)
