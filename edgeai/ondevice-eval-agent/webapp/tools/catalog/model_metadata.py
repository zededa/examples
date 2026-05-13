"""
Get Model Metadata Tool

Retrieves detailed metadata for a specific model.
"""

import logging
from typing import Dict, Any

from tools.base import ok, error_response, get_client
from tools.registry import register_tool

logger = logging.getLogger(__name__)


def get_model_metadata(model_name: str) -> Dict[str, Any]:
    """
    Get detailed metadata for a specific model.
    
    Args:
        model_name: Name of the model to inspect
        
    Returns:
        Dict containing model metadata including input/output specifications
    """
    try:
        client = get_client()
        
        # Get full model info
        full_info = client.get_full_model_info(model_name)
        
        # Extract key information
        input_spec = full_info.get('input_spec', {})
        output_spec = full_info.get('output_spec', {})
        metadata = full_info.get('metadata', {})
        
        return ok(
            model_name=model_name,
            ready=full_info.get('ready', False),
            server_type=full_info.get('server_type', 'unknown'),
            input_spec={
                "name": input_spec.get('name'),
                "shape": input_spec.get('shape'),
                "datatype": input_spec.get('datatype'),
                "format": input_spec.get('format'),
                "height": input_spec.get('height'),
                "width": input_spec.get('width'),
                "channels": input_spec.get('channels')
            },
            output_spec={
                "name": output_spec.get('name'),
                "shape": output_spec.get('shape'),
                "datatype": output_spec.get('datatype'),
                "num_classes": output_spec.get('num_classes')
            },
            raw_metadata=metadata
        )
    except Exception as e:
        logger.error(f"Error getting model metadata for {model_name}: {e}")
        return error_response(e, operation="get_metadata", model_name=model_name)


# Register the tool
register_tool(
    name="get_model_metadata",
    func=get_model_metadata,
    description="Get detailed metadata for a specific model including input/output tensor specifications, data types, and shapes. Essential for understanding how to interact with the model.",
    input_schema={
        "type": "object",
        "properties": {
            "model_name": {
                "type": "string",
                "description": "Name of the model to inspect"
            }
        },
        "required": ["model_name"]
    }
)
