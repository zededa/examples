"""
Get API Examples Tool

Provides API endpoint information and curl command examples.
"""

import logging
from typing import Dict, Any

from tools.base import ok, error_response, get_client
from tools.registry import register_tool

logger = logging.getLogger(__name__)


def get_api_examples(model_name: str) -> Dict[str, Any]:
    """
    Get API endpoint examples and curl commands for a model.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Dict containing API endpoints and example commands
    """
    try:
        client = get_client()
        
        endpoints_info = client.get_api_endpoints_info(model_name)
        
        return ok(
            model_name=model_name,
            endpoints=endpoints_info
        )
    except Exception as e:
        logger.error(f"Error getting API examples for {model_name}: {e}")
        return error_response(e, operation="get_api_examples", model_name=model_name)


# Register the tool
register_tool(
    name="get_api_examples",
    func=get_api_examples,
    description="Get API endpoint information and curl command examples for interacting with a specific model. Useful for developers who want to test the API directly.",
    input_schema={
        "type": "object",
        "properties": {
            "model_name": {
                "type": "string",
                "description": "Name of the model"
            }
        },
        "required": ["model_name"]
    }
)
