"""
Check Model Ready Tool

Lightweight readiness check for a specific model.
"""

import logging
from typing import Dict, Any

from tools.base import ok, error_response, get_client
from tools.registry import register_tool

logger = logging.getLogger(__name__)


def check_model_ready(model_name: str) -> Dict[str, Any]:
    """
    Check if a specific model is ready for inference.

    Performs a lightweight boolean readiness probe without pulling
    full metadata.  Useful before calling ``run_inference`` or when
    troubleshooting "model not found" errors.

    Args:
        model_name: Name of the model to check.

    Returns:
        Dict with readiness status and model name.
    """
    try:
        client = get_client()
        ready = client.check_model_ready(model_name)
        server_type = client.detect_server_type()

        return ok(
            model_name=model_name,
            ready=ready,
            server_type=server_type,
            message=(
                f"Model '{model_name}' is ready for inference."
                if ready
                else f"Model '{model_name}' is NOT ready. It may still be loading or is not deployed."
            ),
        )
    except Exception as e:
        logger.error(f"Error checking model readiness for {model_name}: {e}")
        return error_response(e, operation="check_model_ready", model_name=model_name)


# Register the tool
register_tool(
    name="check_model_ready",
    func=check_model_ready,
    description=(
        "Quickly check whether a specific model is ready for inference. "
        "Returns a simple ready/not-ready status without fetching full metadata. "
        "Use this before running inference or to diagnose 'model not found' issues."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "model_name": {
                "type": "string",
                "description": "Name of the model to check",
            }
        },
        "required": ["model_name"],
    },
)
