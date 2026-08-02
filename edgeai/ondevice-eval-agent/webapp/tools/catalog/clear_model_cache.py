"""
Clear Model Cache Tool

Invalidates all cached metadata and server information so that
subsequent queries fetch fresh data from the inference server.
"""

import logging
from typing import Dict, Any

from tools.base import ok, error_response, get_client
from tools.registry import register_tool

logger = logging.getLogger(__name__)


def clear_model_cache() -> Dict[str, Any]:
    """
    Clear all cached model metadata and server information.

    When models are reloaded, swapped, or redeployed on the inference
    server, cached metadata becomes stale.  Call this tool to force
    the client to re-fetch everything on the next request.

    Returns:
        Dict confirming the cache was cleared.
    """
    try:
        client = get_client()
        client.clear_cache()

        return ok(
            cleared=True,
            message=(
                "All model metadata and server info caches have been cleared. "
                "The next tool call will fetch fresh data from the server."
            ),
        )
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        return error_response(e, operation="clear_model_cache")


# Register the tool
register_tool(
    name="clear_model_cache",
    func=clear_model_cache,
    description=(
        "Clear all cached model metadata and server information. "
        "Use this after models are reloaded, swapped, or redeployed on the "
        "inference server to ensure subsequent queries return fresh data."
    ),
    input_schema={
        "type": "object",
        "properties": {},
        "required": [],
    },
)
