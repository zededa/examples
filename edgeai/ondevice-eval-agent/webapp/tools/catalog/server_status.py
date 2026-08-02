"""
Get Server Status Tool

Checks inference server health and status information.
"""

import logging
from typing import Dict, Any

from tools.base import ok, error_response, get_client
from tools.registry import register_tool

logger = logging.getLogger(__name__)


def get_server_status() -> Dict[str, Any]:
    """
    Get inference server health and status information.
    
    Returns:
        Dict containing server health status and metadata
    """
    try:
        client = get_client()
        
        is_healthy, health_message = client.check_server_health()
        server_info = client.get_server_info()
        server_type = client.detect_server_type()
        device_info = client.get_server_device_info()
        
        return ok(
            healthy=is_healthy,
            message=health_message,
            server_type=server_type,
            server_info=server_info,
            device=device_info,
            server_url=client.server_url
        )
    except Exception as e:
        logger.error(f"Error getting server status: {e}")
        return error_response(e, operation="get_server_status", healthy=False)


# Register the tool
register_tool(
    name="get_server_status",
    func=get_server_status,
    description="Check the health and status of the inference server, including server type (Triton/OpenVINO), version, and device information (CPU/GPU).",
    input_schema={
        "type": "object",
        "properties": {},
        "required": []
    }
)
