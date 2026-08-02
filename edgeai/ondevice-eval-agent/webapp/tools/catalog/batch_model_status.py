"""
Batch Model Status Tool

Returns readiness, type, and input shape for every discovered model
in a single call, reducing round-trips when many models are deployed.
"""

import logging
from typing import Any, Dict, List

from tools.base import ok, error_response, get_client
from tools.registry import register_tool

logger = logging.getLogger(__name__)


def batch_model_status() -> Dict[str, Any]:
    """
    Get status information for all discovered models in one call.

    For each model, reports:
    - readiness (ready / not ready)
    - input shape (height × width)
    - number of output tensors
    - server type

    Returns:
        Dict with a list of per-model status records.
    """
    try:
        client = get_client()
        models = client.get_available_models()
        server_type = client.detect_server_type()

        statuses: List[Dict[str, Any]] = []
        for name in models:
            entry: Dict[str, Any] = {
                "model_name": name,
                "server_type": server_type,
            }

            # Readiness
            try:
                entry["ready"] = client.check_model_ready(name)
            except Exception:
                entry["ready"] = None

            # Input shape
            try:
                h, w = client.get_model_input_shape(name)
                entry["input_shape"] = {"height": h, "width": w}
            except Exception:
                entry["input_shape"] = None

            # Output count
            try:
                all_out = client.get_all_output_specs(name)
                entry["output_count"] = len(all_out)
            except Exception:
                entry["output_count"] = None

            statuses.append(entry)

        ready_count = sum(1 for s in statuses if s.get("ready") is True)

        return ok(
            models=statuses,
            total=len(statuses),
            ready_count=ready_count,
            server_type=server_type,
            message=(
                f"Found {len(statuses)} model(s), {ready_count} ready for inference."
            ),
        )
    except Exception as e:
        logger.error(f"Error getting batch model status: {e}")
        return error_response(e, operation="batch_model_status")


# Register the tool
register_tool(
    name="batch_model_status",
    func=batch_model_status,
    description=(
        "Get readiness, input shape, and output count for ALL discovered models "
        "in a single call. Much faster than checking each model individually. "
        "Use this to get a quick overview of everything deployed on the server."
    ),
    input_schema={
        "type": "object",
        "properties": {},
        "required": [],
    },
)
