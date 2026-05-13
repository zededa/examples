"""
Get Model Config Tool

Retrieves the model configuration (config.pbtxt equivalent) from the inference server.
Returns both JSON and a pbtxt-style text rendering for readability.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict

from tools.base import ok, error_response, get_client
from tools.registry import register_tool

logger = logging.getLogger(__name__)


def _to_pbtxt_like(obj: Any, key: str | None = None, indent: int = 0, lines: list[str] | None = None) -> str:
    """Render a nested dict/list as a pbtxt-style string."""
    if lines is None:
        lines = []
    pad = " " * indent

    if isinstance(obj, dict):
        if key is not None:
            # Wrap dict in a named block
            lines.append(f"{pad}{key} {{")
            for k, v in obj.items():
                _to_pbtxt_like(v, k, indent + 2, lines)
            lines.append(f"{pad}}}")
        else:
            # Top-level dict or nested without key: render children directly
            for k, v in obj.items():
                _to_pbtxt_like(v, k, indent, lines)
    elif isinstance(obj, list):
        # For lists, render each item with the same key (repeated field in pbtxt)
        for item in obj:
            _to_pbtxt_like(item, key, indent, lines)
    else:
        # Scalar value
        value = json.dumps(obj)
        if key is None:
            lines.append(f"{pad}{value}")
        else:
            lines.append(f"{pad}{key}: {value}")

    return "\n".join(lines)


def get_model_config(model_name: str) -> Dict[str, Any]:
    """Fetch the model config (config.pbtxt equivalent) from the server."""
    try:
        client = get_client()
        config = client.get_model_config(model_name)

        if not config:
            return error_response(
                ValueError(f"Config not available for model '{model_name}'"),
                operation="get_model_config",
                model_name=model_name,
            )

        pbtxt_view = _to_pbtxt_like(config)

        return ok(
            model_name=model_name,
            config=config,
            config_pretty=json.dumps(config, indent=2),
            config_pbtxt=pbtxt_view,
            message="Model configuration retrieved. 'config' is the raw JSON; 'config_pbtxt' is a pbtxt-style rendering for readability.",
        )
    except Exception as e:
        logger.error(f"Error getting model config for {model_name}: {e}")
        return error_response(e, operation="get_model_config", model_name=model_name)


register_tool(
    name="get_model_config",
    func=get_model_config,
    description="Retrieve the model configuration (config.pbtxt equivalent) for a model from the inference server.",
    input_schema={
        "type": "object",
        "properties": {
            "model_name": {
                "type": "string",
                "description": "Name of the model to retrieve config for",
            }
        },
        "required": ["model_name"],
    },
)
