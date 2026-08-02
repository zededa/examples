"""
LLM List Models Tool

Lists available LLM models served by vLLM or llama.cpp backends.
"""

import logging
from typing import Any, Dict

from tools.base import ok, error_response
from tools.registry import register_tool

logger = logging.getLogger(__name__)


def _get_llm_client():
    """Lazy import to avoid circular imports."""
    from client.llm_client import get_llm_client
    return get_llm_client()


def llm_list_models() -> Dict[str, Any]:
    """
    List LLM models available on the serving backend (vLLM / llama.cpp).

    Returns:
        List of model IDs and their metadata.
    """
    try:
        client = _get_llm_client()

        if not client.is_healthy():
            return error_response(
                ConnectionError(
                    f"LLM server at {client.base_url} is not reachable"
                ),
                operation="llm_list_models",
            )

        models = client.list_models()

        model_list = []
        for m in models:
            entry: Dict[str, Any] = {"id": m.id}
            if m.owned_by:
                entry["owned_by"] = m.owned_by
            if m.created:
                entry["created"] = m.created
            model_list.append(entry)

        return ok(
            data=model_list,
            count=len(model_list),
            server_url=client.base_url,
            server_type=client.server_type.value,
            message=f"Found {len(model_list)} LLM model(s) on {client.server_type.value} server",
        )

    except Exception as e:
        logger.error("Error listing LLM models: %s", e, exc_info=True)
        return error_response(e, operation="llm_list_models")


register_tool(
    name="llm_list_models",
    func=llm_list_models,
    description=(
        "List LLM models available on the serving backend (vLLM or llama.cpp). "
        "Use this to discover which language models are deployed and available for "
        "chat or text completion. Returns model IDs, server type, and server URL."
    ),
    input_schema={
        "type": "object",
        "properties": {},
        "required": [],
    },
)
