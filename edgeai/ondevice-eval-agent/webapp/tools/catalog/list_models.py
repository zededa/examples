"""
List Available Models Tool

Discovers all available models across both inference backends:
- Classical ML server (Triton / OpenVINO) for vision / detection / classification models.
- LLM server (vLLM / llama.cpp) for language models.

A deployment typically runs both servers side-by-side, so the agent needs to
see both when a user asks "what models are available".
"""

import logging
from typing import Dict, Any, List

from tools.base import ok, error_response, get_client
from tools.registry import register_tool

logger = logging.getLogger(__name__)


def _list_classical_models() -> Dict[str, Any]:
    """Query the Triton/OpenVINO server for classical ML models."""
    try:
        client = get_client()
        models = client.get_available_models()
        server_info = client.get_server_info()
        server_type = client.detect_server_type()
        return {
            "reachable": True,
            "models": models,
            "count": len(models),
            "server_type": server_type,
            "server_info": server_info,
        }
    except Exception as e:
        logger.warning(f"Classical inference server unreachable: {e}")
        return {
            "reachable": False,
            "models": [],
            "count": 0,
            "error": str(e),
        }


def _list_llm_models() -> Dict[str, Any]:
    """Query the vLLM/llama.cpp server for language models."""
    try:
        # Lazy import to avoid circular imports at module load time.
        from client.llm_client import get_llm_client
        client = get_llm_client()

        if not client.is_healthy():
            return {
                "reachable": False,
                "models": [],
                "count": 0,
                "server_url": client.base_url,
                "error": f"LLM server at {client.base_url} is not reachable",
            }

        models = client.list_models()
        model_list: List[Dict[str, Any]] = []
        for m in models:
            entry: Dict[str, Any] = {"id": m.id}
            if m.owned_by:
                entry["owned_by"] = m.owned_by
            if m.created:
                entry["created"] = m.created
            model_list.append(entry)

        return {
            "reachable": True,
            "models": model_list,
            "count": len(model_list),
            "server_type": client.server_type.value,
            "server_url": client.base_url,
        }
    except Exception as e:
        logger.warning(f"LLM server unreachable: {e}")
        return {
            "reachable": False,
            "models": [],
            "count": 0,
            "error": str(e),
        }


def list_available_models() -> Dict[str, Any]:
    """
    Discover all available models across both backends (classical + LLM).

    Returns a combined view so the agent can answer "what models are available"
    without needing to call two separate tools. Each backend block reports its
    own reachability so the agent can explain partial results truthfully.
    """
    classical = _list_classical_models()
    llm = _list_llm_models()

    total = classical.get("count", 0) + llm.get("count", 0)

    # Backwards-compatible top-level fields mirror the classical server so
    # existing callers that only read `models` / `server_type` still work.
    return ok(
        classical=classical,
        llm=llm,
        total_count=total,
        models=classical.get("models", []),
        count=classical.get("count", 0),
        server_type=classical.get("server_type"),
        server_info=classical.get("server_info"),
    )


# Register the tool
register_tool(
    name="list_available_models",
    func=list_available_models,
    description=(
        "Discover all models deployed across both backends: the classical ML "
        "inference server (Triton/OpenVINO — vision, detection, classification) "
        "and the LLM server (vLLM/llama.cpp — language models). "
        "Use this first to get a complete overview. Returns two blocks — "
        "`classical` and `llm` — each with `reachable`, `models`, `count`, and "
        "server info."
    ),
    input_schema={
        "type": "object",
        "properties": {},
        "required": []
    }
)
