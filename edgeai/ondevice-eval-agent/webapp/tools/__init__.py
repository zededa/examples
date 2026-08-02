"""
In-process tool registry for the agent.

Provides a modular tool framework for LLM agent interactions with ML inference
servers. Each tool is in its own file under `catalog/` for easy maintenance.

Usage:
    from tools import execute_tool, TOOL_SCHEMAS, TOOL_FUNCTIONS
    from tools.catalog import list_available_models, get_model_metadata

Session lifecycle and usage tracking live in the `sessions` package.
"""

from .registry import (
    TOOL_SCHEMAS,
    TOOL_FUNCTIONS,
    execute_tool,
    dispatch_tool_calls,
    register_tool,
)

from .base import (
    ToolResult,
    error_response,
    ok,
    get_client,
)

__all__ = [
    "TOOL_SCHEMAS",
    "TOOL_FUNCTIONS",
    "execute_tool",
    "dispatch_tool_calls",
    "register_tool",
    "ToolResult",
    "error_response",
    "ok",
    "get_client",
]
