"""
Tool Registry.

Manages tool registration, schemas, and execution. New tools can be added
by importing them and calling register_tool().

Parallel dispatch: `dispatch_tool_calls(tool_calls)` runs a batch of
tool_calls from a single assistant turn concurrently (ThreadPoolExecutor,
because the rest of the stack is sync + threading). Results are returned
in the same order as the input list, which is what every provider's tool
result message format requires.
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional

from .base import error_response

logger = logging.getLogger(__name__)

# Global registries
TOOL_SCHEMAS: List[Dict[str, Any]] = []
TOOL_FUNCTIONS: Dict[str, Callable] = {}


def register_tool(
    name: str,
    func: Callable,
    description: str,
    input_schema: Dict[str, Any]
) -> None:
    """
    Register a tool function with its schema.
    
    Args:
        name: Unique tool name
        func: The tool function
        description: Human-readable description for the AI agent
        input_schema: JSON Schema for input parameters
    """
    TOOL_FUNCTIONS[name] = func
    # Update existing schema entry in-place if the tool was already
    # registered, instead of blindly appending a duplicate.
    for existing in TOOL_SCHEMAS:
        if existing["name"] == name:
            existing["description"] = description
            existing["input_schema"] = input_schema
            logger.debug(f"Updated tool: {name}")
            return
    TOOL_SCHEMAS.append({
        "name": name,
        "description": description,
        "input_schema": input_schema
    })
    logger.debug(f"Registered tool: {name}")


def execute_tool(tool_name: str, tool_input: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute a tool function by name.

    Args:
        tool_name: Name of the tool to execute
        tool_input: Input parameters for the tool

    Returns:
        Result from the tool function (standardized ToolResult format)
    """
    # Lazy import so importing tools/registry.py doesn't pull in config/observability.
    try:
        from observability.tracing import get_tracing
        _tracing = get_tracing()
    except Exception:
        _tracing = None

    if tool_name not in TOOL_FUNCTIONS:
        return error_response(
            ValueError(f"Unknown tool: {tool_name}"),
            operation="execute_tool",
            available_tools=list(TOOL_FUNCTIONS.keys())
        )

    span_cm = _tracing.tool_call(tool_name=tool_name, args=tool_input) if _tracing else None

    try:
        if span_cm is not None:
            span_cm.__enter__()
        tool_func = TOOL_FUNCTIONS[tool_name]
        result = tool_func(**tool_input)
        if _tracing is not None and _tracing.enabled:
            try:
                from langfuse import get_client as _get_lf_client
                _get_lf_client().update_current_span(
                    output={"success": result.get("success", True) if isinstance(result, dict) else True},
                )
            except Exception:
                pass
        return result
    except TypeError as e:
        logger.error(f"Invalid arguments for tool {tool_name}: {e}")
        return error_response(
            e,
            operation="execute_tool",
            tool_name=tool_name,
            provided_args=list(tool_input.keys())
        )
    except Exception as e:
        logger.error(f"Error executing tool {tool_name}: {e}")
        return error_response(
            e,
            operation="execute_tool",
            tool_name=tool_name
        )
    finally:
        if span_cm is not None:
            try:
                span_cm.__exit__(None, None, None)
            except Exception:
                pass


def dispatch_tool_calls(
    tool_calls: List[Dict[str, Any]],
    *,
    max_workers: Optional[int] = None,
    parallel: Optional[bool] = None,
) -> List[Dict[str, Any]]:
    """
    Execute a batch of tool calls from a single assistant turn.

    Each entry in `tool_calls` is a dict with at least:
        {"id": "<tool_call_id>", "name": "<tool_name>", "arguments": {...}}
    ("arguments" may also be named "input" depending on the provider.)

    Returns a list of result entries in the SAME ORDER as the input, each
    shaped as:
        {"id": "<tool_call_id>", "name": "<tool_name>", "result": <ToolResult>}

    Concurrency:
        - When `parallel` is None (default), reads TOOLS_PARALLEL_EXECUTION
          from config. When False, runs serially (identical to the pre-PR-4
          behavior of iterating execute_tool in a for-loop).
        - Workers are capped at min(len(tool_calls), max_parallel_tools).
        - Preserves original order so provider tool-result messages line up.
    """
    if not tool_calls:
        return []

    # Lazy-import settings so tests that don't boot the app don't need config.
    if parallel is None or max_workers is None:
        try:
            from config import get_settings
            tools_cfg = get_settings().tools
            if parallel is None:
                parallel = tools_cfg.parallel_execution
            if max_workers is None:
                max_workers = tools_cfg.max_parallel_tools
        except Exception:
            parallel = True if parallel is None else parallel
            max_workers = 8 if max_workers is None else max_workers

    def _args_of(tc: Dict[str, Any]) -> Dict[str, Any]:
        args = tc.get("input") if "input" in tc else tc.get("arguments")
        if isinstance(args, dict):
            return args
        if isinstance(args, str):
            import json as _json
            try:
                parsed = _json.loads(args)
                return parsed if isinstance(parsed, dict) else {}
            except Exception:
                return {}
        return {}

    # Serial path: simple, preserves original behavior exactly.
    if not parallel or len(tool_calls) <= 1:
        out: List[Dict[str, Any]] = []
        for tc in tool_calls:
            out.append({
                "id": tc.get("id"),
                "name": tc.get("name"),
                "result": execute_tool(tc.get("name", ""), _args_of(tc)),
            })
        return out

    # Parallel path: fan out. Propagate the current thread's ContextVars
    # (request_id, session_id) to workers so tracing spans stay nested.
    # A Context object can only be entered once at a time, so snapshot
    # the current vars and rebind them in each worker thread instead of
    # using ctx.run on a single Context.
    import contextvars as _cv
    try:
        snapshot = {var: var.get() for var, _ in _cv.copy_context().items()}
    except Exception:
        snapshot = {}

    def _execute_one(tc: Dict[str, Any]) -> Dict[str, Any]:
        tokens = []
        for var, value in snapshot.items():
            try:
                tokens.append((var, var.set(value)))
            except Exception:
                pass
        try:
            return {
                "id": tc.get("id"),
                "name": tc.get("name"),
                "result": execute_tool(tc.get("name", ""), _args_of(tc)),
            }
        finally:
            for var, token in tokens:
                try:
                    var.reset(token)
                except Exception:
                    pass

    worker_count = max(1, min(max_workers or 8, len(tool_calls)))
    # Launch
    with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="tool") as pool:
        futures = {pool.submit(_execute_one, tc): idx for idx, tc in enumerate(tool_calls)}
        by_index: Dict[int, Dict[str, Any]] = {}
        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                by_index[idx] = fut.result()
            except Exception as exc:
                tc = tool_calls[idx]
                logger.exception("Parallel tool dispatch failed for %s", tc.get("name"))
                by_index[idx] = {
                    "id": tc.get("id"),
                    "name": tc.get("name"),
                    "result": error_response(
                        exc,
                        operation="execute_tool",
                        tool_name=tc.get("name"),
                    ),
                }

    return [by_index[i] for i in range(len(tool_calls))]


# Import all tools to register them
# This must be at the end to avoid circular imports
from .catalog import (
    list_available_models,
    get_model_metadata,
    get_model_config,
    get_model_input_requirements,
    get_model_output_interpretation,
    analyze_model_type,
    get_server_status,
    get_api_examples,
    get_frontend_integration_guide,
    recommend_next_steps,
    run_inference,
    list_processing_types,
    get_inference_latency,
    web_search,
    search_model_info,
    view_image,
    analyze_inference_result,
    check_model_ready,
    get_all_model_outputs,
    clear_model_cache,
    configure_preprocessing,
    compare_models,
    run_detr_inference,
    batch_model_status,
    manage_class_names,
    llm_list_models,
    llm_get_performance,
    llm_inference,
    probe_model_io,
    diagnose_failed_models,
    fix_model_config,
    llm_run_benchmark,
    llm_evaluate,
    llm_compare_models,
    get_deployment_health,
)
