"""
Agent Tools - Re-exports from tools and sessions packages.

This module provides backward compatibility by re-exporting the tool registry
and session management APIs. Existing callers (including tests that monkey-patch
`agent.tools`) should keep working.

For new code, prefer importing directly:
    from tools import execute_tool, TOOL_SCHEMAS
    from tools.catalog import list_available_models
    from sessions import get_or_create_session, check_session_warnings
    from sessions import get_session_config
"""

from tools import (
    TOOL_SCHEMAS,
    TOOL_FUNCTIONS,
    execute_tool,
    register_tool,
    ToolResult,
    error_response,
    ok,
    get_client,
)

from sessions import (
    get_session_storage_path,
    check_session_storage_limit,
    cleanup_session_storage,
    SESSION_STORAGE_ROOT,
    SESSION_STORAGE_LIMIT_MB,
    get_or_create_session,
    get_session,
    remove_session,
    check_session_warnings,
    is_session_over_hard_limit,
    cleanup_inactive_sessions,
    get_session_status,
    SessionCapacityError,
)

from tools.catalog import (
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
)

__all__ = [
    "TOOL_SCHEMAS",
    "TOOL_FUNCTIONS",
    "execute_tool",
    "register_tool",
    "ToolResult",
    "error_response",
    "ok",
    "get_client",
    "get_session_storage_path",
    "check_session_storage_limit",
    "cleanup_session_storage",
    "SESSION_STORAGE_ROOT",
    "SESSION_STORAGE_LIMIT_MB",
    "get_or_create_session",
    "get_session",
    "remove_session",
    "check_session_warnings",
    "is_session_over_hard_limit",
    "cleanup_inactive_sessions",
    "get_session_status",
    "SessionCapacityError",
    "list_available_models",
    "get_model_metadata",
    "get_model_config",
    "get_model_input_requirements",
    "get_model_output_interpretation",
    "analyze_model_type",
    "get_server_status",
    "get_api_examples",
    "get_frontend_integration_guide",
    "recommend_next_steps",
    "run_inference",
    "list_processing_types",
    "get_inference_latency",
    "web_search",
    "search_model_info",
    "view_image",
    "analyze_inference_result",
    "check_model_ready",
    "get_all_model_outputs",
    "clear_model_cache",
    "configure_preprocessing",
    "compare_models",
    "run_detr_inference",
    "batch_model_status",
    "manage_class_names",
]
