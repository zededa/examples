"""
Agent Package - AI Agent for ML Model Exploration

This package contains the AI agent components:
- prompts: LLM integration and conversation management
- tools: MCP-style tool functions for model exploration

Usage:
    from webapp.agent import execute_tool, TOOL_SCHEMAS, get_backend_info
    
    # Check if agent is enabled
    info = get_backend_info()
    
    # Execute a tool
    result = execute_tool("list_available_models", {})
"""

from .tools import (
    TOOL_SCHEMAS,
    TOOL_FUNCTIONS,
    execute_tool,
    get_client,
    # Tool functions
    list_available_models,
    get_model_metadata,
    get_model_config,
    analyze_model_type,
    get_model_input_requirements,
    get_model_output_interpretation,
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

from .prompts import (
    SYSTEM_PROMPT,
    check_agent_enabled,
    get_backend_info,
    LLMManager,
)

__all__ = [
    # Tools
    "TOOL_SCHEMAS",
    "TOOL_FUNCTIONS",
    "execute_tool",
    "get_client",
    "list_available_models",
    "get_model_metadata",
    "get_model_config",
    "analyze_model_type",
    "get_model_input_requirements",
    "get_model_output_interpretation",
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
    # Prompts
    "SYSTEM_PROMPT",
    "check_agent_enabled",
    "get_backend_info",
    "LLMManager",
]
