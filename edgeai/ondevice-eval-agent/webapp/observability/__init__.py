"""
Observability: logging, request context, tracing (PR 2).

Structure:
    logging.py           - in-process endpoint and processing log queues
                           (moved from utils/logging.py)
    request_context.py   - per-request ContextVar propagation (request_id,
                           session_id) for logging and Langfuse tracing

Langfuse tracing wiring (TracingService) is added in PR 2.
"""

from .logging import (
    endpoint_logs,
    processing_logs,
    endpoint_logs_lock,
    processing_logs_lock,
    log_endpoint_call,
    log_processing_step,
    init_log_queues,
    clear_all_logs,
)

from .request_context import (
    request_id_var,
    session_id_var,
    get_request_id,
    get_session_id,
    set_request_context,
    clear_request_context,
    new_request_id,
)

__all__ = [
    "endpoint_logs",
    "processing_logs",
    "endpoint_logs_lock",
    "processing_logs_lock",
    "log_endpoint_call",
    "log_processing_step",
    "init_log_queues",
    "clear_all_logs",
    "request_id_var",
    "session_id_var",
    "get_request_id",
    "get_session_id",
    "set_request_context",
    "clear_request_context",
    "new_request_id",
]
