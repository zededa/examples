"""
Per-request context propagation.

Exposes ContextVars that carry the current request_id and session_id across
Flask handlers, thread-pool workers (when propagated), and observability
hooks. Also used by the Langfuse tracing service (PR 2) so every span
carries the same request_id as the structured logs.

Usage:
    from observability.request_context import set_request_context, new_request_id

    req_id = new_request_id()
    token = set_request_context(request_id=req_id, session_id=session_id)
    try:
        ...  # handle request
    finally:
        clear_request_context(token)
"""

from __future__ import annotations

import uuid
from contextvars import ContextVar, Token
from typing import Optional, Tuple

request_id_var: ContextVar[str] = ContextVar("request_id", default="")
session_id_var: ContextVar[str] = ContextVar("session_id", default="")


def new_request_id() -> str:
    """Return a fresh opaque request id."""
    return uuid.uuid4().hex


def get_request_id() -> str:
    """Read the request id bound to the current context, or empty string."""
    return request_id_var.get()


def get_session_id() -> str:
    """Read the session id bound to the current context, or empty string."""
    return session_id_var.get()


def set_request_context(
    *,
    request_id: Optional[str] = None,
    session_id: Optional[str] = None,
) -> Tuple[Optional[Token], Optional[Token]]:
    """
    Bind request_id/session_id to the current context.

    Returns a pair of reset tokens (one per var) that should be passed to
    `clear_request_context` on the way out.
    """
    rid_token = request_id_var.set(request_id) if request_id is not None else None
    sid_token = session_id_var.set(session_id) if session_id is not None else None
    return rid_token, sid_token


def clear_request_context(
    tokens: Tuple[Optional[Token], Optional[Token]],
) -> None:
    """Undo the ContextVar bindings set by `set_request_context`."""
    rid_token, sid_token = tokens
    if rid_token is not None:
        request_id_var.reset(rid_token)
    if sid_token is not None:
        session_id_var.reset(sid_token)
