"""
Langfuse Cloud tracing.

Singleton TracingService that wraps the Langfuse Python SDK v3. The agent
uses raw provider SDKs (anthropic, openai, google-genai) rather than
LangChain, so we create spans manually via the SDK's context-manager
surface — not via LangChain's CallbackHandler.

Design rules:
    - Disabled by default. `LANGFUSE_ENABLED=true` plus keys to turn on.
    - Graceful degradation: import failure, missing keys, or SDK errors
      quietly flip the service to a no-op shell for the process lifetime.
    - Never raise from inside a span; never block a response on tracing.
    - Zero hot-path overhead when disabled (all methods become no-ops).

Usage:
    from observability.tracing import get_tracing

    tracing = get_tracing()
    with tracing.chat_turn(session_id=sid, request_id=rid) as turn:
        with tracing.llm_call(provider="anthropic", model="claude-sonnet-4-6",
                              messages=msgs, tools=tools) as llm_span:
            response = anthropic_client.messages.create(...)
            if llm_span is not None:
                llm_span.update(
                    output={"text": response.content},
                    usage={"input": response.usage.input_tokens,
                           "output": response.usage.output_tokens},
                )
        for tc in response.tool_calls:
            with tracing.tool_call(tool_name=tc.name, args=tc.args):
                result = execute_tool(tc.name, tc.args)
    tracing.flush()
"""

from __future__ import annotations

import logging
import threading
from contextlib import contextmanager
from typing import Any, Dict, Generator, List, Optional

from config import get_settings
from observability.request_context import get_request_id, get_session_id

logger = logging.getLogger(__name__)


class TracingService:
    """
    Singleton wrapper around the Langfuse SDK client.

    On first use, reads config from `Settings.langfuse`, sets the SDK's
    expected environment variables, and calls `langfuse.get_client()`. Any
    failure (missing package, missing keys, network issue) disables the
    service for the process lifetime — subsequent method calls return
    no-op context managers.
    """

    _instance: "TracingService | None" = None
    _lock = threading.Lock()

    def __new__(cls) -> "TracingService":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._initialized = True
        self._client = None
        self._enabled = False
        self._init_client()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def enabled(self) -> bool:
        return self._enabled

    @contextmanager
    def chat_turn(
        self,
        *,
        session_id: Optional[str] = None,
        request_id: Optional[str] = None,
        user_metadata: Optional[Dict[str, Any]] = None,
    ) -> Generator[Any, None, None]:
        """Root span for one chat request. Child spans nest under it automatically."""
        if not self._enabled or self._client is None:
            yield None
            return

        metadata = self._base_metadata(user_metadata)
        session_id = session_id or get_session_id() or None
        request_id = request_id or get_request_id() or None
        if session_id:
            metadata["session_id"] = session_id
        if request_id:
            metadata["request_id"] = request_id

        try:
            ctx = self._client.start_as_current_span(
                name="chat_turn",
                input={"session_id": session_id, "request_id": request_id},
                metadata=metadata,
            )
            with ctx as span:
                if session_id and hasattr(span, "update_trace"):
                    try:
                        span.update_trace(session_id=session_id, user_id=session_id)
                    except Exception:
                        pass
                yield span
        except Exception as exc:
            logger.warning("langfuse_chat_turn_failed: %s", exc)
            yield None

    @contextmanager
    def llm_call(
        self,
        *,
        provider: str,
        model: str,
        messages: Optional[List[Dict[str, Any]]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Generator[Any, None, None]:
        """Generation span for a single LLM request (inside a chat_turn)."""
        if not self._enabled or self._client is None:
            yield None
            return

        try:
            ctx = self._client.start_as_current_generation(
                name=f"{provider}:{model}",
                model=model,
                input=messages if messages is not None else None,
                metadata={
                    "provider": provider,
                    "tool_count": len(tools) if tools else 0,
                },
            )
            with ctx as span:
                yield span
        except Exception as exc:
            logger.warning("langfuse_llm_call_failed: %s", exc)
            yield None

    @contextmanager
    def tool_call(
        self,
        *,
        tool_name: str,
        args: Optional[Dict[str, Any]] = None,
    ) -> Generator[Any, None, None]:
        """Span around a single tool execution."""
        if not self._enabled or self._client is None:
            yield None
            return

        try:
            ctx = self._client.start_as_current_span(
                name=f"tool:{tool_name}",
                input=args,
                metadata={"tool_name": tool_name},
            )
            with ctx as span:
                yield span
        except Exception as exc:
            logger.warning("langfuse_tool_call_failed: %s", exc)
            yield None

    def flush(self) -> None:
        """Flush pending traces. Safe to call when disabled."""
        if not self._enabled or self._client is None:
            return
        try:
            self._client.flush()
        except Exception as exc:
            logger.warning("langfuse_flush_failed: %s", exc)

    # Testing / reconfiguration helper
    def reinit(self) -> None:
        """Force a re-read of Settings and re-attempt SDK init."""
        with self._lock:
            self._client = None
            self._enabled = False
            self._init_client()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _init_client(self) -> None:
        settings = get_settings().langfuse
        if not settings.enabled:
            return
        if not settings.public_key or not settings.secret_key:
            logger.warning(
                "LANGFUSE_ENABLED=true but LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY "
                "are not set; tracing will stay disabled.",
            )
            return

        # SDK v3 reads from env vars, not constructor args.
        import os as _os
        _os.environ["LANGFUSE_PUBLIC_KEY"] = settings.public_key
        _os.environ["LANGFUSE_SECRET_KEY"] = settings.secret_key
        _os.environ["LANGFUSE_HOST"] = settings.host

        try:
            from langfuse import get_client
        except ImportError as exc:
            logger.warning("langfuse_import_failed: %s (is the package installed?)", exc)
            return
        except Exception as exc:  # pragma: no cover
            logger.warning("langfuse_import_error: %s", exc)
            return

        try:
            self._client = get_client()
        except Exception as exc:
            logger.warning("langfuse_client_init_failed: %s", exc)
            self._client = None
            return

        self._enabled = True
        logger.info("Langfuse tracing enabled (host=%s)", settings.host)

    def _base_metadata(self, extra: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        settings = get_settings().langfuse
        meta: Dict[str, Any] = {
            "service": "ondevice-eval-agent",
        }
        if settings.deployment_tag:
            meta["deployment_tag"] = settings.deployment_tag
        if extra:
            meta.update(extra)
        return meta


_tracing: Optional[TracingService] = None
_tracing_lock = threading.Lock()


def get_tracing() -> TracingService:
    """Return the process-wide TracingService, creating it on first use."""
    global _tracing
    if _tracing is None:
        with _tracing_lock:
            if _tracing is None:
                _tracing = TracingService()
    return _tracing


__all__ = [
    "TracingService",
    "get_tracing",
]
