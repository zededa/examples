"""
Token counting for the overflow pipeline.

Prefers langchain-core's `count_tokens_approximately` (char-based heuristic)
when available. Falls back to the same chars/4 heuristic already used by
`router.resilience.estimation` when langchain-core is missing — so the
pipeline still works if the optional dependency isn't installed.

Messages are OpenAI-style dicts (`{"role": ..., "content": ...}`); content
may be a string or a list of parts (e.g. vision messages). We convert them
to LangChain BaseMessage objects for counting only; the pipeline continues
to pass plain dicts to provider SDKs.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Union

try:
    from langchain_core.messages import (
        AIMessage,
        HumanMessage,
        SystemMessage,
        ToolMessage,
    )
    from langchain_core.messages.utils import count_tokens_approximately

    _LANGCHAIN_AVAILABLE = True
except Exception:  # pragma: no cover - graceful degradation
    _LANGCHAIN_AVAILABLE = False
    count_tokens_approximately = None  # type: ignore[assignment]


def _content_to_str(content: Any) -> str:
    """Flatten structured content (vision, tool parts) into a plain string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict):
                parts.append(json.dumps(item, default=str))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content)


def _dicts_to_lc_messages(messages: List[Dict[str, Any]]):
    """Best-effort mapping from our message dicts to LangChain BaseMessages."""
    out = []
    for msg in messages:
        role = msg.get("role", "user")
        content = _content_to_str(msg.get("content", ""))
        if role == "system":
            out.append(SystemMessage(content=content))
        elif role == "assistant":
            out.append(AIMessage(content=content))
        elif role == "tool":
            out.append(
                ToolMessage(
                    content=content,
                    tool_call_id=msg.get("tool_call_id", ""),
                )
            )
        else:
            out.append(HumanMessage(content=content))
    return out


def _fallback_count(text: str) -> int:
    """chars / 3.5 + 1 — matches router.resilience.estimation.estimate_tokens."""
    if not text:
        return 0
    return int(len(text) / 3.5) + 1


def count_message_tokens(message: Dict[str, Any]) -> int:
    """Count tokens in a single message dict."""
    if _LANGCHAIN_AVAILABLE:
        lc = _dicts_to_lc_messages([message])
        return count_tokens_approximately(lc)
    content_str = _content_to_str(message.get("content", ""))
    return _fallback_count(content_str) + 4  # per-message structural overhead


def count_messages_tokens(messages: List[Dict[str, Any]]) -> int:
    """Count total tokens across a list of messages."""
    if not messages:
        return 0
    if _LANGCHAIN_AVAILABLE:
        return count_tokens_approximately(_dicts_to_lc_messages(messages))
    return sum(count_message_tokens(m) for m in messages)


def is_langchain_available() -> bool:
    return _LANGCHAIN_AVAILABLE


__all__ = [
    "count_message_tokens",
    "count_messages_tokens",
    "is_langchain_available",
]
