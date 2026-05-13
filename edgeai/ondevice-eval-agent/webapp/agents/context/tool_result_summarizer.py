"""
Layer 2 — tool-result summarization.

When total context grows large AND an individual `tool` message carries a
huge payload (e.g. a model dump or inference result), rewrite that tool
result in place with a short summary. Summaries carry a marker so Layer 2
won't re-summarize them on subsequent turns.

Summarization uses the same LLMRouter the agent already has configured;
defaults to a cheap model via OVERFLOW_SUMMARY_MODEL or the active provider's
default. When no provider is available we log a warning and leave the
tool result unchanged — Layer 3/4 are still in place to protect the
request.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from config import get_settings

from .token_counter import _content_to_str, count_message_tokens

logger = logging.getLogger(__name__)

_SUMMARIZED_MARKER = "_overflow_summarized"

SUMMARY_PROMPT = (
    "You are compressing an oversized tool-call output so the conversation "
    "stays within the model's context window.\n\n"
    "Rules:\n"
    "- Preserve all error messages and exception text verbatim.\n"
    "- Preserve model/device IDs, file paths, numeric results, and any "
    "tokens that look like identifiers.\n"
    "- Collapse long repetitive structures (tables, lists of similar rows) "
    "into a 1-2 sentence description + one or two representative rows.\n"
    "- Output plain text, no markdown headers. Aim for under "
    "{max_tokens} tokens.\n\n"
    "Tool result to summarize:\n{content}\n"
)


def _is_already_summarized(message: Dict[str, Any]) -> bool:
    meta = message.get("metadata")
    if isinstance(meta, dict) and meta.get(_SUMMARIZED_MARKER):
        return True
    return False


def _mark_summarized(message: Dict[str, Any], original_tokens: int) -> Dict[str, Any]:
    meta = dict(message.get("metadata") or {})
    meta[_SUMMARIZED_MARKER] = True
    meta["original_token_estimate"] = original_tokens
    message["metadata"] = meta
    return message


def _call_summary_llm(
    content: str,
    *,
    max_tokens: int,
) -> Optional[str]:
    """Call the configured LLM router to produce a short summary."""
    try:
        from router import get_router
    except Exception as exc:
        logger.warning("overflow_summarizer_router_unavailable: %s", exc)
        return None

    try:
        router = get_router()
        settings = get_settings().overflow

        prompt = SUMMARY_PROMPT.format(content=content, max_tokens=max_tokens)
        response = router.chat(
            messages=[{"role": "user", "content": prompt}],
            model=settings.summary_model,
            max_tokens=max_tokens + 50,
            temperature=0.0,
        )
        if response and hasattr(response, "content"):
            return response.content
        if isinstance(response, dict):
            return response.get("content") or response.get("text")
    except Exception as exc:
        logger.warning("overflow_summarizer_call_failed: %s", exc)
    return None


def summarize_large_tool_results(
    messages: List[Dict[str, Any]],
    *,
    total_tokens: int,
) -> List[Dict[str, Any]]:
    """
    Possibly replace large `tool` messages with summaries. Returns the
    (possibly mutated) message list.
    """
    settings = get_settings().overflow
    if total_tokens < settings.tool_context_threshold_tokens:
        return messages

    updated = False
    for msg in messages:
        if msg.get("role") != "tool":
            continue
        if _is_already_summarized(msg):
            continue

        msg_tokens = count_message_tokens(msg)
        if msg_tokens < settings.tool_result_threshold_tokens:
            continue

        content_str = _content_to_str(msg.get("content", ""))
        summary = _call_summary_llm(
            content_str,
            max_tokens=settings.tool_summary_max_tokens,
        )
        if not summary:
            continue

        msg["content"] = (
            f"[Summarized from ~{msg_tokens} tokens by overflow-layer2]\n\n{summary}"
        )
        _mark_summarized(msg, original_tokens=msg_tokens)
        updated = True
        logger.info(
            "overflow_layer2_summarized_tool_result",
            extra={
                "tool_call_id": msg.get("tool_call_id"),
                "original_tokens": msg_tokens,
                "summary_max_tokens": settings.tool_summary_max_tokens,
            },
        )

    return messages if updated or not messages else messages


__all__ = [
    "summarize_large_tool_results",
    "_SUMMARIZED_MARKER",
]
