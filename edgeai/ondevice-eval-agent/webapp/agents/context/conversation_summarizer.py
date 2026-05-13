"""
Layer 1 — conversation summarization.

When the running conversation is large (past `conversation_trigger_tokens`),
collapse everything except the last `keep_messages` messages into a single
summary, preserving the original system prompt.

Output shape:
    [ system (if present),
      summary_system_message,  # NEW: compressed older context
      ...last N turns... ]

The summary is emitted as a system-role message so providers treat it as
instruction context rather than chat history.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from config import get_settings

from .token_counter import _content_to_str

logger = logging.getLogger(__name__)

_SUMMARY_MARKER = "_overflow_conversation_summary"

SUMMARY_PROMPT = (
    "You are compressing the earlier turns of a technical chat between a "
    "developer and an ML-inference assistant so the context fits in the "
    "model's window.\n\n"
    "Rules:\n"
    "- Preserve decisions, chosen models, configuration, error messages, "
    "file paths, and identifiers.\n"
    "- Keep exact numbers (latencies, accuracies, thresholds).\n"
    "- Drop pleasantries and restatements.\n"
    "- Output a numbered list of the key facts the assistant must "
    "remember. Under 800 tokens.\n\n"
    "Turns to summarize:\n{content}\n"
)


def _format_history(messages: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for m in messages:
        role = m.get("role", "user")
        text = _content_to_str(m.get("content", ""))
        lines.append(f"[{role}] {text}")
    return "\n\n".join(lines)


def _call_summary_llm(content: str) -> Optional[str]:
    try:
        from router import get_router
    except Exception as exc:
        logger.warning("overflow_conv_summarizer_router_unavailable: %s", exc)
        return None

    try:
        router = get_router()
        settings = get_settings().overflow
        response = router.chat(
            messages=[{"role": "user", "content": SUMMARY_PROMPT.format(content=content)}],
            model=settings.summary_model,
            max_tokens=1000,
            temperature=0.0,
        )
        if response and hasattr(response, "content"):
            return response.content
        if isinstance(response, dict):
            return response.get("content") or response.get("text")
    except Exception as exc:
        logger.warning("overflow_conv_summarizer_call_failed: %s", exc)
    return None


def _already_summarized(message: Dict[str, Any]) -> bool:
    meta = message.get("metadata")
    return isinstance(meta, dict) and meta.get(_SUMMARY_MARKER) is True


def summarize_older_turns(
    messages: List[Dict[str, Any]],
    *,
    total_tokens: int,
) -> List[Dict[str, Any]]:
    """
    If over the conversation threshold, collapse older turns into one
    summary message. Returns the new message list.
    """
    settings = get_settings().overflow
    if total_tokens < settings.conversation_trigger_tokens:
        return messages
    if len(messages) <= settings.keep_messages:
        return messages

    system_prefix: List[Dict[str, Any]] = []
    rest: List[Dict[str, Any]] = []
    for m in messages:
        if m.get("role") == "system" and not rest:
            system_prefix.append(m)
        else:
            rest.append(m)

    if len(rest) <= settings.keep_messages:
        return messages

    older = rest[: -settings.keep_messages]
    keep = rest[-settings.keep_messages :]

    # Skip if older slice is already a summary.
    if len(older) == 1 and _already_summarized(older[0]):
        return messages

    body = _format_history(older)
    summary_text = _call_summary_llm(body)
    if not summary_text:
        return messages

    summary_msg: Dict[str, Any] = {
        "role": "system",
        "content": (
            "[Summary of earlier conversation, produced by overflow-layer1]\n\n"
            + summary_text
        ),
        "metadata": {_SUMMARY_MARKER: True, "original_turn_count": len(older)},
    }

    logger.info(
        "overflow_layer1_summarized_conversation",
        extra={
            "original_turn_count": len(older),
            "kept_turn_count": len(keep),
        },
    )
    return system_prefix + [summary_msg] + keep


__all__ = ["summarize_older_turns"]
