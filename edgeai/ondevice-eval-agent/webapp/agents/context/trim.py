"""
Layer 4 — hard-trim safety net.

If total tokens still exceed the hard ceiling after Layers 1 + 2 (and even
Layer 3 failed to help, e.g. non-Anthropic provider), drop oldest messages
until we're under the ceiling. Preserves the system prompt and the most
recent user/assistant turns.

Uses langchain-core's `trim_messages(strategy="last")` when available;
falls back to a manual tail-preserving truncation otherwise.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from .token_counter import (
    _dicts_to_lc_messages,
    count_messages_tokens,
    is_langchain_available,
)

logger = logging.getLogger(__name__)

try:
    from langchain_core.messages.utils import (
        count_tokens_approximately,
        trim_messages,
    )

    _TRIM_AVAILABLE = True
except Exception:  # pragma: no cover
    _TRIM_AVAILABLE = False


def _fallback_trim(
    messages: List[Dict[str, Any]],
    *,
    max_tokens: int,
) -> List[Dict[str, Any]]:
    """Drop oldest non-system messages until under `max_tokens`."""
    if not messages:
        return messages

    # Keep the system message (first one, if present) at the head.
    system_msgs: List[Dict[str, Any]] = []
    rest: List[Dict[str, Any]] = []
    for m in messages:
        if m.get("role") == "system" and not rest:
            system_msgs.append(m)
        else:
            rest.append(m)

    # Drop from the front of `rest` until we fit.
    while rest and count_messages_tokens(system_msgs + rest) > max_tokens:
        rest.pop(0)

    return system_msgs + rest


def trim_to_max_tokens(
    messages: List[Dict[str, Any]],
    *,
    max_tokens: int,
) -> List[Dict[str, Any]]:
    """
    Trim `messages` so the total token estimate is <= `max_tokens`.

    No-op if already under. Preserves the system prompt. Drops oldest
    non-system messages first.
    """
    if not messages:
        return messages

    current = count_messages_tokens(messages)
    if current <= max_tokens:
        return messages

    if _TRIM_AVAILABLE and is_langchain_available():
        try:
            lc_msgs = _dicts_to_lc_messages(messages)
            trimmed = trim_messages(
                lc_msgs,
                max_tokens=max_tokens,
                token_counter=count_tokens_approximately,
                strategy="last",
                start_on="human",
                include_system=True,
                allow_partial=False,
            )
            # Map back onto the original dicts by identity-preserving index.
            # trim_messages returns the tail of lc_msgs; compute how many the
            # tail covers and slice `messages` accordingly. This keeps all
            # the original dict fields (tool_call_id, metadata, etc.).
            tail_count = len(trimmed)
            if tail_count == len(lc_msgs):
                return messages
            # system + tail of non-system
            sys_prefix = [m for m in messages if m.get("role") == "system"][:1]
            non_sys = [m for m in messages if m.get("role") != "system"]
            keep = non_sys[-(tail_count - len(sys_prefix)) :] if tail_count > len(sys_prefix) else []
            trimmed_msgs = sys_prefix + keep
            logger.warning(
                "overflow_layer4_trimmed",
                extra={
                    "original_count": len(messages),
                    "kept_count": len(trimmed_msgs),
                    "original_tokens": current,
                    "max_tokens": max_tokens,
                },
            )
            return trimmed_msgs
        except Exception as exc:
            logger.warning("overflow_layer4_trim_messages_failed: %s", exc)

    trimmed = _fallback_trim(messages, max_tokens=max_tokens)
    if len(trimmed) < len(messages):
        logger.warning(
            "overflow_layer4_fallback_trimmed",
            extra={
                "original_count": len(messages),
                "kept_count": len(trimmed),
                "original_tokens": current,
                "max_tokens": max_tokens,
            },
        )
    return trimmed


__all__ = ["trim_to_max_tokens"]
