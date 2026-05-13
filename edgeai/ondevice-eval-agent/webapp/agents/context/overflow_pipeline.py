"""
Orchestration for the 4-layer overflow pipeline.

`apply(messages, provider=..., model=...)` runs the Flask-side layers
(2 -> 1 -> 4) and returns the (possibly modified) message list. Layer 3
(Anthropic server-side compaction) is kwargs-level and handled by the
Anthropic adapter; see `agents.context.anthropic_compaction.build_kwargs`.

The pipeline is a no-op when OVERFLOW_ENABLED=false so the existing
sliding-window behavior in api/agent.py keeps working untouched.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from config import get_settings

from .conversation_summarizer import summarize_older_turns
from .token_counter import count_messages_tokens
from .tool_result_summarizer import summarize_large_tool_results
from .trim import trim_to_max_tokens

logger = logging.getLogger(__name__)


def apply(
    messages: List[Dict[str, Any]],
    *,
    provider: Optional[str] = None,
    model: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Run the Flask-side overflow layers.

    Order:
        1. Tool-result summarization — shrink individual giant tool payloads.
        2. Conversation summarization — collapse older non-recent turns.
        3. (skipped here; handled kwargs-side in the Anthropic adapter.)
        4. Hard trim — enforce the absolute token ceiling as a safety net.
    """
    settings = get_settings().overflow
    if not settings.enabled or not messages:
        return messages

    try:
        total = count_messages_tokens(messages)

        # Layer 2: shrink giant tool results first — often recovers the
        # most tokens with the cheapest summary call.
        messages = summarize_large_tool_results(messages, total_tokens=total)

        # Recount; Layer 1 decision uses the post-Layer-2 total.
        total = count_messages_tokens(messages)
        messages = summarize_older_turns(messages, total_tokens=total)

        # Layer 4 safety net.
        messages = trim_to_max_tokens(
            messages,
            max_tokens=settings.hard_ceiling_tokens,
        )
    except Exception as exc:
        # The pipeline must never block a real request on its own failure.
        logger.warning(
            "overflow_pipeline_error_falling_through: %s",
            exc,
            extra={"provider": provider, "model": model},
        )

    return messages


__all__ = ["apply"]
