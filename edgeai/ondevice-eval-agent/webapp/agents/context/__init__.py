"""
4-layer context-overflow protection for the chat pipeline.

Entry point:
    from agents.context import overflow_pipeline
    messages = overflow_pipeline.apply(messages, provider=..., model=...)

Layers (runs in this order):
    1. Tool-result summarization — replace huge `tool` results with summaries
       when total context > threshold AND any single result > threshold.
    2. Conversation summarization — collapse older non-recent messages into
       one summary message when total context > threshold.
    3. Hard trim safety net — `trim_messages(strategy="last")` enforces an
       absolute ceiling. Should almost never fire.

Layer 3 (Anthropic server-side compaction) is NOT applied here — it's a
kwargs-level change in the Anthropic adapter. Call
`agents.context.anthropic_compaction.build_kwargs(...)` to get the
dict to merge into the Anthropic SDK call.
"""

from __future__ import annotations

from . import overflow_pipeline
from . import anthropic_compaction
from . import token_counter

__all__ = [
    "overflow_pipeline",
    "anthropic_compaction",
    "token_counter",
]
