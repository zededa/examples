"""
Token estimation heuristics.

Character-based approximations used to gate context-size decisions in the
resilience layer (truncation, deduplication hashing context). In PR 3
the overflow pipeline wraps `count_tokens_approximately` from langchain-core
and shares the same heuristic surface; until then, callers can import
`estimate_tokens` / `estimate_messages_tokens` here.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List


def estimate_tokens(text: str, model: str = "claude") -> int:
    """
    Estimate token count for text.

    Heuristic: ~3.5 characters per token. Slightly conservative; good enough
    for threshold gating without pulling in a tokenizer dependency.
    """
    if not text:
        return 0
    chars_per_token = 3.5
    return int(len(text) / chars_per_token) + 1


def estimate_messages_tokens(messages: List[Dict[str, Any]], model: str = "claude") -> int:
    """Estimate total tokens for a list of chat messages (dict form)."""
    total = 0
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            total += estimate_tokens(content, model)
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict):
                    total += estimate_tokens(json.dumps(item), model)
                else:
                    total += estimate_tokens(str(item), model)
        # Per-message overhead for role, structure
        total += 4
    return total


__all__ = [
    "estimate_tokens",
    "estimate_messages_tokens",
]
