"""
Layer 3 — Anthropic server-side context compaction.

Beta feature (`compact-2026-01-12`): passes the raw message list through
as-is; Anthropic's servers compact older turns into a summary on their
side when `input_tokens >= trigger.value`. The caller sees a normal
response. We don't touch the local message list at all for this layer.

Delivery mechanism:
    The stable `client.messages.create(...)` call does NOT accept a
    `betas` kwarg — that's reserved for `client.beta.messages.create()`.
    Beta features on the stable endpoint are enabled via:
        extra_headers={"anthropic-beta": "compact-2026-01-12"}
        extra_body={"context_management": {...}}
    Passing `betas` directly raises:
        TypeError: Messages.create() got an unexpected keyword argument 'betas'
    which is what broke v2 images shipping the old behavior.

Usage (from the Anthropic adapter):
    from agents.context.anthropic_compaction import build_kwargs

    extra = build_kwargs()
    anthropic_client.messages.create(
        model=model,
        messages=messages,
        **extra,  # adds extra_headers + extra_body when enabled
    )

When OVERFLOW_ANTHROPIC_COMPACTION_ENABLED=false (or OVERFLOW_ENABLED=false),
`build_kwargs()` returns `{}` and nothing changes.
"""

from __future__ import annotations

from typing import Any, Dict

from config import get_settings

DEFAULT_INSTRUCTIONS = (
    "Preserve tool-call decisions, model identifiers, inference results, "
    "and error messages verbatim. Summarize long tool-result payloads but "
    "keep their key numbers and error strings."
)

BETA_HEADER = "compact-2026-01-12"


def build_kwargs(instructions: str | None = None) -> Dict[str, Any]:
    """
    Produce the kwargs to merge into Anthropic's messages.create(...) call.

    Returns {} when disabled so callers can always do `**build_kwargs()`.
    """
    settings = get_settings().overflow
    if not settings.enabled or not settings.anthropic_compaction_enabled:
        return {}

    return {
        "extra_headers": {"anthropic-beta": BETA_HEADER},
        "extra_body": {
            "context_management": {
                "edits": [
                    {
                        "type": "compact_20260112",
                        "trigger": {
                            "type": "input_tokens",
                            "value": settings.anthropic_compaction_tokens,
                        },
                        "instructions": instructions or DEFAULT_INSTRUCTIONS,
                    }
                ]
            }
        },
    }


__all__ = ["build_kwargs", "DEFAULT_INSTRUCTIONS", "BETA_HEADER"]
