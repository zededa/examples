"""
Response Scoring — Pure-Python scorers for LLM evaluation.

Each scorer compares an LLM response against an expected answer and
returns a standardised result dict.  No ML dependencies — only string
operations and regex.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Dict

# Type alias for scorer functions
ScorerFn = Callable[..., Dict[str, Any]]

_RESULT_KEYS = ("correct", "score", "method", "detail")


def _result(correct: bool, method: str, detail: str = "") -> Dict[str, Any]:
    return {
        "correct": correct,
        "score": 1.0 if correct else 0.0,
        "method": method,
        "detail": detail,
    }


# =============================================================================
# Scorers
# =============================================================================

def exact_match(response: str, expected: str, **_: Any) -> Dict[str, Any]:
    """Case-insensitive exact match after stripping whitespace."""
    r = response.strip().lower()
    e = expected.strip().lower()
    return _result(r == e, "exact_match", f"got='{r[:80]}' expected='{e[:80]}'")


def contains_match(response: str, expected: str, **_: Any) -> Dict[str, Any]:
    """Check if the expected answer appears anywhere in the response."""
    r = response.lower()
    e = expected.strip().lower()
    found = e in r
    return _result(found, "contains", f"expected='{e[:80]}' found={found}")


def multiple_choice(response: str, expected: str, **_: Any) -> Dict[str, Any]:
    """
    Extract a single letter (A-D) from the response and compare.

    Tries several extraction strategies:
    1. Explicit "Answer: X" or "answer is X" patterns
    2. Standalone letter at the start of the response
    3. First A-D letter surrounded by word boundaries
    """
    e = expected.strip().upper()
    if len(e) != 1 or e not in "ABCD":
        return _result(False, "multiple_choice", f"invalid expected='{e}'")

    resp = response.strip()

    # Strategy 1: "answer is X", "Answer: X", "(X)" at end
    patterns = [
        r"(?:answer|choice)\s*(?:is|:)\s*\(?([A-Da-d])\)?",
        r"^\s*\(?([A-Da-d])\)?[\s\.\),:]",
        r"\b([A-Da-d])\b",
    ]
    for pattern in patterns:
        m = re.search(pattern, resp, re.IGNORECASE)
        if m:
            extracted = m.group(1).upper()
            return _result(
                extracted == e,
                "multiple_choice",
                f"extracted='{extracted}' expected='{e}'",
            )

    return _result(False, "multiple_choice", "no letter A-D found in response")


def numeric_match(
    response: str, expected: str, tolerance: float = 0.01, **_: Any
) -> Dict[str, Any]:
    """
    Extract the last number from the response and compare within tolerance.

    Uses the *last* number to handle chain-of-thought responses where the
    final answer appears at the end (e.g. "The answer is 42").
    """
    try:
        expected_num = float(expected.strip().replace(",", ""))
    except ValueError:
        return _result(False, "numeric", f"expected is not a number: '{expected}'")

    # Find all numbers in response
    numbers = re.findall(r"-?[\d,]+\.?\d*", response)
    if not numbers:
        return _result(False, "numeric", "no number found in response")

    # Use the last number (most likely the final answer)
    try:
        extracted = float(numbers[-1].replace(",", ""))
    except ValueError:
        return _result(False, "numeric", f"could not parse '{numbers[-1]}'")

    # Compare with tolerance
    if expected_num == 0:
        correct = abs(extracted) < tolerance
    else:
        correct = abs(extracted - expected_num) / abs(expected_num) <= tolerance

    return _result(
        correct,
        "numeric",
        f"extracted={extracted} expected={expected_num} tol={tolerance}",
    )


def regex_match(response: str, expected: str, **_: Any) -> Dict[str, Any]:
    """Use ``expected`` as a regex pattern and search the response."""
    try:
        match = re.search(expected, response, re.IGNORECASE)
        found = match is not None
        return _result(found, "regex", f"pattern='{expected[:60]}' found={found}")
    except re.error as e:
        return _result(False, "regex", f"invalid pattern: {e}")


# =============================================================================
# Dispatcher
# =============================================================================

SCORERS: Dict[str, ScorerFn] = {
    "exact": exact_match,
    "exact_match": exact_match,
    "contains": contains_match,
    "multiple_choice": multiple_choice,
    "numeric": numeric_match,
    "regex": regex_match,
}


def score_response(
    response: str,
    expected: str,
    score_type: str = "contains",
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Score an LLM response against the expected answer.

    Args:
        response: The LLM's response text.
        expected: The expected/ground-truth answer.
        score_type: Scoring method — one of ``exact``, ``contains``,
            ``multiple_choice``, ``numeric``, ``regex``.

    Returns:
        Dict with keys: ``correct`` (bool), ``score`` (float 0-1),
        ``method`` (str), ``detail`` (str).
    """
    scorer = SCORERS.get(score_type, contains_match)
    return scorer(response, expected, **kwargs)
