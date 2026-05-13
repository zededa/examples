"""
Result Store — Persist evaluation and benchmark results to session storage.

Results are saved as JSON files in the session's storage directory, using
the existing ``mcp.session.get_session_storage_path()`` infrastructure.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_VALID_TYPES = {"benchmark", "eval", "comparison"}


def _get_results_dir(session_id: str) -> str:
    """Get or create the results subdirectory for a session."""
    from sessions.registry import get_session_storage_path

    session_dir = get_session_storage_path(session_id)
    results_dir = os.path.join(session_dir, "eval_results")
    os.makedirs(results_dir, exist_ok=True)
    return results_dir


def save_result(
    session_id: str,
    result_type: str,
    result: Dict[str, Any],
) -> str:
    """
    Save an evaluation result to session storage.

    Args:
        session_id: Session identifier.
        result_type: One of ``benchmark``, ``eval``, ``comparison``.
        result: Result data to persist.

    Returns:
        Filename of the saved result.
    """
    if result_type not in _VALID_TYPES:
        raise ValueError(f"Invalid result_type '{result_type}', must be one of {_VALID_TYPES}")

    results_dir = _get_results_dir(session_id)
    timestamp = int(time.time() * 1000)
    filename = f"{result_type}_{timestamp}.json"
    filepath = os.path.join(results_dir, filename)

    # Add metadata envelope
    envelope = {
        "result_type": result_type,
        "saved_at": time.time(),
        "session_id": session_id,
        "data": result,
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(envelope, f, indent=2, default=str)

    logger.info("Saved %s result: %s", result_type, filepath)
    return filename


def list_results(
    session_id: str,
    result_type: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    List saved results for a session.

    Args:
        session_id: Session identifier.
        result_type: Filter by type (optional).

    Returns:
        List of result metadata dicts with ``filename``, ``result_type``,
        ``saved_at``, and a brief ``summary``.
    """
    results_dir = _get_results_dir(session_id)
    entries: List[Dict[str, Any]] = []

    if not os.path.isdir(results_dir):
        return entries

    for filename in sorted(os.listdir(results_dir), reverse=True):
        if not filename.endswith(".json"):
            continue

        # Filter by type if requested
        if result_type and not filename.startswith(f"{result_type}_"):
            continue

        filepath = os.path.join(results_dir, filename)
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            entries.append({
                "filename": filename,
                "result_type": data.get("result_type", "unknown"),
                "saved_at": data.get("saved_at"),
                "summary": _extract_summary(data),
            })
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to read result %s: %s", filename, e)

    return entries


def load_result(session_id: str, filename: str) -> Dict[str, Any]:
    """
    Load a specific result by filename.

    Args:
        session_id: Session identifier.
        filename: Result filename (e.g. ``benchmark_1713200000000.json``).

    Returns:
        Full result data.

    Raises:
        FileNotFoundError: If the result file doesn't exist.
    """
    # Sanitize filename to prevent path traversal
    safe_filename = os.path.basename(filename)
    results_dir = _get_results_dir(session_id)
    filepath = os.path.join(results_dir, safe_filename)

    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Result '{safe_filename}' not found")

    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def _extract_summary(envelope: Dict[str, Any]) -> str:
    """Extract a brief summary from a result envelope."""
    data = envelope.get("data", {})
    rtype = envelope.get("result_type", "")

    if rtype == "benchmark":
        model = data.get("model_name", "?")
        agg = data.get("aggregate", {})
        tps = agg.get("tokens_per_second", {}).get("mean", "?")
        return f"{model}: {tps} tok/s"

    if rtype == "eval":
        model = data.get("model_name", "?")
        dataset = data.get("dataset", "?")
        accuracy = data.get("accuracy", "?")
        if isinstance(accuracy, float):
            accuracy = f"{accuracy:.1%}"
        return f"{model} on {dataset}: {accuracy}"

    if rtype == "comparison":
        a = data.get("model_a", {}).get("model_name", "?")
        b = data.get("model_b", {}).get("model_name", "?")
        return f"{a} vs {b}"

    return ""
