"""
Dataset Loader — Load and list built-in evaluation datasets.

Datasets are JSON files stored in the ``datasets/`` subdirectory.
Each file contains a list of evaluation items with the schema::

    [{"prompt": "...", "expected": "...", "category": "...", "score_type": "..."}]
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

_DATASETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")

_REQUIRED_KEYS = {"prompt", "expected", "score_type"}


def list_datasets() -> List[Dict[str, Any]]:
    """
    List available evaluation datasets.

    Returns:
        List of dicts with ``name``, ``item_count``, and ``categories``.
    """
    result: List[Dict[str, Any]] = []
    if not os.path.isdir(_DATASETS_DIR):
        return result

    for filename in sorted(os.listdir(_DATASETS_DIR)):
        if not filename.endswith(".json"):
            continue
        name = filename[:-5]  # strip .json
        filepath = os.path.join(_DATASETS_DIR, filename)
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            categories = sorted({item.get("category", "unknown") for item in data})
            result.append({
                "name": name,
                "item_count": len(data),
                "categories": categories,
            })
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to read dataset %s: %s", filename, e)

    return result


def load_dataset(name: str) -> List[Dict[str, Any]]:
    """
    Load an evaluation dataset by name.

    Args:
        name: Dataset name (without ``.json`` extension).

    Returns:
        List of evaluation items.

    Raises:
        ValueError: If the dataset is not found or has invalid format.
    """
    # Sanitize name to prevent path traversal
    safe_name = os.path.basename(name)
    filepath = os.path.join(_DATASETS_DIR, f"{safe_name}.json")

    if not os.path.isfile(filepath):
        available = [d["name"] for d in list_datasets()]
        raise ValueError(
            f"Dataset '{name}' not found. Available: {available}"
        )

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Dataset '{name}' must be a JSON array")

    # Validate required keys on first few items
    for i, item in enumerate(data[:5]):
        missing = _REQUIRED_KEYS - set(item.keys())
        if missing:
            raise ValueError(
                f"Dataset '{name}' item {i} missing keys: {missing}"
            )

    return data
