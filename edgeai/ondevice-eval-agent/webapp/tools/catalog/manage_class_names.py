"""
Manage Class Names Tool

View or set the class label mappings used when interpreting
classification and detection results.
"""

import logging
from typing import Any, Dict, List, Optional

from tools.base import ok, error_response, get_client
from tools.registry import register_tool

logger = logging.getLogger(__name__)


def manage_class_names(
    action: str = "get",
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    View or update the class label mappings used for predictions.

    Actions:
    - ``"get"``: Return the current class names (if any).
    - ``"set"``: Replace the class names with the provided list.
    - ``"clear"``: Remove all custom class names.

    Args:
        action: One of ``"get"``, ``"set"``, or ``"clear"``.
        class_names: List of class label strings (required when action is ``"set"``).

    Returns:
        Dict with the current class names and count.
    """
    try:
        client = get_client()

        if action == "set":
            if not class_names or not isinstance(class_names, list):
                return error_response(
                    ValueError("class_names must be a non-empty list of strings when action='set'"),
                    operation="manage_class_names",
                )
            client.class_names = class_names
            logger.info(f"Class names updated: {len(class_names)} labels")
            return ok(
                action="set",
                class_names=class_names,
                count=len(class_names),
                message=f"Class names updated with {len(class_names)} labels.",
            )

        if action == "clear":
            client.class_names = None
            logger.info("Class names cleared")
            return ok(
                action="clear",
                class_names=None,
                count=0,
                message="Class names cleared. Predictions will use numeric indices.",
            )

        # Default: "get"
        current = client.class_names
        return ok(
            action="get",
            class_names=current,
            count=len(current) if current else 0,
            message=(
                f"Currently {len(current)} class names configured."
                if current
                else "No custom class names set. Predictions use numeric indices."
            ),
        )

    except Exception as e:
        logger.error(f"Error managing class names: {e}")
        return error_response(e, operation="manage_class_names", action=action)


# Register the tool
register_tool(
    name="manage_class_names",
    func=manage_class_names,
    description=(
        "View, set, or clear the class label mappings used for classification "
        "and detection results. Use 'get' to see current labels, 'set' to provide "
        "custom labels (e.g., for a custom-trained model), or 'clear' to revert "
        "to numeric indices."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["get", "set", "clear"],
                "default": "get",
                "description": "Action to perform: 'get' (view), 'set' (update), or 'clear' (remove)",
            },
            "class_names": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of class label strings (required when action is 'set')",
            },
        },
        "required": [],
    },
)
