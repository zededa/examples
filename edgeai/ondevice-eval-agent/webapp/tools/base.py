"""
Base utilities for MCP tools.

Provides common classes and functions used across all tools:
- ToolResult: Standardized result container
- error_response/ok: Response builders
- get_client: ModelServerClient singleton
"""

import os
import sys
import logging
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, List, Any, Optional

# Add parent directories to path for imports
_current_dir = os.path.dirname(os.path.abspath(__file__))
_webapp_dir = os.path.dirname(_current_dir)
_business_logic_dir = os.path.dirname(_webapp_dir)
if _business_logic_dir not in sys.path:
    sys.path.insert(0, _business_logic_dir)

from client import ModelServerClient

logger = logging.getLogger(__name__)


@dataclass
class ToolResult:
    """
    Standardized result container for all tool functions.
    
    Provides consistent structure for agent-side processing, retries,
    and user-facing explanations.
    """
    success: bool
    payload: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, omitting None values."""
        result = {
            "success": self.success,
            **self.payload
        }
        if self.warnings:
            result["warnings"] = self.warnings
        if self.context:
            result["context"] = self.context
        if self.error:
            result["error"] = self.error
        return result


def error_response(error: Exception, **context) -> Dict[str, Any]:
    """
    Create a standardized error response.
    
    Args:
        error: The exception that occurred
        **context: Additional context fields (model_name, operation, etc.)
        
    Returns:
        Consistently structured error dictionary
    """
    return ToolResult(
        success=False,
        error=str(error),
        context=context
    ).to_dict()


def ok(warnings: Optional[List[str]] = None, **payload) -> Dict[str, Any]:
    """
    Create a standardized success response.
    
    Args:
        warnings: Optional list of warning messages
        **payload: Response data fields
        
    Returns:
        Consistently structured success dictionary
    """
    return ToolResult(
        success=True,
        payload=payload,
        warnings=warnings or []
    ).to_dict()


@lru_cache(maxsize=1)
def get_client() -> ModelServerClient:
    """
    Get or create the shared ModelServerClient instance.
    
    Uses lru_cache for singleton pattern. Thread-safe in CPython due to GIL,
    but not suitable for async event loops or multiprocess worker pools where
    each worker needs its own client or connection pooling.
    
    Note: For multi-process deployments (gunicorn workers, etc.), each process
    will have its own client instance, which is typically the desired behavior.
    """
    return ModelServerClient()
