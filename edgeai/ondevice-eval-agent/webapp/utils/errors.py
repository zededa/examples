"""
Error handling utilities for consistent API responses.

This module provides standardized error handling patterns for the web application,
ensuring consistent error response formats across all endpoints.

Features:
    - Exception hierarchy for HTTP status codes
    - Decorator for automatic exception handling
    - Response helpers for success/error responses
    - Request validation utilities
"""

from __future__ import annotations

import functools
import logging
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Final, List, Optional, Tuple, TypeVar

from flask import jsonify

logger = logging.getLogger(__name__)


# =============================================================================
# Type Definitions
# =============================================================================

# Type variable for generic function decorator
F = TypeVar("F", bound=Callable[..., Any])

# Response tuple type: (response_dict, status_code)
ResponseTuple = Tuple[Dict[str, Any], int]


# =============================================================================
# HTTP Status Codes
# =============================================================================

class HTTPStatus:
    """HTTP status code constants for common responses."""
    OK: Final[int] = 200
    CREATED: Final[int] = 201
    BAD_REQUEST: Final[int] = 400
    UNAUTHORIZED: Final[int] = 401
    FORBIDDEN: Final[int] = 403
    NOT_FOUND: Final[int] = 404
    CONFLICT: Final[int] = 409
    GONE: Final[int] = 410
    UNPROCESSABLE_ENTITY: Final[int] = 422
    INTERNAL_SERVER_ERROR: Final[int] = 500
    SERVICE_UNAVAILABLE: Final[int] = 503


# =============================================================================
# Exception Hierarchy
# =============================================================================

@dataclass
class APIError(Exception):
    """
    Base exception for API errors.
    
    Provides consistent error response format with status code.
    All API-related exceptions should inherit from this class.
    
    Attributes:
        message: Human-readable error description.
        status_code: HTTP status code for the response.
        error_code: Machine-readable error code for clients.
        details: Additional context for debugging.
    
    Example:
        >>> raise APIError("Resource not found", status_code=404)
        >>> raise BadRequestError("Invalid input", details={"field": "email"})
    """
    message: str
    status_code: int = HTTPStatus.INTERNAL_SERVER_ERROR
    error_code: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """Initialize exception with message."""
        super().__init__(self.message)
        if self.error_code is None:
            self.error_code = self.__class__.__name__
    
    def to_response(self) -> ResponseTuple:
        """
        Convert to Flask JSON response tuple.
        
        Returns:
            Tuple of (response_dict, status_code) for Flask jsonify.
        """
        response: Dict[str, Any] = {
            "success": False,
            "error": self.message,
            "error_code": self.error_code,
        }
        if self.details:
            response["details"] = self.details
        return response, self.status_code
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return self.to_response()[0]


class BadRequestError(APIError):
    """
    Request validation error (HTTP 400).
    
    Use when request data is malformed, missing required fields,
    or fails validation rules.
    """
    
    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            message=message,
            status_code=HTTPStatus.BAD_REQUEST,
            details=details or {},
        )


class NotFoundError(APIError):
    """
    Resource not found error (HTTP 404).
    
    Use when the requested resource does not exist.
    """
    
    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            message=message,
            status_code=HTTPStatus.NOT_FOUND,
            details=details or {},
        )


class ServiceUnavailableError(APIError):
    """
    Service unavailable error (HTTP 503).
    
    Use when a required service (e.g., inference server) is not available.
    """
    
    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            message=message,
            status_code=HTTPStatus.SERVICE_UNAVAILABLE,
            details=details or {},
        )


class InternalServerError(APIError):
    """
    Internal server error (HTTP 500).
    
    Use for unexpected server-side errors that aren't user-actionable.
    """
    
    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            message=message,
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            details=details or {},
        )


class ConflictError(APIError):
    """
    Conflict error (HTTP 409).
    
    Use when the request conflicts with current state of the server.
    """
    
    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            message=message,
            status_code=HTTPStatus.CONFLICT,
            details=details or {},
        )


class UnauthorizedError(APIError):
    """
    Unauthorized error (HTTP 401).
    
    Use when authentication is required but not provided or invalid.
    """
    
    def __init__(
        self,
        message: str = "Authentication required",
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            message=message,
            status_code=HTTPStatus.UNAUTHORIZED,
            details=details or {},
        )


# =============================================================================
# Response Helpers
# =============================================================================

def create_error_response(
    message: str,
    status_code: int = HTTPStatus.INTERNAL_SERVER_ERROR,
    error_code: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
) -> ResponseTuple:
    """
    Create a standardized error response.
    
    Use this function when you need to create an error response without
    raising an exception.
    
    Args:
        message: Human-readable error message.
        status_code: HTTP status code.
        error_code: Machine-readable error code.
        details: Additional error context.
        
    Returns:
        Tuple of (response_dict, status_code) for Flask.
    
    Example:
        >>> return jsonify(*create_error_response("Invalid input", 400))
    """
    response: Dict[str, Any] = {
        "success": False,
        "error": message,
    }
    if error_code:
        response["error_code"] = error_code
    if details:
        response["details"] = details
    return response, status_code


def create_success_response(
    data: Dict[str, Any],
    message: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Create a standardized success response.
    
    Wraps response data with success=True for consistent API format.
    
    Args:
        data: Response payload dictionary.
        message: Optional success message.
        
    Returns:
        Response dictionary with success=True and data merged.
    
    Example:
        >>> return jsonify(create_success_response({"user": user_data}))
        >>> return jsonify(create_success_response({"count": 5}, "Items retrieved"))
    """
    response = {"success": True, **data}
    if message:
        response["message"] = message
    return response


# =============================================================================
# Exception Handling Decorator
# =============================================================================

def handle_exceptions(
    default_error: str = "Internal server error",
    log_traceback: bool = True,
) -> Callable[[F], F]:
    """
    Decorator for consistent exception handling in route functions.
    
    Catches exceptions and returns standardized error responses. Supports
    the APIError hierarchy for typed exceptions, plus generic handling
    for unexpected errors.
    
    Args:
        default_error: Default error message prefix for unhandled exceptions.
        log_traceback: Whether to log full traceback for errors.
        
    Returns:
        Decorated function with error handling.
    
    Example:
        >>> @app.route("/api/resource")
        >>> @handle_exceptions("Failed to process resource")
        >>> def get_resource():
        ...     # ... code that might raise exceptions
        ...     return jsonify({"data": result})
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            
            except APIError as e:
                # Handle our custom API errors
                logger.warning(f"API error in {func.__name__}: {e.message}")
                response, status_code = e.to_response()
                return jsonify(response), status_code
            
            except ImportError as e:
                # Handle missing dependencies gracefully
                logger.warning(f"Import error in {func.__name__}: {e}")
                return jsonify({
                    "success": False,
                    "error": "Required module not available",
                    "error_code": "ImportError",
                }), HTTPStatus.INTERNAL_SERVER_ERROR
            
            except Exception as e:
                # Handle unexpected errors - log full details but don't
                # expose internal error messages to clients
                if log_traceback:
                    logger.error(f"Error in {func.__name__}: {e}")
                    logger.error(traceback.format_exc())
                else:
                    logger.error(f"Error in {func.__name__}: {e}")

                return jsonify({
                    "success": False,
                    "error": default_error,
                    "error_code": "INTERNAL_ERROR",
                }), HTTPStatus.INTERNAL_SERVER_ERROR
        
        return wrapper  # type: ignore[return-value]
    
    return decorator


# =============================================================================
# Request Validation
# =============================================================================

def validate_request_json(
    required_fields: Optional[List[str]] = None,
    request_data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Validate JSON request data.
    
    Checks that the request has a JSON body and contains all required fields.
    
    Args:
        required_fields: List of required field names.
        request_data: Request JSON data (or None to get from current request).
        
    Returns:
        Validated request data dictionary.
        
    Raises:
        BadRequestError: If validation fails.
    
    Example:
        >>> data = validate_request_json(["name", "email"])
        >>> user_name = data["name"]
    """
    from flask import request
    
    if request_data is None:
        request_data = request.get_json()
    
    if not request_data:
        raise BadRequestError("Missing request body")
    
    if required_fields:
        missing = [f for f in required_fields if f not in request_data]
        if missing:
            raise BadRequestError(
                f"Missing required fields: {missing}",
                details={"missing_fields": missing},
            )
    
    return request_data


def validate_query_params(
    required_params: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Validate query string parameters.
    
    Args:
        required_params: List of required parameter names.
        
    Returns:
        Dictionary of query parameters.
        
    Raises:
        BadRequestError: If required parameters are missing.
    """
    from flask import request
    
    params = dict(request.args)
    
    if required_params:
        missing = [p for p in required_params if p not in params]
        if missing:
            raise BadRequestError(
                f"Missing required query parameters: {missing}",
                details={"missing_params": missing},
            )
    
    return params


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # HTTP status codes
    "HTTPStatus",
    # Exception hierarchy
    "APIError",
    "BadRequestError",
    "NotFoundError",
    "ServiceUnavailableError",
    "InternalServerError",
    "ConflictError",
    "UnauthorizedError",
    # Response helpers
    "create_error_response",
    "create_success_response",
    # Decorators
    "handle_exceptions",
    # Validators
    "validate_request_json",
    "validate_query_params",
]
