"""
Custom exceptions for Model Server Client.

This module provides a hierarchy of exceptions for consistent error handling
across the client codebase.
"""

from __future__ import annotations

from typing import Any, Optional


class ModelServerError(Exception):
    """
    Base exception for all model server errors.
    
    Attributes:
        message: Human-readable error description
        details: Optional dict with additional error context
        status_code: Optional HTTP status code if applicable
    """
    
    def __init__(
        self,
        message: str,
        details: Optional[dict[str, Any]] = None,
        status_code: Optional[int] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.status_code = status_code
    
    def to_dict(self) -> dict[str, Any]:
        """Convert exception to a dictionary for JSON responses."""
        result: dict[str, Any] = {
            "error": self.__class__.__name__,
            "message": self.message,
        }
        if self.details:
            result["details"] = self.details
        if self.status_code:
            result["status_code"] = self.status_code
        return result


class InferenceError(ModelServerError):
    """
    Raised when inference fails.
    
    This can be due to model execution errors, invalid input, or server issues.
    """
    
    def __init__(
        self,
        message: str,
        model_name: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
        status_code: Optional[int] = None,
    ) -> None:
        super().__init__(message, details, status_code)
        self.model_name = model_name
        if model_name:
            self.details["model_name"] = model_name


class ModelNotReadyError(ModelServerError):
    """
    Raised when a model is not ready for inference.
    
    This typically means the model is loading, unloaded, or in an error state.
    """
    
    def __init__(
        self,
        model_name: str,
        message: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        msg = message or f"Model '{model_name}' is not ready for inference"
        super().__init__(msg, details, status_code=503)
        self.model_name = model_name
        self.details["model_name"] = model_name


class ServerConnectionError(ModelServerError):
    """
    Raised when connection to the inference server fails.
    
    This covers network errors, timeouts, and server unreachable conditions.
    """
    
    def __init__(
        self,
        server_url: str,
        message: Optional[str] = None,
        cause: Optional[Exception] = None,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        msg = message or f"Failed to connect to server at '{server_url}'"
        super().__init__(msg, details, status_code=503)
        self.server_url = server_url
        self.cause = cause
        self.details["server_url"] = server_url
        if cause:
            self.details["cause"] = str(cause)


class ImagePreprocessingError(ModelServerError):
    """
    Raised when image preprocessing fails.
    
    This covers format errors, invalid images, and preprocessing failures.
    """
    
    def __init__(
        self,
        message: str,
        image_source: Optional[str] = None,
        cause: Optional[Exception] = None,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(message, details, status_code=400)
        self.image_source = image_source
        self.cause = cause
        if image_source:
            self.details["image_source"] = image_source
        if cause:
            self.details["cause"] = str(cause)


class ModelMetadataError(ModelServerError):
    """
    Raised when model metadata retrieval fails.
    
    This can be due to invalid model names or server configuration issues.
    """
    
    def __init__(
        self,
        model_name: str,
        message: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        msg = message or f"Failed to retrieve metadata for model '{model_name}'"
        super().__init__(msg, details, status_code=404)
        self.model_name = model_name
        self.details["model_name"] = model_name


class ConfigurationError(ModelServerError):
    """
    Raised when there is a configuration error.
    
    This covers invalid settings, missing required configuration, etc.
    """
    
    def __init__(
        self,
        message: str,
        config_key: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(message, details, status_code=400)
        self.config_key = config_key
        if config_key:
            self.details["config_key"] = config_key


__all__ = [
    "ModelServerError",
    "InferenceError",
    "ModelNotReadyError",
    "ServerConnectionError",
    "ImagePreprocessingError",
    "ModelMetadataError",
    "ConfigurationError",
]
