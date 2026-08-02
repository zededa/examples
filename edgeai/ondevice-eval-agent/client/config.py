"""
Constants and configuration for Model Server Client.

This module centralizes all constants, default values, and configuration
dataclasses used across the client modules.

Organization:
    - Server Types: Enum and constants for server identification
    - Image Preprocessing: Default values for image normalization
    - Network Configuration: Timeouts and retry settings
    - API Paths: URL templates for KServe v2 and TensorFlow Serving APIs
    - Specifications: Dataclasses for input/output tensor metadata
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Final, List, Optional, Tuple


# =============================================================================
# Server Types
# =============================================================================

class ServerType(str, Enum):
    """
    Inference server types supported by the client.
    
    The client automatically detects the server type, but users can
    also explicitly specify their preference via INFERENCE_BACKEND.
    """
    TRITON = "triton"
    OPENVINO = "openvino"
    UNKNOWN = "unknown"


# Legacy constants for backward compatibility with existing code
SERVER_TYPE_TRITON: Final[str] = ServerType.TRITON.value
SERVER_TYPE_OPENVINO: Final[str] = ServerType.OPENVINO.value
SERVER_TYPE_UNKNOWN: Final[str] = ServerType.UNKNOWN.value


# =============================================================================
# Image Preprocessing Defaults
# =============================================================================

# ImageNet normalization constants (standard for pretrained vision models)
DEFAULT_IMAGENET_MEAN: Final[tuple[float, float, float]] = (0.485, 0.456, 0.406)
DEFAULT_IMAGENET_STD: Final[tuple[float, float, float]] = (0.229, 0.224, 0.225)

# Default image dimensions (standard ImageNet input size)
DEFAULT_TARGET_SIZE: Final[tuple[int, int]] = (224, 224)  # (height, width)

# Data format (batch, channels, height, width)
DEFAULT_DATA_FORMAT: Final[str] = "NCHW"

# Maximum pixel value for normalization (8-bit images)
PIXEL_VALUE_MAX: Final[float] = 255.0

# Common channel configurations for format detection
# Used to distinguish NCHW from NHWC based on dimension values
COMMON_CHANNEL_COUNTS: Final[frozenset[int]] = frozenset({1, 3, 4})


# =============================================================================
# Network Configuration
# =============================================================================

# HTTP request timeouts (in seconds)
DEFAULT_TIMEOUT_SECONDS: Final[int] = 30
DEFAULT_INFERENCE_TIMEOUT_SECONDS: Final[int] = 60

# Retry configuration
MAX_RETRIES: Final[int] = 3
RETRY_BACKOFF_FACTOR: Final[float] = 0.5

# Default gRPC ports for inference servers
DEFAULT_GRPC_PORT_TRITON: Final[int] = 8001
DEFAULT_GRPC_PORT_OPENVINO: Final[int] = 9000
DEFAULT_GRPC_PORT: Final[int] = 8001  # Default assumes Triton

# Triton metrics endpoint (Prometheus format, HTTP only)
DEFAULT_METRICS_PORT: Final[int] = 8002
DEFAULT_METRICS_PATH: Final[str] = "/metrics"


# =============================================================================
# API Path Templates
# =============================================================================

class APIPath:
    """
    API endpoint path templates for inference servers.
    
    Supports both KServe v2 API (Triton and OpenVINO) and
    TensorFlow Serving v1 API (OpenVINO fallback).
    
    Usage:
        >>> url = f"{base_url}{APIPath.V2_MODEL.format(model_name='resnet50')}"
    """
    
    # -------------------------------------------------------------------------
    # KServe v2 API paths (both Triton and OpenVINO)
    # -------------------------------------------------------------------------
    
    # Server endpoints
    V2_ROOT: Final[str] = "/v2"
    V2_HEALTH_READY: Final[str] = "/v2/health/ready"
    V2_HEALTH_LIVE: Final[str] = "/v2/health/live"
    
    # Model endpoints (requires model_name parameter)
    V2_MODEL: Final[str] = "/v2/models/{model_name}"
    V2_MODEL_READY: Final[str] = "/v2/models/{model_name}/ready"
    V2_MODEL_INFER: Final[str] = "/v2/models/{model_name}/infer"
    V2_MODEL_CONFIG: Final[str] = "/v2/models/{model_name}/config"
    
    # Repository management (Triton-specific)
    V2_REPO_INDEX: Final[str] = "/v2/repository/index"
    
    # -------------------------------------------------------------------------
    # OpenVINO v1 API paths (TensorFlow Serving format)
    # -------------------------------------------------------------------------
    
    V1_CONFIG: Final[str] = "/v1/config"
    V1_MODEL: Final[str] = "/v1/models/{model_name}"
    V1_MODEL_PREDICT: Final[str] = "/v1/models/{model_name}:predict"


# =============================================================================
# Specification Dataclasses
# =============================================================================

@dataclass(frozen=True)
class InputSpec:
    """
    Model input tensor specification.
    
    Describes the expected input format for a model, including shape,
    data type, and layout format (NCHW vs NHWC).
    
    Attributes:
        name: Input tensor name (e.g., 'images', 'input_0').
        shape: Full tensor shape including batch dimension.
        datatype: Data type string ('FP32', 'FP16', 'INT8', etc.).
        format: Layout format ('NCHW' or 'NHWC').
        channels: Number of color channels (typically 3 for RGB).
        height: Input image height in pixels.
        width: Input image width in pixels.
    """
    name: str = "images"
    shape: tuple[int, ...] = (-1, 3, 640, 640)
    datatype: str = "FP32"
    format: str = "NCHW"
    channels: int = 3
    height: int = 640
    width: int = 640
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for backward compatibility and JSON serialization."""
        return {
            "name": self.name,
            "shape": list(self.shape),
            "datatype": self.datatype,
            "format": self.format,
            "channels": self.channels,
            "height": self.height,
            "width": self.width,
        }


@dataclass(frozen=True)
class OutputSpec:
    """
    Model output tensor specification.
    
    Describes the output format of a model, including shape
    and number of classes for classification models.
    
    Attributes:
        name: Output tensor name (e.g., 'output0', 'predictions').
        shape: Full tensor shape including batch dimension.
        datatype: Data type string ('FP32', 'FP16', etc.).
        num_classes: Number of classes for classification (None for non-classification).
    """
    name: str = "output0"
    shape: tuple[int, ...] = (-1, 84, 8400)
    datatype: str = "FP32"
    num_classes: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for backward compatibility and JSON serialization."""
        return {
            "name": self.name,
            "shape": list(self.shape),
            "datatype": self.datatype,
            "num_classes": self.num_classes,
        }


@dataclass
class PreprocessingConfig:
    """
    Image preprocessing configuration.
    
    Controls how images are prepared for model inference, including
    resizing, normalization, and format conversion.
    
    Attributes:
        target_size: Target (height, width) for resizing.
        normalize: Whether to apply ImageNet normalization.
        mean: Per-channel mean values for normalization.
        std: Per-channel standard deviation values for normalization.
        format: Output format ('NCHW' or 'NHWC').
    
    Example:
        >>> config = PreprocessingConfig(target_size=(224, 224), normalize=True)
        >>> preprocessor = ImagePreprocessor(config)
    """
    target_size: tuple[int, int] = DEFAULT_TARGET_SIZE
    normalize: bool = True
    mean: List[float] = field(default_factory=lambda: list(DEFAULT_IMAGENET_MEAN))
    std: List[float] = field(default_factory=lambda: list(DEFAULT_IMAGENET_STD))
    format: str = DEFAULT_DATA_FORMAT
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for backward compatibility."""
        return {
            "target_size": self.target_size,
            "normalize": self.normalize,
            "mean": self.mean,
            "std": self.std,
            "format": self.format,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PreprocessingConfig":
        """
        Create PreprocessingConfig from dictionary.
        
        Args:
            data: Configuration dictionary with optional keys.
            
        Returns:
            New PreprocessingConfig instance.
        """
        return cls(
            target_size=tuple(data.get("target_size", DEFAULT_TARGET_SIZE)),
            normalize=data.get("normalize", True),
            mean=data.get("mean", list(DEFAULT_IMAGENET_MEAN)),
            std=data.get("std", list(DEFAULT_IMAGENET_STD)),
            format=data.get("format", DEFAULT_DATA_FORMAT),
        )


# =============================================================================
# Default Specifications
# =============================================================================

# Default specifications as dicts for backward compatibility with existing code
DEFAULT_INPUT_SPEC: Final[Dict[str, Any]] = InputSpec().to_dict()
DEFAULT_OUTPUT_SPEC: Final[Dict[str, Any]] = OutputSpec().to_dict()


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Server types
    "ServerType",
    "SERVER_TYPE_TRITON",
    "SERVER_TYPE_OPENVINO",
    "SERVER_TYPE_UNKNOWN",
    # Image preprocessing
    "DEFAULT_IMAGENET_MEAN",
    "DEFAULT_IMAGENET_STD",
    "DEFAULT_TARGET_SIZE",
    "DEFAULT_DATA_FORMAT",
    "PIXEL_VALUE_MAX",
    "COMMON_CHANNEL_COUNTS",
    # Network
    "DEFAULT_TIMEOUT_SECONDS",
    "DEFAULT_INFERENCE_TIMEOUT_SECONDS",
    "MAX_RETRIES",
    "RETRY_BACKOFF_FACTOR",
    # gRPC
    "DEFAULT_GRPC_PORT_TRITON",
    "DEFAULT_GRPC_PORT_OPENVINO",
    "DEFAULT_GRPC_PORT",
    "DEFAULT_METRICS_PORT",
    "DEFAULT_METRICS_PATH",
    # API paths
    "APIPath",
    # Specifications
    "InputSpec",
    "OutputSpec",
    "PreprocessingConfig",
    "DEFAULT_INPUT_SPEC",
    "DEFAULT_OUTPUT_SPEC",
]
