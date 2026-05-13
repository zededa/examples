"""
Model metadata retrieval and management via gRPC.

This module handles model metadata operations including input/output
specification detection and thread-safe caching, using the KServe v2
gRPC protocol.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, Final, List, Optional

import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException

from .config import (
    COMMON_CHANNEL_COUNTS,
    DEFAULT_INPUT_SPEC,
    DEFAULT_OUTPUT_SPEC,
    DEFAULT_TARGET_SIZE,
    DEFAULT_TIMEOUT_SECONDS,
)
from .grpc_client import model_metadata_to_dict, model_config_to_dict

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

# Minimum shape length for valid tensor specifications
_MIN_SHAPE_LENGTH: Final[int] = 4

# Index positions for NCHW format
_NCHW_CHANNEL_IDX: Final[int] = 1
_NCHW_HEIGHT_IDX: Final[int] = 2
_NCHW_WIDTH_IDX: Final[int] = 3


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class TensorSpec:
    """
    Tensor specification for model inputs/outputs.

    Provides structured access to tensor metadata from KServe v2 API.
    """
    name: str
    shape: List[int]
    datatype: str

    # Derived properties (computed from shape)
    format: str = "NCHW"
    channels: int = 3
    height: int = 224
    width: int = 224
    num_classes: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for backward compatibility."""
        return {
            "name": self.name,
            "shape": self.shape,
            "datatype": self.datatype,
            "format": self.format,
            "channels": self.channels,
            "height": self.height,
            "width": self.width,
            "num_classes": self.num_classes,
        }

    @classmethod
    def from_input_info(
        cls,
        input_info: Dict[str, Any],
        default_size: int = DEFAULT_TARGET_SIZE[0],
    ) -> "TensorSpec":
        """
        Create TensorSpec from KServe v2 input metadata.

        Automatically detects NCHW vs NHWC format based on shape.
        """
        name = input_info.get("name", "input")
        shape = input_info.get("shape", [-1, 3, *DEFAULT_TARGET_SIZE])
        datatype = input_info.get("datatype", "FP32")

        format_str, channels, height, width = _parse_input_shape(shape, default_size)

        logger.debug(f"Detected input spec: {format_str} {height}x{width}x{channels}")

        return cls(
            name=name,
            shape=shape,
            datatype=datatype,
            format=format_str,
            channels=channels,
            height=height,
            width=width,
        )

    @classmethod
    def from_output_info(cls, output_info: Dict[str, Any]) -> "TensorSpec":
        """Create TensorSpec from KServe v2 output metadata."""
        name = output_info.get("name", "output")
        shape = output_info.get("shape", [-1, 1000])
        datatype = output_info.get("datatype", "FP32")

        num_classes = shape[-1] if len(shape) >= 2 and shape[-1] > 0 else None

        logger.debug(f"Detected output spec: {name}, shape={shape}, classes={num_classes}")

        return cls(
            name=name,
            shape=shape,
            datatype=datatype,
            num_classes=num_classes,
        )


# =============================================================================
# Shape Parsing Utilities
# =============================================================================

def _parse_input_shape(
    shape: List[int],
    default_size: int,
) -> tuple[str, int, int, int]:
    """
    Parse input shape to extract format and dimensions.

    Handles both NCHW and NHWC formats by detecting channel position.
    """
    if len(shape) < _MIN_SHAPE_LENGTH:
        return "NCHW", 3, default_size, default_size

    if shape[_NCHW_CHANNEL_IDX] in COMMON_CHANNEL_COUNTS:
        return (
            "NCHW",
            _resolve_dim(shape[1], 3),
            _resolve_dim(shape[2], default_size),
            _resolve_dim(shape[3], default_size),
        )

    if shape[-1] in COMMON_CHANNEL_COUNTS:
        return (
            "NHWC",
            _resolve_dim(shape[-1], 3),
            _resolve_dim(shape[1], default_size),
            _resolve_dim(shape[2], default_size),
        )

    return (
        "NCHW",
        _resolve_dim(shape[1], 3),
        _resolve_dim(shape[2], default_size),
        _resolve_dim(shape[3], default_size),
    )


def _resolve_dim(value: int, default: int) -> int:
    """Resolve dimension value, using default for dynamic (-1) dimensions."""
    return value if value > 0 else default


# =============================================================================
# Model Metadata Manager
# =============================================================================

class ModelMetadataManager:
    """
    Manages model metadata retrieval and caching via gRPC.

    Provides thread-safe access to model metadata from inference servers
    with automatic caching to reduce redundant gRPC calls.

    Thread Safety:
        All cache operations are protected by a lock.
    """

    __slots__ = ("_grpc_client", "_timeout", "_cache_lock", "_metadata_cache", "_config_cache")

    def __init__(
        self,
        grpc_client: grpcclient.InferenceServerClient,
        timeout: int = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        """
        Initialize the metadata manager.

        Args:
            grpc_client: gRPC inference-server client instance.
            timeout: Request timeout in seconds.
        """
        self._grpc_client = grpc_client
        self._timeout = timeout

        # Thread-safe cache
        self._cache_lock = threading.Lock()
        self._metadata_cache: Dict[str, Dict[str, Any]] = {}
        self._config_cache: Dict[str, Dict[str, Any]] = {}

    # =========================================================================
    # Public API - Cache Management
    # =========================================================================

    def clear_cache(self) -> None:
        """Clear all cached metadata. Thread-safe."""
        with self._cache_lock:
            self._metadata_cache.clear()
            self._config_cache.clear()
        logger.info("Model metadata cache cleared")

    # =========================================================================
    # Public API - Metadata Retrieval
    # =========================================================================

    def get_metadata(
        self,
        model_name: str,
        use_cache: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """
        Get detailed model metadata from inference server via gRPC.

        Args:
            model_name: Name of the model.
            use_cache: Whether to use cached metadata.

        Returns:
            Model metadata with input/output specifications, or None on error.
        """
        if use_cache:
            with self._cache_lock:
                if model_name in self._metadata_cache:
                    return self._metadata_cache[model_name]

        try:
            logger.debug(f"Getting model metadata via gRPC for: {model_name}")
            grpc_metadata = self._grpc_client.get_model_metadata(model_name)
            metadata = model_metadata_to_dict(grpc_metadata)

            with self._cache_lock:
                self._metadata_cache[model_name] = metadata

            logger.info(f"Model metadata retrieved and cached for {model_name} (gRPC)")
            return metadata

        except InferenceServerException as e:
            logger.error(f"gRPC error getting metadata for {model_name}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error getting model metadata for {model_name}: {e}")
            return None

    def get_model_config(self, model_name: str, use_cache: bool = True) -> Optional[Dict[str, Any]]:
        """Fetch model configuration from the server via gRPC."""
        if use_cache:
            with self._cache_lock:
                if model_name in self._config_cache:
                    return self._config_cache[model_name]

        try:
            logger.debug(f"Getting model config via gRPC for: {model_name}")
            grpc_config = self._grpc_client.get_model_config(model_name)
            config = model_config_to_dict(grpc_config)

            with self._cache_lock:
                self._config_cache[model_name] = config

            logger.info(f"Model config retrieved and cached for {model_name} (gRPC)")
            return config

        except InferenceServerException as e:
            logger.error(f"gRPC error getting model config for {model_name}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error getting model config for {model_name}: {e}")
            return None

    # =========================================================================
    # Public API - Input/Output Specifications
    # =========================================================================

    def get_input_spec(self, model_name: str) -> Dict[str, Any]:
        """Auto-detect model input specifications from server metadata."""
        try:
            metadata = self.get_metadata(model_name)

            if not metadata:
                logger.warning(f"No metadata for {model_name}, using defaults")
                return self._get_default_input_spec()

            inputs = metadata.get("inputs", [])
            if inputs:
                return TensorSpec.from_input_info(inputs[0]).to_dict()

            return self._get_default_input_spec()

        except (KeyError, IndexError, TypeError) as e:
            logger.error(f"Error getting input spec for {model_name}: {e}")
            return self._get_default_input_spec()

    def get_output_spec(self, model_name: str) -> Dict[str, Any]:
        """Auto-detect model output specifications from server metadata."""
        try:
            metadata = self.get_metadata(model_name)

            if not metadata:
                return self._get_default_output_spec()

            outputs = metadata.get("outputs", [])
            if outputs:
                return TensorSpec.from_output_info(outputs[0]).to_dict()

            return self._get_default_output_spec()

        except (KeyError, IndexError, TypeError) as e:
            logger.error(f"Error getting output spec for {model_name}: {e}")
            return self._get_default_output_spec()

    def get_all_output_specs(self, model_name: str) -> List[Dict[str, Any]]:
        """Get specifications for ALL model outputs (for multi-output models)."""
        try:
            metadata = self.get_metadata(model_name)

            if not metadata:
                return [self._get_default_output_spec()]

            outputs = metadata.get("outputs", [])
            if not outputs:
                return [self._get_default_output_spec()]

            return [TensorSpec.from_output_info(output).to_dict() for output in outputs]

        except (KeyError, TypeError) as e:
            logger.error(f"Error getting all output specs for {model_name}: {e}")
            return [self._get_default_output_spec()]

    def get_input_shape(self, model_name: str) -> tuple[int, int]:
        """Get the input shape (height, width) for a specific model."""
        input_spec = self.get_input_spec(model_name)
        return (input_spec["height"], input_spec["width"])

    # =========================================================================
    # Private - Defaults
    # =========================================================================

    @staticmethod
    def _get_default_input_spec() -> Dict[str, Any]:
        """Return default input specification."""
        return DEFAULT_INPUT_SPEC.copy()

    @staticmethod
    def _get_default_output_spec() -> Dict[str, Any]:
        """Return default output specification."""
        return DEFAULT_OUTPUT_SPEC.copy()


__all__ = [
    "ModelMetadataManager",
    "TensorSpec",
]
