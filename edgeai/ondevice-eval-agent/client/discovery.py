"""
Server discovery and health checking via gRPC.

This module handles inference server detection, health checking,
and model discovery operations for both Triton and OpenVINO servers
using the KServe v2 gRPC protocol.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Final, List, Literal, Optional

import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException

from .config import (
    DEFAULT_TIMEOUT_SECONDS,
    SERVER_TYPE_OPENVINO,
    SERVER_TYPE_TRITON,
    SERVER_TYPE_UNKNOWN,
)
from .grpc_client import (
    server_metadata_to_dict,
    repository_index_to_list,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

class ModelState(str, Enum):
    """Model readiness states from inference servers."""
    READY = "READY"
    AVAILABLE = "AVAILABLE"
    LOADING = "LOADING"
    UNLOADING = "UNLOADING"


# Server name patterns for auto-detection
_OPENVINO_PATTERNS: Final[frozenset[str]] = frozenset({"openvino"})
_TRITON_PATTERNS: Final[frozenset[str]] = frozenset({"triton"})

# GPU indicators in server extensions
_GPU_INDICATORS: Final[frozenset[str]] = frozenset({"cuda", "gpu", "tensorrt"})


# =============================================================================
# Data Classes
# =============================================================================

@dataclass(frozen=True)
class ServerInfo:
    """Immutable server information container."""
    name: str
    version: str
    extensions: tuple[str, ...]
    raw_data: Dict[str, Any]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ServerInfo":
        """Create ServerInfo from a dict (e.g. converted gRPC metadata)."""
        return cls(
            name=data.get("name", "Unknown"),
            version=data.get("version", "Unknown"),
            extensions=tuple(data.get("extensions", [])),
            raw_data=data,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return self.raw_data


@dataclass(frozen=True)
class HealthStatus:
    """Server health check result."""
    is_healthy: bool
    message: str

    def __iter__(self):
        """Allow unpacking as tuple for backward compatibility."""
        return iter((self.is_healthy, self.message))


# =============================================================================
# Server Discovery
# =============================================================================

class ServerDiscovery:
    """
    Handles inference server discovery and health checking via gRPC.

    Supports both NVIDIA Triton Inference Server and OpenVINO Model Server.

    Thread Safety:
        All mutable state is protected by locks. Safe for concurrent access
        from multiple threads.
    """

    __slots__ = (
        "_grpc_client",
        "_timeout",
        "_inference_backend",
        "_lock",
        "_server_type",
        "_server_info",
    )

    def __init__(
        self,
        grpc_client: grpcclient.InferenceServerClient,
        timeout: int = DEFAULT_TIMEOUT_SECONDS,
        inference_backend: str = "",
    ) -> None:
        """
        Initialize server discovery.

        Args:
            grpc_client: gRPC inference-server client instance.
            timeout: Request timeout in seconds.
            inference_backend: Preferred backend ('triton', 'openvino', or '' for auto).
        """
        self._grpc_client = grpc_client
        self._timeout = timeout
        self._inference_backend = inference_backend.lower().strip()

        # Thread-safe state
        self._lock = threading.Lock()
        self._server_type: Optional[str] = None
        self._server_info: Optional[ServerInfo] = None

    # =========================================================================
    # Public API - Cache Management
    # =========================================================================

    def clear_cache(self) -> None:
        """Clear cached server information. Thread-safe."""
        with self._lock:
            self._server_type = None
            self._server_info = None
        logger.info("Server discovery cache cleared")

    # =========================================================================
    # Public API - Connectivity & Health
    # =========================================================================

    def test_connectivity(self) -> bool:
        """
        Test basic connectivity to the model server via gRPC.

        Returns:
            True if server is reachable and live.
        """
        try:
            if self._grpc_client.is_server_live():
                metadata = self._grpc_client.get_server_metadata()
                info = server_metadata_to_dict(metadata)
                logger.info(
                    f"Connected to {info.get('name', 'Unknown')} "
                    f"v{info.get('version', 'Unknown')} (gRPC)"
                )
                return True
        except InferenceServerException as e:
            logger.warning(f"gRPC connectivity test failed: {e}")
        except Exception as e:
            logger.warning(f"Could not connect to model server via gRPC: {e}")
        return False

    def check_server_health(self) -> HealthStatus:
        """
        Check if the inference server is healthy and ready.

        Returns:
            HealthStatus with is_healthy flag and message.
        """
        try:
            if self._grpc_client.is_server_ready():
                return HealthStatus(True, "Server is ready")
            return HealthStatus(False, "Server not ready")
        except InferenceServerException as e:
            return HealthStatus(False, f"Health check failed: {e}")
        except Exception as e:
            return HealthStatus(False, f"Health check failed: {e}")

    # =========================================================================
    # Public API - Server Type Detection
    # =========================================================================

    def detect_server_type(self) -> str:
        """
        Detect the type of inference server (Triton or OpenVINO).

        Detection strategy:
            1. Return cached result if available.
            2. Use INFERENCE_BACKEND preference if explicitly set.
            3. Auto-detect from server metadata via gRPC.
            4. Probe Triton-specific repository index as fallback.

        Returns:
            Server type: 'triton', 'openvino', or 'unknown'.
        """
        with self._lock:
            if self._server_type is not None:
                return self._server_type

            if self._inference_backend in ("triton", "openvino"):
                self._server_type = self._inference_backend
                logger.info(f"Using server type from preference: {self._server_type}")
                return self._server_type

        detected = self._auto_detect_server_type()

        with self._lock:
            self._server_type = detected

        return detected

    def get_server_info(self) -> Optional[Dict[str, Any]]:
        """Get server information via gRPC."""
        with self._lock:
            if self._server_info is not None:
                return self._server_info.to_dict()

        try:
            metadata = self._grpc_client.get_server_metadata()
            info_dict = server_metadata_to_dict(metadata)
            info = ServerInfo.from_dict(info_dict)
            with self._lock:
                self._server_info = info
            return info.to_dict()
        except InferenceServerException as e:
            logger.error(f"Failed to get server info via gRPC: {e}")
        except Exception as e:
            logger.error(f"Failed to get server info: {e}")
        return None

    def get_server_device_info(self) -> Literal["CPU", "GPU"]:
        """
        Detect compute device (CPU/GPU) from the inference server.

        Returns:
            'GPU' if CUDA/TensorRT detected, otherwise 'CPU'.
        """
        try:
            server_type = self.detect_server_type()

            if server_type == SERVER_TYPE_TRITON:
                metadata = self._grpc_client.get_server_metadata()
                extensions = list(metadata.extensions)
                extensions_str = " ".join(ext.lower() for ext in extensions)

                if any(indicator in extensions_str for indicator in _GPU_INDICATORS):
                    logger.debug("Triton server using GPU (detected from extensions)")
                    return "GPU"

            logger.debug(f"{server_type} server using CPU")
            return "CPU"

        except Exception as e:
            logger.debug(f"Error detecting server device: {e}")
            return "CPU"

    # =========================================================================
    # Public API - Model Discovery
    # =========================================================================

    def check_model_ready(self, model_name: str) -> bool:
        """
        Check if a specific model is ready for inference.

        Args:
            model_name: Name of the model to check.

        Returns:
            True if model is ready, False otherwise.
        """
        try:
            ready = self._grpc_client.is_model_ready(model_name)
            if ready:
                logger.debug(f"Model {model_name} is ready (gRPC)")
                return True
        except InferenceServerException:
            pass
        except Exception:
            pass

        logger.debug(f"Model {model_name} not ready")
        return False

    def get_available_models(
        self,
        known_models: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Get list of available models from the inference server.

        Discovery strategy:
            1. Try gRPC repository index (Triton & compatible OVMS).
            2. Fall back to checking known models individually.

        Args:
            known_models: Optional list of model names to check as fallback.

        Returns:
            List of model names that are ready for inference.
        """
        models = self._discover_via_repository_index()
        if models:
            return models

        if known_models:
            return self._discover_via_known_models(known_models)

        return []

    # =========================================================================
    # Private - Server Type Detection
    # =========================================================================

    def _auto_detect_server_type(self) -> str:
        """Auto-detect server type from gRPC server metadata."""
        try:
            metadata = self._grpc_client.get_server_metadata()
            info_dict = server_metadata_to_dict(metadata)
            info = ServerInfo.from_dict(info_dict)

            with self._lock:
                self._server_info = info

            server_name_lower = info.name.lower()

            if any(pattern in server_name_lower for pattern in _OPENVINO_PATTERNS):
                logger.info(
                    f"Detected OpenVINO Model Server: "
                    f"{info.name} v{info.version}"
                )
                return SERVER_TYPE_OPENVINO

            if any(pattern in server_name_lower for pattern in _TRITON_PATTERNS):
                logger.info(
                    f"Detected Triton Inference Server: "
                    f"{info.name} v{info.version}"
                )
                return SERVER_TYPE_TRITON

            # Probe Triton-specific endpoint
            return self._detect_by_repository_index()

        except InferenceServerException as e:
            logger.warning(f"Failed to detect server type via gRPC: {e}")
            return SERVER_TYPE_UNKNOWN
        except Exception as e:
            logger.warning(f"Failed to detect server type: {e}")
            return SERVER_TYPE_UNKNOWN

    def _detect_by_repository_index(self) -> str:
        """Detect server type by probing Triton-specific repository index."""
        try:
            self._grpc_client.get_model_repository_index()
            logger.info("Detected Triton via repository index (gRPC)")
            return SERVER_TYPE_TRITON
        except InferenceServerException:
            pass
        except Exception:
            pass

        logger.info("Assuming OpenVINO (no repository index via gRPC)")
        return SERVER_TYPE_OPENVINO

    # =========================================================================
    # Private - Model Discovery
    # =========================================================================

    def _discover_via_repository_index(self) -> List[str]:
        """Discover models via gRPC repository index."""
        try:
            index = self._grpc_client.get_model_repository_index()
            index_list = repository_index_to_list(index)

            models: List[str] = []
            for entry in index_list:
                name = entry.get("name")
                if not name:
                    continue
                state = entry.get("state", "").upper()
                if state == "" or state == "READY":
                    models.append(name)
                    logger.debug(f"Found model: {name} (state: {state or 'not specified'})")

            if models:
                logger.info(f"Discovered {len(models)} models via gRPC repository index: {models}")
            return models

        except InferenceServerException as e:
            logger.debug(f"Repository index not available via gRPC: {e}")
        except Exception as e:
            logger.warning(f"Repository index failed: {e}")
        return []

    def _discover_via_known_models(self, known_models: List[str]) -> List[str]:
        """Check known models and return those that are ready."""
        logger.info("Trying known models discovery")
        available: List[str] = []

        for model_name in known_models:
            if self.check_model_ready(model_name):
                available.append(model_name)
                logger.info(f"Found ready model (known): {model_name}")

        if available:
            logger.info(f"Discovered {len(available)} models via known models")

        return available


__all__ = [
    "ServerDiscovery",
    "ServerInfo",
    "HealthStatus",
    "ModelState",
]
