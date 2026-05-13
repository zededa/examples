"""
Model Server Client - Main Facade.

This module provides the main ModelServerClient class that combines all
components into a cohesive, easy-to-use interface for inference operations.

The client communicates with NVIDIA Triton Inference Server and OpenVINO
Model Server via the KServe v2 gRPC protocol for low-latency binary
tensor transfer.  An optional HTTP session is maintained for fetching
Prometheus metrics from the Triton metrics endpoint.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, BinaryIO, Dict, Final, List, Literal, Optional, Tuple, Union

import numpy as np
import requests
from numpy.typing import NDArray

from .config import (
    DEFAULT_GRPC_PORT,
    DEFAULT_INFERENCE_TIMEOUT_SECONDS,
    DEFAULT_METRICS_PATH,
    DEFAULT_METRICS_PORT,
    DEFAULT_TIMEOUT_SECONDS,
    MAX_RETRIES,
    PreprocessingConfig,
    ServerType,
)
from .discovery import ServerDiscovery, HealthStatus
from .exceptions import (
    ImagePreprocessingError,
    InferenceError,
)
from .grpc_client import (
    create_grpc_client,
    grpc_url_from_http,
    parse_prometheus_metrics,
    get_triton_latency_metrics,
    repository_index_to_list,
    _TRITON_TO_NUMPY,
    InferenceServerException,
)
import tritonclient.grpc as grpcclient
from .http_session import create_session
from .inference import InferenceRunner
from .metadata import ModelMetadataManager
from .preprocessing import ImagePreprocessor

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

# Default server URLs
_DEFAULT_SERVER_URL: Final[str] = "http://localhost:8000"
_DEFAULT_GRPC_URL: Final[str] = f"localhost:{DEFAULT_GRPC_PORT}"

# Environment variable names
_ENV_MODEL_SERVER_URL: Final[str] = "MODEL_SERVER_URL"
_ENV_GRPC_URL: Final[str] = "MODEL_SERVER_GRPC_URL"
_ENV_METRICS_URL: Final[str] = "MODEL_SERVER_METRICS_URL"
_ENV_INFERENCE_BACKEND: Final[str] = "INFERENCE_BACKEND"
_ENV_KNOWN_MODELS: Final[str] = "KNOWN_MODELS"
_ENV_MODEL_NAME: Final[str] = "MODEL_NAME"

# Class names file
_CLASS_NAMES_FILENAME: Final[str] = "class_names.json"


# =============================================================================
# Model Server Client
# =============================================================================

class ModelServerClient:
    """
    Client for communicating with NVIDIA Triton or OpenVINO Model Server.

    Uses gRPC (KServe v2 protocol) for all inference, metadata, and
    health operations.  An HTTP session is kept solely for fetching
    Prometheus metrics from the Triton metrics endpoint (port 8002).

    Features:
        - gRPC binary tensor transfer (no JSON serialization overhead)
        - Automatic server type detection (Triton vs OpenVINO)
        - Auto-detection of model input/output specifications
        - Image preprocessing with configurable normalization
        - Thread-safe caching of metadata
        - Prometheus metrics integration for accurate server-side latency
        - Context manager support for resource cleanup

    Thread Safety:
        All mutable caches are protected by locks. Multiple threads can
        safely share a single client instance.

    Example:
        >>> client = ModelServerClient(grpc_url="localhost:8001")
        >>> models = client.get_available_models()
        >>> result = client.infer_image(image_bytes, models[0])

        >>> with ModelServerClient() as client:
        ...     result = client.infer_image("image.jpg", "resnet50")
    """

    __slots__ = (
        "server_url",
        "grpc_url",
        "metrics_url",
        "timeout",
        "inference_timeout",
        "inference_backend",
        "_known_models",
        "_grpc_client",
        "_http_session",
        "_preprocessor",
        "_metadata_manager",
        "_discovery",
        "_inference_runner",
    )

    def __init__(
        self,
        server_url: Optional[str] = None,
        *,
        grpc_url: Optional[str] = None,
        metrics_url: Optional[str] = None,
        timeout: int = DEFAULT_TIMEOUT_SECONDS,
        inference_timeout: int = DEFAULT_INFERENCE_TIMEOUT_SECONDS,
        max_retries: int = MAX_RETRIES,
        test_connectivity: bool = True,
    ) -> None:
        """
        Initialize the client.

        Args:
            server_url: HTTP base URL (used to derive gRPC/metrics URLs
                        when not given explicitly).  Falls back to
                        ``MODEL_SERVER_URL`` env var or ``http://localhost:8000``.
            grpc_url: ``host:port`` for gRPC.  Falls back to
                      ``MODEL_SERVER_GRPC_URL`` env var or derived from
                      *server_url* (same host, port 8001).
            metrics_url: Full URL for Triton metrics endpoint.  Falls
                         back to ``MODEL_SERVER_METRICS_URL`` env var or
                         derived from *server_url* (same host, port 8002).
            timeout: Default timeout for API requests in seconds.
            inference_timeout: Timeout for inference requests.
            max_retries: Maximum retry attempts for HTTP requests.
            test_connectivity: Whether to test server connectivity on init.
        """
        # Resolve URLs
        self.server_url = self._resolve_server_url(server_url)
        self.grpc_url = self._resolve_grpc_url(grpc_url, self.server_url)
        self.metrics_url = self._resolve_metrics_url(metrics_url, self.server_url)
        self.timeout = timeout
        self.inference_timeout = inference_timeout

        # Load configuration from environment
        self.inference_backend = os.environ.get(_ENV_INFERENCE_BACKEND, "").lower()
        self._known_models = self._parse_known_models()

        # Create gRPC client (primary communication channel)
        self._grpc_client = create_grpc_client(self.grpc_url)

        # Create HTTP session (only for metrics endpoint)
        self._http_session = create_session(max_retries)

        # Initialize components with gRPC client
        self._preprocessor = ImagePreprocessor()
        self._metadata_manager = ModelMetadataManager(
            self._grpc_client, timeout
        )
        self._discovery = ServerDiscovery(
            self._grpc_client, timeout, self.inference_backend
        )
        self._inference_runner = InferenceRunner(
            self._grpc_client, inference_timeout
        )

        # Load class names if available
        self._load_class_names()

        # Log initialization
        logger.info(f"Model server client initialized (gRPC: {self.grpc_url})")
        if self.inference_backend:
            logger.info(f"Inference backend preference: {self.inference_backend}")

        # Test connectivity if requested
        if test_connectivity:
            self._discovery.test_connectivity()

    # =========================================================================
    # Context Manager Protocol
    # =========================================================================

    def __enter__(self) -> "ModelServerClient":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit - close resources."""
        self.close()

    def close(self) -> None:
        """Close the HTTP session and release resources."""
        if hasattr(self, "_http_session") and self._http_session:
            self._http_session.close()
            logger.debug("HTTP session closed")
        if hasattr(self, "_grpc_client") and self._grpc_client:
            try:
                self._grpc_client.close()
            except Exception:
                pass
            logger.debug("gRPC client closed")

    # =========================================================================
    # Cache Management
    # =========================================================================

    def clear_cache(self) -> None:
        """Clear all cached metadata. Thread-safe."""
        self._metadata_manager.clear_cache()
        self._discovery.clear_cache()
        logger.info("All caches cleared")

    # =========================================================================
    # Configuration Properties
    # =========================================================================

    @property
    def preprocessing_config(self) -> Dict[str, Any]:
        """Get preprocessing config as dict (backward compatibility)."""
        return self._preprocessor.config.to_dict()

    @preprocessing_config.setter
    def preprocessing_config(self, value: Dict[str, Any]) -> None:
        """Set preprocessing config from dict (backward compatibility)."""
        self._preprocessor.config = PreprocessingConfig.from_dict(value)

    def set_preprocessing_config(self, config: Dict[str, Any]) -> None:
        """Update preprocessing configuration."""
        self._preprocessor.update_config(config)

    @property
    def class_names(self) -> Optional[List[str]]:
        """Get class names for labeling predictions."""
        return self._inference_runner.class_names

    @class_names.setter
    def class_names(self, value: Optional[List[str]]) -> None:
        """Set class names for labeling predictions."""
        self._inference_runner.class_names = value

    # =========================================================================
    # Server Discovery
    # =========================================================================

    def detect_server_type(self) -> str:
        """Detect the type of inference server (Triton or OpenVINO)."""
        return self._discovery.detect_server_type()

    def get_server_info(self) -> Optional[Dict[str, Any]]:
        """Get server information. Thread-safe."""
        return self._discovery.get_server_info()

    def check_server_health(self) -> Tuple[bool, str]:
        """Check if the inference server is healthy and ready."""
        health = self._discovery.check_server_health()
        return (health.is_healthy, health.message)

    def get_server_device_info(self) -> Literal["CPU", "GPU"]:
        """Detect compute device (CPU/GPU) from the inference server."""
        return self._discovery.get_server_device_info()

    def check_model_ready(self, model_name: str) -> bool:
        """Check if a specific model is ready for inference."""
        return self._discovery.check_model_ready(model_name)

    def get_available_models(self) -> List[str]:
        """Get list of available models from the inference server."""
        return self._discovery.get_available_models(self._known_models)

    # =========================================================================
    # Model Repository Management
    # =========================================================================

    def get_repository_index(self) -> List[Dict[str, Any]]:
        """
        Get the full model repository index (all states).

        Unlike ``get_available_models()`` which only returns READY models,
        this returns every entry including UNAVAILABLE or LOADING models
        with their ``state`` and ``reason`` fields.

        Returns:
            List of dicts with keys: ``name``, ``version``, ``state``, ``reason``.

        Raises:
            InferenceServerException: If the repository index is not supported
                (e.g. OpenVINO Model Server without repository index).
        """
        index = self._grpc_client.get_model_repository_index()
        return repository_index_to_list(index)

    def load_model(
        self,
        model_name: str,
        config: Optional[str] = None,
        files: Optional[Dict[str, bytes]] = None,
    ) -> None:
        """
        Load or reload a model on the inference server.

        Requires Triton to be started with ``--model-control-mode=explicit``
        or ``--model-control-mode=poll``.

        Args:
            model_name: Name of the model to load.
            config: Optional JSON string of a model config override.
                    When provided, this config is used instead of config.pbtxt
                    on disk.
            files: Optional dict mapping file paths to bytes content.
                   Requires *config* to also be provided.

        Raises:
            InferenceServerException: If loading fails or model control
                mode does not allow it.
        """
        try:
            self._grpc_client.load_model(
                model_name, config=config, files=files,
            )
            # Clear stale metadata for the loaded model
            self._metadata_manager.clear_cache()
            logger.info(f"Model '{model_name}' load request sent")
        except InferenceServerException as e:
            err_msg = str(e).lower()
            if "model control" in err_msg or "not allowed" in err_msg:
                raise InferenceServerException(
                    f"Cannot load model: Triton model control mode does not "
                    f"allow API-driven load. Start Triton with "
                    f"--model-control-mode=explicit or poll. "
                    f"Original error: {e}"
                ) from e
            raise

    def unload_model(self, model_name: str) -> None:
        """
        Unload a model from the inference server.

        Args:
            model_name: Name of the model to unload.

        Raises:
            InferenceServerException: If unloading fails.
        """
        self._grpc_client.unload_model(model_name)
        self._metadata_manager.clear_cache()
        logger.info(f"Model '{model_name}' unload request sent")

    def send_raw_inference(
        self,
        model_name: str,
        inputs: List[Tuple[str, "NDArray", str]],
    ) -> Dict[str, Any]:
        """
        Send a raw multi-input inference request via gRPC.

        Unlike ``send_inference_request()`` which is single-input and
        image-oriented, this accepts arbitrary inputs for probing
        unknown models.

        Args:
            inputs: List of ``(name, numpy_array, triton_dtype_string)``
                    tuples.  Example: ``[("input", data, "FP32")]``.

        Returns:
            Dict with ``model_name`` and ``outputs`` list, each output
            having ``name``, ``shape``, ``datatype``, and ``data`` keys.

        Raises:
            InferenceServerException: On gRPC errors.
        """
        grpc_inputs: List[grpcclient.InferInput] = []
        for name, data, dtype in inputs:
            inp = grpcclient.InferInput(name, list(data.shape), dtype)
            # Map Triton dtype to numpy dtype for correct casting
            np_dtype = _TRITON_TO_NUMPY.get(dtype, np.dtype("float32"))
            inp.set_data_from_numpy(data.astype(np_dtype))
            grpc_inputs.append(inp)

        result = self._grpc_client.infer(
            model_name=model_name,
            inputs=grpc_inputs,
            client_timeout=self.inference_timeout,
        )

        # Convert result to dict by enumerating the response outputs
        outputs: List[Dict[str, Any]] = []
        response = result.get_response()
        if hasattr(response, "outputs"):
            for idx, out_meta in enumerate(response.outputs):
                out_name = out_meta.name if hasattr(out_meta, "name") else f"output_{idx}"
                out_data = result.as_numpy(out_name)
                outputs.append({
                    "name": out_name,
                    "shape": list(out_data.shape),
                    "datatype": out_meta.datatype if hasattr(out_meta, "datatype") else "FP32",
                    "data": out_data,
                })

        return {"model_name": model_name, "outputs": outputs}

    # =========================================================================
    # Model Metadata
    # =========================================================================

    def get_model_metadata(
        self,
        model_name: str,
        use_cache: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """Get detailed model metadata from inference server."""
        return self._metadata_manager.get_metadata(model_name, use_cache)

    def get_model_config(
        self,
        model_name: str,
        use_cache: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """Get model configuration (config.pbtxt equivalent) from the server."""
        return self._metadata_manager.get_model_config(model_name, use_cache)

    def get_model_input_spec(self, model_name: str) -> Dict[str, Any]:
        """Auto-detect model input specifications from server metadata."""
        return self._metadata_manager.get_input_spec(model_name)

    def get_model_output_spec(self, model_name: str) -> Dict[str, Any]:
        """Auto-detect model output specifications."""
        return self._metadata_manager.get_output_spec(model_name)

    def get_all_output_specs(self, model_name: str) -> List[Dict[str, Any]]:
        """Get specifications for ALL model outputs."""
        return self._metadata_manager.get_all_output_specs(model_name)

    def get_model_input_shape(self, model_name: str) -> Tuple[int, int]:
        """Get the input shape (height, width) for a specific model."""
        return self._metadata_manager.get_input_shape(model_name)

    # =========================================================================
    # Image Preprocessing
    # =========================================================================

    def preprocess_image_bytes(
        self,
        image_bytes: Union[bytes, BinaryIO],
        model_name: Optional[str] = None,
        target_size: Optional[Tuple[int, int]] = None,
    ) -> Optional[NDArray[np.floating[Any]]]:
        """Preprocess image from bytes for model inference."""
        try:
            input_spec = self.get_model_input_spec(model_name) if model_name else None
            return self._preprocessor.preprocess_bytes(image_bytes, input_spec, target_size)
        except ImagePreprocessingError as e:
            logger.error(str(e))
            return None

    def preprocess_image(
        self,
        image_path: str,
        model_name: Optional[str] = None,
        target_size: Optional[Tuple[int, int]] = None,
    ) -> Optional[NDArray[np.floating[Any]]]:
        """Preprocess image from file path for model inference."""
        try:
            input_spec = self.get_model_input_spec(model_name) if model_name else None
            return self._preprocessor.preprocess_file(image_path, input_spec, target_size)
        except ImagePreprocessingError as e:
            logger.error(str(e))
            return None

    # =========================================================================
    # Inference
    # =========================================================================

    def send_inference_request(
        self,
        image_array: NDArray[np.floating[Any]],
        model_name: str,
        measure_latency: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """Send inference request to inference server via gRPC."""
        try:
            input_spec = self.get_model_input_spec(model_name)
            server_type = self.detect_server_type()
            return self._inference_runner.send_inference_request(
                image_array, model_name, input_spec, server_type, measure_latency
            )
        except InferenceError as e:
            logger.error(str(e))
            return None

    def process_prediction(
        self,
        response: Optional[Dict[str, Any]],
        model_name: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Process the prediction response from inference server."""
        if response is None:
            return None
        try:
            return self._inference_runner.process_prediction(response, model_name)
        except InferenceError as e:
            logger.error(str(e))
            return None

    def infer_image(
        self,
        image_data: Union[bytes, BinaryIO, str],
        model_name: str,
        *,
        measure_latency: bool = False,
        process_result: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """
        High-level convenience method: preprocess image and run inference.

        This is the recommended API for most use cases.
        """
        # Step 1: Preprocess image
        image_array = self._preprocess_image_data(image_data, model_name)
        if image_array is None:
            return None

        # Step 2: Run inference
        response = self.send_inference_request(
            image_array, model_name, measure_latency=measure_latency
        )
        if response is None:
            return None

        # Step 3: Process results
        if process_result:
            result = self.process_prediction(response, model_name)
            if result and measure_latency and "latency" in response:
                result["latency"] = response["latency"]
            return result

        return response

    # =========================================================================
    # Metrics (Prometheus / HTTP)
    # =========================================================================

    def get_metrics_raw(self) -> Optional[str]:
        """
        Fetch raw Prometheus metrics text from the Triton metrics endpoint.

        Returns:
            Raw metrics text, or None if unavailable.
        """
        try:
            response = self._http_session.get(
                self.metrics_url, timeout=self.timeout
            )
            if response.status_code == 200:
                return response.text
        except requests.RequestException as e:
            logger.debug(f"Metrics endpoint unavailable: {e}")
        return None

    def get_model_metrics(self, model_name: str) -> Optional[Dict[str, float]]:
        """
        Fetch Triton server-side latency metrics for a specific model.

        Returns a dict with keys like ``queue_ms``, ``compute_infer_ms``,
        ``compute_input_ms``, ``compute_output_ms``, ``request_duration_ms``,
        and ``request_count``.  All durations are in milliseconds.

        Returns:
            Metrics dict, or None if the endpoint is unavailable.
        """
        raw = self.get_metrics_raw()
        if raw is None:
            return None
        parsed = parse_prometheus_metrics(raw, model_name=model_name)
        return get_triton_latency_metrics(parsed)

    # =========================================================================
    # API Information
    # =========================================================================

    def get_api_endpoints_info(self, model_name: str) -> Dict[str, Any]:
        """Get API endpoint information for developers."""
        input_spec = self.get_model_input_spec(model_name)
        output_spec = self.get_model_output_spec(model_name)
        server_type = self.detect_server_type()

        endpoints: Dict[str, Any] = {
            "server_type": server_type,
            "protocol": "gRPC (KServe v2)",
            "grpc_url": self.grpc_url,
            "metrics_url": self.metrics_url,
            "detected_input_spec": input_spec,
            "detected_output_spec": output_spec,
        }

        if server_type == ServerType.TRITON.value:
            endpoints.update(self._build_triton_endpoints(model_name))
        else:
            endpoints.update(self._build_openvino_endpoints(model_name))

        return endpoints

    def get_full_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get comprehensive model information."""
        return {
            "model_name": model_name,
            "server_type": self.detect_server_type(),
            "server_info": self.get_server_info(),
            "ready": self.check_model_ready(model_name),
            "input_spec": self.get_model_input_spec(model_name),
            "output_spec": self.get_model_output_spec(model_name),
            "metadata": self.get_model_metadata(model_name),
        }

    # =========================================================================
    # Private - Initialization Helpers
    # =========================================================================

    @staticmethod
    def _resolve_server_url(server_url: Optional[str]) -> str:
        """Resolve HTTP server URL from parameter or environment."""
        url = server_url or os.environ.get(_ENV_MODEL_SERVER_URL, _DEFAULT_SERVER_URL)
        return url.rstrip("/")

    @staticmethod
    def _resolve_grpc_url(grpc_url: Optional[str], server_url: str) -> str:
        """Resolve gRPC URL from parameter, env var, or derived from HTTP URL."""
        if grpc_url:
            # Strip scheme if present
            if "://" in grpc_url:
                from urllib.parse import urlparse
                parsed = urlparse(grpc_url)
                return f"{parsed.hostname or 'localhost'}:{parsed.port or DEFAULT_GRPC_PORT}"
            return grpc_url

        env_grpc = os.environ.get(_ENV_GRPC_URL, "")
        if env_grpc:
            return env_grpc

        # Derive from HTTP server_url: same host, gRPC port
        return grpc_url_from_http(server_url, DEFAULT_GRPC_PORT)

    @staticmethod
    def _resolve_metrics_url(metrics_url: Optional[str], server_url: str) -> str:
        """Resolve Triton metrics URL."""
        if metrics_url:
            return metrics_url

        env_metrics = os.environ.get(_ENV_METRICS_URL, "")
        if env_metrics:
            return env_metrics

        # Derive from HTTP server_url: same host, metrics port
        from urllib.parse import urlparse
        parsed = urlparse(server_url)
        host = parsed.hostname or "localhost"
        return f"http://{host}:{DEFAULT_METRICS_PORT}{DEFAULT_METRICS_PATH}"

    @staticmethod
    def _parse_known_models() -> List[str]:
        """Parse known model names from environment variables."""
        models: List[str] = []

        models_str = os.environ.get(_ENV_KNOWN_MODELS, "")
        if models_str:
            for model in models_str.split(","):
                model = model.strip()
                if model and model not in models:
                    models.append(model)

        model_name = os.environ.get(_ENV_MODEL_NAME, "").strip()
        if model_name and model_name not in models:
            models.append(model_name)

        if models:
            logger.info(f"Known models from environment: {models}")

        return models

    def _load_class_names(self) -> None:
        """Load class names from class_names.json if available."""
        try:
            class_names_path = Path(__file__).parent.parent / _CLASS_NAMES_FILENAME
            if class_names_path.exists():
                with open(class_names_path, encoding="utf-8") as f:
                    class_names = json.load(f)
                self._inference_runner.class_names = class_names
                logger.info(f"Loaded {len(class_names)} class names from file")
        except (OSError, json.JSONDecodeError) as e:
            logger.debug(f"Could not load class names: {e}")

    def _preprocess_image_data(
        self,
        image_data: Union[bytes, BinaryIO, str],
        model_name: str,
    ) -> Optional[NDArray[np.floating[Any]]]:
        """Preprocess image data from any supported format."""
        if isinstance(image_data, str):
            return self.preprocess_image(image_data, model_name)
        return self.preprocess_image_bytes(image_data, model_name)

    # =========================================================================
    # Private - Endpoint Documentation
    # =========================================================================

    def _build_triton_endpoints(self, model_name: str) -> Dict[str, Any]:
        """Build Triton-specific endpoint documentation."""
        return {
            "grpc_inference": {
                "endpoint": f"{self.grpc_url}",
                "protocol": "gRPC",
                "description": "Send inference via gRPC (binary tensor transfer)",
                "python_example": (
                    f"import tritonclient.grpc as grpcclient\n"
                    f"client = grpcclient.InferenceServerClient(url='{self.grpc_url}')\n"
                    f"inputs = [grpcclient.InferInput('input', shape, 'FP32')]\n"
                    f"inputs[0].set_data_from_numpy(np_array)\n"
                    f"result = client.infer('{model_name}', inputs)"
                ),
            },
            "metrics": {
                "endpoint": self.metrics_url,
                "method": "GET",
                "description": "Prometheus metrics (latency, throughput, etc.)",
                "curl_command": f"curl {self.metrics_url}",
            },
            "rest_inference": {
                "endpoint": f"{self.server_url}/v2/models/{model_name}/infer",
                "method": "POST",
                "description": "REST inference (fallback, higher latency than gRPC)",
                "curl_command": (
                    f'curl -X POST {self.server_url}/v2/models/{model_name}/infer '
                    f'-H "Content-Type: application/json" -d \'{{"inputs": [...]}}\''
                ),
            },
        }

    def _build_openvino_endpoints(self, model_name: str) -> Dict[str, Any]:
        """Build OpenVINO-specific endpoint documentation."""
        return {
            "grpc_inference": {
                "endpoint": f"{self.grpc_url}",
                "protocol": "gRPC",
                "description": "Send inference via gRPC (KServe v2 protocol)",
                "python_example": (
                    f"import tritonclient.grpc as grpcclient\n"
                    f"client = grpcclient.InferenceServerClient(url='{self.grpc_url}')\n"
                    f"inputs = [grpcclient.InferInput('input', shape, 'FP32')]\n"
                    f"inputs[0].set_data_from_numpy(np_array)\n"
                    f"result = client.infer('{model_name}', inputs)"
                ),
            },
            "rest_inference": {
                "endpoint": f"{self.server_url}/v2/models/{model_name}/infer",
                "method": "POST",
                "description": "REST inference (KServe v2, higher latency than gRPC)",
                "curl_command": (
                    f'curl -X POST {self.server_url}/v2/models/{model_name}/infer '
                    f'-H "Content-Type: application/json" -d \'{{"inputs": [...]}}\''
                ),
            },
        }


__all__ = [
    "ModelServerClient",
]
