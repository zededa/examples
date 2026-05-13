"""
Model Server Client Package.

A flexible client for NVIDIA Triton Inference Server and OpenVINO Model Server.
Automatically detects model input/output specifications for any image model.
Communicates via the KServe v2 gRPC protocol for low-latency binary tensor
transfer, with an optional HTTP session for Prometheus metrics.

Thread Safety:
    This client uses threading.Lock for mutable caches. Multiple threads can
    safely share a single client instance.

Quick Start:
    >>> from client import ModelServerClient
    >>>
    >>> client = ModelServerClient(grpc_url="localhost:8001")
    >>> models = client.get_available_models()
    >>> result = client.infer_image(image_bytes, models[0])

Context Manager:
    >>> with ModelServerClient() as client:
    ...     result = client.infer_image("image.jpg", "resnet50")

Modules:
    client: Main ModelServerClient facade class
    config: Constants, configuration dataclasses, API paths
    exceptions: Exception hierarchy for error handling
    preprocessing: Image preprocessing and normalization
    metadata: Model metadata retrieval and caching (gRPC)
    discovery: Server type detection and health checking (gRPC)
    inference: Inference request handling and response processing (gRPC)
    grpc_client: gRPC client factory and response conversion utilities
    http_session: HTTP session creation (metrics endpoint only)
"""

from .client import ModelServerClient
from .config import (
    APIPath,
    COMMON_CHANNEL_COUNTS,
    DEFAULT_DATA_FORMAT,
    DEFAULT_GRPC_PORT,
    DEFAULT_GRPC_PORT_OPENVINO,
    DEFAULT_GRPC_PORT_TRITON,
    DEFAULT_IMAGENET_MEAN,
    DEFAULT_IMAGENET_STD,
    DEFAULT_INFERENCE_TIMEOUT_SECONDS,
    DEFAULT_INPUT_SPEC,
    DEFAULT_METRICS_PATH,
    DEFAULT_METRICS_PORT,
    DEFAULT_OUTPUT_SPEC,
    DEFAULT_TARGET_SIZE,
    DEFAULT_TIMEOUT_SECONDS,
    InputSpec,
    MAX_RETRIES,
    OutputSpec,
    PIXEL_VALUE_MAX,
    PreprocessingConfig,
    RETRY_BACKOFF_FACTOR,
    SERVER_TYPE_OPENVINO,
    SERVER_TYPE_TRITON,
    SERVER_TYPE_UNKNOWN,
    ServerType,
)
from .discovery import HealthStatus, ModelState, ServerDiscovery, ServerInfo
from .exceptions import (
    ConfigurationError,
    ImagePreprocessingError,
    InferenceError,
    ModelMetadataError,
    ModelNotReadyError,
    ModelServerError,
    ServerConnectionError,
)
from .grpc_client import (
    create_grpc_client,
    grpc_url_from_http,
    parse_prometheus_metrics,
    get_triton_latency_metrics,
)
from .http_session import SessionManager, create_session
from .inference import ClassificationResult, InferenceRequest, InferenceResult, InferenceRunner
from .metadata import ModelMetadataManager, TensorSpec
from .preprocessing import ImagePreprocessor, PreprocessingParams
from .llm_client import (
    LLMInferenceClient,
    LLMModelInfo,
    LLMPerformanceMetrics,
    LLMServerMetrics,
    LLMServerType,
    get_llm_client,
)

__all__ = [
    # Main client
    "ModelServerClient",
    # Server types
    "ServerType",
    "SERVER_TYPE_TRITON",
    "SERVER_TYPE_OPENVINO",
    "SERVER_TYPE_UNKNOWN",
    # Specifications
    "InputSpec",
    "OutputSpec",
    "PreprocessingConfig",
    "DEFAULT_INPUT_SPEC",
    "DEFAULT_OUTPUT_SPEC",
    # Constants
    "DEFAULT_IMAGENET_MEAN",
    "DEFAULT_IMAGENET_STD",
    "DEFAULT_TARGET_SIZE",
    "DEFAULT_DATA_FORMAT",
    "DEFAULT_TIMEOUT_SECONDS",
    "DEFAULT_INFERENCE_TIMEOUT_SECONDS",
    "MAX_RETRIES",
    "RETRY_BACKOFF_FACTOR",
    "PIXEL_VALUE_MAX",
    "COMMON_CHANNEL_COUNTS",
    "APIPath",
    # gRPC
    "DEFAULT_GRPC_PORT",
    "DEFAULT_GRPC_PORT_TRITON",
    "DEFAULT_GRPC_PORT_OPENVINO",
    "DEFAULT_METRICS_PORT",
    "DEFAULT_METRICS_PATH",
    "create_grpc_client",
    "grpc_url_from_http",
    "parse_prometheus_metrics",
    "get_triton_latency_metrics",
    # Exceptions
    "ModelServerError",
    "InferenceError",
    "ModelNotReadyError",
    "ServerConnectionError",
    "ImagePreprocessingError",
    "ModelMetadataError",
    "ConfigurationError",
    # Discovery
    "ServerDiscovery",
    "ServerInfo",
    "HealthStatus",
    "ModelState",
    # Metadata
    "ModelMetadataManager",
    "TensorSpec",
    # Preprocessing
    "ImagePreprocessor",
    "PreprocessingParams",
    # Inference
    "InferenceRunner",
    "InferenceRequest",
    "InferenceResult",
    "ClassificationResult",
    # HTTP Session
    "create_session",
    "SessionManager",
    # LLM Client
    "LLMInferenceClient",
    "LLMModelInfo",
    "LLMPerformanceMetrics",
    "LLMServerMetrics",
    "LLMServerType",
    "get_llm_client",
]

__version__ = "3.0.0"
