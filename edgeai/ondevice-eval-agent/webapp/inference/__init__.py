"""
Inference Package - Model Server Client

This package provides the client for interacting with inference servers:
- Triton Inference Server
- OpenVINO Model Server
- Any KServe v2 compatible server

Usage:
    from inference import ModelServerClient
    
    client = ModelServerClient()
    models = client.get_available_models()
    result = client.infer_image(image_bytes, "my_model")
"""

# Re-export from the client package for convenience
# The client package is now at /app/client/ in the container
from client import (
    ModelServerClient,
    ServerType,
    SERVER_TYPE_TRITON,
    SERVER_TYPE_OPENVINO,
    SERVER_TYPE_UNKNOWN,
    InputSpec,
    OutputSpec,
    PreprocessingConfig,
    DEFAULT_INPUT_SPEC,
    DEFAULT_OUTPUT_SPEC,
    DEFAULT_IMAGENET_MEAN,
    DEFAULT_IMAGENET_STD,
    DEFAULT_TARGET_SIZE,
    DEFAULT_DATA_FORMAT,
    ModelServerError,
    InferenceError,
    ModelNotReadyError,
    ServerConnectionError,
    ImagePreprocessingError,
    ModelMetadataError,
    ConfigurationError,
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
    # Exceptions
    "ModelServerError",
    "InferenceError",
    "ModelNotReadyError",
    "ServerConnectionError",
    "ImagePreprocessingError",
    "ModelMetadataError",
    "ConfigurationError",
]
