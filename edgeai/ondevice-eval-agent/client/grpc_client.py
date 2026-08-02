"""
gRPC client wrapper for inference servers.

This module provides a thin wrapper around tritonclient.grpc for
communicating with Triton and OpenVINO Model Server via the KServe v2
gRPC protocol.  Both servers implement the same gRPC interface, so a
single client works for either backend.

Key benefits over HTTP:
    - Binary tensor transfer (no JSON serialization of large arrays)
    - Persistent HTTP/2 connections with lower per-request overhead
    - Native streaming support for future use
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, Final, List, Optional, Tuple
from urllib.parse import urlparse

import numpy as np
import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException

from .config import DEFAULT_GRPC_PORT, DEFAULT_TIMEOUT_SECONDS

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

# Triton datatype string -> numpy dtype mapping
_TRITON_TO_NUMPY: Final[Dict[str, np.dtype]] = {
    "BOOL": np.dtype("bool"),
    "UINT8": np.dtype("uint8"),
    "UINT16": np.dtype("uint16"),
    "UINT32": np.dtype("uint32"),
    "UINT64": np.dtype("uint64"),
    "INT8": np.dtype("int8"),
    "INT16": np.dtype("int16"),
    "INT32": np.dtype("int32"),
    "INT64": np.dtype("int64"),
    "FP16": np.dtype("float16"),
    "FP32": np.dtype("float32"),
    "FP64": np.dtype("float64"),
    "BYTES": np.dtype("object"),
}

# Numpy dtype -> Triton datatype string mapping
_NUMPY_TO_TRITON: Final[Dict[np.dtype, str]] = {
    v: k for k, v in _TRITON_TO_NUMPY.items()
}

# Triton metadata dtype (e.g. "FP32") -> config.pbtxt dtype (e.g. "TYPE_FP32")
_TRITON_DTYPE_TO_CONFIG: Final[Dict[str, str]] = {
    "BOOL": "TYPE_BOOL",
    "UINT8": "TYPE_UINT8",
    "UINT16": "TYPE_UINT16",
    "UINT32": "TYPE_UINT32",
    "UINT64": "TYPE_UINT64",
    "INT8": "TYPE_INT8",
    "INT16": "TYPE_INT16",
    "INT32": "TYPE_INT32",
    "INT64": "TYPE_INT64",
    "FP16": "TYPE_FP16",
    "FP32": "TYPE_FP32",
    "FP64": "TYPE_FP64",
    "BYTES": "TYPE_STRING",
    "BF16": "TYPE_BF16",
}

# Reverse: config.pbtxt dtype -> Triton metadata dtype
_CONFIG_TO_TRITON_DTYPE: Final[Dict[str, str]] = {
    v: k for k, v in _TRITON_DTYPE_TO_CONFIG.items()
}


# =============================================================================
# Factory
# =============================================================================

def create_grpc_client(
    url: str = f"localhost:{DEFAULT_GRPC_PORT}",
    *,
    verbose: bool = False,
) -> grpcclient.InferenceServerClient:
    """
    Create a gRPC inference-server client.

    Args:
        url: ``host:port`` of the gRPC endpoint (no scheme prefix).
             Defaults to ``localhost:8001``.
        verbose: Enable verbose logging in the underlying Triton client.

    Returns:
        A ready-to-use ``tritonclient.grpc.InferenceServerClient``.
    """
    # Strip scheme if the caller accidentally included one
    url = _strip_scheme(url)
    logger.info(f"Creating gRPC client for {url}")
    return grpcclient.InferenceServerClient(url=url, verbose=verbose)


# =============================================================================
# URL helpers
# =============================================================================

def _strip_scheme(url: str) -> str:
    """Remove ``http://`` or ``grpc://`` prefix, returning ``host:port``."""
    if "://" in url:
        parsed = urlparse(url)
        host = parsed.hostname or "localhost"
        port = parsed.port or DEFAULT_GRPC_PORT
        return f"{host}:{port}"
    return url


def grpc_url_from_http(http_url: str, grpc_port: int = DEFAULT_GRPC_PORT) -> str:
    """
    Derive a gRPC ``host:port`` from an HTTP base URL.

    Example:
        >>> grpc_url_from_http("http://192.168.1.10:8000")
        '192.168.1.10:8001'
    """
    parsed = urlparse(http_url)
    host = parsed.hostname or "localhost"
    return f"{host}:{grpc_port}"


# =============================================================================
# Response conversion helpers
# =============================================================================

def server_metadata_to_dict(metadata: Any) -> Dict[str, Any]:
    """
    Convert a gRPC ``ServerMetadataResponse`` to a plain dict matching
    the KServe v2 JSON schema used by the rest of the codebase.
    """
    return {
        "name": metadata.name,
        "version": metadata.version,
        "extensions": list(metadata.extensions),
    }


def model_metadata_to_dict(metadata: Any) -> Dict[str, Any]:
    """
    Convert a gRPC ``ModelMetadataResponse`` to a dict matching the
    KServe v2 REST ``/v2/models/{name}`` JSON response.
    """
    inputs: List[Dict[str, Any]] = []
    for inp in metadata.inputs:
        inputs.append({
            "name": inp.name,
            "datatype": inp.datatype,
            "shape": list(inp.shape),
        })

    outputs: List[Dict[str, Any]] = []
    for out in metadata.outputs:
        outputs.append({
            "name": out.name,
            "datatype": out.datatype,
            "shape": list(out.shape),
        })

    return {
        "name": metadata.name,
        "versions": list(metadata.versions),
        "platform": metadata.platform,
        "inputs": inputs,
        "outputs": outputs,
    }


def model_config_to_dict(config: Any) -> Dict[str, Any]:
    """
    Convert a gRPC ``ModelConfigResponse`` to a plain dict.

    The config protobuf is complex; we serialise the most commonly
    inspected fields and fall back to ``str()`` for anything exotic.
    """
    try:
        from google.protobuf.json_format import MessageToDict
        return MessageToDict(config, preserving_proto_field_name=True)
    except Exception:
        # Fallback: manually extract the top-level fields
        result: Dict[str, Any] = {"name": getattr(config, "name", "")}
        if hasattr(config, "platform"):
            result["platform"] = config.platform
        if hasattr(config, "backend"):
            result["backend"] = config.backend
        if hasattr(config, "max_batch_size"):
            result["max_batch_size"] = config.max_batch_size
        return result


def repository_index_to_list(index: Any) -> List[Dict[str, Any]]:
    """
    Convert a gRPC repository-index response to the list-of-dicts
    format returned by the REST ``POST /v2/repository/index`` endpoint.
    """
    models: List[Dict[str, Any]] = []
    for entry in index:
        models.append({
            "name": entry.name,
            "version": getattr(entry, "version", ""),
            "state": getattr(entry, "state", ""),
            "reason": getattr(entry, "reason", ""),
        })
    return models


def infer_result_to_dict(
    result: grpcclient.InferResult,
    model_name: str,
) -> Dict[str, Any]:
    """
    Convert a gRPC ``InferResult`` into the dict format matching the
    KServe v2 REST inference response used by the rest of the codebase.

    This allows downstream code (prediction processing, etc.) to remain
    unchanged.
    """
    output = result.get_output(0)
    outputs: List[Dict[str, Any]] = []

    # Iterate through all outputs
    idx = 0
    while True:
        try:
            out_meta = result.get_output(idx)
        except IndexError:
            break
        if out_meta is None:
            break

        out_name = out_meta.name if hasattr(out_meta, "name") else f"output_{idx}"
        out_data = result.as_numpy(out_name)
        outputs.append({
            "name": out_name,
            "shape": list(out_data.shape),
            "datatype": out_meta.datatype if hasattr(out_meta, "datatype") else "FP32",
            "data": out_data.flatten().tolist(),
        })
        idx += 1

    return {
        "model_name": model_name,
        "outputs": outputs,
    }


# =============================================================================
# Metrics parsing (Prometheus text format)
# =============================================================================

# Regex for Prometheus metric lines:  metric_name{labels} value
_METRIC_LINE_RE = re.compile(
    r'^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)'
    r'(?:\{(?P<labels>[^}]*)\})?\s+'
    r'(?P<value>[0-9eE.+\-]+)$'
)


def parse_prometheus_metrics(
    text: str,
    model_name: Optional[str] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Parse Prometheus text-format metrics into a nested dict.

    Args:
        text: Raw Prometheus metrics text (from ``/metrics``).
        model_name: If given, only return metrics for this model.

    Returns:
        ``{metric_name: {label_key: value, ...}, ...}``
        When *model_name* is specified the outer dict is filtered to
        metrics whose ``model`` label matches.
    """
    metrics: Dict[str, Dict[str, float]] = {}

    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        m = _METRIC_LINE_RE.match(line)
        if not m:
            continue

        name = m.group("name")
        labels_str = m.group("labels") or ""
        try:
            value = float(m.group("value"))
        except ValueError:
            continue

        # Parse labels
        labels: Dict[str, str] = {}
        if labels_str:
            for pair in labels_str.split(","):
                k, _, v = pair.partition("=")
                labels[k.strip()] = v.strip().strip('"')

        # Filter by model if requested
        if model_name and labels.get("model") != model_name:
            continue

        # Store with version suffix for uniqueness
        version = labels.get("version", "")
        key = f"{name}" if not version else f"{name}:v{version}"
        metrics[key] = {"value": value, **labels}

    return metrics


def get_triton_latency_metrics(
    metrics: Dict[str, Dict[str, float]],
) -> Dict[str, float]:
    """
    Extract Triton-specific latency counters (in microseconds) from
    parsed Prometheus metrics and convert to milliseconds.

    Returns a dict with keys like ``queue_ms``, ``compute_infer_ms``, etc.
    Missing metrics are omitted rather than defaulted.
    """
    mapping = {
        "nv_inference_request_duration_us": "request_duration_ms",
        "nv_inference_queue_duration_us": "queue_ms",
        "nv_inference_compute_input_duration_us": "compute_input_ms",
        "nv_inference_compute_infer_duration_us": "compute_infer_ms",
        "nv_inference_compute_output_duration_us": "compute_output_ms",
    }

    result: Dict[str, float] = {}
    for prom_name, friendly_name in mapping.items():
        # Try without version suffix first, then with :v1
        for key in (prom_name, f"{prom_name}:v1"):
            if key in metrics:
                result[friendly_name] = metrics[key]["value"] / 1000.0
                break

    # Also grab request count for computing per-request averages
    for key in ("nv_inference_request_success", "nv_inference_request_success:v1"):
        if key in metrics:
            result["request_count"] = metrics[key]["value"]
            break

    return result


__all__ = [
    "create_grpc_client",
    "grpc_url_from_http",
    "server_metadata_to_dict",
    "model_metadata_to_dict",
    "model_config_to_dict",
    "repository_index_to_list",
    "infer_result_to_dict",
    "parse_prometheus_metrics",
    "get_triton_latency_metrics",
    "InferenceServerException",
    "_TRITON_TO_NUMPY",
    "_NUMPY_TO_TRITON",
    "_TRITON_DTYPE_TO_CONFIG",
    "_CONFIG_TO_TRITON_DTYPE",
]
