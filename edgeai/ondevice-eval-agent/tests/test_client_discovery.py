"""
Tests for client/discovery.py — ServerDiscovery, HealthStatus, ServerInfo.

Covers server type detection (auto and manual), health checks, model
discovery, device info, caching, and thread safety.
"""

from __future__ import annotations

import concurrent.futures
from unittest.mock import MagicMock

import pytest
from tritonclient.utils import InferenceServerException

from client.discovery import (
    HealthStatus,
    ServerDiscovery,
    ServerInfo,
)


# =============================================================================
# HealthStatus
# =============================================================================


class TestHealthStatus:
    """Tests for the HealthStatus dataclass."""

    def test_tuple_unpacking(self):
        """HealthStatus should support tuple unpacking (is_healthy, message)."""
        status = HealthStatus(is_healthy=True, message="OK")
        healthy, msg = status
        assert healthy is True
        assert msg == "OK"

    def test_tuple_unpacking_unhealthy(self):
        """Unhealthy status should unpack correctly."""
        healthy, msg = HealthStatus(is_healthy=False, message="down")
        assert healthy is False
        assert msg == "down"

    def test_attributes(self):
        """Named attributes should be accessible directly."""
        status = HealthStatus(is_healthy=True, message="Server is ready")
        assert status.is_healthy is True
        assert status.message == "Server is ready"


# =============================================================================
# ServerInfo
# =============================================================================


class TestServerInfo:
    """Tests for the ServerInfo dataclass."""

    def test_from_dict(self):
        """from_dict should populate all fields from a metadata dict."""
        data = {
            "name": "triton",
            "version": "2.40.0",
            "extensions": ["classification", "model_repository"],
        }
        info = ServerInfo.from_dict(data)
        assert info.name == "triton"
        assert info.version == "2.40.0"
        assert "classification" in info.extensions

    def test_from_dict_defaults(self):
        """from_dict with empty dict should use 'Unknown' defaults."""
        info = ServerInfo.from_dict({})
        assert info.name == "Unknown"
        assert info.version == "Unknown"
        assert info.extensions == ()

    def test_to_dict_roundtrip(self):
        """to_dict should return the original raw_data dict."""
        data = {
            "name": "openvino",
            "version": "2024.1",
            "extensions": [],
        }
        info = ServerInfo.from_dict(data)
        assert info.to_dict() == data


# =============================================================================
# ServerDiscovery — detect_server_type
# =============================================================================


class TestDetectServerType:
    """Tests for ServerDiscovery.detect_server_type()."""

    def test_triton_in_name(self, mock_grpc_client):
        """Server name containing 'triton' should be detected as triton."""
        mock_grpc_client.get_server_metadata.return_value.name = "triton"
        disc = ServerDiscovery(mock_grpc_client)
        assert disc.detect_server_type() == "triton"

    def test_openvino_in_name(self, mock_grpc_client):
        """Server name containing 'openvino' should be detected as openvino."""
        mock_grpc_client.get_server_metadata.return_value.name = "OpenVINO Model Server"
        disc = ServerDiscovery(mock_grpc_client)
        assert disc.detect_server_type() == "openvino"

    def test_cached_second_call(self, mock_grpc_client):
        """Second call should return the cached result without hitting gRPC."""
        mock_grpc_client.get_server_metadata.return_value.name = "triton"
        disc = ServerDiscovery(mock_grpc_client)

        first = disc.detect_server_type()
        call_count_after_first = mock_grpc_client.get_server_metadata.call_count

        second = disc.detect_server_type()
        call_count_after_second = mock_grpc_client.get_server_metadata.call_count

        assert first == second == "triton"
        assert call_count_after_second == call_count_after_first

    def test_from_inference_backend_param(self, mock_grpc_client):
        """Explicit inference_backend should override auto-detection."""
        mock_grpc_client.get_server_metadata.return_value.name = "triton"
        disc = ServerDiscovery(mock_grpc_client, inference_backend="openvino")
        assert disc.detect_server_type() == "openvino"
        # gRPC should NOT be called when backend is explicitly set
        mock_grpc_client.get_server_metadata.assert_not_called()

    def test_grpc_failure_returns_unknown(self, mock_grpc_client):
        """If gRPC metadata call raises, server type should be 'unknown'."""
        mock_grpc_client.get_server_metadata.side_effect = InferenceServerException(
            "connection refused"
        )
        disc = ServerDiscovery(mock_grpc_client)
        assert disc.detect_server_type() == "unknown"


# =============================================================================
# ServerDiscovery — check_server_health
# =============================================================================


class TestCheckServerHealth:
    """Tests for ServerDiscovery.check_server_health()."""

    def test_server_ready(self, mock_grpc_client):
        """Ready server should return HealthStatus(True, ...)."""
        mock_grpc_client.is_server_ready.return_value = True
        disc = ServerDiscovery(mock_grpc_client)
        status = disc.check_server_health()
        assert status.is_healthy is True
        assert "ready" in status.message.lower()

    def test_server_not_ready(self, mock_grpc_client):
        """Not-ready server should return HealthStatus(False, ...)."""
        mock_grpc_client.is_server_ready.return_value = False
        disc = ServerDiscovery(mock_grpc_client)
        status = disc.check_server_health()
        assert status.is_healthy is False
        assert "not ready" in status.message.lower()

    def test_server_health_exception(self, mock_grpc_client):
        """gRPC exception during health check returns unhealthy status."""
        mock_grpc_client.is_server_ready.side_effect = InferenceServerException(
            "timeout"
        )
        disc = ServerDiscovery(mock_grpc_client)
        status = disc.check_server_health()
        assert status.is_healthy is False
        assert "failed" in status.message.lower()


# =============================================================================
# ServerDiscovery — get_available_models
# =============================================================================


class TestGetAvailableModels:
    """Tests for ServerDiscovery.get_available_models()."""

    def test_from_repository_index(self, mock_grpc_client):
        """Models should be discovered from the gRPC repository index."""
        disc = ServerDiscovery(mock_grpc_client)
        models = disc.get_available_models()
        assert "test_model" in models

    def test_known_models_fallback(self, mock_grpc_client):
        """When repository index fails, known_models should be checked."""
        mock_grpc_client.get_model_repository_index.side_effect = (
            InferenceServerException("not supported")
        )
        mock_grpc_client.is_model_ready.return_value = True
        disc = ServerDiscovery(mock_grpc_client)
        models = disc.get_available_models(known_models=["fallback_model"])
        assert "fallback_model" in models

    def test_no_models_returns_empty(self, mock_grpc_client):
        """When no discovery source yields results, return empty list."""
        mock_grpc_client.get_model_repository_index.side_effect = (
            InferenceServerException("not supported")
        )
        disc = ServerDiscovery(mock_grpc_client)
        models = disc.get_available_models()
        assert models == []


# =============================================================================
# ServerDiscovery — check_model_ready
# =============================================================================


class TestCheckModelReady:
    """Tests for ServerDiscovery.check_model_ready()."""

    def test_delegates_to_grpc(self, mock_grpc_client):
        """check_model_ready should delegate to the gRPC client."""
        mock_grpc_client.is_model_ready.return_value = True
        disc = ServerDiscovery(mock_grpc_client)
        assert disc.check_model_ready("test_model") is True
        mock_grpc_client.is_model_ready.assert_called_with("test_model")

    def test_not_ready(self, mock_grpc_client):
        """Model not ready should return False."""
        mock_grpc_client.is_model_ready.return_value = False
        disc = ServerDiscovery(mock_grpc_client)
        assert disc.check_model_ready("missing_model") is False


# =============================================================================
# ServerDiscovery — get_server_device_info
# =============================================================================


class TestGetServerDeviceInfo:
    """Tests for ServerDiscovery.get_server_device_info()."""

    def test_cuda_in_extensions_returns_gpu(self, mock_grpc_client):
        """Extensions containing 'cuda' should yield 'GPU'."""
        mock_grpc_client.get_server_metadata.return_value.name = "triton"
        mock_grpc_client.get_server_metadata.return_value.extensions = [
            "classification",
            "cuda_shared_memory",
        ]
        disc = ServerDiscovery(mock_grpc_client)
        assert disc.get_server_device_info() == "GPU"

    def test_no_gpu_indicators_returns_cpu(self, mock_grpc_client):
        """No GPU indicators should yield 'CPU'."""
        mock_grpc_client.get_server_metadata.return_value.name = "triton"
        mock_grpc_client.get_server_metadata.return_value.extensions = [
            "classification",
            "model_repository",
        ]
        disc = ServerDiscovery(mock_grpc_client)
        assert disc.get_server_device_info() == "CPU"


# =============================================================================
# ServerDiscovery — clear_cache
# =============================================================================


class TestClearCache:
    """Tests for ServerDiscovery.clear_cache()."""

    def test_clear_cache_re_detects(self, mock_grpc_client):
        """After clear_cache(), detect_server_type should query gRPC again."""
        mock_grpc_client.get_server_metadata.return_value.name = "triton"
        disc = ServerDiscovery(mock_grpc_client)

        disc.detect_server_type()
        calls_before = mock_grpc_client.get_server_metadata.call_count

        disc.clear_cache()
        disc.detect_server_type()
        calls_after = mock_grpc_client.get_server_metadata.call_count

        assert calls_after > calls_before


# =============================================================================
# Thread Safety
# =============================================================================


class TestThreadSafety:
    """Verify concurrent access does not corrupt state."""

    def test_concurrent_detect_server_type(self, mock_grpc_client):
        """Multiple threads calling detect_server_type must all get the same result."""
        mock_grpc_client.get_server_metadata.return_value.name = "triton"
        disc = ServerDiscovery(mock_grpc_client)

        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(disc.detect_server_type) for _ in range(20)]
            results = [f.result() for f in futures]

        assert all(r == "triton" for r in results)
