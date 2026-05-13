"""
Tests for client/client.py — ModelServerClient facade.

Covers URL resolution, environment variable parsing, context manager
protocol, delegation to sub-components, and property round-trips.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock

import pytest

from client.client import ModelServerClient


# =============================================================================
# Static helpers — _resolve_server_url
# =============================================================================


class TestResolveServerUrl:
    """Tests for ModelServerClient._resolve_server_url."""

    def test_from_param(self):
        """Explicit parameter should be used as-is (after trailing slash strip)."""
        url = ModelServerClient._resolve_server_url("http://myhost:8000")
        assert url == "http://myhost:8000"

    def test_from_env(self, monkeypatch):
        """MODEL_SERVER_URL env var should be used when no param is given."""
        monkeypatch.setenv("MODEL_SERVER_URL", "http://envhost:9000")
        url = ModelServerClient._resolve_server_url(None)
        assert url == "http://envhost:9000"

    def test_default(self, clean_env):
        """With no param and no env var, default localhost:8000 is used."""
        url = ModelServerClient._resolve_server_url(None)
        assert url == "http://localhost:8000"

    def test_strips_trailing_slash(self):
        """Trailing slash should be removed."""
        url = ModelServerClient._resolve_server_url("http://host:8000/")
        assert not url.endswith("/")


# =============================================================================
# Static helpers — _resolve_grpc_url
# =============================================================================


class TestResolveGrpcUrl:
    """Tests for ModelServerClient._resolve_grpc_url."""

    def test_from_param(self, clean_env):
        """Explicit grpc_url parameter should be returned directly."""
        url = ModelServerClient._resolve_grpc_url("myhost:8001", "http://ignored:8000")
        assert url == "myhost:8001"

    def test_strips_scheme(self, clean_env):
        """A grpc_url with http:// scheme should have it stripped."""
        url = ModelServerClient._resolve_grpc_url(
            "http://myhost:8001", "http://ignored:8000"
        )
        assert url == "myhost:8001"

    def test_derived_from_http_url(self, clean_env):
        """When no grpc_url is given, derive host from server_url with port 8001."""
        url = ModelServerClient._resolve_grpc_url(None, "http://modelhost:8000")
        assert "modelhost" in url
        assert "8001" in url

    def test_from_env(self, monkeypatch):
        """MODEL_SERVER_GRPC_URL env var should be used when no param is given."""
        monkeypatch.setenv("MODEL_SERVER_GRPC_URL", "envhost:9001")
        url = ModelServerClient._resolve_grpc_url(None, "http://localhost:8000")
        assert url == "envhost:9001"


# =============================================================================
# Static helpers — _resolve_metrics_url
# =============================================================================


class TestResolveMetricsUrl:
    """Tests for ModelServerClient._resolve_metrics_url."""

    def test_derived_from_server_url(self, clean_env):
        """Metrics URL should be derived from server_url host with port 8002."""
        url = ModelServerClient._resolve_metrics_url(None, "http://myhost:8000")
        assert "myhost" in url
        assert "8002" in url
        assert "/metrics" in url

    def test_from_param(self, clean_env):
        """Explicit metrics_url parameter should be used directly."""
        url = ModelServerClient._resolve_metrics_url(
            "http://custom:9002/metrics", "http://ignored:8000"
        )
        assert url == "http://custom:9002/metrics"

    def test_from_env(self, monkeypatch):
        """MODEL_SERVER_METRICS_URL env var should be used when no param."""
        monkeypatch.setenv("MODEL_SERVER_METRICS_URL", "http://envhost:9002/metrics")
        url = ModelServerClient._resolve_metrics_url(None, "http://localhost:8000")
        assert url == "http://envhost:9002/metrics"


# =============================================================================
# Static helpers — _parse_known_models
# =============================================================================


class TestParseKnownModels:
    """Tests for ModelServerClient._parse_known_models."""

    def test_from_env_var(self, monkeypatch):
        """KNOWN_MODELS env var should be split on commas."""
        monkeypatch.setenv("KNOWN_MODELS", "resnet50,mobilenet_v2")
        monkeypatch.delenv("MODEL_NAME", raising=False)
        models = ModelServerClient._parse_known_models()
        assert "resnet50" in models
        assert "mobilenet_v2" in models

    def test_deduplicates(self, monkeypatch):
        """Duplicate model names should be removed."""
        monkeypatch.setenv("KNOWN_MODELS", "resnet50,resnet50,mobilenet")
        monkeypatch.delenv("MODEL_NAME", raising=False)
        models = ModelServerClient._parse_known_models()
        assert models.count("resnet50") == 1

    def test_empty_when_unset(self, clean_env):
        """Returns empty list when no env vars are set."""
        models = ModelServerClient._parse_known_models()
        assert models == []

    def test_model_name_env_appended(self, monkeypatch):
        """MODEL_NAME env var should be appended if not already present."""
        monkeypatch.setenv("KNOWN_MODELS", "resnet50")
        monkeypatch.setenv("MODEL_NAME", "efficientnet")
        models = ModelServerClient._parse_known_models()
        assert "resnet50" in models
        assert "efficientnet" in models


# =============================================================================
# Context manager
# =============================================================================


class TestContextManager:
    """Tests for the context manager protocol."""

    def test_context_manager_enter_exit(self, mock_model_client):
        """Using the client as a context manager should not raise."""
        # mock_model_client is already constructed; simulate with/as
        client = mock_model_client
        entered = client.__enter__()
        assert entered is client
        # __exit__ should not raise
        client.__exit__(None, None, None)


# =============================================================================
# Instance methods — close
# =============================================================================


class TestClose:
    """Tests for ModelServerClient.close()."""

    def test_close_calls_session_and_grpc(self, mock_model_client):
        """close() must close both the HTTP session and the gRPC client."""
        client = mock_model_client
        client.close()
        client._http_session.close.assert_called_once()
        client._grpc_client.close.assert_called_once()


# =============================================================================
# Instance methods — delegation
# =============================================================================


class TestDelegation:
    """Tests verifying facade methods delegate to sub-components."""

    def test_get_available_models(self, mock_model_client):
        """get_available_models should delegate to the discovery component."""
        models = mock_model_client.get_available_models()
        # The mock gRPC client's repo index returns ["test_model"]
        assert isinstance(models, list)
        assert "test_model" in models

    def test_infer_image_with_bytes(self, mock_model_client, sample_image_bytes):
        """infer_image with raw bytes should return a result dict (not None)."""
        result = mock_model_client.infer_image(sample_image_bytes, "test_model")
        # With the mock gRPC returning 1000-class output, we expect classification
        assert result is not None

    def test_infer_image_with_bad_data(self, mock_model_client):
        """infer_image with clearly invalid data should return None."""
        result = mock_model_client.infer_image(b"not-an-image", "test_model")
        # Preprocessing should fail, yielding None
        assert result is None


# =============================================================================
# Properties
# =============================================================================


class TestProperties:
    """Tests for configuration properties."""

    def test_preprocessing_config_roundtrip(self, mock_model_client):
        """Setting and getting preprocessing_config should round-trip."""
        config = {
            "target_size": (320, 320),
            "normalize": False,
            "mean": [0.0, 0.0, 0.0],
            "std": [1.0, 1.0, 1.0],
            "format": "NHWC",
        }
        mock_model_client.preprocessing_config = config
        retrieved = mock_model_client.preprocessing_config
        assert retrieved["target_size"] == (320, 320)
        assert retrieved["normalize"] is False

    def test_class_names_propagates(self, mock_model_client):
        """Setting class_names on the facade should propagate to the runner."""
        names = ["cat", "dog", "bird"]
        mock_model_client.class_names = names
        assert mock_model_client.class_names == names
