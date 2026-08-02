"""
Tests for individual MCP tool functions in webapp/mcp/tools/.

All tools use get_client() from tools.base to obtain a ModelServerClient.
We patch each tool module's get_client reference to avoid network calls.
"""

import pytest
from unittest.mock import patch, MagicMock

from tools.registry import TOOL_SCHEMAS, TOOL_FUNCTIONS


# ============================================================================
# Helpers
# ============================================================================

def _make_mock_client(**overrides):
    """Build a MagicMock mimicking ModelServerClient with sensible defaults."""
    client = MagicMock()
    client.get_available_models.return_value = ["resnet50"]
    client.detect_server_type.return_value = "triton"
    client.get_server_info.return_value = {"name": "triton", "version": "2.40.0"}
    client.check_server_health.return_value = (True, "Server is healthy")
    client.get_server_device_info.return_value = "CPU"
    client.server_url = "localhost:8001"
    client.check_model_ready.return_value = True
    client.get_full_model_info.return_value = {
        "input_spec": {"name": "images", "shape": [1, 3, 224, 224], "datatype": "FP32"},
        "output_spec": {"name": "output0", "shape": [1, 1000], "datatype": "FP32"},
        "metadata": {},
        "ready": True,
        "server_type": "triton",
    }
    client.get_model_input_spec.return_value = {"name": "images", "shape": [1, 3, 224, 224]}
    client.get_model_output_spec.return_value = {"name": "output0", "shape": [1, 1000]}
    client.get_all_output_specs.return_value = [{"name": "output0", "shape": [1, 1000]}]
    for key, val in overrides.items():
        setattr(client, key, val)
    return client


# ============================================================================
# list_available_models
# ============================================================================

class TestListAvailableModels:
    def test_success_returns_models(self):
        mock_client = _make_mock_client()
        with patch("tools.catalog.list_models.get_client", return_value=mock_client):
            from tools.catalog.list_models import list_available_models
            result = list_available_models()
        assert result["success"] is True
        assert "models" in result
        assert result["models"] == ["resnet50"]

    def test_error_path(self):
        mock_client = _make_mock_client()
        mock_client.get_available_models.side_effect = RuntimeError("connection refused")
        with patch("tools.catalog.list_models.get_client", return_value=mock_client):
            from tools.catalog.list_models import list_available_models
            result = list_available_models()
        assert result["success"] is False


# ============================================================================
# get_model_metadata
# ============================================================================

class TestGetModelMetadata:
    def test_returns_input_and_output_spec(self):
        mock_client = _make_mock_client()
        with patch("tools.catalog.model_metadata.get_client", return_value=mock_client):
            from tools.catalog.model_metadata import get_model_metadata
            result = get_model_metadata(model_name="resnet50")
        assert result["success"] is True
        assert result["model_name"] == "resnet50"
        assert "input_spec" in result
        assert "output_spec" in result


# ============================================================================
# get_server_status
# ============================================================================

class TestGetServerStatus:
    def test_healthy(self):
        mock_client = _make_mock_client()
        with patch("tools.catalog.server_status.get_client", return_value=mock_client):
            from tools.catalog.server_status import get_server_status
            result = get_server_status()
        assert result["success"] is True
        assert result["healthy"] is True
        assert "server_type" in result

    def test_unhealthy(self):
        mock_client = _make_mock_client()
        mock_client.check_server_health.return_value = (False, "Server is down")
        with patch("tools.catalog.server_status.get_client", return_value=mock_client):
            from tools.catalog.server_status import get_server_status
            result = get_server_status()
        assert result["success"] is True
        assert result["healthy"] is False


# ============================================================================
# compare_models
# ============================================================================

class TestCompareModels:
    def test_returns_both_models_and_differences(self):
        mock_client = _make_mock_client()
        # The tool module's `get_client` is a local ref from `from tools.base import get_client`.
        # Access the actual module object via sys.modules to patch the local ref.
        import sys as _sys
        mod = _sys.modules["tools.catalog.compare_models"]
        with patch.object(mod, "get_client", return_value=mock_client):
            result = mod.compare_models(model_a="resnet50", model_b="mobilenet")
        assert result["success"] is True
        assert "model_a" in result
        assert "model_b" in result
        assert "differences" in result


# ============================================================================
# check_model_ready
# ============================================================================

class TestCheckModelReady:
    def test_ready_model(self):
        mock_client = _make_mock_client()
        import sys as _sys
        mod = _sys.modules["tools.catalog.check_model_ready"]
        with patch.object(mod, "get_client", return_value=mock_client):
            result = mod.check_model_ready(model_name="resnet50")
        assert result["success"] is True
        assert result["model_name"] == "resnet50"
        assert result["ready"] is True

    def test_not_ready_model(self):
        mock_client = _make_mock_client()
        mock_client.check_model_ready.return_value = False
        import sys as _sys
        mod = _sys.modules["tools.catalog.check_model_ready"]
        with patch.object(mod, "get_client", return_value=mock_client):
            result = mod.check_model_ready(model_name="broken_model")
        assert result["success"] is True
        assert result["ready"] is False


# ============================================================================
# list_processing_types
# ============================================================================

class TestListProcessingTypes:
    def test_returns_processing_types(self):
        from tools.catalog.run_inference import list_processing_types
        result = list_processing_types()
        assert result["success"] is True
        # processing_types is nested under "data"
        assert "data" in result
        assert "processing_types" in result["data"]


# ============================================================================
# Cross-cutting: registration consistency
# ============================================================================

class TestToolRegistration:
    def test_all_schemas_have_matching_function(self):
        """Every entry in TOOL_SCHEMAS should have a callable in TOOL_FUNCTIONS."""
        for schema in TOOL_SCHEMAS:
            name = schema["name"]
            assert name in TOOL_FUNCTIONS, f"Schema '{name}' has no matching function"
            assert callable(TOOL_FUNCTIONS[name])
