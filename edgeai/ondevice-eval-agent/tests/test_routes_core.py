"""
Tests for Flask core endpoints in webapp/routes/core.py.

Covers: /, /health, /models, /predict, /debug/config, /config.
"""

import io
import json

import pytest


# ============================================================================
# GET / (chat UI page)
# ============================================================================

class TestIndex:
    def test_index_returns_200(self, flask_test_client):
        resp = flask_test_client.get("/")
        assert resp.status_code == 200

    def test_index_returns_html(self, flask_test_client):
        resp = flask_test_client.get("/")
        assert "text/html" in resp.content_type


# ============================================================================
# GET /health
# ============================================================================

class TestHealth:
    def test_health_healthy_with_models(self, flask_test_client):
        resp = flask_test_client.get("/health")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is True
        assert data["status"] == "healthy"
        assert "available_models" in data

    def test_health_degraded_no_models(self, flask_test_client, mock_grpc_client):
        """Server live but no models => 200 / degraded."""
        mock_grpc_client.get_model_repository_index.return_value = []
        mock_grpc_client.is_model_ready.return_value = False
        resp = flask_test_client.get("/health")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "degraded"

    def test_health_unavailable(self, flask_test_client, mock_grpc_client):
        """Server not healthy and no models => 503."""
        mock_grpc_client.get_model_repository_index.return_value = []
        mock_grpc_client.is_model_ready.return_value = False
        mock_grpc_client.is_server_ready.return_value = False
        resp = flask_test_client.get("/health")
        assert resp.status_code == 503


# ============================================================================
# GET /models
# ============================================================================

class TestModels:
    def test_models_returns_200(self, flask_test_client):
        resp = flask_test_client.get("/models")
        assert resp.status_code == 200

    def test_models_response_shape(self, flask_test_client):
        resp = flask_test_client.get("/models")
        data = resp.get_json()
        assert "models" in data
        assert "server_type" in data


# ============================================================================
# POST /predict
# ============================================================================

class TestPredict:
    def test_predict_missing_image(self, flask_test_client):
        resp = flask_test_client.post(
            "/predict",
            data={"model": "test_model"},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 400
        data = resp.get_json()
        assert data["error_code"] == "MISSING_IMAGE"

    def test_predict_missing_model(self, flask_test_client, sample_image_bytes):
        resp = flask_test_client.post(
            "/predict",
            data={"image": (io.BytesIO(sample_image_bytes), "test.png")},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 400
        data = resp.get_json()
        assert data["error_code"] == "MISSING_MODEL"

    def test_predict_invalid_format(self, flask_test_client):
        resp = flask_test_client.post(
            "/predict",
            data={
                "image": (io.BytesIO(b"not-an-image"), "test.txt"),
                "model": "test_model",
            },
            content_type="multipart/form-data",
        )
        assert resp.status_code == 400
        data = resp.get_json()
        assert data["error_code"] == "INVALID_FILE_FORMAT"

    def test_predict_success(self, flask_test_client, sample_image_bytes, monkeypatch):
        success_result = {
            "success": True,
            "model_name": "test_model",
            "model_type": "classification",
            "predictions": [{"class": "cat", "confidence": 0.95}],
        }
        monkeypatch.setattr(
            "api.core.execute_prediction",
            lambda filepath, file_bytes, model_name, task_type="auto": success_result,
        )
        resp = flask_test_client.post(
            "/predict",
            data={
                "image": (io.BytesIO(sample_image_bytes), "test.png"),
                "model": "test_model",
            },
            content_type="multipart/form-data",
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is True


# ============================================================================
# GET /debug/config  (Bug fix #13)
# ============================================================================

class TestDebugConfig:
    def test_debug_config_forbidden_by_default(self, flask_test_client, clean_env):
        resp = flask_test_client.get("/debug/config")
        assert resp.status_code == 403
        data = resp.get_json()
        assert data["error_code"] == "DEBUG_DISABLED"

    def test_debug_config_enabled(self, flask_test_client, monkeypatch):
        monkeypatch.setenv("FLASK_DEBUG", "1")
        resp = flask_test_client.get("/debug/config")
        assert resp.status_code != 403


# ============================================================================
# GET /config
# ============================================================================

class TestConfig:
    def test_config_returns_200(self, flask_test_client, monkeypatch):
        # The config fixture uses a set for allowed_extensions which isn't
        # JSON-serializable.  Patch the config to use a list instead.
        import api.core as rc
        patched = dict(rc._app_config)
        patched["allowed_extensions"] = list(patched.get("allowed_extensions", []))
        monkeypatch.setattr(rc, "_app_config", patched)
        resp = flask_test_client.get("/config")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is True
