"""
Tests for LLM provider endpoints in webapp/routes/llm.py.

Covers: /llm/providers, /llm/health, /llm/credentials, /llm/strategy,
        /llm/chat, /llm/credentials/export.
"""

import json
import threading

import pytest


# ============================================================================
# Helpers
# ============================================================================

def _register_provider(client, name="test-ollama", provider_type="ollama",
                       url="http://localhost:11434", model="llama3.2"):
    """Helper to register a provider via POST."""
    return client.post(
        "/llm/providers",
        data=json.dumps({
            "name": name,
            "provider_type": provider_type,
            "url": url,
            "model": model,
            "priority": 1,
        }),
        content_type="application/json",
    )


# ============================================================================
# GET /llm/providers
# ============================================================================

class TestLLMProviders:
    """Tests for LLM provider listing."""

    def test_list_providers_empty(self, flask_test_client, reset_router):
        resp = flask_test_client.get("/llm/providers")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "providers" in data
        assert "count" in data
        assert isinstance(data["providers"], list)

    def test_list_providers_after_register(self, flask_test_client, reset_router):
        _register_provider(flask_test_client)
        resp = flask_test_client.get("/llm/providers")
        data = resp.get_json()
        assert data["count"] >= 1


# ============================================================================
# POST /llm/providers
# ============================================================================

class TestLLMRegisterProvider:
    """Tests for provider registration."""

    def test_register_valid(self, flask_test_client, reset_router):
        resp = _register_provider(flask_test_client)
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["registered"] is True
        assert data["provider_name"] == "test-ollama"

    def test_register_missing_fields(self, flask_test_client, reset_router):
        resp = flask_test_client.post(
            "/llm/providers",
            data=json.dumps({"name": "incomplete"}),
            content_type="application/json",
        )
        assert resp.status_code == 400

    def test_register_no_json_body(self, flask_test_client, reset_router):
        resp = flask_test_client.post(
            "/llm/providers",
            data=json.dumps({}),
            content_type="application/json",
        )
        # Missing required fields (name, provider_type) returns 400
        assert resp.status_code == 400


# ============================================================================
# PATCH /llm/providers/<name>
# ============================================================================

class TestLLMUpdateProvider:
    """Tests for provider update."""

    def test_update_existing(self, flask_test_client, reset_router):
        _register_provider(flask_test_client, name="updatable")
        resp = flask_test_client.patch(
            "/llm/providers/updatable",
            data=json.dumps({"model": "llama3.3"}),
            content_type="application/json",
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["updated"] is True

    def test_update_nonexistent(self, flask_test_client, reset_router):
        resp = flask_test_client.patch(
            "/llm/providers/nonexistent",
            data=json.dumps({"model": "x"}),
            content_type="application/json",
        )
        assert resp.status_code == 404


# ============================================================================
# DELETE /llm/providers/<name>
# ============================================================================

class TestLLMDeleteProvider:
    """Tests for provider deletion."""

    def test_delete_provider(self, flask_test_client, reset_router):
        _register_provider(flask_test_client, name="deleteme")
        resp = flask_test_client.delete("/llm/providers/deleteme")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["provider_name"] == "deleteme"


# ============================================================================
# GET /llm/health
# ============================================================================

class TestLLMHealth:
    """Tests for LLM provider health check."""

    def test_health_returns_200(self, flask_test_client, reset_router):
        resp = flask_test_client.get("/llm/health")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "available" in data
        assert "unavailable" in data


# ============================================================================
# Credential Endpoints
# ============================================================================

@pytest.fixture()
def mock_secure_storage(monkeypatch, tmp_path):
    """Provide a real SecureStorage backed by a temp directory."""
    from storage import SecureStorage, reset_secure_storage

    reset_secure_storage()
    storage = SecureStorage(storage_dir=str(tmp_path / "secure"))

    monkeypatch.setattr(
        "storage.credentials._storage_instance", storage
    )
    yield storage
    reset_secure_storage()


def _store_credential(client, name="test-cred", provider_type="openai",
                      api_key="sk-testkey1234567890"):
    """Helper to store a credential via POST."""
    return client.post(
        "/llm/credentials",
        data=json.dumps({
            "name": name,
            "provider_type": provider_type,
            "api_key": api_key,
        }),
        content_type="application/json",
    )


class TestLLMCredentials:
    """Tests for credential storage endpoints."""

    def test_store_credential_valid(self, flask_test_client, mock_secure_storage):
        resp = _store_credential(flask_test_client)
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["stored"] is True
        assert data["credential_name"] == "test-cred"

    def test_list_credentials_no_api_key(self, flask_test_client, mock_secure_storage):
        """GET /llm/credentials must NOT expose api_key."""
        _store_credential(flask_test_client)
        resp = flask_test_client.get("/llm/credentials")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["count"] >= 1
        for cred in data["credentials"]:
            assert "api_key" not in cred

    def test_get_credential_masked_key(self, flask_test_client, mock_secure_storage):
        """GET /llm/credentials/<name> masks the api_key."""
        _store_credential(flask_test_client, name="masked-test",
                          api_key="sk-abcdefghijklmnop")
        resp = flask_test_client.get("/llm/credentials/masked-test")
        assert resp.status_code == 200
        data = resp.get_json()
        cred = data["credential"]
        masked = cred["api_key_masked"]
        # Should be partially masked, not the full key
        assert masked is not None
        assert "..." in masked
        assert masked != "sk-abcdefghijklmnop"

    def test_delete_credential(self, flask_test_client, mock_secure_storage):
        _store_credential(flask_test_client, name="to-delete")
        resp = flask_test_client.delete("/llm/credentials/to-delete")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["deleted"] is True

    def test_export_defaults_exclude_keys(self, flask_test_client, mock_secure_storage):
        """Bug fix #3: POST /llm/credentials/export defaults to include_keys=False."""
        _store_credential(flask_test_client, name="export-test",
                          api_key="sk-secretvalue12345678")
        resp = flask_test_client.post(
            "/llm/credentials/export",
            data=json.dumps({}),
            content_type="application/json",
        )
        assert resp.status_code == 200
        data = resp.get_json()
        bundle = data["bundle"]
        # Credentials in the bundle should NOT have api_key
        for cred in bundle.get("credentials", []):
            assert "api_key" not in cred

    def test_export_with_keys_when_requested(self, flask_test_client, mock_secure_storage):
        """When include_keys=True, api_key should be present."""
        _store_credential(flask_test_client, name="export-keys",
                          api_key="sk-secretvalue12345678")
        resp = flask_test_client.post(
            "/llm/credentials/export",
            data=json.dumps({"include_keys": True}),
            content_type="application/json",
        )
        assert resp.status_code == 200
        data = resp.get_json()
        bundle = data["bundle"]
        found_key = False
        for cred in bundle.get("credentials", []):
            if cred.get("api_key"):
                found_key = True
        assert found_key


# ============================================================================
# POST /llm/chat
# ============================================================================

class TestLLMChat:
    """Tests for the LLM chat endpoint."""

    def test_chat_missing_messages(self, flask_test_client, reset_router):
        resp = flask_test_client.post(
            "/llm/chat",
            data=json.dumps({}),
            content_type="application/json",
        )
        assert resp.status_code == 400


# ============================================================================
# PUT /llm/strategy
# ============================================================================

class TestLLMStrategy:
    """Tests for the routing strategy endpoint."""

    def test_set_valid_strategy(self, flask_test_client, reset_router):
        resp = flask_test_client.put(
            "/llm/strategy",
            data=json.dumps({"strategy": "round_robin"}),
            content_type="application/json",
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["new_strategy"] == "round_robin"

    def test_set_invalid_strategy(self, flask_test_client, reset_router):
        resp = flask_test_client.put(
            "/llm/strategy",
            data=json.dumps({"strategy": "banana"}),
            content_type="application/json",
        )
        assert resp.status_code == 400

    def test_set_strategy_missing_field(self, flask_test_client, reset_router):
        resp = flask_test_client.put(
            "/llm/strategy",
            data=json.dumps({}),
            content_type="application/json",
        )
        assert resp.status_code == 400
