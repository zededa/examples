"""
Tests for agent chat endpoints in webapp/routes/agent.py.

Covers: _sanitize_filename, cleanup throttling, /agent/chat,
/agent/chat/stream, /agent/session/config.
"""

import json
import time
from unittest.mock import MagicMock, patch

import pytest


# ============================================================================
# _sanitize_filename
# ============================================================================

class TestSanitizeFilename:
    """Tests for the _sanitize_filename helper (Bug fix #2 related)."""

    def _fn(self, name: str) -> str:
        from api.agent import _sanitize_filename
        return _sanitize_filename(name)

    def test_ascii_passthrough(self):
        assert self._fn("photo.png") == "photo.png"

    def test_preserves_dashes_and_underscores(self):
        assert self._fn("my-file_name.jpg") == "my-file_name.jpg"

    def test_unicode_normalization(self):
        # NFKD decomposition strips accents
        result = self._fn("caf\u00e9.png")
        assert result == "cafe.png"

    def test_special_chars_replaced(self):
        result = self._fn("hello world!@#$.png")
        # Spaces and special chars become underscores
        assert "!" not in result
        assert "@" not in result
        assert "#" not in result
        assert "$" not in result

    def test_truncation_to_128(self):
        long_name = "a" * 200 + ".png"
        result = self._fn(long_name)
        assert len(result) <= 128

    def test_empty_returns_upload(self):
        assert self._fn("") == "upload"

    def test_only_special_chars_returns_upload(self):
        # All chars get stripped, leaving empty => "upload"
        assert self._fn("!!!") == "upload"

    def test_leading_trailing_dots_stripped(self):
        result = self._fn(".hidden_file.")
        assert not result.startswith(".")
        assert not result.endswith(".")

    def test_japanese_chars_stripped(self):
        """Non-ASCII characters are removed after NFKD normalization."""
        result = self._fn("\u30c6\u30b9\u30c8.png")
        # Japanese katakana won't survive ascii encoding; result may be
        # "png" (dot stripped), ".png", or "upload"
        assert result in ("png", "upload") or ".png" in result

    def test_mixed_unicode_and_ascii(self):
        result = self._fn("r\u00e9sum\u00e9_v2.pdf")
        assert "resume_v2.pdf" == result


# ============================================================================
# Cleanup throttling (Bug fix #11)
# ============================================================================

class TestCleanupThrottling:
    """Verify _cleanup_old_sessions is throttled to once per 60 seconds."""

    def test_cleanup_skipped_when_recent(self, monkeypatch):
        import api.agent as agent_mod

        # Simulate that cleanup just ran
        agent_mod._last_cleanup_time = time.time()

        tracker = MagicMock()
        monkeypatch.setattr(
            "api.agent._cleanup_old_sessions_legacy", tracker
        )

        # Patch out the sessions.registry import so it falls through cleanly
        monkeypatch.setattr(
            "api.agent._CLEANUP_INTERVAL_SECONDS",
            agent_mod._CLEANUP_INTERVAL_SECONDS,
        )

        agent_mod._cleanup_old_sessions()
        # Because _last_cleanup_time is recent, the actual cleanup should NOT fire
        tracker.assert_not_called()

    def test_cleanup_runs_when_stale(self, monkeypatch):
        import api.agent as agent_mod

        # Simulate stale timestamp (well past the interval)
        agent_mod._last_cleanup_time = 0.0

        # Patch the import-based cleanup to avoid needing sessions.registry
        called = {"count": 0}

        def fake_cleanup():
            called["count"] += 1

        monkeypatch.setattr(
            "api.agent._cleanup_old_sessions_legacy", fake_cleanup
        )

        # Force ImportError for sessions.registry path
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "sessions.registry":
                raise ImportError("mocked")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)

        agent_mod._cleanup_old_sessions()
        assert called["count"] == 1


# ============================================================================
# POST /agent/chat
# ============================================================================

class TestAgentChat:
    """Tests for the /agent/chat endpoint."""

    def test_chat_missing_message_json(self, flask_test_client, monkeypatch):
        """Missing message in JSON body. Agent modules must be mocked so the
        code reaches the validation check (otherwise import fails first)."""
        import sys
        fake_prompts = MagicMock()
        fake_prompts.check_agent_enabled = MagicMock(return_value=True)
        fake_prompts.process_chat_message = MagicMock()
        fake_tools = MagicMock()
        fake_tools.get_session_storage_path = MagicMock(return_value="/tmp")
        fake_tools.check_session_storage_limit = MagicMock(return_value=(True, 0))
        monkeypatch.setitem(sys.modules, "agents.prompts", fake_prompts)
        monkeypatch.setitem(sys.modules, "agents.tools", fake_tools)

        resp = flask_test_client.post(
            "/agent/chat",
            data=json.dumps({}),
            content_type="application/json",
        )
        assert resp.status_code == 400

    def test_chat_missing_message_form(self, flask_test_client, monkeypatch):
        """Missing message in form data."""
        import sys
        fake_prompts = MagicMock()
        fake_prompts.check_agent_enabled = MagicMock(return_value=True)
        fake_prompts.process_chat_message = MagicMock()
        fake_tools = MagicMock()
        fake_tools.get_session_storage_path = MagicMock(return_value="/tmp")
        fake_tools.check_session_storage_limit = MagicMock(return_value=(True, 0))
        monkeypatch.setitem(sys.modules, "agents.prompts", fake_prompts)
        monkeypatch.setitem(sys.modules, "agents.tools", fake_tools)

        resp = flask_test_client.post(
            "/agent/chat",
            data={},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 400

    def test_chat_agent_disabled(self, flask_test_client, monkeypatch):
        """When check_agent_enabled returns False, response has enabled=False."""
        monkeypatch.setattr(
            "api.agent.check_agent_enabled",
            lambda: False,
            raising=False,
        )
        # Also need to make the import succeed inside the endpoint
        # The endpoint uses lazy imports, so we patch the modules directly
        mock_process = MagicMock()
        mock_check = MagicMock(return_value=False)
        mock_tools = MagicMock()

        import sys
        fake_prompts = MagicMock()
        fake_prompts.process_chat_message = mock_process
        fake_prompts.check_agent_enabled = mock_check
        fake_tools = MagicMock()
        fake_tools.get_session_storage_path = MagicMock(return_value="/tmp")
        fake_tools.check_session_storage_limit = MagicMock(return_value=(True, 0))

        monkeypatch.setitem(sys.modules, "agents.prompts", fake_prompts)
        monkeypatch.setitem(sys.modules, "agents.tools", fake_tools)

        resp = flask_test_client.post(
            "/agent/chat",
            data=json.dumps({"message": "hello"}),
            content_type="application/json",
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["enabled"] is False

    def test_chat_success(self, flask_test_client, monkeypatch):
        """Successful chat returns 200 with response text."""
        import sys

        fake_prompts = MagicMock()
        fake_prompts.check_agent_enabled = MagicMock(return_value=True)
        fake_prompts.process_chat_message = MagicMock(return_value={
            "success": True,
            "response": "Hello! I can help you explore models.",
            "enabled": True,
            "tool_calls": [],
            "tokens": {"prompt_tokens": 10, "completion_tokens": 20},
        })
        fake_tools = MagicMock()
        fake_tools.get_session_storage_path = MagicMock(return_value="/tmp")
        fake_tools.check_session_storage_limit = MagicMock(return_value=(True, 0))

        monkeypatch.setitem(sys.modules, "agents.prompts", fake_prompts)
        monkeypatch.setitem(sys.modules, "agents.tools", fake_tools)

        resp = flask_test_client.post(
            "/agent/chat",
            data=json.dumps({"message": "list models"}),
            content_type="application/json",
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is True
        assert "response" in data
        assert data["session_id"] is not None

    def test_chat_returns_session_id(self, flask_test_client, monkeypatch):
        """Response always includes a session_id."""
        import sys

        fake_prompts = MagicMock()
        fake_prompts.check_agent_enabled = MagicMock(return_value=True)
        fake_prompts.process_chat_message = MagicMock(return_value={
            "success": True,
            "response": "ok",
            "enabled": True,
        })
        fake_tools = MagicMock()
        fake_tools.get_session_storage_path = MagicMock(return_value="/tmp")
        fake_tools.check_session_storage_limit = MagicMock(return_value=(True, 0))

        monkeypatch.setitem(sys.modules, "agents.prompts", fake_prompts)
        monkeypatch.setitem(sys.modules, "agents.tools", fake_tools)

        resp = flask_test_client.post(
            "/agent/chat",
            data=json.dumps({"message": "hi", "session_id": "my-session-123"}),
            content_type="application/json",
        )
        data = resp.get_json()
        assert data["session_id"] == "my-session-123"


# ============================================================================
# POST /agent/chat/stream (Bug fix #2 - image_path from JSON)
# ============================================================================

class TestAgentChatStream:
    """Tests for the streaming SSE endpoint."""

    def test_stream_returns_event_stream(self, flask_test_client, monkeypatch):
        """The streaming endpoint should return text/event-stream content type."""
        import sys

        fake_prompts = MagicMock()
        fake_prompts.check_agent_enabled = MagicMock(return_value=True)
        fake_prompts.process_chat_message_stream = MagicMock(return_value=iter([
            {"type": "done", "response": "hi", "tool_calls": [], "meta": {}},
        ]))
        fake_tools = MagicMock()
        fake_tools.get_session_storage_path = MagicMock(return_value="/tmp")

        monkeypatch.setitem(sys.modules, "agents.prompts", fake_prompts)
        monkeypatch.setitem(sys.modules, "agents.tools", fake_tools)

        resp = flask_test_client.post(
            "/agent/chat/stream",
            data=json.dumps({"message": "hello"}),
            content_type="application/json",
        )
        assert resp.status_code == 200
        assert "text/event-stream" in resp.content_type

    def test_stream_missing_message(self, flask_test_client):
        resp = flask_test_client.post(
            "/agent/chat/stream",
            data=json.dumps({}),
            content_type="application/json",
        )
        assert resp.status_code == 400

    def test_stream_ignores_image_path_from_json(self, flask_test_client, monkeypatch):
        """Bug fix #2: image_path must NOT be accepted from JSON input."""
        import sys

        captured_kwargs = {}

        def capturing_stream(message, history, session_id=None, image_path=None):
            captured_kwargs["image_path"] = image_path
            yield {"type": "done", "response": "ok", "tool_calls": [], "meta": {}}

        fake_prompts = MagicMock()
        fake_prompts.check_agent_enabled = MagicMock(return_value=True)
        fake_prompts.process_chat_message_stream = capturing_stream
        fake_tools = MagicMock()
        fake_tools.get_session_storage_path = MagicMock(return_value="/tmp")

        monkeypatch.setitem(sys.modules, "agents.prompts", fake_prompts)
        monkeypatch.setitem(sys.modules, "agents.tools", fake_tools)

        resp = flask_test_client.post(
            "/agent/chat/stream",
            data=json.dumps({
                "message": "hello",
                "image_path": "/etc/passwd",  # malicious attempt
            }),
            content_type="application/json",
        )
        # Consume the response to trigger the generator
        _ = resp.get_data(as_text=True)

        assert captured_kwargs.get("image_path") is None


# ============================================================================
# GET /agent/session/config (Bug fix #9 - fallback returns 30.0)
# ============================================================================

class TestSessionConfig:
    """Tests for session configuration endpoint."""

    def test_session_config_fallback(self, flask_test_client, monkeypatch):
        """Bug fix #9: when sessions.config is unavailable, fallback
        must return max_storage_mb = 30.0."""
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if "sessions.config" in name:
                raise ImportError("mocked")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)

        resp = flask_test_client.get("/agent/session/config")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is True
        assert data["config"]["limits"]["max_storage_mb"] == 30.0

    def test_session_config_returns_200(self, flask_test_client):
        resp = flask_test_client.get("/agent/session/config")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is True
        assert "config" in data
