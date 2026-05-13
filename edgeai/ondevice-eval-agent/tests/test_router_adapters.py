"""
Tests for webapp/router/adapters/ — OllamaAdapter, OpenAICompatibleAdapter,
AnthropicAdapter, and the LLMAdapter contract.
"""

import threading
from unittest.mock import MagicMock, patch

import pytest

from router.base import LLMAdapter
from router.config import LLMProviderConfig, LLMProviderType, ChatResponse
from router.adapters.ollama import OllamaAdapter
from router.adapters.openai_compatible import OpenAICompatibleAdapter
from router.adapters.anthropic import (
    AnthropicAdapter,
    _convert_tools_to_anthropic_format,
)


# ============================================================================
# OllamaAdapter
# ============================================================================

class TestOllamaAdapter:

    def test_check_availability_success(self, reset_rate_limit_config):
        adapter = OllamaAdapter()
        mock_session = MagicMock()
        mock_response = MagicMock(status_code=200)
        mock_response.json.return_value = {"models": [{"name": "llama3"}]}
        mock_session.get.return_value = mock_response

        config = LLMProviderConfig(name="oll", provider_type="ollama", url="http://localhost:11434")
        with patch.object(adapter, '_get_session', return_value=mock_session):
            available, latency, error = adapter.check_availability(config)

        assert available is True
        assert latency > 0 or latency == pytest.approx(0, abs=50)
        assert error is None

    def test_check_availability_failure(self, reset_rate_limit_config):
        adapter = OllamaAdapter()
        mock_session = MagicMock()
        mock_session.get.side_effect = ConnectionError("refused")

        config = LLMProviderConfig(name="oll", provider_type="ollama", url="http://localhost:11434")
        with patch.object(adapter, '_get_session', return_value=mock_session):
            available, latency, error = adapter.check_availability(config)

        assert available is False
        assert latency == pytest.approx(0.0)
        assert error is not None and "refused" in error

    def test_chat_no_model_raises_value_error(self, reset_rate_limit_config):
        adapter = OllamaAdapter()
        config = LLMProviderConfig(name="oll", provider_type="ollama", model=None,
                                   url="http://localhost:11434")
        with pytest.raises(ValueError, match="No model specified"):
            adapter.chat(config, messages=[{"role": "user", "content": "hi"}])

    def test_default_url(self):
        assert OllamaAdapter.DEFAULT_URL == "http://localhost:11434"


# ============================================================================
# OpenAICompatibleAdapter._normalize_url
# ============================================================================

class TestOpenAICompatibleNormalizeUrl:

    def _normalize(self, url):
        return OpenAICompatibleAdapter()._normalize_url(url)

    def test_bare_host_gets_v1(self):
        assert self._normalize("http://localhost:1234") == "http://localhost:1234/v1"

    def test_url_with_path_no_v1(self):
        """Bug fix #17 - URLs that already have a non-trivial path must not get /v1."""
        assert self._normalize("http://localhost:1234/api") == "http://localhost:1234/api"

    def test_already_has_v1(self):
        assert self._normalize("http://localhost:1234/v1") == "http://localhost:1234/v1"

    def test_trailing_slash_stripped(self):
        result = self._normalize("http://localhost:1234/")
        assert not result.endswith("//v1")
        assert result.endswith("/v1")


# ============================================================================
# OpenAICompatibleAdapter.check_availability
# ============================================================================

class TestOpenAICompatibleAvailability:

    def test_cloud_provider_with_key(self, reset_rate_limit_config):
        adapter = OpenAICompatibleAdapter()
        config = LLMProviderConfig(
            name="groq", provider_type="groq", api_key="gsk-key",
        )
        available, latency, error = adapter.check_availability(config)
        assert available is True
        assert error is None

    def test_cloud_provider_without_key(self, reset_rate_limit_config):
        adapter = OpenAICompatibleAdapter()
        config = LLMProviderConfig(
            name="groq", provider_type="groq", api_key=None,
        )
        available, _, error = adapter.check_availability(config)
        assert available is False
        assert "API key" in error


# ============================================================================
# OpenAICompatibleAdapter.chat_stream — mid-stream SSE error handling
# ============================================================================

class TestOpenAICompatibleSSEErrorHandling:
    """The EdgeAI endpoint emits rate_limit_exceeded as an SSE error
    payload AFTER the initial 200 OK, since rate limiting only kicks in
    once usage bookkeeping completes. The adapter must surface this as a
    structured error event with retry_after preserved."""

    def _stream_response(self, lines):
        """Build a fake requests.Response that yields the given SSE lines."""
        mock = MagicMock(status_code=200)
        mock.iter_lines.return_value = iter(lines)
        return mock

    def _make_config(self):
        return LLMProviderConfig(
            name="edgeai-builtin",
            provider_type=LLMProviderType.OPENAI_COMPATIBLE,
            url="https://edgeai.example.com/api/v1/openai",
            model="edgeai-default",
            api_key="jwt",
        )

    def test_mid_stream_rate_limit_event_surfaced(self, reset_rate_limit_config):
        adapter = OpenAICompatibleAdapter()
        config = self._make_config()
        body = [
            'data: {"choices":[{"delta":{"content":"hello"}}]}',
            'data: {"error":{"type":"rate_limit_exceeded","message":"RPM cap","retry_after":12}}',
        ]
        mock_session = MagicMock()
        mock_session.post.return_value = self._stream_response(body)

        with patch.object(adapter, '_get_session', return_value=mock_session):
            events = list(adapter.chat_stream(
                config, messages=[{"role": "user", "content": "hi"}],
            ))

        # First event is the token "hello"; then the rate-limit error.
        assert events[0] == {"type": "token", "content": "hello"}
        rate_evt = next(e for e in events if e.get("type") == "error")
        assert rate_evt["status_code"] == 429
        assert rate_evt["error_code"] == "rate_limit_exceeded"
        assert rate_evt["retry_after"] == pytest.approx(12.0)
        assert "RPM cap" in rate_evt["error"]

    def test_mid_stream_generic_error_event(self, reset_rate_limit_config):
        adapter = OpenAICompatibleAdapter()
        config = self._make_config()
        body = [
            'data: {"error":{"type":"server_error","message":"upstream blew up"}}',
        ]
        mock_session = MagicMock()
        mock_session.post.return_value = self._stream_response(body)

        with patch.object(adapter, '_get_session', return_value=mock_session):
            events = list(adapter.chat_stream(
                config, messages=[{"role": "user", "content": "hi"}],
            ))

        err = next(e for e in events if e.get("type") == "error")
        # Generic error doesn't claim a 429 status_code.
        assert err.get("status_code") != 429
        assert err.get("error_code") != "rate_limit_exceeded"
        assert "upstream blew up" in err["error"]


# ============================================================================
# AnthropicAdapter
# ============================================================================

class TestAnthropicAdapter:

    def test_check_availability_with_valid_client(self, reset_rate_limit_config):
        """Bug fix #6 & #16 - check_availability makes a lightweight API call."""
        adapter = AnthropicAdapter()
        mock_client = MagicMock()
        # models.list returns a page-like object
        mock_client.models.list.return_value = MagicMock(data=[])

        config = LLMProviderConfig(
            name="anth", provider_type="anthropic", api_key="sk-test",
        )
        with patch.object(adapter, '_get_client', return_value=mock_client):
            available, latency, error = adapter.check_availability(config)

        assert available is True
        assert error is None
        mock_client.models.list.assert_called_once_with(limit=1)

    def test_check_availability_no_client(self, reset_rate_limit_config):
        adapter = AnthropicAdapter()
        config = LLMProviderConfig(name="anth", provider_type="anthropic")
        with patch.object(adapter, '_get_client', return_value=None):
            available, _, error = adapter.check_availability(config)
        assert available is False
        assert "not installed" in error or "API key" in error

    def test_convert_tools_to_anthropic_format_openai_input(self):
        """_convert_tools_to_anthropic_format handles OpenAI function-calling style."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
        result = _convert_tools_to_anthropic_format(tools)
        assert len(result) == 1
        assert result[0]["name"] == "get_weather"
        assert "input_schema" in result[0]

    def test_convert_tools_to_anthropic_format_native(self):
        """Already-Anthropic-format tools pass through cleanly."""
        tools = [
            {
                "name": "search",
                "description": "Search",
                "input_schema": {"type": "object"},
            }
        ]
        result = _convert_tools_to_anthropic_format(tools)
        assert result[0]["name"] == "search"
        assert result[0]["input_schema"] == {"type": "object"}


# ============================================================================
# Thread-safe model cache (Bug fix #7)
# ============================================================================

class TestThreadSafeModelCache:

    def test_concurrent_list_models_no_corruption(self, reset_rate_limit_config):
        """Bug fix #7 - concurrent list_models must not corrupt the cache."""
        adapter = OpenAICompatibleAdapter()
        # Clear class-level cache
        OpenAICompatibleAdapter._models_cache.clear()
        OpenAICompatibleAdapter._models_cache_time.clear()

        mock_session = MagicMock()
        mock_response = MagicMock(status_code=200)
        mock_response.json.return_value = {
            "data": [{"id": "model-a"}, {"id": "model-b"}]
        }
        mock_session.get.return_value = mock_response

        config = LLMProviderConfig(
            name="local", provider_type="openai-compatible",
            url="http://localhost:1234", api_key=None,
        )

        results = []
        errors = []

        def fetch():
            try:
                with patch.object(adapter, '_get_session', return_value=mock_session):
                    models = adapter.list_models(config)
                results.append(models)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=fetch) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Threads raised errors: {errors}"
        # Every thread should see the same consistent model list
        for r in results:
            assert sorted(r) == ["model-a", "model-b"]


# ============================================================================
# Adapter contract: all concrete adapters inherit from LLMAdapter
# ============================================================================

class TestAdapterContract:

    @pytest.mark.parametrize("cls", [
        OllamaAdapter,
        OpenAICompatibleAdapter,
        AnthropicAdapter,
    ])
    def test_inherits_from_llm_adapter(self, cls):
        assert issubclass(cls, LLMAdapter)

    @pytest.mark.parametrize("cls", [
        OllamaAdapter,
        OpenAICompatibleAdapter,
        AnthropicAdapter,
    ])
    def test_has_required_methods(self, cls):
        for method in ("check_availability", "list_models", "chat"):
            assert hasattr(cls, method), f"{cls.__name__} missing {method}"
