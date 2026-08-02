"""
Tests for webapp/router/llm_router.py — AgentLLMRouter singleton,
routing strategies, auto-discovery, and token tracking.
"""

import logging
import threading
from unittest.mock import MagicMock, patch

import pytest

from router.config import (
    ChatResponse,
    LLMProviderConfig,
    LLMProviderType,
    RoutingStrategy,
)
from router.llm_router import (
    AgentLLMRouter,
    TokenUsageTracker,
    get_router,
    reset_token_usage,
    get_token_usage,
)


# ---------------------------------------------------------------------------
# Helper: register a provider with mocked availability
# ---------------------------------------------------------------------------

def _register_with_mock(router, name, provider_type="ollama", priority=10,
                        available=True, model="test-model", api_key=None):
    """Register a provider, mocking _check_provider_availability."""
    config = LLMProviderConfig(
        name=name,
        provider_type=provider_type,
        model=model,
        priority=priority,
        api_key=api_key,
        url="http://localhost:11434",
    )
    with patch.object(router, '_get_adapter') as mock_get:
        mock_adapter = MagicMock()
        mock_adapter.check_availability.return_value = (available, 10.0, None)
        mock_get.return_value = mock_adapter
        router.register_provider(config)
    return config


# ============================================================================
# Singleton behaviour
# ============================================================================

class TestSingleton:

    def test_get_router_returns_same_instance(self, reset_router):
        r1 = get_router()
        r2 = get_router()
        assert r1 is r2


# ============================================================================
# Provider registration
# ============================================================================

class TestProviderRegistration:

    def test_register_provider_returns_true(self, reset_router):
        router = AgentLLMRouter(auto_discover=False)
        with patch.object(router, '_get_adapter') as mg:
            mg.return_value = MagicMock(
                check_availability=MagicMock(return_value=(True, 5.0, None))
            )
            result = router.register_provider(
                LLMProviderConfig(name="a", provider_type="ollama", url="http://localhost:11434")
            )
        assert result is True

    def test_register_duplicate_updates(self, reset_router):
        router = AgentLLMRouter(auto_discover=False)
        _register_with_mock(router, "dup", priority=10)
        _register_with_mock(router, "dup", priority=1)
        providers = router.list_providers()
        assert len(providers) == 1
        assert providers[0]["priority"] == 1

    def test_unregister_provider(self, reset_router):
        router = AgentLLMRouter(auto_discover=False)
        _register_with_mock(router, "removeme")
        assert router.unregister_provider("removeme") is True
        assert router.list_providers() == []

    def test_list_providers_returns_dicts_with_status(self, reset_router):
        router = AgentLLMRouter(auto_discover=False)
        _register_with_mock(router, "prov")
        items = router.list_providers()
        assert isinstance(items, list)
        assert len(items) == 1
        assert "status" in items[0]


# ============================================================================
# Routing strategies
# ============================================================================

class TestRouting:

    def test_priority_selects_lowest(self, reset_router):
        router = AgentLLMRouter(routing_strategy=RoutingStrategy.PRIORITY, auto_discover=False)
        _register_with_mock(router, "high", priority=10)
        _register_with_mock(router, "low", priority=1)
        selected = router._select_provider()
        assert selected is not None
        assert selected.name == "low"

    def test_round_robin_rotates(self, reset_router):
        router = AgentLLMRouter(routing_strategy=RoutingStrategy.ROUND_ROBIN, auto_discover=False)
        _register_with_mock(router, "a", priority=1)
        _register_with_mock(router, "b", priority=2)
        first = router._select_provider()
        second = router._select_provider()
        assert first is not None and second is not None
        assert first.name != second.name

    def test_failover_tries_next_on_failure(self, reset_router):
        router = AgentLLMRouter(routing_strategy=RoutingStrategy.FAILOVER, auto_discover=False)
        _register_with_mock(router, "primary", priority=1)
        _register_with_mock(router, "secondary", priority=2)

        call_count = {"n": 0}

        def side_effect(config, messages, tools, **kw):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise RuntimeError("primary down")
            return ChatResponse(content="ok", provider=config.name, model="m")

        mock_adapter = MagicMock()
        mock_adapter.chat.side_effect = side_effect

        with patch.object(router, '_get_adapter', return_value=mock_adapter):
            resp = router.chat(messages=[{"role": "user", "content": "hi"}])

        assert resp.content == "ok"
        assert resp.provider == "secondary"

    def test_no_providers_raises_runtime_error(self, reset_router):
        router = AgentLLMRouter(auto_discover=False)
        with pytest.raises(RuntimeError, match="No LLM providers available"):
            router.chat(messages=[{"role": "user", "content": "hi"}])


# ============================================================================
# Auto-discovery
# ============================================================================

class TestAutoDiscovery:

    def test_anthropic_env_registers_provider(self, reset_router, clean_env):
        clean_env.setenv("ANTHROPIC_API_KEY", "sk-test")
        clean_env.setenv("ANTHROPIC_MODEL", "claude-3-sonnet")
        with patch.object(AgentLLMRouter, '_check_provider_availability', return_value=True):
            router = AgentLLMRouter()
        names = [p["name"] for p in router.list_providers()]
        assert "anthropic" in names

    def test_key_without_model_logs_warning_no_registration(self, reset_router, clean_env, caplog):
        clean_env.setenv("ANTHROPIC_API_KEY", "sk-test")
        # No ANTHROPIC_MODEL set
        with patch.object(AgentLLMRouter, '_check_provider_availability', return_value=True):
            with caplog.at_level(logging.WARNING):
                router = AgentLLMRouter()
        names = [p["name"] for p in router.list_providers()]
        assert "anthropic" not in names
        assert any("ANTHROPIC_MODEL not specified" in r.message for r in caplog.records)

    def test_no_env_vars_empty_providers(self, reset_router, clean_env):
        with patch.object(AgentLLMRouter, '_check_provider_availability', return_value=True):
            router = AgentLLMRouter()
        assert router.list_providers() == []

    def test_auto_discovery_passes_api_key(self, reset_router, clean_env):
        """Bug fix #15 - all auto-discovered providers must carry their api_key."""
        clean_env.setenv("ANTHROPIC_API_KEY", "sk-key-123")
        clean_env.setenv("ANTHROPIC_MODEL", "claude-3-sonnet")
        with patch.object(AgentLLMRouter, '_check_provider_availability', return_value=True):
            router = AgentLLMRouter()
        cfg = router.get_provider("anthropic")
        assert cfg is not None
        assert cfg.api_key == "sk-key-123"

    def test_auto_discovery_all_cloud_providers(self, reset_router, clean_env):
        """Verify OpenAI, Google, Groq also register when env vars are set."""
        clean_env.setenv("OPENAI_API_KEY", "sk-openai")
        clean_env.setenv("OPENAI_MODEL", "gpt-4")
        clean_env.setenv("GOOGLE_API_KEY", "gkey")
        clean_env.setenv("GOOGLE_MODEL", "gemini")
        clean_env.setenv("GROQ_API_KEY", "gsk-groq")
        clean_env.setenv("GROQ_MODEL", "llama3")
        with patch.object(AgentLLMRouter, '_check_provider_availability', return_value=True):
            router = AgentLLMRouter()
        names = {p["name"] for p in router.list_providers()}
        assert {"openai", "google", "groq"}.issubset(names)

    def test_edgeai_builtin_registers_with_eip_token(self, reset_router, clean_env):
        """EIP_ACCESS_TOKEN + LLM_SERVER_URL auto-registers an
        openai-compatible provider at priority 1 with /openai appended."""
        clean_env.setenv("EIP_ACCESS_TOKEN", "jwt-platform-token")
        clean_env.setenv("LLM_SERVER_URL", "https://edgeai.example.com/api/v1")
        clean_env.setenv("LLM_MODEL_NAME", "edgeai-default")
        with patch.object(AgentLLMRouter, '_check_provider_availability', return_value=True):
            router = AgentLLMRouter()
        cfg = router.get_provider("edgeai-builtin")
        assert cfg is not None
        assert cfg.provider_type == LLMProviderType.OPENAI_COMPATIBLE
        assert cfg.url == "https://edgeai.example.com/api/v1/openai"
        assert cfg.api_key == "jwt-platform-token"
        assert cfg.model == "edgeai-default"
        assert cfg.priority == 1
        assert cfg.metadata.get("builtin") is True
        assert cfg.metadata.get("managed_by") == "edgeai-platform"

    def test_edgeai_builtin_default_model(self, reset_router, clean_env):
        """LLM_MODEL_NAME defaults to 'edgeai-default' when omitted."""
        clean_env.setenv("EIP_ACCESS_TOKEN", "jwt")
        clean_env.setenv("LLM_SERVER_URL", "https://edgeai.example.com/api/v1")
        with patch.object(AgentLLMRouter, '_check_provider_availability', return_value=True):
            router = AgentLLMRouter()
        cfg = router.get_provider("edgeai-builtin")
        assert cfg is not None
        assert cfg.model == "edgeai-default"

    def test_edgeai_builtin_does_not_double_append_openai(self, reset_router, clean_env):
        """If LLM_SERVER_URL already ends in /openai we leave it alone."""
        clean_env.setenv("EIP_ACCESS_TOKEN", "jwt")
        clean_env.setenv("LLM_SERVER_URL", "https://edgeai.example.com/api/v1/openai")
        with patch.object(AgentLLMRouter, '_check_provider_availability', return_value=True):
            router = AgentLLMRouter()
        cfg = router.get_provider("edgeai-builtin")
        assert cfg is not None
        assert cfg.url == "https://edgeai.example.com/api/v1/openai"

    def test_edgeai_builtin_suppresses_local_llm_registration(self, reset_router, clean_env):
        """When EIP_ACCESS_TOKEN is set, we don't also register the legacy
        local-llm entry from LLM_API_KEY — that would mean two providers
        on the same URL with conflicting auth."""
        clean_env.setenv("EIP_ACCESS_TOKEN", "jwt")
        clean_env.setenv("LLM_SERVER_URL", "https://edgeai.example.com/api/v1")
        clean_env.setenv("LLM_MODEL_NAME", "edgeai-default")
        clean_env.setenv("LLM_API_KEY", "user-key")
        with patch.object(AgentLLMRouter, '_check_provider_availability', return_value=True):
            router = AgentLLMRouter()
        names = {p["name"] for p in router.list_providers()}
        assert "edgeai-builtin" in names
        assert "local-llm" not in names

    def test_eip_token_without_url_logs_warning(self, reset_router, clean_env, caplog):
        clean_env.setenv("EIP_ACCESS_TOKEN", "jwt")
        with patch.object(AgentLLMRouter, '_check_provider_availability', return_value=True):
            with caplog.at_level(logging.WARNING):
                router = AgentLLMRouter()
        names = {p["name"] for p in router.list_providers()}
        assert "edgeai-builtin" not in names
        assert any("LLM_SERVER_URL not specified" in r.message for r in caplog.records)


# ============================================================================
# Token usage tracking
# ============================================================================

class TestTokenTracking:

    def test_record_and_get_usage(self):
        tracker = TokenUsageTracker()
        tracker.record("prov", "model-a", {"prompt_tokens": 10, "completion_tokens": 20})
        usage = tracker.get_usage()
        assert "prov/model-a" in usage
        assert usage["prov/model-a"]["total_tokens"] == 30

    def test_reset_clears_stats(self):
        tracker = TokenUsageTracker()
        tracker.record("prov", "model-a", {"prompt_tokens": 5, "completion_tokens": 5})
        tracker.reset()
        assert tracker.get_usage() == {}

    def test_get_usage_filter_by_provider(self):
        tracker = TokenUsageTracker()
        tracker.record("alpha", "m1", {"prompt_tokens": 1, "completion_tokens": 1})
        tracker.record("beta", "m2", {"prompt_tokens": 2, "completion_tokens": 2})
        alpha_only = tracker.get_usage(provider="alpha")
        assert "alpha/m1" in alpha_only
        assert "beta/m2" not in alpha_only

    def test_get_totals(self):
        tracker = TokenUsageTracker()
        tracker.record("a", "m", {"prompt_tokens": 10, "completion_tokens": 5})
        tracker.record("b", "m", {"prompt_tokens": 20, "completion_tokens": 10})
        totals = tracker.get_totals()
        assert totals["prompt_tokens"] == 30
        assert totals["completion_tokens"] == 15
        assert totals["total_tokens"] == 45
        assert totals["request_count"] == 2
