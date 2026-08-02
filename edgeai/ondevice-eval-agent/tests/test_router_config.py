"""
Tests for webapp/router/config.py — pure dataclass and enum tests, no mocking.
"""

import pytest

from router.config import (
    LLMProviderType,
    RoutingStrategy,
    LLMProviderConfig,
    ProviderStatus,
    ChatResponse,
)


# ============================================================================
# LLMProviderType enum
# ============================================================================

@pytest.mark.parametrize("member", [
    "ANTHROPIC",
    "OPENAI",
    "GOOGLE",
    "GROQ",
    "OLLAMA",
    "VLLM",
    "TGI",
    "LMSTUDIO",
    "OPENAI_COMPATIBLE",
])
def test_llm_provider_type_members(member):
    """Every expected provider type exists as an enum member."""
    assert hasattr(LLMProviderType, member)
    assert isinstance(LLMProviderType[member], LLMProviderType)


# ============================================================================
# RoutingStrategy enum
# ============================================================================

@pytest.mark.parametrize("member", [
    "PRIORITY",
    "ROUND_ROBIN",
    "FAILOVER",
    "LATENCY",
    "COST",
])
def test_routing_strategy_members(member):
    """Every expected routing strategy exists as an enum member."""
    assert hasattr(RoutingStrategy, member)
    assert isinstance(RoutingStrategy[member], RoutingStrategy)


# ============================================================================
# LLMProviderConfig defaults
# ============================================================================

class TestLLMProviderConfigDefaults:

    def test_default_priority(self):
        cfg = LLMProviderConfig(name="t", provider_type="ollama")
        assert cfg.priority == 10

    def test_default_temperature(self):
        cfg = LLMProviderConfig(name="t", provider_type="ollama")
        assert cfg.temperature == pytest.approx(0.1)

    def test_default_max_tokens(self):
        cfg = LLMProviderConfig(name="t", provider_type="ollama")
        assert cfg.max_tokens == 4096

    def test_default_timeout(self):
        cfg = LLMProviderConfig(name="t", provider_type="ollama")
        assert cfg.timeout == 60


# ============================================================================
# LLMProviderConfig __post_init__ conversions
# ============================================================================

class TestLLMProviderConfigPostInit:

    def test_string_to_enum_conversion(self):
        cfg = LLMProviderConfig(name="x", provider_type="ollama")
        assert cfg.provider_type is LLMProviderType.OLLAMA

    def test_url_normalization_adds_scheme(self):
        cfg = LLMProviderConfig(name="x", provider_type="ollama", url="localhost:11434")
        assert cfg.url == "http://localhost:11434"

    def test_url_normalization_strips_trailing_slash(self):
        cfg = LLMProviderConfig(name="x", provider_type="ollama", url="http://localhost:11434/")
        assert cfg.url == "http://localhost:11434"

    def test_url_normalization_preserves_https(self):
        cfg = LLMProviderConfig(name="x", provider_type="ollama", url="https://api.example.com")
        assert cfg.url.startswith("https://")

    def test_url_normalization_none_stays_none(self):
        cfg = LLMProviderConfig(name="x", provider_type="ollama", url=None)
        assert cfg.url is None


# ============================================================================
# LLMProviderConfig.model default is None, not empty string
# ============================================================================

def test_model_default_is_none():
    cfg = LLMProviderConfig(name="x", provider_type="ollama")
    assert cfg.model is None


# ============================================================================
# LLMProviderConfig.to_dict / from_dict
# ============================================================================

class TestLLMProviderConfigSerialization:

    def test_to_dict_hides_api_key(self):
        cfg = LLMProviderConfig(name="x", provider_type="ollama", api_key="secret-123")
        d = cfg.to_dict()
        assert "api_key" not in d
        assert d["has_api_key"] is True

    def test_from_dict_roundtrip(self):
        original = LLMProviderConfig(
            name="roundtrip",
            provider_type="ollama",
            url="http://localhost:11434",
            model="llama3",
            priority=5,
            max_tokens=2048,
            temperature=0.7,
            timeout=30,
        )
        d = original.to_dict()
        # from_dict needs provider_type as string and won't have api_key
        restored = LLMProviderConfig.from_dict(d)
        assert restored.name == original.name
        assert restored.provider_type == original.provider_type
        assert restored.url == original.url
        assert restored.model == original.model
        assert restored.priority == original.priority
        assert restored.max_tokens == original.max_tokens
        assert restored.temperature == pytest.approx(original.temperature)
        assert restored.timeout == original.timeout


# ============================================================================
# ProviderStatus.to_dict
# ============================================================================

def test_provider_status_to_dict_keys():
    status = ProviderStatus(name="p", available=True, last_check=1.0)
    d = status.to_dict()
    expected_keys = {
        "name", "available", "last_check", "latency_ms",
        "total_requests", "error_count", "last_error", "models_available",
    }
    assert set(d.keys()) == expected_keys


# ============================================================================
# ChatResponse.to_dict
# ============================================================================

class TestChatResponseToDict:

    def test_minimal_response_has_required_keys(self):
        resp = ChatResponse(content="hello", provider="test", model="m1")
        d = resp.to_dict()
        assert d["content"] == "hello"
        assert d["provider"] == "test"
        assert d["model"] == "m1"

    def test_with_tool_calls(self):
        tc = [{"id": "1", "name": "fn", "arguments": "{}"}]
        resp = ChatResponse(content="", provider="p", model="m", tool_calls=tc)
        d = resp.to_dict()
        assert "tool_calls" in d
        assert d["tool_calls"] == tc

    def test_omits_none_optional_fields(self):
        resp = ChatResponse(content="ok", provider="p", model="m")
        d = resp.to_dict()
        assert "tool_calls" not in d
        assert "usage" not in d
        assert "finish_reason" not in d
