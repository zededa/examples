"""
LLM Adapters Package

Provider-specific adapters for different LLM backends.
Each adapter implements the LLMAdapter interface.

Adapters are imported lazily so that a missing optional dependency for one
provider does not prevent the rest from being used. Each import lives in its
own try/except so (e.g.) a missing `google-genai` install only disables
GoogleAdapter — the rest continue to register.
"""

import logging

logger = logging.getLogger(__name__)

__all__ = [
    "OllamaAdapter",
    "VLLMAdapter",
    "TGIAdapter",
    "OpenAICompatibleAdapter",
    "AnthropicAdapter",
    "OpenAIAdapter",
    "GoogleAdapter",
]

# Default every symbol to None, then try to import each in turn. The router's
# ADAPTER_REGISTRY filters out None entries, so unavailable adapters simply
# aren't registered.
OllamaAdapter = None  # type: ignore[assignment,misc]
VLLMAdapter = None  # type: ignore[assignment,misc]
TGIAdapter = None  # type: ignore[assignment,misc]
OpenAICompatibleAdapter = None  # type: ignore[assignment,misc]
AnthropicAdapter = None  # type: ignore[assignment,misc]
OpenAIAdapter = None  # type: ignore[assignment,misc]
GoogleAdapter = None  # type: ignore[assignment,misc]

try:
    from .anthropic import AnthropicAdapter
except ImportError as e:
    logger.warning(f"AnthropicAdapter unavailable: {e}")

try:
    from .openai_compatible import OpenAICompatibleAdapter
except ImportError as e:
    logger.warning(f"OpenAICompatibleAdapter unavailable: {e}")

try:
    from .openai import OpenAIAdapter
except ImportError as e:
    logger.warning(f"OpenAIAdapter unavailable: {e}")

try:
    from .google import GoogleAdapter
except ImportError as e:
    logger.warning(f"GoogleAdapter unavailable: {e}")

try:
    from .ollama import OllamaAdapter
except ImportError as e:
    logger.debug(f"OllamaAdapter unavailable: {e}")

try:
    from .vllm import VLLMAdapter
except ImportError as e:
    logger.debug(f"VLLMAdapter unavailable: {e}")

try:
    from .tgi import TGIAdapter
except ImportError as e:
    logger.debug(f"TGIAdapter unavailable: {e}")
