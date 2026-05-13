"""
Thread-safe LLM client manager.

Centralizes per-thread client instantiation for all provider SDKs
(Anthropic, OpenAI cloud, OpenAI-compatible, Groq, Google Gemini),
backend detection from environment, and the BACKEND_CAPABILITIES table
consumed by the chat orchestration in prompts.py.
"""

import logging
import os
import threading
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


# SDK availability flags. Imported conditionally so a minimal install can
# still boot without every provider SDK present.
OPENAI_AVAILABLE = False
ANTHROPIC_AVAILABLE = False
GOOGLE_AVAILABLE = False

OpenAI = None
anthropic = None
genai = None

try:
    from openai import OpenAI as _OpenAI
    OpenAI = _OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    logger.info("OpenAI SDK not installed. OpenAI backends will be unavailable.")

try:
    import anthropic as _anthropic
    anthropic = _anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    logger.info("Anthropic SDK not installed. Anthropic backend will be unavailable.")

try:
    import google.generativeai as _genai
    genai = _genai
    GOOGLE_AVAILABLE = True
except ImportError:
    logger.info("Google Generative AI SDK not installed. Google backend will be unavailable.")


# NOTE: Models must be explicitly configured via environment variables.
# No hardcoded defaults - this ensures users always specify the model they want.
DEFAULT_MODELS = {
    'anthropic': '',          # Set via ANTHROPIC_MODEL
    'openai': '',             # Set via OPENAI_MODEL
    'google': '',             # Set via GOOGLE_MODEL
    'groq': '',               # Set via GROQ_MODEL
    'openai-compatible': '',  # Set via LLM_MODEL_NAME
}


# ============================================================================
# LLM Manager - Thread-safe client management
# ============================================================================

class LLMManager:
    """
    Thread-safe manager for LLM client instances.
    Uses thread-local storage to ensure safe concurrent access in web servers.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """Singleton pattern with thread safety."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        # Thread-local storage for client instances
        self._local = threading.local()
        self._initialized = True
        self._backend = None
        logger.info("LLMManager initialized")
    
    def _get_model_name(self, backend: str, env_var: str) -> str:
        """Get model name from environment variable."""
        model = os.environ.get(env_var, DEFAULT_MODELS.get(backend, ''))
        
        if not model:
            logger.warning(f"No model configured for {backend}. Set {env_var} environment variable.")
        
        return model
    
    @property
    def backend(self) -> Optional[str]:
        """
        Detect and cache the active backend.
        
        Note: Backend is cached at first access for performance. If environment
        variables change (hot reload, tests, multi-tenant), call reset_backend()
        to re-detect.
        """
        if self._backend is None:
            self._backend = self._detect_backend()
        return self._backend
    
    def reset_backend(self):
        """
        Reset cached backend to force re-detection on next access.
        
        Use this when:
        - Environment variables change at runtime
        - Running tests that switch backends
        - Hot-reloading configuration
        - Multi-tenant scenarios with different configs
        """
        self._backend = None
        # Also clear thread-local clients since they may be for old backend
        if hasattr(self._local, 'anthropic_client'):
            self._local.anthropic_client = None
        if hasattr(self._local, 'openai_cloud_client'):
            self._local.openai_cloud_client = None
        if hasattr(self._local, 'openai_compatible_client'):
            self._local.openai_compatible_client = None
        if hasattr(self._local, 'google_model'):
            self._local.google_model = None
        if hasattr(self._local, 'groq_client'):
            self._local.groq_client = None
        logger.info("LLMManager backend cache reset")
    
    def _detect_backend(self) -> Optional[str]:
        """
        Detect which backend to use based on environment variables.
        Priority: ANTHROPIC_API_KEY > OPENAI_API_KEY > GOOGLE_API_KEY > GROQ_API_KEY > LLM_SERVER_URL (+ LLM_API_KEY or EIP_ACCESS_TOKEN)

        For the openai-compatible backend, LLM_SERVER_URL must be paired
        with credentials — either LLM_API_KEY (user-supplied) or
        EIP_ACCESS_TOKEN (injected by the EdgeAI platform). A bare
        LLM_SERVER_URL must not by itself make the UI claim that a local
        LLM is "configured".
        """
        if os.environ.get('ANTHROPIC_API_KEY') and ANTHROPIC_AVAILABLE:
            return 'anthropic'

        if os.environ.get('OPENAI_API_KEY') and OPENAI_AVAILABLE:
            return 'openai'

        if os.environ.get('GOOGLE_API_KEY') and GOOGLE_AVAILABLE:
            return 'google'

        if os.environ.get('GROQ_API_KEY') and OPENAI_AVAILABLE:
            return 'groq'

        if (
            os.environ.get('LLM_SERVER_URL')
            and (os.environ.get('LLM_API_KEY') or os.environ.get('EIP_ACCESS_TOKEN'))
            and OPENAI_AVAILABLE
        ):
            return 'openai-compatible'

        # Log warnings for misconfigurations
        self._log_configuration_warnings()
        return None
    
    def _log_configuration_warnings(self):
        """Log warnings for API keys set without corresponding SDKs."""
        if os.environ.get('ANTHROPIC_API_KEY') and not ANTHROPIC_AVAILABLE:
            logger.warning("ANTHROPIC_API_KEY is set but anthropic SDK is not installed. Install with: pip install anthropic")
        
        if os.environ.get('OPENAI_API_KEY') and not OPENAI_AVAILABLE:
            logger.warning("OPENAI_API_KEY is set but openai SDK is not installed. Install with: pip install openai")
        
        if os.environ.get('GOOGLE_API_KEY') and not GOOGLE_AVAILABLE:
            logger.warning("GOOGLE_API_KEY is set but google-generativeai SDK is not installed. Install with: pip install google-generativeai")
        
        if os.environ.get('LLM_SERVER_URL') and not OPENAI_AVAILABLE:
            logger.warning("LLM_SERVER_URL is set but openai SDK is not installed. Install with: pip install openai")
        
        if os.environ.get('GROQ_API_KEY') and not OPENAI_AVAILABLE:
            logger.warning("GROQ_API_KEY is set but openai SDK is not installed. Install with: pip install openai")
    
    def get_anthropic_client(self):
        """Get or create thread-local Anthropic client."""
        if not ANTHROPIC_AVAILABLE:
            return None
        
        if not hasattr(self._local, 'anthropic_client') or self._local.anthropic_client is None:
            api_key = os.environ.get('ANTHROPIC_API_KEY')
            if not api_key:
                return None
            
            self._local.anthropic_client = anthropic.Anthropic(api_key=api_key)
            logger.info("Initialized thread-local Anthropic client")
        
        return self._local.anthropic_client
    
    def get_openai_cloud_client(self):
        """Get or create thread-local OpenAI cloud client."""
        if not OPENAI_AVAILABLE:
            return None
        
        if not hasattr(self._local, 'openai_cloud_client') or self._local.openai_cloud_client is None:
            api_key = os.environ.get('OPENAI_API_KEY')
            if not api_key:
                return None
            
            self._local.openai_cloud_client = OpenAI(api_key=api_key)
            logger.info("Initialized thread-local OpenAI cloud client")
        
        return self._local.openai_cloud_client
    
    def get_openai_compatible_client(self):
        """Get or create thread-local OpenAI-compatible client."""
        if not OPENAI_AVAILABLE:
            return None
        
        if not hasattr(self._local, 'openai_compatible_client') or self._local.openai_compatible_client is None:
            server_url = os.environ.get('LLM_SERVER_URL')
            if not server_url:
                logger.warning("LLM_SERVER_URL not set. OpenAI-compatible backend will be disabled.")
                return None

            # Prefer the EdgeAI-injected JWT when present; falls back to the
            # user-supplied LLM_API_KEY, then to a sentinel for genuinely
            # anonymous local servers (Ollama, LM Studio).
            eip_token = os.environ.get('EIP_ACCESS_TOKEN')
            user_key = os.environ.get('LLM_API_KEY')

            if eip_token and not user_key:
                # EdgeAI proxy lives under /openai of the API base.
                server_url = server_url.rstrip('/')
                if not server_url.endswith('/openai'):
                    server_url = f"{server_url}/openai"
                api_key = eip_token
            else:
                # Ensure URL ends with /v1 for OpenAI compatibility
                if not server_url.endswith('/v1'):
                    server_url = server_url.rstrip('/') + '/v1'
                api_key = user_key or 'not-needed'

            self._local.openai_compatible_client = OpenAI(
                base_url=server_url,
                api_key=api_key
            )
            logger.info(f"Initialized thread-local OpenAI-compatible client: {server_url}")
        
        return self._local.openai_compatible_client
    
    def get_groq_client(self):
        """Get or create thread-local Groq client (uses OpenAI SDK)."""
        if not OPENAI_AVAILABLE:
            return None
        
        if not hasattr(self._local, 'groq_client') or self._local.groq_client is None:
            api_key = os.environ.get('GROQ_API_KEY')
            if not api_key:
                return None
            
            self._local.groq_client = OpenAI(
                base_url="https://api.groq.com/openai/v1",
                api_key=api_key
            )
            logger.info("Initialized thread-local Groq client")
        
        return self._local.groq_client
    
    def get_google_model(self):
        """Get or create thread-local Google Gemini model."""
        if not GOOGLE_AVAILABLE:
            return None
        
        if not hasattr(self._local, 'google_model') or self._local.google_model is None:
            api_key = os.environ.get('GOOGLE_API_KEY')
            if not api_key:
                return None
            
            genai.configure(api_key=api_key)
            model_name = self._get_model_name('google', 'GOOGLE_MODEL')
            self._local.google_model = genai.GenerativeModel(model_name)
            logger.info(f"Initialized thread-local Google Gemini model: {model_name}")
        
        return self._local.google_model
    
    def get_backend_info(self) -> Dict[str, Any]:
        """Get information about the configured LLM backend."""
        backend = self.backend
        
        if not backend:
            return {
                "enabled": False,
                "backend": None,
                "message": "No LLM backend configured"
            }
        
        if backend == 'anthropic':
            model = self._get_model_name('anthropic', 'ANTHROPIC_MODEL')
            return {
                "enabled": True,
                "backend": "anthropic",
                "model": model,
                "message": f"Using Anthropic Claude ({model})"
            }
        elif backend == 'openai':
            model = self._get_model_name('openai', 'OPENAI_MODEL')
            return {
                "enabled": True,
                "backend": "openai",
                "model": model,
                "message": f"Using OpenAI ({model})"
            }
        elif backend == 'google':
            model = self._get_model_name('google', 'GOOGLE_MODEL')
            return {
                "enabled": True,
                "backend": "google",
                "model": model,
                "message": f"Using Google Gemini ({model})"
            }
        elif backend == 'groq':
            model = self._get_model_name('groq', 'GROQ_MODEL')
            return {
                "enabled": True,
                "backend": "groq",
                "model": model,
                "message": f"Using Groq ({model})"
            }
        else:  # openai-compatible
            server_url = os.environ.get('LLM_SERVER_URL', '')
            model = self._get_model_name('openai-compatible', 'LLM_MODEL_NAME')
            return {
                "enabled": True,
                "backend": "openai-compatible",
                "server_url": server_url,
                "model": model,
                "message": f"Using {model} via {server_url}"
            }


# Global manager instance (singleton, thread-safe)
_llm_manager = LLMManager()


# Backend capability flags - describes what each backend reliably supports
BACKEND_CAPABILITIES = {
    'anthropic': {
        'tool_calling': True,
        'tool_choice_required': True,
        'streaming': True,
        'token_counting': True,
        'multimodal': True,
        'max_context': 200000,
        'reliability': 'high'
    },
    'openai': {
        'tool_calling': True,
        'tool_choice_required': True,
        'streaming': True,
        'token_counting': True,
        'multimodal': True,
        'max_context': 128000,
        'reliability': 'high'
    },
    'google': {
        'tool_calling': True,
        'tool_choice_required': False,  # Gemini doesn't support required tool choice
        'streaming': True,
        'token_counting': False,  # Often not returned
        'multimodal': True,
        'max_context': 1000000,
        'reliability': 'medium'
    },
    'groq': {
        'tool_calling': True,
        'tool_choice_required': True,
        'streaming': True,
        'token_counting': True,
        'multimodal': False,  # Groq doesn't support vision yet
        'max_context': 128000,
        'reliability': 'high'
    },
    'openai-compatible': {
        'tool_calling': 'varies',  # Depends on server/model
        'tool_choice_required': False,
        'streaming': True,
        'token_counting': 'varies',
        'multimodal': 'varies',
        'max_context': 'varies',
        'reliability': 'varies'
    }
}


def get_backend_capabilities(backend: Optional[str] = None) -> Dict[str, Any]:
    """
    Get capability flags for the specified or current backend.
    
    Args:
        backend: Backend name, or None to use current backend
        
    Returns:
        Dict of capability flags
    """
    if backend is None:
        backend = _llm_manager.backend
    
    if backend is None:
        return {'enabled': False}
    
    caps = BACKEND_CAPABILITIES.get(backend, {}).copy()
    caps['backend'] = backend
    caps['enabled'] = True
    return caps




__all__ = [
    "OPENAI_AVAILABLE",
    "ANTHROPIC_AVAILABLE",
    "GOOGLE_AVAILABLE",
    "OpenAI",
    "anthropic",
    "genai",
    "DEFAULT_MODELS",
    "LLMManager",
    "_llm_manager",
    "BACKEND_CAPABILITIES",
    "get_backend_capabilities",
]
