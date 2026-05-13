"""
Pytest configuration and shared fixtures for ondevice-eval-agent tests.

All external dependencies (gRPC, HTTP, LLM SDKs) are mocked so tests
run fully offline without any inference or LLM servers.
"""

import io
import os
import sys
import threading
import time
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest
from PIL import Image

# ---------------------------------------------------------------------------
# Path setup — add both project root and webapp/ so all imports resolve.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
_WEBAPP_DIR = os.path.join(_PROJECT_ROOT, "webapp")

for _p in (os.path.abspath(_PROJECT_ROOT), os.path.abspath(_WEBAPP_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ============================================================================
# Environment Isolation
# ============================================================================

_LLM_ENV_VARS = [
    "MODEL_SERVER_URL", "MODEL_SERVER_GRPC_URL", "MODEL_SERVER_METRICS_URL",
    "INFERENCE_BACKEND", "KNOWN_MODELS", "MODEL_NAME",
    "ANTHROPIC_API_KEY", "ANTHROPIC_MODEL",
    "OPENAI_API_KEY", "OPENAI_MODEL",
    "GOOGLE_API_KEY", "GOOGLE_MODEL",
    "GROQ_API_KEY", "GROQ_MODEL",
    "OLLAMA_URL", "OLLAMA_MODEL", "USE_OLLAMA",
    "LLM_SERVER_URL", "LLM_MODEL_NAME", "LLM_API_KEY",
    "EIP_ACCESS_TOKEN",
    "OPENAI_API_BASE_URLS",
    "LLM_SUPPORTS_TOOLS", "LLM_PROVIDERS",
    "FLASK_DEBUG",
    "LLM_MAX_RETRIES", "LLM_BACKOFF_BASE", "LLM_MAX_CONCURRENCY",
]


@pytest.fixture()
def clean_env(monkeypatch):
    """Remove all inference / LLM env vars so tests start from a known state."""
    for var in _LLM_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    return monkeypatch


# ============================================================================
# Sample Images
# ============================================================================

def _make_png_bytes(width: int = 4, height: int = 4) -> bytes:
    """Create a minimal RGB PNG in memory."""
    img = Image.new("RGB", (width, height), color=(255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture()
def sample_image_bytes():
    """4x4 red-pixel PNG as raw bytes."""
    return _make_png_bytes()


@pytest.fixture()
def sample_image_path(tmp_path):
    """4x4 red-pixel PNG written to a temporary file."""
    path = tmp_path / "test_image.png"
    path.write_bytes(_make_png_bytes())
    return str(path)


# ============================================================================
# Mock gRPC helpers
# ============================================================================

def _make_mock_output(name: str = "output0", shape=(1, 1000), datatype: str = "FP32"):
    out = MagicMock()
    out.name = name
    out.shape = list(shape)
    out.datatype = datatype
    return out


def _make_mock_input(name: str = "images", shape=(1, 3, 224, 224), datatype: str = "FP32"):
    inp = MagicMock()
    inp.name = name
    inp.shape = list(shape)
    inp.datatype = datatype
    return inp


def make_grpc_metadata(
    name: str = "test_model",
    inputs: Optional[List[Dict]] = None,
    outputs: Optional[List[Dict]] = None,
    platform: str = "onnxruntime_onnx",
    versions: Optional[List[str]] = None,
):
    """Build a mock gRPC ModelMetadataResponse-like object."""
    if inputs is None:
        inputs = [{"name": "images", "shape": [1, 3, 224, 224], "datatype": "FP32"}]
    if outputs is None:
        outputs = [{"name": "output0", "shape": [1, 1000], "datatype": "FP32"}]

    meta = MagicMock()
    meta.name = name
    meta.platform = platform
    meta.versions = versions or ["1"]
    meta.inputs = [_make_mock_input(**i) for i in inputs]
    meta.outputs = [_make_mock_output(**o) for o in outputs]
    return meta


def make_inference_response(
    model_name: str = "test_model",
    outputs: Optional[List[Dict]] = None,
) -> Dict[str, Any]:
    """Build the dict format produced by InferenceRunner._grpc_result_to_dict."""
    if outputs is None:
        outputs = [{
            "name": "output0",
            "shape": [1, 1000],
            "datatype": "FP32",
            "data": np.random.randn(1000).tolist(),
        }]
    return {"model_name": model_name, "outputs": outputs}


# ============================================================================
# Mock gRPC Client
# ============================================================================

@pytest.fixture()
def mock_grpc_client():
    """MagicMock mimicking tritonclient.grpc.InferenceServerClient."""
    client = MagicMock()
    client.is_server_live.return_value = True
    client.is_server_ready.return_value = True

    # Server metadata
    server_meta = MagicMock()
    server_meta.name = "triton"
    server_meta.version = "2.40.0"
    server_meta.extensions = ["classification", "model_repository"]
    client.get_server_metadata.return_value = server_meta

    # Model metadata
    client.get_model_metadata.return_value = make_grpc_metadata()

    # Model config
    cfg = MagicMock()
    cfg.config = MagicMock()
    cfg.config.name = "test_model"
    cfg.config.platform = "onnxruntime_onnx"
    cfg.config.backend = "onnxruntime"
    cfg.config.max_batch_size = 1
    client.get_model_config.return_value = cfg

    # Model readiness
    client.is_model_ready.return_value = True

    # Repository index
    repo_entry = MagicMock()
    repo_entry.name = "test_model"
    repo_entry.version = "1"
    repo_entry.state = "READY"
    repo_entry.reason = ""
    client.get_model_repository_index.return_value = [repo_entry]

    # Inference result
    infer_result = MagicMock()
    response_obj = MagicMock()
    out_meta = MagicMock()
    out_meta.name = "output0"
    out_meta.datatype = "FP32"
    response_obj.outputs = [out_meta]
    infer_result.get_response.return_value = response_obj
    infer_result.as_numpy.return_value = np.random.randn(1, 1000).astype(np.float32)
    client.infer.return_value = infer_result

    client.close.return_value = None
    return client


# ============================================================================
# Mock ModelServerClient
# ============================================================================

@pytest.fixture()
def mock_model_client(mock_grpc_client, monkeypatch):
    """A real ModelServerClient wired to the mock gRPC client."""
    monkeypatch.setattr(
        "client.client.create_grpc_client",
        lambda *a, **kw: mock_grpc_client,
    )
    monkeypatch.setattr(
        "client.client.create_session",
        lambda *a, **kw: MagicMock(),
    )
    from client import ModelServerClient
    return ModelServerClient(test_connectivity=False)


# ============================================================================
# Flask Test Client
# ============================================================================

@pytest.fixture()
def flask_test_app(mock_model_client, monkeypatch, tmp_path):
    """Create a Flask app with all blueprints, wired to mock client."""
    from flask import Flask

    webapp_dir = os.path.abspath(_WEBAPP_DIR)
    app = Flask(
        __name__,
        static_folder=os.path.join(webapp_dir, "static"),
        template_folder=os.path.join(webapp_dir, "templates"),
    )
    app.config["TESTING"] = True
    app.config["UPLOAD_FOLDER"] = str(tmp_path / "uploads")
    app.config["ALLOWED_EXTENSIONS"] = {"png", "jpg", "jpeg", "gif", "bmp", "webp"}
    app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

    test_config = {
        "title": "Test",
        "description": "Test",
        "logo_url": "",
        "primary_color": "#333",
        "upload_folder": app.config["UPLOAD_FOLDER"],
        "max_content_mb": 16,
        "allowed_extensions": app.config["ALLOWED_EXTENSIONS"],
        "max_log_entries": 10,
    }

    from api import core_bp, agent_bp, llm_bp
    from api.core import init_core_routes

    init_core_routes(test_config, mock_model_client)
    app.register_blueprint(core_bp)
    app.register_blueprint(agent_bp)
    app.register_blueprint(llm_bp)

    from observability.logging import init_log_queues
    init_log_queues(10)

    return app


@pytest.fixture()
def flask_test_client(flask_test_app):
    """A Flask test client ready for request assertions."""
    return flask_test_app.test_client()


# ============================================================================
# Router Reset
# ============================================================================

@pytest.fixture()
def reset_router(clean_env):
    """Reset the AgentLLMRouter singleton so each test gets a fresh instance."""
    from router.llm_router import AgentLLMRouter
    AgentLLMRouter._instance = None
    AgentLLMRouter._lock = threading.Lock()
    yield
    AgentLLMRouter._instance = None
    AgentLLMRouter._lock = threading.Lock()


# ============================================================================
# Rate-limit Config Reset
# ============================================================================

@pytest.fixture(autouse=False)
def reset_rate_limit_config():
    """Reset global rate-limit singletons so tests don't leak state."""
    from router.rate_limit_config import reset_config
    from router.resilience import reset_resilience_stats
    reset_config()
    reset_resilience_stats()
    yield
    reset_config()
    reset_resilience_stats()
