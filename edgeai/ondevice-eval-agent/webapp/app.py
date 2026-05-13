#!/usr/bin/env python3
"""
ZEDEDA On-Device AI Agent - Web Application
Flask web app that provides a web interface for ML model predictions.
Automatically detects model type (classification, detection, segmentation)
and displays appropriate results with raw tensor information.

This file serves as the main entry point that imports and registers
all route blueprints from modular components.
"""

import os
import sys
import logging

from flask import Flask

# Add parent directory to path to import client package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Add webapp directory to path so modules can use absolute imports (e.g., from processing import ...)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__, static_folder='static', static_url_path='/static')

# Application configuration from environment variables
APP_CONFIG = {
    'title': os.environ.get('APP_TITLE', 'OnDevice Eval Agent'),
    'description': os.environ.get('APP_DESCRIPTION', 'ZEDEDA\'s ML Model Inference and Evaluation Interface Agent'),
    'logo_url': os.environ.get('LOGO_URL', ''),
    'primary_color': os.environ.get('PRIMARY_COLOR', '#3498db'),
    'upload_folder': os.environ.get('UPLOAD_FOLDER', '/tmp/uploads/'),
    'max_content_mb': int(os.environ.get('MAX_CONTENT_MB', '16') or '16'),
    'allowed_extensions': set(os.environ.get('ALLOWED_EXTENSIONS', 'png,jpg,jpeg,gif,bmp,webp').split(',')),
    'max_log_entries': int(os.environ.get('MAX_LOG_ENTRIES', '100') or '100'),
}

# Configure Flask app
app.config['UPLOAD_FOLDER'] = APP_CONFIG['upload_folder']
app.config['ALLOWED_EXTENSIONS'] = APP_CONFIG['allowed_extensions']
app.config['MAX_CONTENT_LENGTH'] = APP_CONFIG['max_content_mb'] * 1024 * 1024

# Create upload folder
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'], mode=0o700)

# Import inference client directly from the client package
from client import ModelServerClient

# Initialize the model server client
model_client = ModelServerClient()

# Initialize log queues
from observability.logging import init_log_queues
init_log_queues(APP_CONFIG.get('max_log_entries', 100))

# Import and register blueprints
from api import core_bp, agent_bp, llm_bp, eval_bp, metrics_bp
from api.core import init_core_routes

# Initialize routes with app config and client
init_core_routes(APP_CONFIG, model_client)

# Register blueprints
app.register_blueprint(core_bp)
app.register_blueprint(agent_bp)
app.register_blueprint(llm_bp)
app.register_blueprint(eval_bp)
app.register_blueprint(metrics_bp)

# Auto-activate saved LLM credentials on startup
def _auto_activate_credentials():
    """Automatically activate all enabled credentials on startup."""
    try:
        from storage import get_secure_storage
        from router import get_router, LLMProviderConfig
        
        storage = get_secure_storage()
        router = get_router()
        
        activated = []
        for credential in storage.get_all_enabled():
            try:
                config = LLMProviderConfig(
                    name=credential.name,
                    provider_type=credential.provider_type,
                    url=credential.url,
                    model=credential.model,
                    api_key=credential.api_key,
                    priority=credential.priority,
                    max_tokens=credential.max_tokens,
                    temperature=credential.temperature,
                    enabled=True,
                    supports_tools=credential.supports_tools,
                    supports_vision=credential.supports_vision,
                )
                if router.register_provider(config):
                    activated.append(credential.name)
            except Exception as e:
                logger.warning(f"Failed to activate credential {credential.name}: {e}")
        
        if activated:
            logger.info(f"Auto-activated {len(activated)} LLM credential(s): {', '.join(activated)}")
    except Exception as e:
        logger.warning(f"Could not auto-activate credentials: {e}")

_auto_activate_credentials()

# Kick off per-deployment bootstrap: auto-baseline, scheduled sanity evals,
# and Prometheus identity publishing. Runs in a daemon thread so an
# unavailable Triton never blocks Flask from coming up.
try:
    from deployment import start_bootstrap
    start_bootstrap()
except Exception as e:
    logger.warning(f"Could not start deployment bootstrap: {e}")

# Log startup info
logger.info(f"  ZEDEDA On-Device AI Agent Web Application")
logger.info(f"  Server URL: {model_client.server_url}")
logger.info(f"  Upload folder: {APP_CONFIG['upload_folder']}")
logger.info(f"  Max content size: {APP_CONFIG['max_content_mb']} MB")


if __name__ == '__main__':
    logger.info("Starting ZEDEDA On-Device AI Agent Web Application...")
    debug_env = os.getenv("FLASK_DEBUG", "").lower()
    debug_mode = debug_env in ("1", "true", "yes", "on")
    logger.info(f"Flask debug mode is {'ENABLED' if debug_mode else 'DISABLED'} (FLASK_DEBUG={debug_env!r})")
    app.run(host='0.0.0.0', port=8080, debug=debug_mode)
