"""LLM Router API routes."""

import logging

from flask import Blueprint, jsonify, request

from utils.errors import (
    BadRequestError,
    create_success_response,
    handle_exceptions,
    validate_request_json,
)

logger = logging.getLogger(__name__)

# Create Blueprint
llm_bp = Blueprint('llm', __name__, url_prefix='/llm')


@llm_bp.route('/providers', methods=['GET'])
@handle_exceptions("Failed to list LLM providers")
def llm_list_providers():
    """
    List all registered LLM providers with their status.
    
    Response:
        {
            "success": true,
            "providers": [...],
            "count": 3,
            "active_provider": {...}
        }
    """
    from router import get_router
    router = get_router()
    providers = router.list_providers()
    active = router.get_active_provider()
    
    return jsonify(create_success_response({
        "providers": providers,
        "count": len(providers),
        "active_provider": active
    }))


@llm_bp.route('/providers', methods=['POST'])
@handle_exceptions("Failed to register LLM provider")
def llm_register_provider():
    """
    Register a new LLM provider.
    
    Request body:
        {
            "name": "ollama-local",
            "provider_type": "ollama",
            "url": "http://localhost:11434",
            "model": "llama3.2",
            "priority": 1
        }
    
    Response:
        {
            "success": true,
            "registered": true,
            "provider_name": "ollama-local"
        }
    """
    from router import get_router, LLMProviderConfig, detect_provider_type_from_url

    # `provider_type` is only strictly required when `auto` would be ambiguous.
    # We accept the sentinel "auto" (or a missing field when a `url` is given)
    # and pick the best adapter from the URL host — Anthropic for
    # api.anthropic.com, OpenAI for api.openai.com, etc., falling back to
    # openai-compatible for everything else (Groq, vLLM, LM Studio, proxies).
    data = validate_request_json(required_fields=["name"])

    raw_type = (data.get("provider_type") or "auto").lower()
    if raw_type == "auto":
        provider_type = detect_provider_type_from_url(data.get("url"))
    else:
        provider_type = raw_type

    config = LLMProviderConfig(
        name=data["name"],
        provider_type=provider_type,
        url=data.get("url"),
        model=data.get("model"),
        api_key=data.get("api_key"),
        priority=data.get("priority", 10),
        max_tokens=data.get("max_tokens", 4096),
        temperature=data.get("temperature", 0.7),
        enabled=data.get("enabled", True),
        supports_tools=data.get("supports_tools", True),
        supports_vision=data.get("supports_vision", False),
    )
    
    router = get_router()
    success = router.register_provider(config)
    
    return jsonify(create_success_response({
        "registered": success,
        "provider_name": config.name
    }))


@llm_bp.route('/providers/<name>', methods=['PATCH'])
@handle_exceptions("Failed to update LLM provider")
def llm_update_provider(name):
    """
    Update an existing LLM provider's configuration.
    
    Request body (all fields optional):
        {
            "model": "qwen/qwen3-vl-8b",
            "url": "http://localhost:1234",
            "priority": 1,
            "enabled": true
        }
    
    Response:
        {
            "success": true,
            "updated": true,
            "provider_name": "lmstudio"
        }
    """
    from router import get_router
    
    router = get_router()
    existing = router.get_provider(name)
    
    if not existing:
        from utils.errors import NotFoundError
        raise NotFoundError(f"Provider '{name}' not found")
    
    data = request.get_json() or {}
    
    # Update only provided fields
    from router import LLMProviderConfig
    config = LLMProviderConfig(
        name=name,
        provider_type=existing.provider_type,
        url=data.get("url", existing.url),
        model=data.get("model", existing.model),
        api_key=data.get("api_key", existing.api_key),
        priority=data.get("priority", existing.priority),
        max_tokens=data.get("max_tokens", existing.max_tokens),
        temperature=data.get("temperature", existing.temperature),
        enabled=data.get("enabled", existing.enabled),
        supports_tools=data.get("supports_tools", existing.supports_tools),
        supports_vision=data.get("supports_vision", existing.supports_vision),
    )
    
    success = router.register_provider(config)
    
    return jsonify(create_success_response({
        "updated": success,
        "provider_name": name
    }))


@llm_bp.route('/providers/<name>', methods=['DELETE'])
@handle_exceptions("Failed to unregister LLM provider")
def llm_unregister_provider(name):
    """
    Unregister an LLM provider.
    
    Response:
        {
            "success": true,
            "unregistered": true,
            "provider_name": "ollama-local"
        }
    """
    from router import get_router
    router = get_router()
    success = router.unregister_provider(name)
    
    return jsonify(create_success_response({
        "unregistered": success,
        "provider_name": name
    }))


@llm_bp.route('/providers/check', methods=['POST'])
@handle_exceptions("Failed to check LLM providers")
def llm_check_providers():
    """
    Re-check availability of all registered LLM providers.
    
    Useful for refreshing connection status after network changes
    or when LLM servers come online.
    
    Response:
        {
            "success": true,
            "providers": {"ollama-local": true, "openai": false},
            "available": 1,
            "unavailable": 1
        }
    """
    from router import get_router
    router = get_router()
    health_results = router.check_all_providers()
    
    available = sum(1 for v in health_results.values() if v)
    unavailable = len(health_results) - available
    
    return jsonify(create_success_response({
        "providers": health_results,
        "available": available,
        "unavailable": unavailable
    }))


@llm_bp.route('/health', methods=['GET'])
@handle_exceptions("Failed to check LLM provider health")
def llm_check_health():
    """
    Check health of all LLM providers.
    
    Response:
        {
            "success": true,
            "providers": {...},
            "available": 2,
            "unavailable": 1
        }
    """
    from router import get_router
    router = get_router()
    health_results = router.check_all_providers()
    
    available = sum(1 for v in health_results.values() if v)
    unavailable = len(health_results) - available
    
    return jsonify(create_success_response({
        "providers": health_results,
        "available": available,
        "unavailable": unavailable
    }))


@llm_bp.route('/resilience', methods=['GET'])
@handle_exceptions("Failed to get resilience stats")
def llm_resilience_stats():
    """
    Get rate limit resilience statistics.
    
    Returns information about:
    - Concurrency limiter (active/waiting requests, max concurrent)
    - Request deduplication (cache size, dedup rate)
    - Rate limit configuration
    
    Response:
        {
            "success": true,
            "concurrency": {
                "max_concurrent": 2,
                "active_requests": 0,
                "waiting_requests": 0,
                "total_acquired": 100,
                "total_waited": 5,
                "max_wait_time": 1.5
            },
            "deduplication": {
                "cache_size": 10,
                "window_seconds": 5.0,
                "total_requests": 100,
                "deduplicated": 5,
                "dedup_rate": 0.05
            },
            "config": {
                "max_retries": 5,
                "backoff_base": 2.0,
                "backoff_max": 30.0,
                "max_concurrency": 2,
                "max_prompt_tokens": 100000,
                ...
            }
        }
    """
    try:
        from router.resilience import get_resilience_stats
        stats = get_resilience_stats()
        return jsonify(create_success_response(stats))
    except ImportError:
        return jsonify(create_success_response({
            "error": "Resilience module not available",
            "message": "Rate limit resilience features are not installed"
        }))


@llm_bp.route('/chat', methods=['POST'])
@handle_exceptions("Failed to process LLM chat request")
def llm_chat():
    """
    Send a chat request through the LLM router.
    
    Request body:
        {
            "messages": [
                {"role": "user", "content": "Hello!"}
            ],
            "provider": "ollama-local" (optional)
        }
    
    Response:
        {
            "success": true,
            "response": {...}
        }
    """
    from router import get_router
    
    data = validate_request_json(required_fields=["messages"])
    
    router = get_router()
    response = router.chat(
        messages=data["messages"],
        tools=data.get("tools"),
        provider_name=data.get("provider")
    )
    
    return jsonify(create_success_response({
        "response": response.to_dict()
    }))


@llm_bp.route('/strategy', methods=['PUT'])
@handle_exceptions("Failed to set LLM routing strategy")
def llm_set_strategy():
    """
    Set the LLM routing strategy.
    
    Request body:
        {
            "strategy": "round_robin"
        }
    
    Valid strategies: priority, round_robin, failover, latency, cost
    
    Response:
        {
            "success": true,
            "new_strategy": "round_robin"
        }
    """
    from router import get_router, RoutingStrategy
    
    data = validate_request_json(required_fields=["strategy"])
    
    try:
        strategy = RoutingStrategy(data["strategy"])
    except ValueError:
        valid = [s.value for s in RoutingStrategy]
        raise BadRequestError(
            f"Invalid strategy. Valid options: {valid}",
            details={"valid_strategies": valid}
        )
    
    router = get_router()
    router.set_routing_strategy(strategy)
    
    return jsonify(create_success_response({
        "new_strategy": strategy.value
    }))


@llm_bp.route('/models/fetch', methods=['POST'])
@handle_exceptions("Failed to fetch models")
def fetch_models():
    """
    Fetch available models from a provider without registering it.
    Used for populating model dropdowns in the UI.
    
    Request:
        {
            "provider_type": "groq",
            "api_key": "gsk_...",
            "url": null
        }
    
    Response:
        {
            "success": true,
            "models": ["llama-3.3-70b-versatile", "mixtral-8x7b-32768", ...]
        }
    """
    from router import get_router
    from router.config import LLMProviderConfig, LLMProviderType
    from router.adapters import OpenAICompatibleAdapter, OllamaAdapter
    
    data = request.get_json() or {}
    provider_type_str = data.get('provider_type', 'openai-compatible')
    api_key = data.get('api_key')
    url = data.get('url')
    
    try:
        provider_type = LLMProviderType(provider_type_str)
    except ValueError:
        provider_type = LLMProviderType.OPENAI_COMPATIBLE
    
    # Create a temporary config for fetching models
    temp_config = LLMProviderConfig(
        name="_temp_fetch",
        provider_type=provider_type,
        api_key=api_key,
        url=url,
        model="temp"
    )
    
    # Get the appropriate adapter
    router = get_router()
    adapter = router._get_adapter(provider_type)
    
    # First check availability
    available, latency, error = adapter.check_availability(temp_config)
    
    if not available:
        return jsonify(create_success_response({
            "models": [],
            "error": error or "Provider not available"
        }))
    
    # Fetch models
    try:
        models = adapter.list_models(temp_config)
        return jsonify(create_success_response({
            "models": models
        }))
    except Exception as e:
        logger.warning(f"Failed to list models: {e}")
        return jsonify(create_success_response({
            "models": [],
            "error": str(e)
        }))


@llm_bp.route('/status', methods=['GET'])
@handle_exceptions("Failed to get LLM router status")
def llm_router_status():
    """
    Get comprehensive LLM router status.
    
    Response:
        {
            "success": true,
            "routing_strategy": "failover",
            "providers": [...],
            "active_provider": {...}
        }
    """
    from router import get_router
    router = get_router()
    
    return jsonify(create_success_response(router.to_dict()))


# ============================================================================
# Secure Credential Storage Endpoints
# ============================================================================

@llm_bp.route('/credentials', methods=['GET'])
@handle_exceptions("Failed to list stored credentials")
def list_credentials():
    """
    List all stored credentials (without exposing sensitive data).
    
    Response:
        {
            "success": true,
            "credentials": [
                {
                    "name": "openai-prod",
                    "provider_type": "openai",
                    "has_api_key": true,
                    "created_at": "2024-01-15T10:30:00",
                    "updated_at": "2024-01-15T10:30:00"
                }
            ],
            "count": 1
        }
    """
    from storage import get_secure_storage
    
    storage = get_secure_storage()
    # list_credentials returns dicts when include_keys=False
    credentials = storage.list_credentials(include_keys=False)
    
    return jsonify(create_success_response({
        "credentials": credentials,
        "count": len(credentials)
    }))


@llm_bp.route('/credentials', methods=['POST'])
@handle_exceptions("Failed to store credential")
def store_credential():
    """
    Store a new LLM credential securely.
    
    Request body:
        {
            "name": "openai-prod",
            "provider_type": "openai",
            "api_key": "sk-...",
            "url": "https://api.openai.com/v1",
            "model": "gpt-4",
            "priority": 10,
            "max_tokens": 4096,
            "temperature": 0.7,
            "supports_tools": true,
            "supports_vision": false
        }
    
    Response:
        {
            "success": true,
            "stored": true,
            "credential_name": "openai-prod"
        }
    """
    from storage import get_secure_storage, StoredCredential
    
    data = validate_request_json(required_fields=["name", "provider_type"])
    
    credential = StoredCredential(
        name=data["name"],
        provider_type=data["provider_type"],
        api_key=data.get("api_key"),
        url=data.get("url"),
        model=data.get("model"),
        priority=data.get("priority", 10),
        max_tokens=data.get("max_tokens", 4096),
        temperature=data.get("temperature", 0.7),
        enabled=data.get("enabled", True),
        supports_tools=data.get("supports_tools", True),
        supports_vision=data.get("supports_vision", False),
        metadata=data.get("metadata", {}),
    )
    
    storage = get_secure_storage()
    success = storage.save_credential(credential)
    
    logger.info(f"Stored credential: {credential.name}")
    
    return jsonify(create_success_response({
        "stored": success,
        "credential_name": credential.name
    }))


@llm_bp.route('/credentials/<name>', methods=['GET'])
@handle_exceptions("Failed to retrieve credential")
def get_credential(name: str):
    """
    Retrieve a stored credential by name.
    
    Note: API key is partially masked for security.
    
    Response:
        {
            "success": true,
            "credential": {
                "name": "openai-prod",
                "provider_type": "openai",
                "api_key_masked": "sk-...abc",
                "url": "https://api.openai.com/v1",
                "model": "gpt-4"
            }
        }
    """
    from storage import get_secure_storage
    from utils.errors import NotFoundError
    
    storage = get_secure_storage()
    credential = storage.get_credential(name)
    
    if credential is None:
        raise NotFoundError(f"Credential '{name}' not found")
    
    # Mask API key for response
    masked_key = None
    if credential.api_key:
        key = credential.api_key
        if len(key) > 8:
            masked_key = f"{key[:4]}...{key[-4:]}"
        else:
            masked_key = "****"
    
    return jsonify(create_success_response({
        "credential": {
            "name": credential.name,
            "provider_type": credential.provider_type,
            "api_key_masked": masked_key,
            "url": credential.url,
            "model": credential.model,
            "priority": credential.priority,
            "max_tokens": credential.max_tokens,
            "temperature": credential.temperature,
            "enabled": credential.enabled,
            "supports_tools": credential.supports_tools,
            "supports_vision": credential.supports_vision,
            "metadata": credential.metadata,
            "created_at": credential.created_at,
            "updated_at": credential.updated_at,
        }
    }))


@llm_bp.route('/credentials/<name>', methods=['DELETE'])
@handle_exceptions("Failed to delete credential")
def delete_credential(name: str):
    """
    Delete a stored credential.
    
    Response:
        {
            "success": true,
            "deleted": true,
            "credential_name": "openai-prod"
        }
    """
    from storage import get_secure_storage
    
    storage = get_secure_storage()
    success = storage.delete_credential(name)
    
    if success:
        logger.info(f"Deleted credential: {name}")
    
    return jsonify(create_success_response({
        "deleted": success,
        "credential_name": name
    }))


@llm_bp.route('/credentials/<name>/activate', methods=['POST'])
@handle_exceptions("Failed to activate credential")
def activate_credential(name: str):
    """
    Activate a stored credential by registering it as an LLM provider.
    
    This loads the credential from secure storage and registers it
    with the LLM router for immediate use. The activated credential
    becomes the highest priority (default) provider.
    
    Response:
        {
            "success": true,
            "activated": true,
            "provider_name": "openai-prod"
        }
    """
    from router import get_router, LLMProviderConfig
    from storage import get_secure_storage
    from utils.errors import NotFoundError
    
    storage = get_secure_storage()
    credential = storage.get_credential(name)
    
    if credential is None:
        raise NotFoundError(f"Credential '{name}' not found")
    
    router = get_router()
    
    # Give the newly activated credential the highest priority (0)
    # This makes it the default provider
    priority = 0
    
    # Build provider config from credential
    config = LLMProviderConfig(
        name=credential.name,
        provider_type=credential.provider_type,
        url=credential.url,
        model=credential.model,
        api_key=credential.api_key,
        priority=priority,
        max_tokens=credential.max_tokens,
        temperature=credential.temperature,
        enabled=True,
        supports_tools=credential.supports_tools,
        supports_vision=credential.supports_vision,
    )
    
    success = router.register_provider(config)
    
    logger.info(f"Activated credential as provider: {name} (priority={priority})")
    
    return jsonify(create_success_response({
        "activated": success,
        "provider_name": name
    }))


@llm_bp.route('/credentials/export', methods=['POST'])
@handle_exceptions("Failed to export credentials")
def export_credentials():
    """
    Export all credentials as a JSON bundle.

    API keys are included so the bundle can be restored on another
    machine (the on-disk store is encrypted with a machine-derived
    key, so a keyless export is not portable).

    Response:
        {
            "success": true,
            "bundle": {...},
            "credential_count": 3
        }
    """
    from storage import get_secure_storage

    storage = get_secure_storage()
    bundle = storage.export_credentials(include_keys=True)
    credentials = bundle.get("credentials", [])
    key_count = sum(1 for c in credentials if c.get("api_key"))

    return jsonify(create_success_response({
        "bundle": bundle,
        "credential_count": len(credentials),
        "contains_secrets": key_count > 0,
        "warning": (
            f"This export contains {key_count} plaintext API key(s). "
            "Store the file securely and do not share or commit it."
            if key_count > 0 else None
        ),
    }))


@llm_bp.route('/credentials/import', methods=['POST'])
@handle_exceptions("Failed to import credentials")
def import_credentials():
    """
    Import credentials from an encrypted bundle.
    
    Request body:
        {
            "bundle": "base64-encoded-encrypted-data",
            "password": "custom-encryption-password" (optional),
            "overwrite": false (optional)
        }
    
    Response:
        {
            "success": true,
            "imported_count": 3
        }
    """
    from storage import get_secure_storage
    
    data = validate_request_json(required_fields=["bundle"])
    
    storage = get_secure_storage()
    results = storage.import_credentials(
        data=data["bundle"],
        overwrite=data.get("overwrite", False)
    )
    
    logger.info(f"Imported credentials: {results}")
    
    return jsonify(create_success_response({
        "imported_count": results.get('imported', 0),
        "skipped_count": results.get('skipped', 0),
        "error_count": results.get('errors', 0)
    }))


@llm_bp.route('/credentials/activate-all', methods=['POST'])
@handle_exceptions("Failed to activate all credentials")
def activate_all_credentials():
    """
    Activate all stored credentials by registering them as LLM providers.
    
    Response:
        {
            "success": true,
            "activated": ["openai-prod", "anthropic-main"],
            "failed": [],
            "total": 2
        }
    """
    from router import get_router, LLMProviderConfig
    from storage import get_secure_storage
    
    storage = get_secure_storage()
    router = get_router()
    
    activated = []
    failed = []
    
    # get_all_enabled returns StoredCredential objects
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
            else:
                failed.append(credential.name)
        except Exception as e:
            logger.warning(f"Failed to activate {credential.name}: {e}")
            failed.append(credential.name)
    
    logger.info(f"Activated {len(activated)} credentials, {len(failed)} failed")
    
    return jsonify(create_success_response({
        "activated": activated,
        "failed": failed,
        "total": len(activated)
    }))


# =============================================================================
# Token Usage Tracking
# =============================================================================

@llm_bp.route('/usage', methods=['GET'])
@handle_exceptions("Failed to get token usage")
def llm_get_usage():
    """
    Get token usage statistics.
    
    Query params:
        provider: Optional provider name to filter by
    
    Response:
        {
            "success": true,
            "usage": {
                "by_provider": {
                    "google/gemini-2.0-flash": {
                        "prompt_tokens": 1234,
                        "completion_tokens": 567,
                        "total_tokens": 1801,
                        "request_count": 5
                    }
                },
                "totals": {
                    "prompt_tokens": 1234,
                    "completion_tokens": 567,
                    "total_tokens": 1801,
                    "request_count": 5
                }
            }
        }
    """
    from router import get_token_usage
    
    usage = get_token_usage()
    
    return jsonify(create_success_response({
        "usage": usage
    }))


@llm_bp.route('/usage/reset', methods=['POST'])
@handle_exceptions("Failed to reset token usage")
def llm_reset_usage():
    """
    Reset token usage statistics.
    
    Response:
        {
            "success": true,
            "reset": true
        }
    """
    from router import reset_token_usage
    
    reset_token_usage()
    
    return jsonify(create_success_response({
        "reset": True
    }))
