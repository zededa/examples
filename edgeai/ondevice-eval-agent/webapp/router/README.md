# Agent LLM Router

The Agent LLM Router enables users to interact with the AI agent regardless of which LLM service they're running. It provides a unified interface for routing chat requests to different LLM backends with automatic failover, health monitoring, and multiple routing strategies.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Agent LLM Router                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
│  │  Registry   │  │  Selector   │  │  Health Monitor        │ │
│  │ (providers) │◄─┤  (routing)  │◄─┤  (availability check)  │ │
│  └──────┬──────┘  └─────────────┘  └─────────────────────────┘ │
│         │                                                       │
│  ┌──────▼──────────────────────────────────────────────────┐   │
│  │                    LLM Adapters                          │   │
│  │  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ │   │
│  │  │Anthropic│ │ OpenAI │ │ Ollama │ │  vLLM  │ │  TGI   │ │   │
│  │  └────────┘ └────────┘ └────────┘ └────────┘ └────────┘ │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## Supported LLM Providers

| Provider | Type | Description |
|----------|------|-------------|
| **Anthropic** | Cloud API | Claude models (claude-sonnet-4-20250514, etc.) |
| **OpenAI** | Cloud API | GPT models (gpt-4o, gpt-4-turbo, etc.) |
| **Google** | Cloud API | Gemini models (gemini-1.5-pro, etc.) |
| **Ollama** | Local | Run open-source LLMs locally |
| **vLLM** | Self-hosted | High-throughput LLM serving |
| **TGI** | Self-hosted | Hugging Face Text Generation Inference |
| **LM Studio** | Local | Desktop app for running LLMs |
| **OpenAI-Compatible** | Any | Any API following OpenAI's format |

## Quick Start

### 1. Environment Variables (Auto-Discovery)

The router automatically discovers providers from environment variables:

```bash
# Cloud APIs (set API keys)
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENAI_API_KEY="sk-..."
export GOOGLE_API_KEY="..."

# Local/Self-hosted (set URLs)
export OLLAMA_URL="http://localhost:11434"
export OLLAMA_MODEL="llama3.2"

export LLM_SERVER_URL="http://localhost:8000/v1"
export LLM_MODEL_NAME="my-model"
```

### 2. Programmatic Registration

```python
from webapp.router import AgentLLMRouter, LLMProviderConfig, LLMProviderType

router = AgentLLMRouter()

# Register Ollama
router.register_provider(LLMProviderConfig(
    name="ollama-local",
    provider_type=LLMProviderType.OLLAMA,
    url="http://localhost:11434",
    model="llama3.2",
    priority=1,
    supports_tools=True
))

# Register vLLM
router.register_provider(LLMProviderConfig(
    name="vllm-server",
    provider_type=LLMProviderType.VLLM,
    url="http://gpu-server:8000",
    model="meta-llama/Llama-3.2-8B-Instruct",
    priority=2
))
```

### 3. Send Chat Requests

```python
# Chat through the router (automatic provider selection)
response = router.chat(messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
])

print(f"Response from {response.provider}: {response.content}")

# Chat with specific provider
response = router.chat(
    messages=[{"role": "user", "content": "What's 2+2?"}],
    provider_name="ollama-local"
)

# Chat with function calling
response = router.chat(
    messages=[{"role": "user", "content": "What's the weather?"}],
    tools=[{
        "name": "get_weather",
        "description": "Get current weather",
        "input_schema": {
            "type": "object",
            "properties": {
                "location": {"type": "string"}
            }
        }
    }]
)
```

## REST API Endpoints

### List Providers

```bash
GET /llm/providers

# Response
{
    "success": true,
    "providers": [
        {
            "name": "anthropic",
            "provider_type": "anthropic",
            "model": "claude-sonnet-4-20250514",
            "priority": 1,
            "enabled": true,
            "status": {
                "available": true,
                "latency_ms": 0.0
            }
        }
    ],
    "count": 3,
    "active_provider": {...}
}
```

### Register Provider

```bash
POST /llm/providers
Content-Type: application/json

{
    "name": "ollama-local",
    "provider_type": "ollama",
    "url": "http://localhost:11434",
    "model": "llama3.2",
    "priority": 1,
    "supports_tools": true
}

# Response
{
    "success": true,
    "registered": true,
    "provider_name": "ollama-local"
}
```

### Unregister Provider

```bash
DELETE /llm/providers/{name}

# Response
{
    "success": true,
    "unregistered": true,
    "provider_name": "ollama-local"
}
```

### Check Health

```bash
GET /llm/health

# Response
{
    "success": true,
    "providers": {
        "anthropic": true,
        "ollama-local": false,
        "vllm-server": true
    },
    "available": 2,
    "unavailable": 1
}
```

### Chat

```bash
POST /llm/chat
Content-Type: application/json

{
    "messages": [
        {"role": "user", "content": "Hello!"}
    ],
    "provider": "ollama-local"  // optional
}

# Response
{
    "success": true,
    "response": {
        "content": "Hello! How can I help you today?",
        "provider": "ollama-local",
        "model": "llama3.2",
        "usage": {
            "prompt_tokens": 5,
            "completion_tokens": 10
        }
    }
}
```

### Set Routing Strategy

```bash
PUT /llm/strategy
Content-Type: application/json

{
    "strategy": "round_robin"
}

# Response
{
    "success": true,
    "new_strategy": "round_robin"
}
```

### Get Router Status

```bash
GET /llm/status

# Response
{
    "success": true,
    "routing_strategy": "failover",
    "providers": [...],
    "active_provider": {...}
}
```

## Routing Strategies

| Strategy | Description |
|----------|-------------|
| `priority` | Use highest priority (lowest number) available provider |
| `round_robin` | Rotate between available providers |
| `failover` | Use primary provider, automatically fail over on errors |
| `latency` | Use provider with lowest measured latency |
| `cost` | Use lowest cost provider (based on priority as proxy) |

```python
from webapp.router import RoutingStrategy

router.set_routing_strategy(RoutingStrategy.ROUND_ROBIN)
```

## Provider Configuration Options

```python
LLMProviderConfig(
    # Required
    name="my-provider",           # Unique identifier
    provider_type="ollama",       # Provider type (see supported list)
    
    # Connection
    url="http://localhost:11434", # Server URL (for self-hosted)
    api_key="sk-...",            # API key (for cloud APIs)
    
    # Model
    model="llama3.2",            # Model name
    max_tokens=4096,             # Max output tokens
    temperature=0.7,             # Sampling temperature
    
    # Routing
    priority=10,                 # Lower = higher priority
    enabled=True,                # Enable/disable this provider
    
    # Capabilities
    supports_tools=True,         # Function calling support
    supports_vision=False,       # Image input support
    
    # Connection
    timeout=60,                  # Request timeout in seconds
    
    # Additional
    metadata={}                  # Custom metadata
)
```

## Deploying Multiple LLM Backends

### Docker Compose Example

```yaml
version: '3.8'

services:
  # Your application
  business-logic:
    build: ./business-logic
    environment:
      # Cloud APIs
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      # Local LLMs
      - OLLAMA_URL=http://ollama:11434
      - OLLAMA_MODEL=llama3.2
      - LLM_SERVER_URL=http://vllm:8000/v1
    ports:
      - "8080:8080"

  # Ollama for local inference
  ollama:
    image: ollama/ollama:latest
    volumes:
      - ollama_data:/root/.ollama
    ports:
      - "11434:11434"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  # vLLM for high-throughput inference
  vllm:
    image: vllm/vllm-openai:latest
    command: ["--model", "meta-llama/Llama-3.2-8B-Instruct"]
    ports:
      - "8000:8000"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

volumes:
  ollama_data:
```

### Kubernetes Deployment

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: llm-router-config
data:
  LLM_PROVIDERS: |
    [
      {
        "name": "anthropic",
        "provider_type": "anthropic",
        "priority": 1
      },
      {
        "name": "ollama",
        "provider_type": "ollama",
        "url": "http://ollama-service:11434",
        "model": "llama3.2",
        "priority": 10
      }
    ]
```

## Error Handling and Failover

The router automatically handles failures:

1. **Health Checks**: Periodically checks provider availability
2. **Automatic Failover**: If a provider fails, tries the next available one
3. **Error Tracking**: Tracks error counts and last errors per provider
4. **Graceful Degradation**: Falls back to available providers seamlessly

```python
# With failover strategy (default)
response = router.chat(messages=[...])
# If primary fails, automatically tries secondary providers
```

## Integration with Existing Agent

The router integrates with the existing `agent_prompts.py` LLMManager:

```python
# In agent_prompts.py, the LLMManager can use the router
from webapp.router import get_router, ChatResponse

class LLMManager:
    def __init__(self):
        self.router = get_router()
    
    def chat(self, messages, tools=None):
        return self.router.chat(messages, tools)
```

## Thread Safety

The router is fully thread-safe:
- Uses locks for registry modifications
- Singleton pattern ensures single instance
- Safe for use with multiple Flask workers

## Extending with Custom Adapters

```python
from webapp.router import LLMAdapter, register_adapter, LLMProviderType

class CustomAdapter(LLMAdapter):
    def check_availability(self, config):
        # Your implementation
        return True, 0.0, None
    
    def list_models(self, config):
        return ["model1", "model2"]
    
    def chat(self, config, messages, tools=None, **kwargs):
        # Your chat implementation
        pass

# Register the adapter
register_adapter(LLMProviderType.OPENAI_COMPATIBLE, CustomAdapter)
```

## Environment Variables Reference

| Variable | Description | Default |
|----------|-------------|---------|
| `ANTHROPIC_API_KEY` | Anthropic API key | - |
| `ANTHROPIC_MODEL` | Default Anthropic model | claude-sonnet-4-20250514 |
| `OPENAI_API_KEY` | OpenAI API key | - |
| `OPENAI_MODEL` | Default OpenAI model | gpt-4o |
| `GOOGLE_API_KEY` | Google API key | - |
| `GOOGLE_MODEL` | Default Google model | gemini-1.5-pro |
| `OLLAMA_URL` | Ollama server URL | http://localhost:11434 |
| `OLLAMA_MODEL` | Default Ollama model | llama3.2 |
| `USE_OLLAMA` | Enable Ollama discovery | - |
| `LLM_SERVER_URL` | OpenAI-compatible server URL | - |
| `LLM_MODEL_NAME` | Model name for generic server | default |
| `LLM_API_KEY` | API key for generic server | - |
| `LLM_SUPPORTS_TOOLS` | Enable tools for generic server | true |
| `LLM_PROVIDERS` | JSON array of provider configs | - |
