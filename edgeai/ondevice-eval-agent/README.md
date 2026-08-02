# ZEDEDA On-Device AI Agent - Client Container

Flask-based web application for ML model inference with an AI-powered assistant for model exploration and integration guidance.

## Features

- Web interface for image upload and classification
- Multi-model support with dynamic discovery
- Real-time processing logs
- API endpoints for programmatic access
- Customizable preprocessing
- **AI Agent for model exploration and integration guidance**

## AI Agent (Agentic Demo POC)

The business logic includes an intelligent AI assistant that helps developers understand and integrate with deployed ML models.

### Agent Capabilities

| Capability | Description |
|------------|-------------|
| **Model Discovery** | Identifies available models on Triton/OpenVINO inference servers |
| **Input Requirements** | Explains image formats, preprocessing, and camera feed recommendations |
| **Output Interpretation** | Describes model outputs (bounding boxes, labels, masks) and post-processing |
| **Integration Guidance** | Provides code examples for JavaScript, Python, React, and cURL |

### Example Questions

- "What model is currently running on the server?"
- "How should I structure the frontend/client logic to call this model?"
- "What images or camera feed characteristics will this model respond to reliably?"
- "How do I interpret the bounding box outputs from this detection model?"
- "Show me how to preprocess images for this model"

### Agent API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/agent/chat` | POST | Send a message to the AI agent |
| `/agent/status` | GET | Check if agent is enabled |

#### Chat Request Example

```bash
curl -X POST http://localhost:8080/agent/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What model is running?", "session_id": "my-session"}'
```

#### Response Format

```json
{
  "success": true,
  "response": "Agent's response text...",
  "session_id": "my-session",
  "enabled": true,
  "tool_calls": [...],
  "tokens": {"input": 150, "output": 200}
}
```

### Agent Tools

The agent has access to these tools for real-time model exploration:

| Tool | Purpose |
|------|---------|
| `list_available_models` | Discover all models on the inference server |
| `get_model_metadata` | Get detailed model specifications |
| `get_model_input_requirements` | Get preprocessing and input format guidance |
| `get_model_output_interpretation` | Understand model outputs and post-processing |
| `analyze_model_type` | Infer model type from tensor shapes |
| `get_server_status` | Check inference server health |
| `get_api_examples` | Get cURL commands for API testing |
| `get_frontend_integration_guide` | Get full integration code examples |

### Enabling the Agent

The agent supports multiple LLM backends. Set one of the following:

#### Option 1: Anthropic Claude (Recommended)

Best for reliable tool calling and high-quality responses.

```bash
export ANTHROPIC_API_KEY=sk-ant-your-key-here
export ANTHROPIC_MODEL=claude-sonnet-4-20250514  # optional
```

#### Option 2: OpenAI

Use GPT-4o or other OpenAI models.

```bash
export OPENAI_API_KEY=sk-your-key-here
export OPENAI_MODEL=gpt-4o  # optional, defaults to gpt-4o
```

#### Option 3: Google Gemini

Use Gemini 1.5 Pro or other Google models.

```bash
export GOOGLE_API_KEY=your-key-here
export GOOGLE_MODEL=gemini-1.5-pro  # optional
```

#### Option 4: Local LLM Server (OpenAI-Compatible)

Use Ollama, LM Studio, vLLM, or any OpenAI-compatible API.

```bash
export LLM_SERVER_URL=http://your-llm-server:1234
export LLM_MODEL_NAME=your-model-name  # optional
export LLM_API_KEY=your-api-key        # optional
```

**Server-specific examples:**
```bash
# Ollama
export LLM_SERVER_URL=http://localhost:11434
export LLM_MODEL_NAME=llama3.1

# LM Studio
export LLM_SERVER_URL=http://localhost:1234

# vLLM
export LLM_SERVER_URL=http://localhost:8000
```

> **Priority:** If multiple backends are configured, they are used in this order:
> Anthropic → OpenAI → Google → Local LLM Server

## Configuration

Environment variables:
- `MODEL_SERVER_URL`: URL of the inference server (Triton or OpenVINO)
- `ANTHROPIC_API_KEY`: Anthropic API key (for Claude backend)
- `ANTHROPIC_MODEL`: Claude model to use (default: `claude-sonnet-4-20250514`)
- `OPENAI_API_KEY`: OpenAI API key (for GPT backend)
- `OPENAI_MODEL`: OpenAI model to use (default: `gpt-4o`)
- `GOOGLE_API_KEY`: Google API key (for Gemini backend)
- `GOOGLE_MODEL`: Google model to use (default: `gemini-1.5-pro`)
- `LLM_SERVER_URL`: URL of OpenAI-compatible LLM server
- `LLM_MODEL_NAME`: Model name for OpenAI-compatible server (default: `local-model`)
- `LLM_API_KEY`: API key for OpenAI-compatible server (default: `not-needed`)
- `APP_TITLE`: Application title
- `APP_DESCRIPTION`: Application description
- `LOGO_URL`: URL for logo image
- `PRIMARY_COLOR`: Primary theme color (CSS)

## API Endpoints

- `GET /` - Web interface
- `GET /health` - Health check
- `GET /models` - List available models
- `POST /predict` - Run inference
- `GET /models/<name>/metadata` - Get model metadata
- `POST /agent/chat` - AI agent chat
- `GET /agent/status` - Agent status

## Customization

### Class Names

Edit `class_names.json` with your model's class labels:

```json
[
    "cat",
    "dog",
    "bird",
    ...
]
```

### Preprocessing

Modify `client.py` to adjust preprocessing for your model:

```python
def preprocess_image(self, image_path, ...):
    # Customize resize, normalization, etc.
```

## Local Development

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Option 1: Use Anthropic Claude
export ANTHROPIC_API_KEY=sk-ant-your-key-here

# Option 2: Use local LLM server
export LLM_SERVER_URL=http://localhost:11434  # e.g., Ollama
export LLM_MODEL_NAME=llama3.1

python webapp/app.py
```

## Architecture

```
business-logic/
├── client.py              # Model server client (Triton/OpenVINO)
├── requirements.txt       # Python dependencies
├── Dockerfile            # Container build
└── webapp/
    ├── app.py            # Flask application
    ├── static/           # CSS, JS assets
    ├── templates/        # HTML templates
    ├── agent/            # Agent package (backward compatibility)
    │   ├── tools.py      # Re-exports from mcp package
    │   └── prompts.py    # LLM chat processing
    ├── router/           # LLM Router package
    │   ├── config.py     # Provider configuration
    │   ├── llm_router.py # Multi-provider routing
    │   └── adapters/     # Provider-specific adapters
    ├── inference/        # Inference client wrapper
    └── mcp/              # MCP (Model Context Protocol) tools
        ├── base.py       # Base utilities (ToolResult, ok, error_response)
        ├── session.py    # Session storage management
        ├── registry.py   # Tool registration and execution
        └── tools/        # Individual tool modules
            ├── list_models.py
            ├── model_metadata.py
            ├── model_inputs.py
            ├── model_outputs.py
            ├── model_type.py
            ├── server_status.py
            ├── api_examples.py
            ├── integration_guide.py
            └── recommendations.py
```

## MCP Package

The MCP (Model Context Protocol) package provides a modular tool framework for AI agent interactions with ML inference servers. Each tool is in its own file for easy maintenance and extension.

### Usage

```python
# Direct imports from mcp package
from mcp import execute_tool, TOOL_SCHEMAS, TOOL_FUNCTIONS
from mcp.tools import list_available_models, get_model_metadata

# Or use backward-compatible imports
from agent.tools import TOOL_SCHEMAS, execute_tool
```

### Adding New Tools

To add a new tool, create a file in `webapp/mcp/tools/`:

```python
# mcp/tools/my_new_tool.py
from ..base import ok, error_response, get_client
from ..registry import register_tool

def my_new_tool(param: str) -> Dict[str, Any]:
    """Tool implementation."""
    try:
        client = get_client()
        # Your tool logic here
        return ok(result="success", data={...})
    except Exception as e:
        return error_response(e, operation="my_new_tool")

# Auto-register the tool
register_tool(
    name="my_new_tool",
    func=my_new_tool,
    description="Description for AI agent to understand when to use this tool",
    input_schema={
        "type": "object",
        "properties": {
            "param": {
                "type": "string",
                "description": "Parameter description"
            }
        },
        "required": ["param"]
    }
)
```

Then add the import to `mcp/tools/__init__.py`:

```python
from .my_new_tool import my_new_tool
```

The tool will be automatically available to the AI agent.
