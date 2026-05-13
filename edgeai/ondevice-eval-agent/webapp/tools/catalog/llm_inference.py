"""
LLM Inference Tool

Sends a prompt to an LLM served by vLLM or llama.cpp and returns
the completion along with token usage and performance metrics.
"""

import logging
from typing import Any, Dict, List, Optional

from tools.base import ok, error_response
from tools.registry import register_tool

logger = logging.getLogger(__name__)

DEFAULT_MAX_TOKENS = 512
DEFAULT_TEMPERATURE = 0.7
MAX_MAX_TOKENS = 4096


def _get_llm_client():
    from client.llm_client import get_llm_client
    return get_llm_client()


def llm_inference(
    prompt: str,
    model_name: str = "",
    system_prompt: str = "",
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    mode: str = "chat",
) -> Dict[str, Any]:
    """
    Send a prompt to an LLM and return the completion.

    Args:
        prompt: The user prompt to send to the model.
        model_name: Model ID. If empty, uses the first available model.
        system_prompt: Optional system message (chat mode only).
        max_tokens: Maximum tokens to generate (default 512, max 4096).
        temperature: Sampling temperature (0.0-2.0, default 0.7).
        mode: "chat" for chat completions, "completion" for text completions.

    Returns:
        The model's response, token usage, and performance metrics.
    """
    try:
        if not prompt:
            return error_response(
                ValueError("prompt is required"),
                operation="llm_inference",
            )

        client = _get_llm_client()

        if not client.is_healthy():
            return error_response(
                ConnectionError(
                    f"LLM server at {client.base_url} is not reachable"
                ),
                operation="llm_inference",
            )

        # Resolve model name
        if not model_name:
            models = client.list_models()
            if not models:
                return error_response(
                    ValueError("No LLM models available on the server"),
                    operation="llm_inference",
                )
            model_name = models[0].id

        max_tokens = max(1, min(int(max_tokens), MAX_MAX_TOKENS))
        temperature = max(0.0, min(float(temperature), 2.0))

        if mode == "completion":
            result = client.text_completion(
                model=model_name,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        else:
            messages: List[Dict[str, str]] = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            result = client.chat_completion(
                model=model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )

        perf = result["performance"]
        usage = result["usage"]

        summary = (
            f"Generated {usage['completion_tokens']} tokens in "
            f"{perf['total_time_ms']:.0f}ms "
            f"({perf['tokens_per_second']:.1f} tok/s) "
            f"using {result['model']}"
        )

        return ok(
            response=result["response"],
            model=result["model"],
            usage=usage,
            performance=perf,
            finish_reason=result["finish_reason"],
            mode=mode,
            message=summary,
        )

    except Exception as e:
        logger.error("Error running LLM inference: %s", e, exc_info=True)
        return error_response(
            e,
            operation="llm_inference",
            model_name=model_name,
        )


register_tool(
    name="llm_inference",
    func=llm_inference,
    description=(
        "Send a prompt to an LLM (served by vLLM or llama.cpp) and return the "
        "model's response along with token usage and performance metrics "
        "(tokens/sec, latency). Supports both chat completions and text "
        "completions. Use this when the user wants to send a prompt to the "
        "language model, get a response from the LLM, or test the model with "
        "a specific input."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": "The prompt to send to the LLM.",
            },
            "model_name": {
                "type": "string",
                "description": (
                    "Model ID to use. If empty, uses the first available model."
                ),
            },
            "system_prompt": {
                "type": "string",
                "description": (
                    "Optional system message to set context (chat mode only)."
                ),
            },
            "max_tokens": {
                "type": "integer",
                "default": 512,
                "minimum": 1,
                "maximum": 4096,
                "description": "Maximum number of tokens to generate.",
            },
            "temperature": {
                "type": "number",
                "default": 0.7,
                "minimum": 0.0,
                "maximum": 2.0,
                "description": (
                    "Sampling temperature. 0.0 = deterministic, higher = more creative."
                ),
            },
            "mode": {
                "type": "string",
                "enum": ["chat", "completion"],
                "default": "chat",
                "description": (
                    "Inference mode: 'chat' for chat completions (default), "
                    "'completion' for raw text completions."
                ),
            },
        },
        "required": ["prompt"],
    },
)
