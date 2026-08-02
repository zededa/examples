"""
Get Frontend Integration Guide Tool

Provides comprehensive frontend/client integration guidance.
"""

import logging
from typing import Dict, List, Any, Optional

from tools.base import ok, error_response, get_client
from tools.registry import register_tool
from .model_type import infer_model_type_from_shapes

logger = logging.getLogger(__name__)


def _generate_integration_code(model_name: str, input_spec: Dict, model_type: str, 
                                server_type: str, endpoints_info: Dict, 
                                framework: str) -> Dict[str, str]:
    """Generate framework-specific integration code examples."""
    
    inference_endpoint = endpoints_info.get('inference', {}).get('endpoint', f'/v2/models/{model_name}/infer')
    input_name = input_spec.get('name', 'images')
    
    examples = {}
    
    # JavaScript/Fetch example
    examples["javascript_fetch"] = f'''// JavaScript - Fetch API
async function runInference(imageFile) {{
    const formData = new FormData();
    formData.append('file', imageFile);
    
    try {{
        const response = await fetch('/predict', {{
            method: 'POST',
            body: formData
        }});
        
        if (!response.ok) {{
            throw new Error(`HTTP error! status: ${{response.status}}`);
        }}
        
        const result = await response.json();
        return result;
    }} catch (error) {{
        console.error('Inference failed:', error);
        throw error;
    }}
}}

// Usage with file input
document.getElementById('imageInput').addEventListener('change', async (e) => {{
    const file = e.target.files[0];
    if (file) {{
        showLoading();
        try {{
            const result = await runInference(file);
            displayResults(result);
        }} catch (error) {{
            showError('Failed to process image');
        }} finally {{
            hideLoading();
        }}
    }}
}});
'''
    
    # Python requests example
    examples["python_requests"] = f'''# Python - requests library
import requests
import json

def run_inference(image_path, server_url="http://localhost:8000"):
    """Send inference request to the ML server."""
    
    with open(image_path, 'rb') as f:
        files = {{'file': f}}
        response = requests.post(
            f"{{server_url}}/predict",
            files=files,
            timeout=60
        )
    
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Inference failed: {{response.status_code}}")

# Direct v2 API (requires preprocessing)
def run_inference_v2(preprocessed_array, model_name="{model_name}"):
    payload = {{
        "inputs": [{{
            "name": "{input_name}",
            "shape": list(preprocessed_array.shape),
            "datatype": "FP32",
            "data": preprocessed_array.flatten().tolist()
        }}]
    }}
    
    response = requests.post(
        "{inference_endpoint}",
        json=payload,
        headers={{"Content-Type": "application/json"}},
        timeout=60
    )
    
    return response.json()
'''
    
    # React component example
    examples["react_component"] = f'''// React Component Example
import React, {{ useState, useCallback }} from 'react';

function InferenceComponent() {{
    const [result, setResult] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    
    const handleImageUpload = useCallback(async (event) => {{
        const file = event.target.files[0];
        if (!file) return;
        
        if (!file.type.startsWith('image/')) {{
            setError('Please select an image file');
            return;
        }}
        
        setLoading(true);
        setError(null);
        
        const formData = new FormData();
        formData.append('file', file);
        
        try {{
            const response = await fetch('/predict', {{
                method: 'POST',
                body: formData
            }});
            
            if (!response.ok) {{
                throw new Error(`Server error: ${{response.status}}`);
            }}
            
            const data = await response.json();
            setResult(data);
        }} catch (err) {{
            setError(err.message);
        }} finally {{
            setLoading(false);
        }}
    }}, []);
    
    return (
        <div>
            <input 
                type="file" 
                accept="image/*" 
                onChange={{handleImageUpload}}
                disabled={{loading}}
            />
            {{loading && <div>Processing...</div>}}
            {{error && <div className="error">{{error}}</div>}}
            {{result && <ResultDisplay data={{result}} />}}
        </div>
    );
}}
'''

    # cURL example
    examples["curl"] = f'''# cURL Examples

# Health check
curl -X GET {endpoints_info.get('server_health', {}).get('endpoint', '/v2/health/ready')}

# Get model metadata
curl -X GET {endpoints_info.get('model_metadata', {}).get('endpoint', f'/v2/models/{model_name}')} | jq

# Inference via web app (easiest - handles preprocessing)
curl -X POST http://localhost:5000/predict \\
    -F "file=@/path/to/image.jpg" | jq

# Direct v2 API inference (requires preprocessed tensor)
curl -X POST {inference_endpoint} \\
    -H "Content-Type: application/json" \\
    -d '{{"inputs": [{{"name": "{input_name}", "shape": [1, 3, 640, 640], "datatype": "FP32", "data": [...]}}]}}'
'''

    return examples


def get_frontend_integration_guide(
    model_name: str, 
    framework: str = "javascript",
    detail_level: str = "full",
    sections: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Get comprehensive frontend/client integration guidance for a model.
    
    Provides guidance for implementing frontend/client logic including
    request/response flow, error handling, and UX patterns.
    
    Args:
        model_name: Name of the model to integrate with
        framework: Target framework (javascript, python, react, etc.)
        detail_level: Response verbosity - "minimal", "standard", or "full" (default)
        sections: Specific sections to include. Options:
                  ["input", "output", "code", "ux", "errors", "flow"]
                  If None, includes all sections based on detail_level.
        
    Returns:
        Dict containing integration guide with code examples
    """
    try:
        client = get_client()
        
        # Get model specifications
        input_spec = client.get_model_input_spec(model_name)
        output_specs = client.get_all_output_specs(model_name)
        endpoints_info = client.get_api_endpoints_info(model_name)
        server_type = client.detect_server_type()
        
        # Determine model type for appropriate guidance
        model_type_info = infer_model_type_from_shapes(input_spec, output_specs)
        model_type = model_type_info['type']
        
        # Generate framework-specific code examples
        code_examples = _generate_integration_code(
            model_name, input_spec, model_type, server_type, 
            endpoints_info, framework
        )
        
        # UX patterns and best practices
        ux_patterns = {
            "loading_states": [
                "Show loading spinner during inference",
                "Display progress indicator for image upload",
                "Implement timeout handling (recommend 30s max)"
            ],
            "error_handling": [
                "Handle network errors gracefully",
                "Show user-friendly error messages",
                "Implement retry logic for transient failures",
                "Validate image before upload (size, format)"
            ],
            "performance": [
                "Resize images client-side before upload to reduce bandwidth",
                "Use WebSocket for real-time camera feeds if available",
                "Implement request debouncing for video streams",
                "Cache results when appropriate"
            ],
            "accessibility": [
                "Provide alt text for detection visualizations",
                "Announce results to screen readers",
                "Support keyboard navigation"
            ]
        }
        
        # Request/response flow
        request_flow = {
            "steps": [
                "1. Capture/select image from user",
                "2. Validate image (format, size)",
                "3. Resize/preprocess if needed (optional, server handles this)",
                "4. Convert to base64 or FormData",
                "5. Send POST request to inference endpoint",
                "6. Handle loading state",
                "7. Parse JSON response",
                "8. Post-process results (NMS for detection, etc.)",
                "9. Render visualizations",
                "10. Handle errors appropriately"
            ],
            "recommended_timeouts": {
                "upload": "30 seconds",
                "inference": "60 seconds",
                "total": "90 seconds"
            }
        }
        
        # Build response based on detail_level and sections
        requested_sections = sections or ["input", "output", "code", "ux", "errors", "flow"]
        
        result = {
            "model_name": model_name,
            "model_type": model_type,
            "server_type": server_type,
            "target_framework": framework,
            "detail_level": detail_level
        }
        
        if "input" in requested_sections:
            result["input_requirements"] = {
                "image_size": f"{input_spec.get('width', 640)}x{input_spec.get('height', 640)}",
                "format": input_spec.get('format', 'NCHW'),
                "channels": input_spec.get('channels', 3)
            }
        
        if "output" in requested_sections:
            result["api_endpoint"] = endpoints_info.get('inference', {})
        
        if "code" in requested_sections:
            if detail_level == "minimal":
                primary_key = f"{framework}_fetch" if framework == "javascript" else f"{framework}_requests" if framework == "python" else f"{framework}_component"
                result["code_examples"] = {primary_key: code_examples.get(primary_key, code_examples.get(list(code_examples.keys())[0]))}
            elif detail_level == "standard":
                result["code_examples"] = {
                    k: v for k, v in code_examples.items() 
                    if framework in k.lower() or k == "curl"
                }
            else:
                result["code_examples"] = code_examples
        
        if "ux" in requested_sections and detail_level != "minimal":
            if detail_level == "standard":
                result["ux_patterns"] = {
                    "key_points": [
                        "Show loading state during inference",
                        "Handle errors gracefully with user-friendly messages",
                        "Resize images client-side before upload",
                        "Implement timeout handling (30-60s)"
                    ]
                }
            else:
                result["ux_patterns"] = ux_patterns
        
        if "flow" in requested_sections and detail_level != "minimal":
            result["request_flow"] = request_flow
        
        if "errors" in requested_sections:
            result["error_codes"] = {
                "400": "Bad Request - Invalid input format",
                "404": "Model not found",
                "500": "Server error - Check server logs",
                "503": "Model not ready - Try again later"
            }
        
        return ok(**result)
    except Exception as e:
        logger.error(f"Error generating integration guide for {model_name}: {e}")
        return error_response(e, operation="get_frontend_integration_guide", model_name=model_name)


# Register the tool
register_tool(
    name="get_frontend_integration_guide",
    func=get_frontend_integration_guide,
    description="Get comprehensive frontend/client integration guidance including code examples, request/response flow, error handling patterns, and UX best practices. Use this when users ask how to structure their frontend or client application. Supports detail_level to control response size.",
    input_schema={
        "type": "object",
        "properties": {
            "model_name": {
                "type": "string",
                "description": "Name of the model to integrate with"
            },
            "framework": {
                "type": "string",
                "description": "Target framework (javascript, python, react). Default: javascript",
                "enum": ["javascript", "python", "react"]
            },
            "detail_level": {
                "type": "string",
                "description": "Response verbosity: 'minimal' (code only), 'standard' (code + key points), 'full' (everything). Default: full",
                "enum": ["minimal", "standard", "full"]
            },
            "sections": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": ["input", "output", "code", "ux", "errors", "flow"]
                },
                "description": "Specific sections to include. If omitted, includes all based on detail_level."
            }
        },
        "required": ["model_name"]
    }
)
