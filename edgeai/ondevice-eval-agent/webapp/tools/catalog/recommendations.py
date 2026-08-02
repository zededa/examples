"""
Recommend Next Steps Tool

Meta-tool that suggests what actions to take next based on current state.
"""

import logging
from typing import Dict, Any, Optional

from tools.base import ok, error_response, get_client
from tools.registry import register_tool
from .model_type import infer_model_type_from_shapes

logger = logging.getLogger(__name__)


def recommend_next_steps(model_name: Optional[str] = None, current_context: Optional[str] = None) -> Dict[str, Any]:
    """
    Meta-tool that suggests what actions to take next based on current state.
    
    Analyzes the model and context to provide intelligent recommendations
    for the conversational flow, helping users and agents decide what to explore next.
    
    Args:
        model_name: Name of the model being explored (optional)
        current_context: Description of what has been done so far (optional)
            Options: "initial", "listed_models", "analyzed_type", "checked_inputs", 
                     "checked_outputs", "ready_to_integrate", "troubleshooting"
        
    Returns:
        Dict containing prioritized next step recommendations
    """
    try:
        client = get_client()
        recommendations = []
        warnings = []
        
        # If no model specified, suggest discovery first
        if not model_name:
            models = client.get_available_models()
            is_healthy, _ = client.check_server_health()
            
            if not is_healthy:
                recommendations.append({
                    "priority": 1,
                    "action": "check_server_status",
                    "tool": "get_server_status",
                    "reason": "Server may not be healthy. Check status before proceeding."
                })
                warnings.append("Server health check recommended before model exploration.")
            
            if models:
                recommendations.append({
                    "priority": 2,
                    "action": "explore_model",
                    "tool": "get_model_metadata",
                    "reason": f"Found {len(models)} model(s): {', '.join(models[:3])}{'...' if len(models) > 3 else ''}. Pick one to analyze.",
                    "available_models": models
                })
            else:
                recommendations.append({
                    "priority": 1,
                    "action": "list_models",
                    "tool": "list_available_models",
                    "reason": "Start by discovering what models are available on the server."
                })
            
            return ok(
                warnings=warnings if warnings else None,
                context=current_context or "initial",
                recommendations=recommendations,
                summary="No specific model selected. Start with model discovery."
            )
        
        # Model specified - analyze and recommend based on context
        try:
            input_spec = client.get_model_input_spec(model_name)
            output_specs = client.get_all_output_specs(model_name)
            model_type_info = infer_model_type_from_shapes(input_spec, output_specs)
        except Exception:
            return ok(
                warnings=[f"Model '{model_name}' may not be available or ready."],
                model_name=model_name,
                context=current_context,
                recommendations=[{
                    "priority": 1,
                    "action": "verify_model",
                    "tool": "get_model_metadata",
                    "reason": f"Could not fetch model info. Verify '{model_name}' exists and is ready."
                }]
            )
        
        # Build recommendations based on context
        context = current_context or "initial"
        
        if context in ["initial", "listed_models"]:
            recommendations = [
                {
                    "priority": 1,
                    "action": "analyze_model_type",
                    "tool": "analyze_model_type",
                    "args": {"model_name": model_name},
                    "reason": "Understand what type of model this is (detection, classification, etc.)"
                },
                {
                    "priority": 2,
                    "action": "check_input_requirements",
                    "tool": "get_model_input_requirements",
                    "args": {"model_name": model_name},
                    "reason": "Learn what inputs the model expects"
                }
            ]
        
        elif context == "analyzed_type":
            recommendations = [
                {
                    "priority": 1,
                    "action": "check_input_requirements",
                    "tool": "get_model_input_requirements",
                    "args": {"model_name": model_name},
                    "reason": "Understand preprocessing requirements"
                },
                {
                    "priority": 2,
                    "action": "check_output_interpretation",
                    "tool": "get_model_output_interpretation",
                    "args": {"model_name": model_name},
                    "reason": "Learn how to interpret model outputs"
                }
            ]
            if model_type_info['confidence'] != 'high':
                warnings.append("Model type confidence is not high. Run sample inference to verify.")
        
        elif context == "checked_inputs":
            recommendations = [
                {
                    "priority": 1,
                    "action": "check_output_interpretation",
                    "tool": "get_model_output_interpretation",
                    "args": {"model_name": model_name},
                    "reason": "Understand post-processing for model outputs"
                },
                {
                    "priority": 2,
                    "action": "run_sample_inference",
                    "tool": None,
                    "reason": "Test the model with a sample image to verify preprocessing"
                }
            ]
        
        elif context == "checked_outputs":
            recommendations = [
                {
                    "priority": 1,
                    "action": "get_integration_guide",
                    "tool": "get_frontend_integration_guide",
                    "args": {"model_name": model_name},
                    "reason": "Get code examples for integrating with your application"
                },
                {
                    "priority": 2,
                    "action": "get_api_examples",
                    "tool": "get_api_examples",
                    "args": {"model_name": model_name},
                    "reason": "Get curl commands to test the API directly"
                }
            ]
        
        elif context == "ready_to_integrate":
            recommendations = [
                {
                    "priority": 1,
                    "action": "implement_frontend",
                    "tool": None,
                    "reason": "You have all the information needed. Start implementing your frontend/client."
                },
                {
                    "priority": 2,
                    "action": "run_inference_test",
                    "tool": None,
                    "reason": "Test end-to-end inference with a real image before full integration"
                }
            ]
        
        elif context == "troubleshooting":
            recommendations = [
                {
                    "priority": 1,
                    "action": "check_server_status",
                    "tool": "get_server_status",
                    "reason": "Verify server health and connectivity"
                },
                {
                    "priority": 2,
                    "action": "verify_model_metadata",
                    "tool": "get_model_metadata",
                    "args": {"model_name": model_name},
                    "reason": "Confirm model specifications match your expectations"
                },
                {
                    "priority": 3,
                    "action": "check_input_format",
                    "tool": "get_model_input_requirements",
                    "args": {"model_name": model_name},
                    "reason": "Verify your preprocessing matches model requirements"
                }
            ]
        
        else:
            recommendations = [
                {
                    "priority": 1,
                    "action": "get_metadata",
                    "tool": "get_model_metadata",
                    "args": {"model_name": model_name},
                    "reason": "Get complete model specifications"
                },
                {
                    "priority": 2,
                    "action": "analyze_type",
                    "tool": "analyze_model_type",
                    "args": {"model_name": model_name},
                    "reason": "Determine model type and capabilities"
                }
            ]
        
        return ok(
            warnings=warnings if warnings else None,
            model_name=model_name,
            model_type=model_type_info['type'],
            model_confidence=model_type_info['confidence'],
            context=context,
            recommendations=recommendations,
            summary=f"Analyzing '{model_name}' ({model_type_info['type']}). {len(recommendations)} recommended next steps."
        )
        
    except Exception as e:
        logger.error(f"Error generating recommendations: {e}")
        return error_response(e, operation="recommend_next_steps", model_name=model_name)


# Register the tool
register_tool(
    name="recommend_next_steps",
    func=recommend_next_steps,
    description="Meta-tool that suggests what actions to take next based on current exploration state. Helps guide the conversational flow by recommending which tools to use and in what order. Great for users unsure what to do next. All parameters are optional.",
    input_schema={
        "type": "object",
        "properties": {
            "model_name": {
                "type": ["string", "null"],
                "description": "Name of the model being explored. Optional - omit or pass null for initial discovery when no model is selected yet."
            },
            "current_context": {
                "type": ["string", "null"],
                "description": "What has been done so far. Optional - omit to get general recommendations.",
                "enum": ["initial", "listed_models", "analyzed_type", "checked_inputs", "checked_outputs", "ready_to_integrate", "troubleshooting", None]
            }
        },
        "required": []
    }
)
