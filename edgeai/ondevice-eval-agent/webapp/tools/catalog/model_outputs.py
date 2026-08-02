"""
Get Model Output Interpretation Tool

Provides detailed output interpretation and post-processing guidance for models.
"""

import logging
from typing import Dict, List, Any

from tools.base import ok, error_response, get_client
from tools.registry import register_tool
from .model_type import infer_model_type_from_shapes

logger = logging.getLogger(__name__)


def _interpret_output_shape(shape: List[int], model_type: str, output_name: str) -> Dict[str, Any]:
    """Interpret what an output tensor shape represents."""
    interpretation = {
        "description": "",
        "dimensions_explained": {},
        "value_range": "Typically float32 values"
    }
    
    if model_type == "object_detection":
        if len(shape) == 3:
            interpretation["description"] = "Object detection predictions with bounding boxes, confidence, and class scores"
            interpretation["dimensions_explained"] = {
                "dim_0": f"Batch size ({shape[0]})",
                "dim_1": f"Features per detection ({shape[1]}) - typically [x, y, w, h, conf, class_scores...]",
                "dim_2": f"Number of detection anchors ({shape[2]})"
            }
            num_classes = shape[1] - 5 if shape[1] > 5 else shape[1] - 4
            interpretation["structure"] = f"Each detection: [center_x, center_y, width, height, objectness, {num_classes} class probabilities]"
    
    elif model_type == "classification":
        if len(shape) == 2:
            interpretation["description"] = "Classification logits/probabilities for each class"
            interpretation["dimensions_explained"] = {
                "dim_0": f"Batch size ({shape[0]})",
                "dim_1": f"Number of classes ({shape[1]})"
            }
            interpretation["structure"] = "Apply softmax to convert logits to probabilities, then argmax for predicted class"
            interpretation["value_range"] = "Raw logits (apply softmax for probabilities 0-1)"
    
    elif model_type == "segmentation":
        if len(shape) >= 3:
            interpretation["description"] = "Pixel-wise segmentation mask with class predictions"
            interpretation["dimensions_explained"] = {
                "dim_0": f"Batch size ({shape[0]})",
                "dim_1": f"Number of classes ({shape[1] if len(shape) == 4 else 'N/A'})",
                "dim_2": f"Height ({shape[2] if len(shape) >= 3 else 'N/A'})",
                "dim_3": f"Width ({shape[3] if len(shape) == 4 else 'N/A'})"
            }
            interpretation["structure"] = "Argmax along class dimension for final segmentation mask"
    
    else:
        interpretation["description"] = f"Model output tensor for {model_type}"
        interpretation["dimensions_explained"] = {f"dim_{i}": str(d) for i, d in enumerate(shape)}
    
    return interpretation


def _generate_postprocessing_guide(model_type: str, output_specs: List[Dict]) -> Dict[str, Any]:
    """Generate post-processing guidance based on model type."""
    guides = {
        "object_detection": {
            "steps": [
                "1. Reshape output tensor to [num_detections, features]",
                "2. Extract bounding boxes (x, y, w, h) and convert to (x1, y1, x2, y2)",
                "3. Extract objectness/confidence scores",
                "4. Extract class probabilities and get predicted class",
                "5. Apply Non-Maximum Suppression (NMS) to filter overlapping boxes",
                "6. Filter by confidence threshold (e.g., 0.5)",
                "7. Scale coordinates back to original image dimensions"
            ],
            "common_thresholds": {
                "confidence_threshold": 0.5,
                "nms_iou_threshold": 0.45
            },
            "output_format": "List of detections: [{bbox: [x1,y1,x2,y2], class_id, class_name, confidence}]",
            "warning": "Assumes YOLOv5/v8-style output. Verify tensor layout for other architectures."
        },
        "classification": {
            "steps": [
                "1. Apply softmax to convert logits to probabilities",
                "2. Get top-k predictions using argsort",
                "3. Map class indices to class names",
                "4. Return predictions with confidence scores"
            ],
            "common_thresholds": {
                "top_k": 5,
                "min_confidence": 0.1
            },
            "output_format": "List of predictions: [{class_id, class_name, probability}]"
        },
        "segmentation": {
            "steps": [
                "1. Apply argmax along class dimension to get class per pixel",
                "2. Resize mask to original image dimensions",
                "3. Apply color map for visualization",
                "4. Optionally compute class areas/percentages"
            ],
            "output_format": "2D array of class indices (H x W)"
        },
        "embedding": {
            "steps": [
                "1. Extract feature vector from model output",
                "2. L2-normalize the embedding (recommended for cosine similarity)",
                "3. Store or compare against database of known embeddings",
                "4. Use cosine similarity or Euclidean distance for matching"
            ],
            "common_thresholds": {
                "similarity_threshold": 0.7,
                "top_k_matches": 5
            },
            "output_format": "1D feature vector of shape [feature_dim], typically 128-2048 dimensions",
            "use_cases": [
                "Image similarity/retrieval: Find similar images in a database",
                "Face recognition: Compare against known face embeddings",
                "Clustering: Group similar images together",
                "Anomaly detection: Flag embeddings far from normal distribution"
            ]
        }
    }
    
    return guides.get(model_type, {
        "steps": ["Post-processing depends on specific model architecture"],
        "output_format": "Refer to model documentation"
    })


def _generate_postprocessing_code(model_type: str, output_specs: List[Dict]) -> str:
    """Generate example post-processing code."""
    if model_type == "object_detection":
        return '''# Python post-processing for object detection (YOLO-style)
# WARNING: This example assumes YOLOv5/v8-style output layout.
import numpy as np

def postprocess_detections(output, conf_threshold=0.5, iou_threshold=0.45):
    # output shape: [1, 84, 8400] -> transpose to [8400, 84]
    predictions = output[0].T
    
    # Extract boxes and scores
    boxes = predictions[:, :4]  # x, y, w, h
    scores = predictions[:, 4:]  # class scores
    
    # Get class with highest score for each detection
    class_ids = np.argmax(scores, axis=1)
    confidences = np.max(scores, axis=1)
    
    # Filter by confidence
    mask = confidences > conf_threshold
    boxes = boxes[mask]
    class_ids = class_ids[mask]
    confidences = confidences[mask]
    
    # Convert xywh to xyxy
    boxes_xyxy = np.zeros_like(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    
    return boxes_xyxy, class_ids, confidences
'''
    elif model_type == "classification":
        return '''# Python post-processing for classification
import numpy as np

def postprocess_classification(output, top_k=5):
    logits = output[0]
    exp_logits = np.exp(logits - np.max(logits))
    probabilities = exp_logits / np.sum(exp_logits)
    
    top_indices = np.argsort(probabilities)[-top_k:][::-1]
    
    results = []
    for idx in top_indices:
        results.append({
            "class_id": int(idx),
            "probability": float(probabilities[idx])
        })
    
    return results
'''
    elif model_type == "embedding":
        return '''# Python post-processing for embeddings
import numpy as np

def postprocess_embedding(output, normalize=True):
    embedding = output[0]
    
    if normalize:
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
    
    return embedding

def compute_similarity(embedding1, embedding2):
    return np.dot(embedding1, embedding2)
'''
    else:
        return "# Post-processing code depends on specific model type"


def get_model_output_interpretation(model_name: str) -> Dict[str, Any]:
    """
    Get detailed output interpretation guide for a model.
    
    Explains what outputs the model returns (labels, bounding boxes, embeddings, etc.)
    and how to interpret and post-process them.
    
    Args:
        model_name: Name of the model to analyze
        
    Returns:
        Dict containing output interpretation guide and post-processing examples
    """
    try:
        client = get_client()
        output_specs = client.get_all_output_specs(model_name)
        input_spec = client.get_model_input_spec(model_name)
        
        # Analyze model type from outputs
        model_type_info = infer_model_type_from_shapes(input_spec, output_specs)
        model_type = model_type_info['type']
        
        outputs_info = []
        for spec in output_specs:
            output_shape = spec.get('shape', [])
            output_name = spec.get('name', 'output')
            datatype = spec.get('datatype', 'FP32')
            
            interpretation = _interpret_output_shape(output_shape, model_type, output_name)
            outputs_info.append({
                "name": output_name,
                "shape": output_shape,
                "datatype": datatype,
                "interpretation": interpretation
            })
        
        # Generate post-processing guidance based on model type
        postprocessing = _generate_postprocessing_guide(model_type, output_specs)
        
        # Build warnings for heuristic inferences
        warnings = []
        if model_type_info['confidence'] != 'high':
            warnings.append("Model type inferred heuristically. Verify with sample inference before production use.")
        if len(output_specs) > 1:
            warnings.append(f"Model has {len(output_specs)} outputs. Post-processing may need to combine multiple outputs.")
        
        return ok(
            warnings=warnings if warnings else None,
            model_name=model_name,
            inferred_model_type=model_type,
            confidence=model_type_info['confidence'],
            reasoning=model_type_info['reasoning'],
            outputs=outputs_info,
            postprocessing_guide=postprocessing,
            code_example=_generate_postprocessing_code(model_type, output_specs),
            inference_warning="Model type inferred heuristically. Verify with sample inference." if model_type_info['confidence'] != 'high' else None
        )
    except Exception as e:
        logger.error(f"Error getting output interpretation for {model_name}: {e}")
        return error_response(e, operation="get_output_interpretation", model_name=model_name)


# Register the tool
register_tool(
    name="get_model_output_interpretation",
    func=get_model_output_interpretation,
    description="Get detailed output interpretation guide explaining what the model returns (labels, bounding boxes, embeddings, etc.) and how to post-process results. Use this when users ask about model outputs or how to interpret results.",
    input_schema={
        "type": "object",
        "properties": {
            "model_name": {
                "type": "string",
                "description": "Name of the model to analyze"
            }
        },
        "required": ["model_name"]
    }
)
