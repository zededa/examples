"""
Analyze Model Type Tool

Infers model type from tensor shape patterns.
"""

import logging
from typing import Dict, List, Any

from tools.base import ok, error_response, get_client
from tools.registry import register_tool

logger = logging.getLogger(__name__)


def _get_model_capabilities(model_type: str) -> Dict[str, Any]:
    """Get capabilities and use cases for a model type."""
    capabilities = {
        "object_detection": {
            "description": "Detects and localizes objects in images with bounding boxes",
            "outputs": ["Bounding boxes (x, y, width, height)", "Class labels", "Confidence scores"],
            "use_cases": ["Security/surveillance", "Inventory counting", "Quality inspection", "Autonomous navigation"],
            "real_time_capable": True
        },
        "classification": {
            "description": "Classifies entire images into predefined categories",
            "outputs": ["Class label", "Probability distribution over classes"],
            "use_cases": ["Product categorization", "Defect classification", "Scene recognition", "Medical imaging"],
            "real_time_capable": True
        },
        "segmentation": {
            "description": "Assigns a class label to each pixel in the image",
            "outputs": ["Pixel-wise class mask", "Class probabilities per pixel"],
            "use_cases": ["Autonomous driving", "Medical image analysis", "Background removal", "Land use mapping"],
            "real_time_capable": "Depends on resolution"
        },
        "pose": {
            "description": "Detects human body keypoints and skeletal structure",
            "outputs": ["Keypoint coordinates", "Confidence per keypoint", "Skeleton connections"],
            "use_cases": ["Fitness tracking", "Gesture recognition", "Sports analysis", "Animation"],
            "real_time_capable": True
        },
        "embedding": {
            "description": "Generates fixed-size feature vectors representing input images",
            "outputs": ["Feature vector (embedding)", "Typically 128-2048 dimensions"],
            "use_cases": ["Image similarity search", "Face recognition", "Content-based retrieval", "Clustering", "Transfer learning"],
            "real_time_capable": True,
            "post_processing": "Normalize vectors (L2) for cosine similarity comparisons"
        }
    }
    return capabilities.get(model_type, {
        "description": f"Model type: {model_type}",
        "outputs": ["Varies by model"],
        "use_cases": ["Refer to model documentation"]
    })


def infer_model_type_from_shapes(input_spec: Dict, output_specs: List[Dict]) -> Dict[str, Any]:
    """
    Infer model type from tensor shape patterns by analyzing ALL outputs.
    
    Common patterns:
    - Detection (YOLO): output shape like [-1, 84, 8400] or [-1, num_boxes, 5+num_classes]
    - Classification: output shape like [-1, num_classes]
    - Segmentation: output shape like [-1, num_classes, H, W]
    
    Note: Confidence is conservative by default. Multiple matching signals
    are required for "high" confidence to prevent over-trust by agents.
    """
    if not output_specs or len(output_specs) == 0:
        return {
            "type": "unknown",
            "confidence": "low",
            "reasoning": "No output specifications available"
        }
    
    # Analyze ALL outputs to gather signals
    signals = {
        "classification": [],
        "object_detection": [],
        "segmentation": [],
        "pose": [],
        "embedding": []
    }
    
    input_height = input_spec.get('height', 0)
    input_width = input_spec.get('width', 0)
    
    for idx, spec in enumerate(output_specs):
        output_shape = spec.get('shape', [])
        output_name = spec.get('name', '').lower()
        
        if not output_shape:
            continue
        
        # Check for classification pattern: [batch, num_classes]
        if len(output_shape) == 2:
            num_classes = output_shape[-1]
            if 2 <= num_classes < 10000:
                signals["classification"].append({
                    "output_idx": idx,
                    "shape": output_shape,
                    "num_classes": num_classes
                })
            if num_classes >= 128:
                signals["embedding"].append({
                    "output_idx": idx,
                    "shape": output_shape,
                    "feature_dim": num_classes
                })
        
        # Check for detection pattern: [batch, features, num_detections]
        elif len(output_shape) == 3:
            dim1, dim2 = output_shape[1], output_shape[2]
            
            if 5 <= dim1 <= 500 and dim2 > 100:
                inferred_classes = dim1 - 4
                signals["object_detection"].append({
                    "output_idx": idx,
                    "shape": output_shape,
                    "pattern": "yolo_style",
                    "inferred_classes": inferred_classes
                })
            elif dim1 > 100 and 5 <= dim2 <= 500:
                inferred_classes = dim2 - 4
                signals["object_detection"].append({
                    "output_idx": idx,
                    "shape": output_shape,
                    "pattern": "anchor_first",
                    "inferred_classes": inferred_classes
                })
            elif dim1 > 10 and dim2 > 10:
                signals["segmentation"].append({
                    "output_idx": idx,
                    "shape": output_shape,
                    "pattern": "single_class_mask"
                })
        
        # Check for 4D outputs: [batch, classes/channels, H, W]
        elif len(output_shape) == 4:
            batch, c, h, w = output_shape
            if 1 <= c <= 256 and h > 1 and w > 1:
                signals["segmentation"].append({
                    "output_idx": idx,
                    "shape": output_shape,
                    "num_classes": c,
                    "spatial_size": f"{h}x{w}"
                })
        
        # Check output names for hints
        if any(kw in output_name for kw in ['box', 'detect', 'bbox', 'yolo']):
            signals["object_detection"].append({"name_hint": output_name, "output_idx": idx})
        if any(kw in output_name for kw in ['class', 'logit', 'prob', 'score']) and 'box' not in output_name:
            signals["classification"].append({"name_hint": output_name, "output_idx": idx})
        if any(kw in output_name for kw in ['mask', 'segment', 'semantic']):
            signals["segmentation"].append({"name_hint": output_name, "output_idx": idx})
        if any(kw in output_name for kw in ['keypoint', 'pose', 'skeleton', 'joint']):
            signals["pose"].append({"name_hint": output_name, "output_idx": idx})
    
    def count_unique_signals(signal_list):
        indices = set()
        has_name_hint = False
        for s in signal_list:
            if 'output_idx' in s:
                indices.add(s['output_idx'])
            if 'name_hint' in s:
                has_name_hint = True
        return len(indices), has_name_hint
    
    best_type = "unknown"
    best_confidence = "low"
    best_reasoning = []
    
    # Check detection signals
    det_count, det_name_hint = count_unique_signals(signals["object_detection"])
    if det_count > 0:
        best_type = "object_detection"
        shape_signals = [s for s in signals["object_detection"] if 'pattern' in s]
        if shape_signals:
            best_reasoning.append(f"Output shape matches detection pattern: {shape_signals[0]['pattern']}")
        if det_name_hint:
            best_reasoning.append("Output name contains detection keywords")
        if det_count >= 1 and (det_name_hint or len(shape_signals) > 0):
            best_confidence = "medium"
        if det_count >= 1 and det_name_hint and len(shape_signals) > 0:
            best_confidence = "high"
    
    # Check classification vs embedding signals
    cls_count, cls_name_hint = count_unique_signals(signals["classification"])
    emb_count, emb_name_hint = count_unique_signals(signals["embedding"])
    
    if best_type == "unknown" and (cls_count > 0 or emb_count > 0):
        shape_signals_cls = [s for s in signals["classification"] if 'num_classes' in s]
        shape_signals_emb = [s for s in signals["embedding"] if 'feature_dim' in s]
        
        is_likely_embedding = False
        if emb_count > 0:
            for spec in output_specs:
                name = spec.get('name', '').lower()
                if any(kw in name for kw in ['embed', 'feature', 'encoding', 'vector', 'representation']):
                    is_likely_embedding = True
                    break
            if shape_signals_emb and shape_signals_emb[0]['feature_dim'] >= 128:
                if not cls_name_hint and len(output_specs) == 1:
                    common_embedding_dims = {128, 256, 384, 512, 768, 1024, 1536, 2048}
                    if shape_signals_emb[0]['feature_dim'] in common_embedding_dims:
                        is_likely_embedding = True
        
        if is_likely_embedding:
            best_type = "embedding"
            if shape_signals_emb:
                best_reasoning.append(f"Output shape [{shape_signals_emb[0]['shape']}] matches embedding with {shape_signals_emb[0]['feature_dim']}-dim features")
            best_confidence = "medium"
            for spec in output_specs:
                name = spec.get('name', '').lower()
                if any(kw in name for kw in ['embed', 'feature', 'encoding']):
                    best_reasoning.append("Output name contains embedding keywords")
                    best_confidence = "high"
                    break
        else:
            best_type = "classification"
            if shape_signals_cls:
                best_reasoning.append(f"Output shape [{shape_signals_cls[0]['shape']}] matches classification with {shape_signals_cls[0]['num_classes']} classes")
            if cls_name_hint:
                best_reasoning.append("Output name contains classification keywords")
            if cls_count == 1 and len(output_specs) == 1:
                best_confidence = "medium"
            if cls_name_hint:
                best_confidence = "medium" if best_confidence == "low" else "high"
    
    # Check segmentation signals
    seg_count, seg_name_hint = count_unique_signals(signals["segmentation"])
    if seg_count > 0 and (seg_count > det_count or seg_name_hint):
        best_type = "segmentation"
        best_confidence = "medium"
        shape_signals = [s for s in signals["segmentation"] if 'spatial_size' in s]
        if shape_signals:
            best_reasoning.append(f"Output has spatial dimensions: {shape_signals[0]['spatial_size']}")
        if seg_name_hint:
            best_reasoning.append("Output name contains segmentation keywords")
            best_confidence = "high" if shape_signals else "medium"
    
    # Check pose signals
    pose_count, pose_name_hint = count_unique_signals(signals["pose"])
    if pose_count > 0 and pose_name_hint:
        best_type = "pose"
        best_confidence = "medium"
        best_reasoning.append("Output name contains pose/keypoint keywords")
    
    if best_type == "unknown" and len(output_specs) > 1:
        best_type = "multi_output"
        best_confidence = "low"
        best_reasoning.append(f"Model has {len(output_specs)} outputs with unclear patterns")
    
    if not best_reasoning:
        output_shapes = [spec.get('shape', []) for spec in output_specs]
        best_reasoning.append(f"Output shapes {output_shapes} don't match common patterns")
    
    return {
        "type": best_type,
        "confidence": best_confidence,
        "reasoning": "; ".join(best_reasoning),
        "signals_found": {k: len(v) for k, v in signals.items() if v}
    }


def analyze_model_type(model_name: str) -> Dict[str, Any]:
    """
    Analyze model type based on input/output tensor shapes.
    
    Infers model type (detection, classification, segmentation, etc.) from
    tensor shape patterns.
    
    Args:
        model_name: Name of the model to analyze
        
    Returns:
        Dict containing inferred model type and reasoning
    """
    try:
        client = get_client()
        
        input_spec = client.get_model_input_spec(model_name)
        output_specs = client.get_all_output_specs(model_name)
        
        # Analyze output shape patterns
        analysis = infer_model_type_from_shapes(input_spec, output_specs)
        
        # Build warnings based on confidence
        warnings = []
        if analysis['confidence'] == 'low':
            warnings.append("Low confidence inference. Run sample inference to verify model behavior.")
        elif analysis['confidence'] == 'medium':
            warnings.append("Model type inferred heuristically. Verify with sample inference before production use.")
        
        return ok(
            warnings=warnings if warnings else None,
            model_name=model_name,
            inferred_type=analysis['type'],
            confidence=analysis['confidence'],
            reasoning=analysis['reasoning'],
            signals_found=analysis.get('signals_found', {}),
            input_shape=input_spec.get('shape'),
            output_shapes=[spec.get('shape') for spec in output_specs],
            model_capabilities=_get_model_capabilities(analysis['type']),
            inference_warning=warnings[0] if warnings else None
        )
    except Exception as e:
        logger.error(f"Error analyzing model type for {model_name}: {e}")
        return error_response(e, operation="analyze_model_type", model_name=model_name)


# Register the tool
register_tool(
    name="analyze_model_type",
    func=analyze_model_type,
    description="Analyze and infer the model type (classification, detection, segmentation, etc.) based on tensor shape patterns. Helps understand what the model does without prior knowledge.",
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
