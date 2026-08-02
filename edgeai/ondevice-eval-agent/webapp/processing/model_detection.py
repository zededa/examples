"""Model type detection based on name patterns and output shapes."""

import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Model type detection thresholds and patterns
MODEL_TYPE_PATTERNS = {
    'detection': ['yolo', 'ssd', 'rcnn', 'fasterrcnn', 'retinanet', 'efficientdet', 'detr', 'detectron'],
    'classification': ['resnet', 'vgg', 'efficientnet', 'mobilenet', 'inception', 'densenet', 'alexnet', 'convnext', 'vit'],
    'segmentation': ['unet', 'deeplab', 'fcn', 'segformer', 'segnet', 'pspnet'],
    'panoptic': ['panoptic', 'mask2former', 'maskformer', 'oneformer'],
    'pose': ['pose', 'hrnet', 'simplepose', 'openpose', 'vitpose', 'rtmpose', 'movenet'],
    'keypoint': ['keypoint', 'landmark', 'hourglass', 'cpn', 'rtmdet-pose'],
    'ocr': ['ocr', 'crnn', 'troc', 'paddle', 'easyocr', 'tesseract', 'parseq', 'trocr', 'text_recognition', 'str_']
}

# Output shape patterns for different model types
OUTPUT_SHAPE_PATTERNS = {
    # Classification: [batch, num_classes]
    'classification': lambda shape: len(shape) == 2 and 1 < shape[-1] < 10000,
    
    # Detection: [batch, anchors, values] or [batch, values, anchors]
    'detection': lambda shape: (
        len(shape) >= 2 and 
        (shape[-1] in [4, 5, 6, 7, 84, 85] or 
         (len(shape) >= 2 and shape[-2] in [4, 5, 6, 7, 84, 85]) or
         (len(shape) >= 2 and (shape[-1] > 1000 or shape[-2] > 1000)))
    ),
    
    # Segmentation: [batch, classes, height, width] or [batch, height, width]
    'segmentation': lambda shape: (
        len(shape) >= 3 and 
        shape[-1] > 16 and shape[-2] > 16 and  # Not just small feature maps
        (len(shape) == 3 or (len(shape) == 4 and shape[1] < 256))  # Classes dimension
    ),
    
    # Panoptic: Usually has multiple outputs or specific shapes
    'panoptic': lambda shape: (
        len(shape) >= 4 and shape[-1] > 16 and shape[-2] > 16
    ),
    
    # Pose/Keypoint: [batch, num_keypoints, 2/3] or [batch, num_people, num_keypoints, 2/3]
    'pose': lambda shape: (
        len(shape) >= 2 and 
        (shape[-1] in [2, 3] and 5 <= shape[-2] <= 200) or  # [batch, keypoints, coords]
        (len(shape) >= 3 and shape[-1] in [2, 3] and 5 <= shape[-2] <= 50)  # With confidence
    ),
    
    # Keypoint: Similar to pose but often includes heatmaps
    'keypoint': lambda shape: (
        (len(shape) == 4 and shape[1] > 5 and shape[-1] > 16 and shape[-2] > 16) or  # Heatmaps
        (len(shape) >= 2 and shape[-1] in [2, 3] and 5 <= shape[-2] <= 100)
    ),
    
    # OCR: [batch, sequence_length, vocab_size] or [batch, sequence_length]
    'ocr': lambda shape: (
        len(shape) >= 2 and 
        10 <= shape[-2] <= 500 and  # Sequence length
        (shape[-1] > 26 if len(shape) == 3 else True)  # Vocab size > alphabet
    )
}


def detect_model_type(
    model_name: str,
    output_spec: Optional[Dict[str, Any]],
    num_outputs: int = 1,
    all_output_specs: Optional[List[Dict[str, Any]]] = None
) -> str:
    """
    Auto-detect model type based on model name, output shape, and number of outputs.
    
    WARNING: This detection uses heuristics and may be incorrect. The detection
    confidence varies based on the method used:
    - Name pattern matching: HIGH confidence
    - Output shape analysis: MEDIUM confidence  
    - Default fallback: LOW confidence
    
    Args:
        model_name: Name of the model
        output_spec: Output specification dictionary with 'shape' key
        num_outputs: Number of model outputs
        all_output_specs: List of all output specifications for multi-output models
    
    Returns:
        Model type string: 'classification', 'detection', 'segmentation', 
        'panoptic', 'pose', 'keypoint', or 'ocr'
    """
    model_name_lower = model_name.lower() if model_name else ''
    
    # Check name patterns first (most reliable)
    for model_type, patterns in MODEL_TYPE_PATTERNS.items():
        for pattern in patterns:
            if pattern in model_name_lower:
                logger.info(f"Model type detected from name pattern '{pattern}': {model_type} (HIGH confidence)")
                return model_type
    
    # Infer from output shape (MEDIUM confidence)
    # Handle None or missing output_spec
    if output_spec is None:
        output_spec = {}
    output_shape = output_spec.get('shape', [])
    if output_shape is None:
        output_shape = []
    
    # Remove batch dimension if present
    if len(output_shape) >= 1 and output_shape[0] == 1:
        shape_without_batch = output_shape[1:] if len(output_shape) > 1 else output_shape
    else:
        shape_without_batch = output_shape
    
    # Check output shape patterns
    for model_type, pattern_fn in OUTPUT_SHAPE_PATTERNS.items():
        try:
            if pattern_fn(shape_without_batch):
                logger.info(f"Model type inferred from output shape {shape_without_batch}: {model_type} (MEDIUM confidence)")
                return model_type
        except Exception:
            continue
    
    # Multi-output models often indicate specific types
    if num_outputs >= 3:
        # Multiple outputs often indicate detection or panoptic
        if all_output_specs:
            has_boxes = any('box' in s.get('name', '').lower() for s in all_output_specs)
            has_masks = any('mask' in s.get('name', '').lower() for s in all_output_specs)
            has_keypoints = any('keypoint' in s.get('name', '').lower() or 'pose' in s.get('name', '').lower() for s in all_output_specs)
            
            if has_masks and has_boxes:
                logger.info(f"Model type detected from multiple outputs with boxes+masks: panoptic (MEDIUM confidence)")
                return 'panoptic'
            if has_keypoints:
                logger.info(f"Model type detected from multiple outputs with keypoints: pose (MEDIUM confidence)")
                return 'pose'
            if has_boxes:
                logger.info(f"Model type detected from multiple outputs with boxes: detection (MEDIUM confidence)")
                return 'detection'
    
    # Default to classification (LOW confidence - may be wrong!)
    logger.warning(f"Model type defaulting to classification for shape {output_shape} (LOW confidence - may be incorrect)")
    return 'classification'
