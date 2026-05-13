"""
DETR (DEtection TRansformer) model processing.

Handles the special requirements for DETR models:
- Two inputs: pixel_values and pixel_mask
- Transformer-based architecture outputs
- Special post-processing for detection results
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import requests

logger = logging.getLogger(__name__)

# DETR uses COCO class labels (91 classes + background)
COCO_CLASSES = [
    "N/A", "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "N/A", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "N/A", "backpack", "umbrella",
    "N/A", "N/A", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "N/A", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "N/A", "dining table", "N/A", "N/A",
    "toilet", "N/A", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "N/A", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush"
]


def preprocess_detr(
    image_bytes: bytes,
    target_size: Tuple[int, int] = (800, 800),
    mask_size: Optional[Tuple[int, int]] = None,
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
    std: Tuple[float, ...] = (0.229, 0.224, 0.225),
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
    """
    Preprocess image for DETR model.
    
    Args:
        image_bytes: Raw image bytes
        target_size: Target size (height, width) for pixel_values
        mask_size: Target size (height, width) for pixel_mask.
                   If None, uses target_size. Some DETR ONNX exports
                   have a fixed mask resolution (e.g. 64x64) that
                   differs from the image resolution.
        mean: Normalization mean (ImageNet)
        std: Normalization std (ImageNet)
        
    Returns:
        pixel_values: Preprocessed image tensor [1, 3, H, W]
        pixel_mask: Mask tensor [1, mask_H, mask_W]
        original_size: Original image size (height, width)
    """
    # Decode image
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image is None:
        raise ValueError("Failed to decode image")
    
    original_size = (image.shape[0], image.shape[1])  # (H, W)
    
    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize while maintaining aspect ratio
    h, w = image.shape[:2]
    target_h, target_w = target_size
    
    # Calculate scale to fit within target size
    scale = min(target_h / h, target_w / w)
    new_h, new_w = int(h * scale), int(w * scale)
    
    # Resize image
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Create padded image (pad to target size)
    padded = np.zeros((target_h, target_w, 3), dtype=np.float32)
    padded[:new_h, :new_w, :] = resized
    
    # Normalize to [0, 1] then apply ImageNet normalization
    padded = padded / 255.0
    padded = (padded - np.array(mean)) / np.array(std)
    
    # Convert to CHW format and add batch dimension
    pixel_values = np.transpose(padded, (2, 0, 1))  # [3, H, W]
    pixel_values = np.expand_dims(pixel_values, axis=0).astype(np.float32)  # [1, 3, H, W]
    
    # Create pixel mask at the required resolution
    # Some DETR ONNX exports have a fixed mask size (e.g. 64x64) that
    # differs from the pixel_values resolution. The mask indicates which
    # spatial locations contain real image content vs. padding.
    mask_h, mask_w = mask_size if mask_size is not None else (target_h, target_w)
    
    if mask_h == target_h and mask_w == target_w:
        # Mask matches image resolution — build directly
        pixel_mask = np.zeros((1, mask_h, mask_w), dtype=np.int64)
        pixel_mask[0, :new_h, :new_w] = 1
    else:
        # Mask has a different (typically smaller) resolution.
        # Build the full-resolution mask first, then resize.
        full_mask = np.zeros((target_h, target_w), dtype=np.uint8)
        full_mask[:new_h, :new_w] = 1
        resized_mask = cv2.resize(
            full_mask, (mask_w, mask_h), interpolation=cv2.INTER_NEAREST
        )
        pixel_mask = np.expand_dims(resized_mask.astype(np.int64), axis=0)  # [1, mask_H, mask_W]
    
    logger.debug(
        f"DETR preprocess: original={original_size}, "
        f"pixel_values={pixel_values.shape}, pixel_mask={pixel_mask.shape}"
    )
    
    return pixel_values, pixel_mask, original_size


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Compute softmax values."""
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def box_cxcywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """
    Convert boxes from center format (cx, cy, w, h) to corner format (x1, y1, x2, y2).
    """
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return np.stack([x1, y1, x2, y2], axis=1)


def postprocess_detr(
    logits: np.ndarray,
    pred_boxes: np.ndarray,
    original_size: Tuple[int, int],
    target_size: Tuple[int, int] = (800, 800),
    threshold: float = 0.7,
    max_detections: int = 100,
) -> List[Dict[str, Any]]:
    """
    Post-process DETR model outputs.
    
    Args:
        logits: Classification logits [batch, num_queries, num_classes]
        pred_boxes: Predicted boxes [batch, num_queries, 4] in cxcywh format (normalized)
        original_size: Original image size (height, width)
        target_size: Model input size (height, width)
        threshold: Confidence threshold
        max_detections: Maximum number of detections to return
        
    Returns:
        List of detection dicts with score, label, box
    """
    # Remove batch dimension
    logits = logits[0]  # [num_queries, num_classes]
    pred_boxes = pred_boxes[0]  # [num_queries, 4]
    
    # Apply softmax to get probabilities
    probs = softmax(logits, axis=-1)
    
    # Get best class for each query (excluding the last class which is "no object")
    # DETR has num_classes + 1 outputs, last one is "no object"
    scores = np.max(probs[:, :-1], axis=-1)  # Best score excluding no-object class
    labels = np.argmax(probs[:, :-1], axis=-1)  # Best class excluding no-object
    
    # Filter by threshold
    keep = scores > threshold
    scores = scores[keep]
    labels = labels[keep]
    boxes = pred_boxes[keep]
    
    if len(scores) == 0:
        return []
    
    # Convert boxes from normalized cxcywh to absolute xyxy
    boxes = box_cxcywh_to_xyxy(boxes)
    
    # Scale boxes to original image size
    orig_h, orig_w = original_size
    target_h, target_w = target_size
    
    # Calculate the actual used size (with aspect ratio)
    scale = min(target_h / orig_h, target_w / orig_w)
    used_h, used_w = int(orig_h * scale), int(orig_w * scale)
    
    # Scale from target coordinates to original coordinates
    # Boxes are normalized [0, 1] relative to target_size
    boxes[:, 0] = boxes[:, 0] * target_w / scale  # x1
    boxes[:, 1] = boxes[:, 1] * target_h / scale  # y1
    boxes[:, 2] = boxes[:, 2] * target_w / scale  # x2
    boxes[:, 3] = boxes[:, 3] * target_h / scale  # y2
    
    # Clip to image bounds
    boxes[:, 0] = np.clip(boxes[:, 0], 0, orig_w)
    boxes[:, 1] = np.clip(boxes[:, 1], 0, orig_h)
    boxes[:, 2] = np.clip(boxes[:, 2], 0, orig_w)
    boxes[:, 3] = np.clip(boxes[:, 3], 0, orig_h)
    
    # Sort by score
    sorted_indices = np.argsort(scores)[::-1][:max_detections]
    
    # Build results
    detections = []
    for idx in sorted_indices:
        label_id = int(labels[idx])
        label_name = COCO_CLASSES[label_id] if label_id < len(COCO_CLASSES) else f"class_{label_id}"
        
        if label_name == "N/A":
            continue
            
        detections.append({
            "score": float(scores[idx]),
            "label": label_name,
            "label_id": label_id,
            "box": {
                "xmin": int(boxes[idx, 0]),
                "ymin": int(boxes[idx, 1]),
                "xmax": int(boxes[idx, 2]),
                "ymax": int(boxes[idx, 3]),
            }
        })
    
    return detections


def _get_detr_mask_size(
    server_url: str,
    model_name: str,
    timeout: float = 10.0,
) -> Optional[Tuple[int, int]]:
    """
    Query Triton model metadata to discover the pixel_mask dimensions.
    
    Returns:
        (height, width) of the pixel_mask input, or None if metadata
        cannot be fetched (falls back to default behaviour).
    """
    try:
        metadata_url = f"{server_url}/v2/models/{model_name}"
        resp = requests.get(metadata_url, timeout=timeout)
        if resp.status_code != 200:
            return None
        meta = resp.json()
        for inp in meta.get("inputs", []):
            if inp.get("name", "").lower() == "pixel_mask":
                shape = inp.get("shape", [])
                # shape is [batch, H, W] — extract H, W (skip dynamic dims)
                dims = [d for d in shape if isinstance(d, int) and d > 0]
                if len(dims) >= 2:
                    return (dims[-2], dims[-1])
        return None
    except Exception as e:
        logger.warning(f"Could not fetch DETR mask size from metadata: {e}")
        return None


def run_detr_inference(
    server_url: str,
    model_name: str,
    image_bytes: bytes,
    threshold: float = 0.7,
    timeout: float = 30.0,
) -> Dict[str, Any]:
    """
    Run DETR inference on an image.
    
    Automatically queries Triton model metadata to discover the exact
    pixel_mask resolution required by the model, so it works with any
    DETR ONNX export (fixed or dynamic mask sizes).
    
    Args:
        server_url: Triton server URL
        model_name: Model name
        image_bytes: Raw image bytes
        threshold: Detection confidence threshold
        timeout: Request timeout
        
    Returns:
        Dict with detections and metadata
    """
    start_time = time.time()
    
    try:
        # Discover the mask resolution the model expects
        mask_size = _get_detr_mask_size(server_url, model_name, timeout=min(timeout, 10.0))
        if mask_size is not None:
            logger.info(f"DETR model '{model_name}' expects pixel_mask at {mask_size}")
        
        # Preprocess image
        preprocess_start = time.time()
        pixel_values, pixel_mask, original_size = preprocess_detr(
            image_bytes, mask_size=mask_size
        )
        preprocess_time = time.time() - preprocess_start
        
        logger.info(f"DETR preprocessing: {preprocess_time*1000:.1f}ms, "
                   f"original_size={original_size}, "
                   f"pixel_values shape={pixel_values.shape}, "
                   f"pixel_mask shape={pixel_mask.shape}")
        
        # Build inference request with both inputs
        inference_url = f"{server_url}/v2/models/{model_name}/infer"
        
        payload = {
            "inputs": [
                {
                    "name": "pixel_values",
                    "shape": list(pixel_values.shape),
                    "datatype": "FP32",
                    "data": pixel_values.flatten().tolist()
                },
                {
                    "name": "pixel_mask",
                    "shape": list(pixel_mask.shape),
                    "datatype": "INT64",
                    "data": pixel_mask.flatten().tolist()
                }
            ]
        }
        
        # Send inference request
        inference_start = time.time()
        response = requests.post(
            inference_url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=timeout
        )
        inference_time = time.time() - inference_start
        
        if response.status_code != 200:
            error_msg = response.text
            try:
                error_data = response.json()
                error_msg = error_data.get("error", error_msg)
            except:
                pass
            return {
                "success": False,
                "error": f"Inference failed: {error_msg}"
            }
        
        result = response.json()
        
        # Extract outputs
        outputs = {out["name"]: out for out in result.get("outputs", [])}
        
        if "logits" not in outputs or "pred_boxes" not in outputs:
            return {
                "success": False,
                "error": f"Expected 'logits' and 'pred_boxes' outputs, got: {list(outputs.keys())}"
            }
        
        # Reshape outputs
        logits_out = outputs["logits"]
        boxes_out = outputs["pred_boxes"]
        
        logits = np.array(logits_out["data"]).reshape(logits_out["shape"])
        pred_boxes = np.array(boxes_out["data"]).reshape(boxes_out["shape"])
        
        logger.info(f"DETR outputs: logits shape={logits.shape}, pred_boxes shape={pred_boxes.shape}")
        
        # Post-process
        postprocess_start = time.time()
        detections = postprocess_detr(
            logits, 
            pred_boxes, 
            original_size,
            threshold=threshold
        )
        postprocess_time = time.time() - postprocess_start
        
        total_time = time.time() - start_time
        
        return {
            "success": True,
            "detections": detections,
            "detection_count": len(detections),
            "threshold": threshold,
            "original_size": {"height": original_size[0], "width": original_size[1]},
            "timing": {
                "preprocess_ms": round(preprocess_time * 1000, 2),
                "inference_ms": round(inference_time * 1000, 2),
                "postprocess_ms": round(postprocess_time * 1000, 2),
                "total_ms": round(total_time * 1000, 2),
            }
        }
        
    except Exception as e:
        logger.error(f"DETR inference error: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e)
        }


def is_detr_model(model_name: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
    """
    Check if a model is a DETR-style model based on name and metadata.
    
    Args:
        model_name: Model name
        metadata: Optional model metadata
        
    Returns:
        True if model appears to be DETR
    """
    # Check name patterns
    name_lower = model_name.lower()
    if "detr" in name_lower:
        return True
    
    # Check metadata for DETR signature (pixel_values + pixel_mask inputs, logits + pred_boxes outputs)
    if metadata:
        inputs = metadata.get("inputs", [])
        outputs = metadata.get("outputs", [])
        
        input_names = {inp.get("name", "").lower() for inp in inputs}
        output_names = {out.get("name", "").lower() for out in outputs}
        
        has_detr_inputs = "pixel_values" in input_names and "pixel_mask" in input_names
        has_detr_outputs = "logits" in output_names and "pred_boxes" in output_names
        
        if has_detr_inputs and has_detr_outputs:
            return True
    
    return False
