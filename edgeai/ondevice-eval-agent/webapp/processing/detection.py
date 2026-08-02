"""Object detection processing with support for multiple formats."""

import logging
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from utils.files import get_class_name
from observability.logging import log_processing_step
from utils.tensor import get_tensor_summary
from utils.visualization import draw_bounding_boxes

logger = logging.getLogger(__name__)


def nms_boxes(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float = 0.45,
    score_threshold: float = 0.25
) -> List[int]:
    """
    Apply Non-Maximum Suppression to filter overlapping boxes.
    Uses cv2.dnn.NMSBoxes for better performance with large anchor counts.
    
    Args:
        boxes: numpy array of shape [N, 4] with [x1, y1, x2, y2] format
        scores: numpy array of shape [N] with confidence scores
        iou_threshold: IoU threshold for suppression
        score_threshold: Minimum score threshold
    
    Returns:
        indices: list of indices to keep
    """
    if len(boxes) == 0:
        return []
    
    # Convert from [x1, y1, x2, y2] to [x, y, w, h] format for cv2.dnn.NMSBoxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    
    w = x2 - x1
    h = y2 - y1
    
    # cv2.dnn.NMSBoxes expects list of [x, y, w, h] and list of scores
    boxes_xywh = np.stack([x1, y1, w, h], axis=1).tolist()
    scores_list = scores.tolist()
    
    try:
        # Use OpenCV's optimized NMS implementation
        indices = cv2.dnn.NMSBoxes(boxes_xywh, scores_list, score_threshold, iou_threshold)
        
        # Handle different OpenCV versions (some return nested list)
        if len(indices) > 0:
            if isinstance(indices[0], (list, np.ndarray)):
                indices = [int(i[0]) for i in indices]
            else:
                indices = [int(i) for i in (indices.flatten() if hasattr(indices, 'flatten') else indices)]
        return indices
    except Exception as e:
        # Fallback to manual NMS if cv2.dnn.NMSBoxes fails
        logger.warning(f"cv2.dnn.NMSBoxes failed, using fallback: {e}")
        return _nms_fallback(boxes, scores, iou_threshold)


def _nms_fallback(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float = 0.45
) -> List[int]:
    """
    Fallback NMS implementation in case cv2.dnn.NMSBoxes is unavailable.
    """
    if len(boxes) == 0:
        return []
    
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        
        if order.size == 1:
            break
        
        # Compute IoU with remaining boxes
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        
        union = areas[i] + areas[order[1:]] - inter
        iou = np.where(union > 0, inter / (union + 1e-6), 0.0)
        
        # Keep boxes with IoU below threshold
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    
    return keep


def detect_output_format(
    output_array: np.ndarray,
    model_name: str
) -> Tuple[str, Dict[str, Any]]:
    """
    Detect the output format of a detection model.
    
    Returns:
        format_type: one of 'yolov8', 'yolov5', 'ssd', 'row_detections', 'unknown'
        info: dict with format-specific information
    """
    shape = output_array.shape
    model_lower = model_name.lower() if model_name else ''
    
    logger.info(f"Detecting output format for shape {shape}, model: {model_name}")
    
    # Remove batch dimension if present
    if len(shape) >= 2 and shape[0] == 1:
        shape = shape[1:]
    
    if len(shape) == 2:
        dim1, dim2 = shape
        
        # Check for YOLOv8/v11 model hints in name
        is_yolo_v8_v11 = any(kw in model_lower for kw in ['yolov8', 'yolov11', 'yolo11', 'ultralytics'])
        
        # YOLOv8/v11 format: [num_features, num_anchors] - features x anchors
        # num_features = 4 (bbox: cx, cy, w, h) + num_classes
        # For COCO (80 classes): [84, 8400]
        # For single-class models (e.g., face detection): [5, 8400] or similar
        # YOLOv5 style with objectness: [85, 8400] = 4 + 1 (objectness) + 80
        
        # YOLOv8/v11 format detection: [features, anchors] where anchors > 1000
        if dim2 > 1000 and dim1 >= 5:
            # This is [features, anchors] format
            has_objectness = dim1 == 85  # Special case for YOLOv5-style output
            num_classes = dim1 - 5 if has_objectness else dim1 - 4
            logger.info(f"Detected YOLOv8/v11 format: {dim1} features, {dim2} anchors, {num_classes} classes")
            return 'yolov8', {'num_classes': num_classes, 'num_anchors': dim2, 'has_objectness': has_objectness}
        
        # YOLOv8/v11 transposed: [anchors, features] where anchors > 1000
        if dim1 > 1000 and dim2 >= 5:
            # This is [anchors, features] format  
            has_objectness = dim2 == 85
            num_classes = dim2 - 5 if has_objectness else dim2 - 4
            logger.info(f"Detected YOLOv8/v11 transposed format: {dim1} anchors, {dim2} features, {num_classes} classes")
            return 'yolov8_transposed', {'num_classes': num_classes, 'num_anchors': dim1, 'has_objectness': has_objectness}
        
        # YOLOv5 format: [num_anchors, 5 + num_classes] where 5 = x,y,w,h,objectness
        if dim2 > 5 and dim1 > 100:
            # Check if second dim looks like 5 + classes (common: 85 for COCO)
            if dim2 in [85, 25, 6, 7, 8]:  # Common values: 85=COCO, 25=20cls, small numbers for custom
                num_classes = dim2 - 5
                return 'yolov5', {'num_classes': num_classes, 'num_anchors': dim1}
        
        # Row-based detections: [N, 4/5/6/7] where N is num detections
        if dim2 <= 10 and dim1 < 10000:
            return 'row_detections', {'num_detections': dim1, 'values_per_det': dim2}
        
        # Large number of detections with few values each
        if dim2 >= 4 and dim2 <= 100 and dim1 > 100:
            # Could be [num_anchors, num_values] - need to check values
            return 'row_detections', {'num_detections': dim1, 'values_per_det': dim2}
    
    elif len(shape) == 3:
        # Some models output [batch, num_detections, values]
        batch, dim1, dim2 = shape
        if batch == 1:
            # Recurse with 2D shape
            return detect_output_format(output_array.reshape(dim1, dim2), model_name)
    
    return 'unknown', {}


def process_yolov8_output(
    output_array: np.ndarray,
    input_width: int,
    input_height: int,
    confidence_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    model_name: Optional[str] = None,
    is_transposed: bool = False
) -> List[Dict[str, Any]]:
    """
    Process YOLOv8/v11 raw output format.
    
    YOLOv8/v11 output: [batch, num_features, num_anchors] where:
    - num_features = 4 (cx, cy, w, h) + num_classes (class probabilities)
    - num_anchors = total anchor predictions (e.g., 8400)
    
    For single-class models (e.g., face detection): [batch, 5, 8400]
    For COCO 80-class models: [batch, 84, 8400]
    
    Args:
        output_array: Raw model output
        input_width, input_height: Model input dimensions for scaling
        confidence_threshold: Minimum confidence to keep
        iou_threshold: NMS IoU threshold
        model_name: Model name for class name lookup
        is_transposed: If True, input is already [anchors, features]
    
    Returns:
        List of detection dicts with normalized [0,1] bounding boxes
    """
    original_shape = output_array.shape
    logger.info(f"YOLOv8/v11 processing - Original shape: {original_shape}, is_transposed: {is_transposed}")
    
    # Remove batch dimension
    if len(output_array.shape) == 3:
        output_array = output_array[0]
    
    logger.info(f"After batch removal: {output_array.shape}")
    
    # Determine the correct orientation
    # YOLOv8/v11 standard format is [features, anchors] where anchors >> features
    # features = 4 (bbox) + num_classes
    dim0, dim1 = output_array.shape
    
    # Heuristic: the larger dimension is the number of anchors
    if not is_transposed:
        if dim0 > dim1:
            # Already [anchors, features] - no transpose needed
            logger.info(f"Shape {output_array.shape}: dim0 > dim1, already [anchors, features]")
            is_transposed = True
        else:
            # [features, anchors] - need to transpose
            logger.info(f"Shape {output_array.shape}: dim0 <= dim1, need to transpose from [features, anchors]")
            output_array = output_array.T
    
    # Now shape should be [num_anchors, num_features]
    num_anchors = output_array.shape[0]
    num_values = output_array.shape[1]
    num_classes = num_values - 4
    
    logger.info(f"Processing: {num_anchors} anchors, {num_values} values per anchor, {num_classes} class(es)")
    
    # Extract bbox and class scores
    boxes_cxcywh = output_array[:, :4]  # [cx, cy, w, h]
    class_scores = output_array[:, 4:]   # [num_classes]
    
    # Log coordinate and score statistics for debugging
    cx_vals = boxes_cxcywh[:, 0]
    cy_vals = boxes_cxcywh[:, 1]
    w_vals = boxes_cxcywh[:, 2]
    h_vals = boxes_cxcywh[:, 3]
    
    logger.info(f"Bbox stats - cx: [{cx_vals.min():.2f}, {cx_vals.max():.2f}], "
                f"cy: [{cy_vals.min():.2f}, {cy_vals.max():.2f}], "
                f"w: [{w_vals.min():.2f}, {w_vals.max():.2f}], "
                f"h: [{h_vals.min():.2f}, {h_vals.max():.2f}]")
    
    # Log class score statistics
    score_min, score_max = class_scores.min(), class_scores.max()
    logger.info(f"Class score range: [{score_min:.4f}, {score_max:.4f}]")
    
    # Check if class scores need sigmoid (raw logits vs probabilities)
    # YOLO typically outputs sigmoid probabilities (0-1), but some exports output logits
    if score_max > 1.0 or score_min < 0.0:
        logger.info("Applying sigmoid to class scores (appear to be raw logits)")
        class_scores = 1.0 / (1.0 + np.exp(-np.clip(class_scores, -500, 500)))
        logger.info(f"After sigmoid - score range: [{class_scores.min():.4f}, {class_scores.max():.4f}]")
    
    # Determine if bbox coordinates are normalized (0-1) or in pixel space
    max_coord = max(cx_vals.max(), cy_vals.max())
    max_size = max(w_vals.max(), h_vals.max())
    
    # Coordinates are in pixel space if max center coord > 1 or max size > 1
    coords_are_normalized = max_coord <= 1.0 and max_size <= 1.0
    
    if coords_are_normalized:
        logger.info("Coordinates appear normalized (0-1 range)")
    else:
        logger.info(f"Coordinates in pixel space (max center: {max_coord:.2f}, max size: {max_size:.2f})")
    
    # Get max class score and class id for each anchor
    class_ids = np.argmax(class_scores, axis=1)
    confidences = np.max(class_scores, axis=1)
    
    # Log unique class IDs found
    unique_before = np.unique(class_ids[confidences > 0.1])
    logger.info(f"Unique class IDs (conf > 0.1): {unique_before[:20]}{'...' if len(unique_before) > 20 else ''}")
    
    # Filter by confidence
    mask = confidences > confidence_threshold
    boxes_cxcywh = boxes_cxcywh[mask]
    class_ids = class_ids[mask]
    confidences = confidences[mask]
    
    logger.info(f"After confidence filter ({confidence_threshold}): {len(boxes_cxcywh)} candidates")
    
    if len(boxes_cxcywh) == 0:
        return []
    
    # Convert cx,cy,w,h to x1,y1,x2,y2
    cx, cy, w, h = boxes_cxcywh[:, 0], boxes_cxcywh[:, 1], boxes_cxcywh[:, 2], boxes_cxcywh[:, 3]
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)
    
    # Log sample boxes before normalization
    if len(boxes_xyxy) > 0:
        sample_box = boxes_xyxy[0]
        logger.info(f"Sample box before norm: [{sample_box[0]:.2f}, {sample_box[1]:.2f}, {sample_box[2]:.2f}, {sample_box[3]:.2f}]")
    
    # Apply NMS per class
    detections = []
    unique_classes = np.unique(class_ids)
    
    for cls_id in unique_classes:
        cls_mask = class_ids == cls_id
        cls_boxes = boxes_xyxy[cls_mask]
        cls_scores = confidences[cls_mask]
        
        keep_indices = nms_boxes(cls_boxes, cls_scores, iou_threshold)
        
        for idx in keep_indices:
            box = cls_boxes[idx]
            
            # Normalize to 0-1 range if coordinates are in pixel space
            if coords_are_normalized:
                # Already normalized, clamp to valid range
                norm_box = [
                    float(max(0.0, min(1.0, box[0]))),
                    float(max(0.0, min(1.0, box[1]))),
                    float(max(0.0, min(1.0, box[2]))),
                    float(max(0.0, min(1.0, box[3])))
                ]
            else:
                # Normalize pixel coordinates to [0, 1]
                norm_box = [
                    float(max(0.0, min(1.0, box[0] / input_width))),
                    float(max(0.0, min(1.0, box[1] / input_height))),
                    float(max(0.0, min(1.0, box[2] / input_width))),
                    float(max(0.0, min(1.0, box[3] / input_height)))
                ]
            
            detections.append({
                'bbox': norm_box,
                'confidence': float(cls_scores[idx]),
                'class_id': int(cls_id),
                'class_name': get_class_name(int(cls_id), model_name)
            })
    
    logger.info(f"After NMS: {len(detections)} final detections")
    if detections:
        det = detections[0]
        logger.info(f"Sample detection: class={det['class_id']}, conf={det['confidence']:.4f}, bbox={det['bbox']}")
    
    # Sort by confidence
    detections.sort(key=lambda x: x['confidence'], reverse=True)
    
    return detections


def process_yolov5_output(
    output_array: np.ndarray,
    input_width: int,
    input_height: int,
    confidence_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    model_name: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Process YOLOv5 raw output format.
    
    YOLOv5 output: [batch, num_anchors, 5 + num_classes] where:
    - 5 = cx, cy, w, h, objectness
    - num_classes = class probabilities
    
    Args:
        output_array: Raw model output
        input_width, input_height: Model input dimensions for scaling
        confidence_threshold: Minimum confidence to keep
        iou_threshold: NMS IoU threshold
        model_name: Model name for class name lookup
    
    Returns:
        List of detection dicts with normalized [0,1] bounding boxes
    """
    # Remove batch dimension
    if len(output_array.shape) == 3:
        output_array = output_array[0]
    
    num_anchors = output_array.shape[0]
    num_values = output_array.shape[1]
    num_classes = num_values - 5
    
    logger.info(f"Processing YOLOv5 output: {num_anchors} anchors, {num_classes} classes, input size: {input_width}x{input_height}")
    
    # Extract components
    boxes_cxcywh = output_array[:, :4]   # [cx, cy, w, h]
    objectness = output_array[:, 4]       # objectness score
    class_scores = output_array[:, 5:]    # class probabilities
    
    # Log coordinate statistics for debugging
    cx_vals = boxes_cxcywh[:, 0]
    cy_vals = boxes_cxcywh[:, 1]
    w_vals = boxes_cxcywh[:, 2]
    h_vals = boxes_cxcywh[:, 3]
    max_coord = max(cx_vals.max(), cy_vals.max(), w_vals.max(), h_vals.max())
    coords_are_normalized = max_coord <= 1.0
    
    logger.info(f"Raw bbox stats - max coord: {max_coord:.2f}, normalized: {coords_are_normalized}")
    
    # Combined confidence = objectness * class_prob
    class_ids = np.argmax(class_scores, axis=1)
    class_probs = np.max(class_scores, axis=1)
    confidences = objectness * class_probs
    
    # Filter by confidence
    mask = confidences > confidence_threshold
    boxes_cxcywh = boxes_cxcywh[mask]
    class_ids = class_ids[mask]
    confidences = confidences[mask]
    
    if len(boxes_cxcywh) == 0:
        return []
    
    # Convert cx,cy,w,h to x1,y1,x2,y2
    cx, cy, w, h = boxes_cxcywh[:, 0], boxes_cxcywh[:, 1], boxes_cxcywh[:, 2], boxes_cxcywh[:, 3]
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)
    
    # Apply NMS per class
    detections = []
    unique_classes = np.unique(class_ids)
    
    for cls_id in unique_classes:
        cls_mask = class_ids == cls_id
        cls_boxes = boxes_xyxy[cls_mask]
        cls_scores = confidences[cls_mask]
        
        keep_indices = nms_boxes(cls_boxes, cls_scores, iou_threshold)
        
        for idx in keep_indices:
            box = cls_boxes[idx]
            
            # Normalize to 0-1 range if coordinates are in pixel space
            if coords_are_normalized:
                norm_box = [
                    float(max(0.0, min(1.0, box[0]))),
                    float(max(0.0, min(1.0, box[1]))),
                    float(max(0.0, min(1.0, box[2]))),
                    float(max(0.0, min(1.0, box[3])))
                ]
            else:
                norm_box = [
                    float(max(0.0, min(1.0, box[0] / input_width))),
                    float(max(0.0, min(1.0, box[1] / input_height))),
                    float(max(0.0, min(1.0, box[2] / input_width))),
                    float(max(0.0, min(1.0, box[3] / input_height)))
                ]
            
            detections.append({
                'bbox': norm_box,
                'confidence': float(cls_scores[idx]),
                'class_id': int(cls_id),
                'class_name': get_class_name(int(cls_id), model_name)
            })
    
    # Sort by confidence
    detections.sort(key=lambda x: x['confidence'], reverse=True)
    
    return detections


def process_row_detections(
    output_array: np.ndarray,
    input_width: int,
    input_height: int,
    confidence_threshold: float = 0.25,
    model_name: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Process row-based detection output format.
    
    Common formats:
    - [N, 4]: [x1, y1, x2, y2]
    - [N, 5]: [x1, y1, x2, y2, conf]
    - [N, 6]: [x1, y1, x2, y2, conf, class_id]
    - [N, 7]: [batch_id, x1, y1, x2, y2, conf, class_id] (some TensorRT models)
    
    Also handles center-format if detected.
    """
    # Remove batch dimension
    if len(output_array.shape) == 3:
        output_array = output_array[0]
    
    num_detections, values_per_det = output_array.shape
    logger.info(f"Processing row detections: {num_detections} x {values_per_det}")
    
    detections = []
    
    for det in output_array:
        # Skip empty/padding rows (all zeros or very low values)
        if np.all(det[:4] == 0) or np.max(np.abs(det[:4])) < 1e-6:
            continue
        
        # Parse based on number of values
        if values_per_det >= 7:
            # [batch_id, x1, y1, x2, y2, conf, class_id]
            x1, y1, x2, y2, conf, class_id = det[1], det[2], det[3], det[4], det[5], det[6]
        elif values_per_det == 6:
            # [x1, y1, x2, y2, conf, class_id]
            x1, y1, x2, y2, conf, class_id = det[0], det[1], det[2], det[3], det[4], det[5]
        elif values_per_det == 5:
            # [x1, y1, x2, y2, conf]
            x1, y1, x2, y2, conf = det[0], det[1], det[2], det[3], det[4]
            class_id = 0
        elif values_per_det == 4:
            # [x1, y1, x2, y2]
            x1, y1, x2, y2 = det[0], det[1], det[2], det[3]
            conf = 1.0
            class_id = 0
        else:
            continue
        
        # Check if this might be center format (w/h instead of x2/y2)
        # Heuristic: if x2 < x1 or y2 < y1, or values are small, might be cx,cy,w,h
        if x2 < x1 or y2 < y1:
            # Assume cx, cy, w, h format
            cx, cy, w, h = x1, y1, x2, y2
            x1 = cx - w / 2
            y1 = cy - h / 2
            x2 = cx + w / 2
            y2 = cy + h / 2
        
        # Skip invalid boxes
        if x2 <= x1 or y2 <= y1:
            continue
        
        if conf > confidence_threshold:
            # Determine if coordinates are in pixel space or normalized [0,1]
            max_coord = max(abs(x1), abs(y1), abs(x2), abs(y2))
            
            # If coordinates are clearly larger than 1, normalize them
            is_pixel_coords = max_coord > 2.0 or (max_coord > 1.0 and (input_width > 10 and input_height > 10))
            
            if is_pixel_coords:
                # Normalize pixel coordinates to [0, 1]
                x1, x2 = x1 / input_width, x2 / input_width
                y1, y2 = y1 / input_height, y2 / input_height
            
            # Clamp to valid range
            x1 = max(0.0, min(1.0, x1))
            y1 = max(0.0, min(1.0, y1))
            x2 = max(0.0, min(1.0, x2))
            y2 = max(0.0, min(1.0, y2))
            
            detections.append({
                'bbox': [float(x1), float(y1), float(x2), float(y2)],
                'confidence': float(conf),
                'class_id': int(class_id),
                'class_name': get_class_name(int(class_id), model_name)
            })
    
    return detections


def process_object_detection(
    prediction: Dict[str, Any],
    response: Dict[str, Any],
    filepath: str,
    filename: str,
    model_name: str,
    inference_time: float,
    start_request_time: float,
    input_spec: Dict[str, Any],
    output_spec: Dict[str, Any],
    image_array: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Process object detection results and draw bounding boxes.
    
    Supports multiple detection model output formats:
    - YOLOv8: [batch, 84, 8400] or [batch, 8400, 84]
    - YOLOv5: [batch, num_anchors, 5 + num_classes]
    - SSD/TF-style: Multiple outputs (boxes, scores, classes)
    - Row-based: [N, 4/5/6/7] format
    """
    try:
        detections: List[Dict[str, Any]] = []
        confidence_threshold = 0.25
        iou_threshold = 0.45
        
        # Get input dimensions for coordinate scaling
        input_width = input_spec.get('width', 640)
        input_height = input_spec.get('height', 640)
        
        # Check if we have outputs
        if 'outputs' not in response or len(response['outputs']) == 0:
            log_processing_step("Detection Error", "No output data found", "error")
            return {'success': False, 'error': 'No detection output found'}
        
        outputs = response['outputs']
        num_outputs = len(outputs)
        
        logger.info(f"Detection model has {num_outputs} output(s)")
        
        # Case 1: Multiple outputs (boxes, scores, classes) - common in TensorFlow/SSD models
        if num_outputs >= 3:
            boxes, scores, classes = None, None, None
            
            for output in outputs:
                data = np.array(output['data'])
                shape = output.get('shape', [])
                name = output.get('name', '').lower()
                
                if shape:
                    data = data.reshape(shape)
                
                # Identify output type by name or shape
                if 'box' in name or 'bbox' in name:
                    boxes = data
                elif 'score' in name or 'conf' in name:
                    scores = data
                elif 'class' in name or 'label' in name:
                    classes = data
                else:
                    # Try to guess by shape
                    if len(shape) >= 2 and shape[-1] == 4:
                        boxes = data
                    elif len(shape) >= 1 and len(data.flatten()) <= 1000:
                        if scores is None:
                            scores = data
                        elif classes is None:
                            classes = data
            
            # Parse detections from separate outputs
            if boxes is not None:
                boxes = boxes.reshape(-1, 4)
                if scores is not None:
                    scores = scores.flatten()
                else:
                    scores = np.ones(len(boxes))
                if classes is not None:
                    classes = classes.flatten()
                else:
                    classes = np.zeros(len(boxes))
                
                for i, (bbox, conf, cls) in enumerate(zip(boxes, scores, classes)):
                    if conf > confidence_threshold:
                        # Normalize if needed
                        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
                        if x1 > 1 or x2 > 1:
                            x1, x2 = x1 / input_width, x2 / input_width
                            y1, y2 = y1 / input_height, y2 / input_height
                        
                        detections.append({
                            'bbox': [float(x1), float(y1), float(x2), float(y2)],
                            'confidence': float(conf),
                            'class_id': int(cls),
                            'class_name': get_class_name(int(cls), model_name)
                        })
            
            log_processing_step("Detection Format", "Multi-output (SSD/TF style)", "info")
        
        # Case 2: Single output - detect format automatically
        else:
            output_array = np.array(outputs[0]['data'])
            output_shape = outputs[0].get('shape', [])
            
            if output_shape:
                output_array = output_array.reshape(output_shape)
            
            logger.info(f"Single output detection shape: {output_array.shape}")
            
            # Detect the output format
            format_type, format_info = detect_output_format(output_array, model_name)
            
            logger.info(f"Detected format: {format_type}, info: {format_info}")
            log_processing_step("Detection Format", f"{format_type}", "info")
            
            if format_type == 'yolov8':
                detections = process_yolov8_output(
                    output_array, input_width, input_height,
                    confidence_threshold, iou_threshold, model_name,
                    is_transposed=False
                )
            elif format_type == 'yolov8_transposed':
                detections = process_yolov8_output(
                    output_array, input_width, input_height,
                    confidence_threshold, iou_threshold, model_name,
                    is_transposed=True
                )
            elif format_type == 'yolov5':
                detections = process_yolov5_output(
                    output_array, input_width, input_height,
                    confidence_threshold, iou_threshold, model_name
                )
            elif format_type == 'row_detections':
                detections = process_row_detections(
                    output_array, input_width, input_height,
                    confidence_threshold, model_name
                )
            else:
                # Unknown format - try row detections as fallback
                logger.warning(f"Unknown detection format, trying row-based parsing")
                try:
                    # Remove batch dim if present
                    if len(output_array.shape) == 3 and output_array.shape[0] == 1:
                        output_array = output_array[0]
                    
                    if len(output_array.shape) == 2:
                        detections = process_row_detections(
                            output_array, input_width, input_height,
                            confidence_threshold, model_name
                        )
                except Exception as e:
                    logger.error(f"Fallback parsing failed: {e}")
        
        log_processing_step("Object Detection", 
                          f"Processed {num_outputs} output(s), found {len(detections)} objects", 
                          "success")
        
        # Draw bounding boxes on image
        annotated_image_base64 = None
        if detections:
            annotated_image_base64 = draw_bounding_boxes(filepath, detections)
        
        log_processing_step("Detections Found", f"Found {len(detections)} objects", "success")
        
        total_time = time.time() - start_request_time
        
        # Extract raw tensor information
        output_tensor_info: Dict[str, Any] = {}
        if outputs and len(outputs) > 0:
            # Summarize all outputs
            all_outputs_info = []
            for idx, output in enumerate(outputs):
                output_array = np.array(output.get('data', []))
                out_shape = output.get('shape', [])
                if out_shape:
                    output_array = output_array.reshape(out_shape)
                info = get_tensor_summary(output_array)
                info['shape'] = out_shape
                info['name'] = output.get('name', f'output_{idx}')
                all_outputs_info.append(info)
            output_tensor_info = all_outputs_info[0] if len(all_outputs_info) == 1 else {'outputs': all_outputs_info}
        
        # Input tensor info
        input_tensor_info: Dict[str, Any] = {}
        if image_array is not None:
            input_tensor_info = get_tensor_summary(image_array)
            input_tensor_info['shape'] = list(image_array.shape)
            input_tensor_info['name'] = input_spec.get('name', 'input')
        
        result = {
            'success': True,
            'task_type': 'detection',
            'detected_type': 'detection',
            'model_name': model_name,
            'latency': inference_time,
            'total_time': total_time,
            'detections': detections,
            'num_detections': len(detections),
            'annotated_image': annotated_image_base64,
            'image_filename': filename,
            'model_spec': {
                'input': {
                    'name': input_spec.get('name', 'input'),
                    'shape': input_spec.get('shape', []),
                    'datatype': input_spec.get('datatype', 'FP32'),
                    'format': input_spec.get('format', 'NCHW'),
                    'size': f"{input_spec.get('width', 'unknown')}x{input_spec.get('height', 'unknown')}"
                },
                'output': {
                    'name': output_spec.get('name', 'output'),
                    'shape': output_spec.get('shape', []),
                    'datatype': output_spec.get('datatype', 'FP32')
                }
            },
            'tensor_info': {
                'input': input_tensor_info,
                'output': output_tensor_info,
                'num_output_tensors': num_outputs
            }
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Error processing object detection: {e}")
        log_processing_step("Detection Error", str(e), "error")
        return {
            'success': False,
            'error': f'Object detection processing failed: {str(e)}'
        }
