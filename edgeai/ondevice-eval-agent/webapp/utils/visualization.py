"""Visualization utilities for drawing annotations on images."""

import base64
import logging
import traceback
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Bounding box colors (BGR format for OpenCV)
BBOX_COLORS = [
    (0, 255, 0),    # Green
    (255, 0, 0),    # Blue
    (0, 0, 255),    # Red
    (255, 255, 0),  # Cyan
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Yellow
    (128, 255, 0),  # Light green
    (255, 128, 0),  # Orange-blue
    (128, 0, 255),  # Purple
    (255, 255, 128),# Light cyan
]

# Pose estimation colors
POSE_COLORS = [
    (255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0),
    (170, 255, 0), (85, 255, 0), (0, 255, 0), (0, 255, 85),
    (0, 255, 170), (0, 255, 255), (0, 170, 255), (0, 85, 255),
    (0, 0, 255), (85, 0, 255), (170, 0, 255), (255, 0, 255), (255, 0, 170)
]

# Color palette for segmentation classes
SEGMENTATION_COLORS = [
    [0, 0, 0],       # Background
    [128, 0, 0],     # Class 1
    [0, 128, 0],     # Class 2
    [128, 128, 0],   # Class 3
    [0, 0, 128],     # Class 4
    [128, 0, 128],   # Class 5
    [0, 128, 128],   # Class 6
    [128, 128, 128], # Class 7
    [64, 0, 0],      # Class 8
    [192, 0, 0],     # Class 9
    [64, 128, 0],    # Class 10
    [192, 128, 0],   # Class 11
    [64, 0, 128],    # Class 12
    [192, 0, 128],   # Class 13
    [64, 128, 128],  # Class 14
    [192, 128, 128], # Class 15
    [0, 64, 0],      # Class 16
    [128, 64, 0],    # Class 17
    [0, 192, 0],     # Class 18
    [128, 192, 0],   # Class 19
    [0, 64, 128],    # Class 20
]

# Standard keypoint connections for COCO pose models
POSE_SKELETON_COCO = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # Head
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
    (5, 11), (6, 12), (11, 12),  # Torso
    (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
]


def draw_bounding_boxes(
    image_path: str,
    detections: List[Dict[str, Any]]
) -> Optional[str]:
    """Draw bounding boxes on image and return base64 encoded result."""
    try:
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Failed to read image: {image_path}")
            return None
        
        height, width = image.shape[:2]
        
        for det in detections:
            bbox = det['bbox']
            conf = det['confidence']
            class_id = det['class_id']
            class_name = det['class_name']
            
            # Convert normalized coordinates to pixel coordinates if needed
            x1, y1, x2, y2 = bbox
            if x1 <= 1.0 and y1 <= 1.0 and x2 <= 1.0 and y2 <= 1.0:  # Normalized coordinates
                x1, x2 = int(x1 * width), int(x2 * width)
                y1, y2 = int(y1 * height), int(y2 * height)
            else:  # Pixel coordinates
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Choose color based on class
            color = BBOX_COLORS[class_id % len(BBOX_COLORS)]
            
            # Draw rectangle
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            # Draw label background
            label = f"{class_name}: {conf:.2f}"
            (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(image, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)
            
            # Draw label text
            cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Convert to base64
        _, buffer = cv2.imencode('.jpg', image)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return img_base64
        
    except Exception as e:
        logger.error(f"Error drawing bounding boxes: {e}")
        traceback.print_exc()
        return None


def draw_pose_keypoints(
    image_path: str,
    poses: List[Dict[str, Any]]
) -> Optional[str]:
    """Draw pose keypoints and skeleton on image."""
    try:
        image = cv2.imread(image_path)
        if image is None:
            return None
        
        height, width = image.shape[:2]
        
        for pose in poses:
            keypoints = pose['keypoints']
            
            # Draw keypoints
            for kp in keypoints:
                if kp['confidence'] > 0.3:
                    x = int(kp['x'] * width)
                    y = int(kp['y'] * height)
                    color = POSE_COLORS[kp['id'] % len(POSE_COLORS)]
                    cv2.circle(image, (x, y), 5, color, -1)
                    cv2.circle(image, (x, y), 7, (255, 255, 255), 1)
            
            # Draw skeleton
            for (start_idx, end_idx) in POSE_SKELETON_COCO:
                if start_idx < len(keypoints) and end_idx < len(keypoints):
                    start_kp = keypoints[start_idx]
                    end_kp = keypoints[end_idx]
                    
                    if start_kp['confidence'] > 0.3 and end_kp['confidence'] > 0.3:
                        start_point = (int(start_kp['x'] * width), int(start_kp['y'] * height))
                        end_point = (int(end_kp['x'] * width), int(end_kp['y'] * height))
                        color = POSE_COLORS[start_idx % len(POSE_COLORS)]
                        cv2.line(image, start_point, end_point, color, 2)
        
        _, buffer = cv2.imencode('.jpg', image)
        return base64.b64encode(buffer).decode('utf-8')
        
    except Exception as e:
        logger.error(f"Error drawing pose: {e}")
        return None


def draw_segmentation_mask(
    image_path: str,
    class_map: np.ndarray
) -> Optional[str]:
    """Draw colored segmentation mask overlaid on image."""
    try:
        image = cv2.imread(image_path)
        if image is None:
            return None
        
        height, width = image.shape[:2]
        
        # Resize class_map to match image size
        class_map_resized = cv2.resize(
            class_map.astype(np.float32), 
            (width, height), 
            interpolation=cv2.INTER_NEAREST
        ).astype(np.int32)
        
        # Create colored mask
        color_mask = np.zeros((height, width, 3), dtype=np.uint8)
        
        for cls_id in np.unique(class_map_resized):
            color = SEGMENTATION_COLORS[int(cls_id) % len(SEGMENTATION_COLORS)]
            color_mask[class_map_resized == cls_id] = color
        
        # Blend with original image
        alpha = 0.5
        blended = cv2.addWeighted(image, 1 - alpha, color_mask, alpha, 0)
        
        _, buffer = cv2.imencode('.jpg', blended)
        return base64.b64encode(buffer).decode('utf-8')
        
    except Exception as e:
        logger.error(f"Error drawing segmentation mask: {e}")
        return None


def draw_keypoints(
    image_path: str,
    keypoint_results: List[Dict[str, Any]]
) -> Optional[str]:
    """Draw keypoints on image."""
    try:
        image = cv2.imread(image_path)
        if image is None:
            return None
        
        height, width = image.shape[:2]
        
        for result in keypoint_results:
            keypoints = result['keypoints']
            color = POSE_COLORS[result['instance_id'] % len(POSE_COLORS)]
            
            for kp in keypoints:
                if kp['confidence'] > 0.3:
                    x = int(kp['x'] * width)
                    y = int(kp['y'] * height)
                    cv2.circle(image, (x, y), 4, color, -1)
                    cv2.circle(image, (x, y), 6, (255, 255, 255), 1)
        
        _, buffer = cv2.imencode('.jpg', image)
        return base64.b64encode(buffer).decode('utf-8')
        
    except Exception as e:
        logger.error(f"Error drawing keypoints: {e}")
        return None


def draw_classification_result(
    image_path: str,
    predictions: List[Dict[str, Any]],
    max_labels: int = 5,
) -> Optional[str]:
    """Draw classification predictions as a label overlay on the image.

    Shows the top-N predictions with confidence bars overlaid on the
    upper-left corner of the image.

    Args:
        image_path: Path to the source image.
        predictions: List of dicts with 'class_name' and 'confidence'.
        max_labels: Maximum number of predictions to show.

    Returns:
        Base64-encoded JPEG string, or None on failure.
    """
    try:
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Failed to read image: {image_path}")
            return None

        height, width = image.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        line_height = 32
        bar_max_width = min(300, width - 20)
        margin = 10
        y_offset = margin

        # Semi-transparent overlay background
        overlay = image.copy()
        num_labels = min(len(predictions), max_labels)
        bg_height = margin + num_labels * line_height + margin
        cv2.rectangle(overlay, (0, 0), (bar_max_width + 2 * margin, bg_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, image, 0.45, 0, image)

        for pred_idx, pred in enumerate(predictions[:max_labels]):
            class_name = pred.get('class_name', f"Class_{pred.get('class_id', '?')}")
            conf = pred.get('confidence', 0.0)
            label = f"{class_name}: {conf:.1%}"

            # Confidence bar
            bar_width = int(conf * bar_max_width)
            bar_y = y_offset + 4
            bar_color = BBOX_COLORS[pred_idx % len(BBOX_COLORS)]
            cv2.rectangle(image, (margin, bar_y), (margin + bar_width, bar_y + 18), bar_color, -1)

            # Label text
            cv2.putText(image, label, (margin + 4, bar_y + 14), font, font_scale, (255, 255, 255), thickness)
            y_offset += line_height

        _, buffer = cv2.imencode('.jpg', image)
        return base64.b64encode(buffer).decode('utf-8')

    except Exception as e:
        logger.error(f"Error drawing classification result: {e}")
        traceback.print_exc()
        return None


def draw_ocr_result(
    image_path: str,
    text: str,
    confidence: float
) -> Optional[str]:
    """Draw OCR result on image."""
    try:
        image = cv2.imread(image_path)
        if image is None:
            return None
        
        height, width = image.shape[:2]
        
        # Create a label box at the bottom
        label = f"Text: {text} ({confidence:.1%})"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        
        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        
        # Draw background rectangle
        cv2.rectangle(image, (0, height - text_h - 20), (width, height), (0, 0, 0), -1)
        
        # Draw text
        cv2.putText(image, label, (10, height - 10), font, font_scale, (255, 255, 255), thickness)
        
        _, buffer = cv2.imencode('.jpg', image)
        return base64.b64encode(buffer).decode('utf-8')
        
    except Exception as e:
        logger.error(f"Error drawing OCR result: {e}")
        return None
