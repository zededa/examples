"""Processing modules for different model types."""

from .model_detection import (
    MODEL_TYPE_PATTERNS,
    OUTPUT_SHAPE_PATTERNS,
    detect_model_type,
)
from .classification import process_image_classification
from .detection import (
    nms_boxes,
    detect_output_format,
    process_yolov8_output,
    process_yolov5_output,
    process_row_detections,
    process_object_detection,
)
from .pose import (
    POSE_SKELETON_COCO,
    POSE_KEYPOINT_NAMES_COCO,
    process_pose_estimation,
)
from .segmentation import process_segmentation
from .panoptic import process_panoptic_segmentation
from .keypoint import process_keypoint_detection
from .ocr import process_ocr

__all__ = [
    # Model detection
    'MODEL_TYPE_PATTERNS',
    'OUTPUT_SHAPE_PATTERNS',
    'detect_model_type',
    # Classification
    'process_image_classification',
    # Detection
    'nms_boxes',
    'detect_output_format',
    'process_yolov8_output',
    'process_yolov5_output',
    'process_row_detections',
    'process_object_detection',
    # Pose
    'POSE_SKELETON_COCO',
    'POSE_KEYPOINT_NAMES_COCO',
    'process_pose_estimation',
    # Segmentation
    'process_segmentation',
    # Panoptic
    'process_panoptic_segmentation',
    # Keypoint
    'process_keypoint_detection',
    # OCR
    'process_ocr',
]
