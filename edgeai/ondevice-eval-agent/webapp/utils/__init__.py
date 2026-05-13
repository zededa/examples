"""Utility modules for the web application."""

from observability.logging import (
    endpoint_logs,
    processing_logs,
    endpoint_logs_lock,
    processing_logs_lock,
    log_endpoint_call,
    log_processing_step,
    init_log_queues,
    clear_all_logs,
)
from .tensor import (
    format_tensor_shape,
    get_tensor_summary,
)
from .files import (
    allowed_file,
    get_class_name,
)
from .visualization import (
    BBOX_COLORS,
    POSE_COLORS,
    SEGMENTATION_COLORS,
    draw_bounding_boxes,
    draw_classification_result,
    draw_pose_keypoints,
    draw_segmentation_mask,
    draw_keypoints,
    draw_ocr_result,
)
from .errors import (
    APIError,
    BadRequestError,
    NotFoundError,
    ServiceUnavailableError,
    InternalServerError,
    create_error_response,
    create_success_response,
    handle_exceptions,
    validate_request_json,
)
__all__ = [
    # Logging
    'endpoint_logs',
    'processing_logs',
    'endpoint_logs_lock',
    'processing_logs_lock',
    'log_endpoint_call',
    'log_processing_step',
    'init_log_queues',
    'clear_all_logs',
    # Tensor
    'format_tensor_shape',
    'get_tensor_summary',
    # Files
    'allowed_file',
    'get_class_name',
    # Visualization
    'BBOX_COLORS',
    'POSE_COLORS',
    'SEGMENTATION_COLORS',
    'draw_bounding_boxes',
    'draw_pose_keypoints',
    'draw_segmentation_mask',
    'draw_keypoints',
    'draw_ocr_result',
    # Error handling
    'APIError',
    'BadRequestError',
    'NotFoundError',
    'ServiceUnavailableError',
    'InternalServerError',
    'create_error_response',
    'create_success_response',
    'handle_exceptions',
    'validate_request_json',
]
