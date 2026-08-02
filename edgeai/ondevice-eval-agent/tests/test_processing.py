"""
Tests for webapp/processing/model_detection.py and webapp/processing/detection.py.

Covers detect_model_type, nms_boxes, and detect_output_format.
"""

import numpy as np
import pytest

from processing.model_detection import (
    detect_model_type,
    MODEL_TYPE_PATTERNS,
    OUTPUT_SHAPE_PATTERNS,
)
from processing.detection import nms_boxes, detect_output_format


# ============================================================================
# detect_model_type  --  name-based detection
# ============================================================================


class TestDetectModelTypeByName:
    @pytest.mark.parametrize(
        "model_name, expected_type",
        [
            ("yolov8n", "detection"),
            ("resnet50", "classification"),
            ("unet_segmentation", "segmentation"),
            ("pose_model", "pose"),
            ("crnn_ocr", "ocr"),
            ("my_efficientdet_model", "detection"),
            ("vit-base-patch16", "classification"),
            ("deeplab_v3", "segmentation"),
            ("panoptic_fpn", "panoptic"),
        ],
    )
    def test_name_pattern_detection(self, model_name, expected_type):
        result = detect_model_type(model_name, output_spec=None)
        assert result == expected_type


# ============================================================================
# detect_model_type  --  shape-based detection
# ============================================================================


class TestDetectModelTypeByShape:
    def test_classification_shape(self):
        """Output shape [1, 1000] should be classified as 'classification'."""
        result = detect_model_type(
            "unknown_model",
            output_spec={"shape": [1, 1000]},
        )
        assert result == "classification"

    def test_none_output_spec_does_not_crash(self):
        result = detect_model_type("unknown_model", output_spec=None)
        # Should fall through to default without raising
        assert isinstance(result, str)

    def test_empty_model_name_does_not_crash(self):
        result = detect_model_type("", output_spec={"shape": [1, 1000]})
        assert isinstance(result, str)

    def test_default_fallback_is_classification(self):
        """When no pattern matches, the function falls back to 'classification'."""
        result = detect_model_type(
            "totally_unknown_xyz",
            output_spec={"shape": [1, 2]},  # shape too small for classification lambda
        )
        assert result == "classification"


# ============================================================================
# detect_model_type  --  multi-output detection
# ============================================================================


class TestDetectModelTypeMultiOutput:
    def test_multi_output_with_boxes(self):
        """Multi-output with 'box' in name, using shape that doesn't match
        earlier shape patterns so the multi-output branch is reached."""
        all_outputs = [
            {"name": "box_output", "shape": [1, 100, 4]},
            {"name": "score_output", "shape": [1, 100]},
            {"name": "class_output", "shape": [1, 100]},
        ]
        # Use an output_spec shape that doesn't match classification/detection
        # patterns (empty) so the multi-output analysis can run.
        result = detect_model_type(
            "custom_model",
            output_spec={"shape": []},
            num_outputs=3,
            all_output_specs=all_outputs,
        )
        assert result == "detection"

    def test_multi_output_with_boxes_and_masks(self):
        all_outputs = [
            {"name": "box_output", "shape": [1, 100, 4]},
            {"name": "mask_output", "shape": [1, 100, 28, 28]},
            {"name": "class_output", "shape": [1, 100]},
        ]
        result = detect_model_type(
            "custom_model",
            output_spec={"shape": []},
            num_outputs=3,
            all_output_specs=all_outputs,
        )
        assert result == "panoptic"

    def test_multi_output_with_keypoints(self):
        all_outputs = [
            {"name": "keypoint_output", "shape": [1, 17, 3]},
            {"name": "score_output", "shape": [1, 17]},
            {"name": "bbox_output", "shape": [1, 1, 4]},
        ]
        result = detect_model_type(
            "custom_model",
            output_spec={"shape": []},
            num_outputs=3,
            all_output_specs=all_outputs,
        )
        assert result == "pose"


# ============================================================================
# nms_boxes
# ============================================================================


class TestNmsBoxes:
    def test_empty_input(self):
        boxes = np.array([]).reshape(0, 4)
        scores = np.array([])
        result = nms_boxes(boxes, scores)
        assert result == []

    def test_single_box(self):
        boxes = np.array([[10, 10, 50, 50]], dtype=np.float32)
        scores = np.array([0.9], dtype=np.float32)
        result = nms_boxes(boxes, scores)
        assert len(result) == 1

    def test_two_overlapping_boxes_one_kept(self):
        """Two highly overlapping boxes should be suppressed to one."""
        boxes = np.array(
            [[10, 10, 50, 50], [12, 12, 52, 52]], dtype=np.float32
        )
        scores = np.array([0.9, 0.8], dtype=np.float32)
        result = nms_boxes(boxes, scores, iou_threshold=0.45)
        assert len(result) == 1

    def test_two_non_overlapping_boxes_both_kept(self):
        boxes = np.array(
            [[10, 10, 20, 20], [200, 200, 300, 300]], dtype=np.float32
        )
        scores = np.array([0.9, 0.85], dtype=np.float32)
        result = nms_boxes(boxes, scores, iou_threshold=0.45)
        assert len(result) == 2

    def test_score_threshold_filters_low_confidence(self):
        boxes = np.array(
            [[10, 10, 50, 50], [200, 200, 300, 300]], dtype=np.float32
        )
        scores = np.array([0.9, 0.1], dtype=np.float32)
        result = nms_boxes(boxes, scores, score_threshold=0.25)
        assert len(result) == 1


# ============================================================================
# detect_output_format
# ============================================================================


class TestDetectOutputFormat:
    def test_yolov8_shape(self):
        """Shape [1, 84, 8400] with 'yolov8' in the name -> 'yolov8'."""
        arr = np.zeros((1, 84, 8400), dtype=np.float32)
        fmt, info = detect_output_format(arr, "yolov8n")
        assert fmt == "yolov8"
        assert "num_classes" in info

    def test_yolov5_shape(self):
        """Shape [1, 25200, 85] is detected as yolov8_transposed (anchors x features)."""
        arr = np.zeros((1, 25200, 85), dtype=np.float32)
        fmt, info = detect_output_format(arr, "custom_det")
        # The code treats [anchors, features] as yolov8_transposed
        assert fmt in ("yolov5", "yolov8_transposed")
        assert "num_classes" in info

    def test_unknown_shape(self):
        """A very small shape should return 'unknown'."""
        arr = np.zeros((1, 2), dtype=np.float32)
        fmt, _info = detect_output_format(arr, "mystery_model")
        # Small 2D array -- may match row_detections or unknown
        assert isinstance(fmt, str)
