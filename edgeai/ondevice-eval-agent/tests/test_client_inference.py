"""
Tests for client/inference.py — InferenceRunner, InferenceRequest, ClassificationResult.

Covers request building, classification post-processing, softmax stability,
latency measurement, and error handling for invalid responses.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from client.inference import (
    ClassificationResult,
    InferenceRequest,
    InferenceRunner,
)
from client.exceptions import InferenceError

# Re-use helpers from conftest (imported automatically by pytest)
from conftest import make_inference_response


# =============================================================================
# InferenceRequest
# =============================================================================


class TestInferenceRequest:
    """Tests for the InferenceRequest dataclass."""

    def test_to_grpc_inputs_returns_list(self):
        """to_grpc_inputs() must return a non-empty list."""
        data = np.random.randn(1, 3, 224, 224).astype(np.float32)
        req = InferenceRequest(
            model_name="resnet50",
            input_name="images",
            input_shape=list(data.shape),
            input_data=data,
            datatype="FP32",
        )
        inputs = req.to_grpc_inputs()
        assert isinstance(inputs, list)
        assert len(inputs) == 1

    def test_to_grpc_inputs_default_datatype(self):
        """Datatype defaults to FP32 when not explicitly set."""
        data = np.zeros((1, 3, 224, 224), dtype=np.float32)
        req = InferenceRequest(
            model_name="m",
            input_name="input",
            input_shape=list(data.shape),
            input_data=data,
        )
        assert req.datatype == "FP32"
        # Should not raise
        req.to_grpc_inputs()


# =============================================================================
# ClassificationResult
# =============================================================================


class TestClassificationResult:
    """Tests for the ClassificationResult dataclass."""

    def test_to_dict_has_expected_keys(self):
        """to_dict() must contain the canonical top-level keys."""
        result = ClassificationResult(
            model_name="resnet50",
            timestamp="2026-01-01 00:00:00",
            num_classes=1000,
            output_name="output0",
            output_shape=[1, 1000],
            predictions=[{"rank": 1, "class_id": 0, "confidence": 0.9}],
        )
        d = result.to_dict()
        expected_keys = {
            "timestamp",
            "model_name",
            "num_classes",
            "output_name",
            "output_shape",
            "top_predictions",
        }
        assert expected_keys == set(d.keys())

    def test_to_dict_predictions_stored_as_top_predictions(self):
        """The predictions list should appear under the 'top_predictions' key."""
        preds = [{"rank": 1, "class_id": 5, "confidence": 0.8}]
        result = ClassificationResult(
            model_name="m",
            timestamp="t",
            num_classes=10,
            output_name="out",
            output_shape=[1, 10],
            predictions=preds,
        )
        assert result.to_dict()["top_predictions"] is preds


# =============================================================================
# InferenceRunner — process_prediction
# =============================================================================


class TestInferenceRunnerProcessPrediction:
    """Tests for InferenceRunner.process_prediction()."""

    @pytest.fixture()
    def runner(self, mock_grpc_client):
        return InferenceRunner(mock_grpc_client)

    # -- Classification output (1D, 1000 classes) --

    def test_classification_output_top_predictions(self, runner):
        """1000-class output should produce at most 5 top_predictions."""
        response = make_inference_response(
            model_name="resnet50",
            outputs=[{
                "name": "output0",
                "shape": [1, 1000],
                "datatype": "FP32",
                "data": np.random.randn(1000).tolist(),
            }],
        )
        result = runner.process_prediction(response, "resnet50")
        assert "top_predictions" in result
        assert len(result["top_predictions"]) <= 5
        assert result["num_classes"] == 1000

    def test_classification_predictions_sorted_descending(self, runner):
        """Top predictions should be ordered by confidence descending."""
        scores = np.zeros(100)
        scores[42] = 10.0  # dominant class
        scores[7] = 5.0
        response = make_inference_response(
            outputs=[{
                "name": "output0",
                "shape": [1, 100],
                "datatype": "FP32",
                "data": scores.tolist(),
            }],
        )
        result = runner.process_prediction(response)
        preds = result["top_predictions"]
        assert preds[0]["class_id"] == 42
        confidences = [p["confidence"] for p in preds]
        assert confidences == sorted(confidences, reverse=True)

    # -- Class names --

    def test_process_prediction_uses_class_names(self, runner):
        """When class_names are set, predictions should use them."""
        runner.class_names = [f"cat_{i}" for i in range(1000)]
        response = make_inference_response(
            outputs=[{
                "name": "output0",
                "shape": [1, 1000],
                "datatype": "FP32",
                "data": np.random.randn(1000).tolist(),
            }],
        )
        result = runner.process_prediction(response, "resnet50")
        for pred in result["top_predictions"]:
            assert pred["class_name"].startswith("cat_")

    def test_process_prediction_without_class_names(self, runner):
        """Without class_names, predictions should use Class_<id> format."""
        runner.class_names = None
        response = make_inference_response(
            outputs=[{
                "name": "output0",
                "shape": [1, 1000],
                "datatype": "FP32",
                "data": np.random.randn(1000).tolist(),
            }],
        )
        result = runner.process_prediction(response, "resnet50")
        for pred in result["top_predictions"]:
            assert pred["class_name"].startswith("Class_")

    # -- Non-classification output --

    def test_non_1d_output_returns_raw_output(self, runner):
        """Multi-dimensional outputs (e.g. detection) should return raw_output."""
        data_2d = np.random.randn(84, 8400).tolist()
        flat = [v for row in data_2d for v in row]
        response = make_inference_response(
            outputs=[{
                "name": "output0",
                "shape": [84, 8400],
                "datatype": "FP32",
                "data": flat,
            }],
        )
        result = runner.process_prediction(response, "yolov8")
        assert "raw_output" in result
        assert result["top_predictions"] == []

    # -- Error handling --

    def test_process_prediction_empty_response_raises(self, runner):
        """Empty response dict should raise InferenceError."""
        with pytest.raises(InferenceError):
            runner.process_prediction({}, "model")

    def test_process_prediction_none_response_raises(self, runner):
        """None response should raise InferenceError."""
        with pytest.raises(InferenceError):
            runner.process_prediction(None, "model")

    def test_process_prediction_missing_outputs_key_raises(self, runner):
        """Response without 'outputs' key should raise InferenceError."""
        with pytest.raises(InferenceError):
            runner.process_prediction({"model_name": "m"}, "m")

    # -- Softmax numerical stability --

    def test_softmax_large_values_no_nan(self, runner):
        """Very large input values should not produce NaN after softmax."""
        large_scores = np.full(1000, 1e6)
        large_scores[0] = 1e6 + 10  # slightly larger
        response = make_inference_response(
            outputs=[{
                "name": "output0",
                "shape": [1, 1000],
                "datatype": "FP32",
                "data": large_scores.tolist(),
            }],
        )
        result = runner.process_prediction(response, "test")
        for pred in result["top_predictions"]:
            assert not np.isnan(pred["confidence"])
            assert not np.isinf(pred["confidence"])


# =============================================================================
# InferenceRunner — send_inference_request
# =============================================================================


class TestInferenceRunnerSendRequest:
    """Tests for InferenceRunner.send_inference_request()."""

    @pytest.fixture()
    def runner(self, mock_grpc_client):
        return InferenceRunner(mock_grpc_client)

    def test_measure_latency_includes_latency_key(self, runner):
        """When measure_latency=True the result dict must contain 'latency'."""
        image_array = np.random.randn(1, 3, 224, 224).astype(np.float32)
        input_spec = {"name": "images", "datatype": "FP32"}
        result = runner.send_inference_request(
            image_array, "test_model", input_spec, "triton", measure_latency=True,
        )
        assert "latency" in result
        assert isinstance(result["latency"], float)
        assert result["latency"] >= 0

    def test_measure_latency_false_omits_latency(self, runner):
        """When measure_latency=False the result dict should not have 'latency'."""
        image_array = np.random.randn(1, 3, 224, 224).astype(np.float32)
        input_spec = {"name": "images", "datatype": "FP32"}
        result = runner.send_inference_request(
            image_array, "test_model", input_spec, "triton", measure_latency=False,
        )
        assert "latency" not in result

    def test_send_returns_outputs_list(self, runner):
        """The returned dict must contain an 'outputs' list."""
        image_array = np.random.randn(1, 3, 224, 224).astype(np.float32)
        input_spec = {"name": "images", "datatype": "FP32"}
        result = runner.send_inference_request(
            image_array, "test_model", input_spec, "triton",
        )
        assert "outputs" in result
        assert isinstance(result["outputs"], list)
