"""Tests for client/metadata.py — TensorSpec, ModelMetadataManager, caching."""

import threading
import pytest
from unittest.mock import MagicMock
from tritonclient.utils import InferenceServerException

from client.metadata import TensorSpec, ModelMetadataManager
from client.config import DEFAULT_INPUT_SPEC, DEFAULT_OUTPUT_SPEC, DEFAULT_TARGET_SIZE
from conftest import make_grpc_metadata


class TestTensorSpec:
    @pytest.mark.parametrize("shape,expected_format,expected_h,expected_w,expected_c", [
        ([1, 3, 640, 640], "NCHW", 640, 640, 3),
        ([1, 640, 640, 3], "NHWC", 640, 640, 3),
        ([1, 1, 224, 224], "NCHW", 224, 224, 1),
    ])
    def test_from_input_info_formats(self, shape, expected_format, expected_h, expected_w, expected_c):
        spec = TensorSpec.from_input_info({"name": "input", "shape": shape, "datatype": "FP32"})
        assert spec.format == expected_format
        assert spec.height == expected_h
        assert spec.width == expected_w
        assert spec.channels == expected_c

    def test_from_input_info_short_shape_uses_defaults(self):
        spec = TensorSpec.from_input_info({"name": "input", "shape": [1, 1000], "datatype": "FP32"})
        assert spec.height == DEFAULT_TARGET_SIZE[0]
        assert spec.width == DEFAULT_TARGET_SIZE[1]

    def test_from_input_info_dynamic_dims_resolved(self):
        spec = TensorSpec.from_input_info({"name": "input", "shape": [-1, 3, -1, -1], "datatype": "FP32"})
        assert spec.channels == 3
        assert spec.height > 0 and spec.width > 0

    def test_from_output_info_classification(self):
        spec = TensorSpec.from_output_info({"name": "output", "shape": [1, 1000], "datatype": "FP32"})
        assert spec.num_classes == 1000

    def test_from_output_info_detection(self):
        spec = TensorSpec.from_output_info({"name": "output", "shape": [1, 84, 8400], "datatype": "FP32"})
        assert spec.num_classes == 8400

    def test_to_dict_keys(self):
        spec = TensorSpec.from_input_info({"name": "x", "shape": [1, 3, 224, 224], "datatype": "FP32"})
        d = spec.to_dict()
        for key in ("name", "shape", "datatype", "format", "channels", "height", "width"):
            assert key in d


class TestModelMetadataManager:
    def _make_manager(self, mock_grpc_client):
        return ModelMetadataManager(mock_grpc_client, timeout=5)

    def test_get_metadata_returns_dict(self, mock_grpc_client):
        mgr = self._make_manager(mock_grpc_client)
        result = mgr.get_metadata("test_model")
        assert isinstance(result, dict)
        assert "inputs" in result or "name" in result

    def test_get_metadata_caches_result(self, mock_grpc_client):
        mgr = self._make_manager(mock_grpc_client)
        mgr.get_metadata("m1")
        mgr.get_metadata("m1")
        assert mock_grpc_client.get_model_metadata.call_count == 1

    def test_get_metadata_bypass_cache(self, mock_grpc_client):
        mgr = self._make_manager(mock_grpc_client)
        mgr.get_metadata("m1", use_cache=True)
        mgr.get_metadata("m1", use_cache=False)
        assert mock_grpc_client.get_model_metadata.call_count == 2

    def test_get_metadata_grpc_error_returns_none(self, mock_grpc_client):
        mock_grpc_client.get_model_metadata.side_effect = InferenceServerException("fail")
        mgr = self._make_manager(mock_grpc_client)
        assert mgr.get_metadata("bad_model") is None

    def test_clear_cache_forces_refresh(self, mock_grpc_client):
        mgr = self._make_manager(mock_grpc_client)
        mgr.get_metadata("m1")
        mgr.clear_cache()
        mgr.get_metadata("m1")
        assert mock_grpc_client.get_model_metadata.call_count == 2

    def test_get_input_spec_returns_dict(self, mock_grpc_client):
        mgr = self._make_manager(mock_grpc_client)
        spec = mgr.get_input_spec("test_model")
        assert isinstance(spec, dict)
        for key in ("name", "shape", "datatype", "format"):
            assert key in spec

    def test_get_output_spec_returns_dict(self, mock_grpc_client):
        mgr = self._make_manager(mock_grpc_client)
        spec = mgr.get_output_spec("test_model")
        assert isinstance(spec, dict)
        assert "datatype" in spec

    def test_get_all_output_specs_returns_list(self, mock_grpc_client):
        mgr = self._make_manager(mock_grpc_client)
        specs = mgr.get_all_output_specs("test_model")
        assert isinstance(specs, list) and len(specs) >= 1

    def test_get_input_shape_returns_tuple(self, mock_grpc_client):
        mgr = self._make_manager(mock_grpc_client)
        shape = mgr.get_input_shape("test_model")
        assert isinstance(shape, tuple) and len(shape) == 2

    def test_get_input_spec_no_metadata_returns_defaults(self, mock_grpc_client):
        mock_grpc_client.get_model_metadata.side_effect = InferenceServerException("fail")
        mgr = self._make_manager(mock_grpc_client)
        spec = mgr.get_input_spec("bad_model")
        assert spec == DEFAULT_INPUT_SPEC

    def test_thread_safety(self, mock_grpc_client):
        mgr = self._make_manager(mock_grpc_client)
        errors = []

        def worker():
            try:
                for _ in range(20):
                    mgr.get_metadata("test_model")
                    mgr.get_input_spec("test_model")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0
