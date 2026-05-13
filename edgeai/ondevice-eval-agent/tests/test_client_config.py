"""Tests for client/config.py — dataclasses, enums, constants."""

import pytest
from client.config import (
    ServerType,
    SERVER_TYPE_TRITON,
    SERVER_TYPE_OPENVINO,
    SERVER_TYPE_UNKNOWN,
    InputSpec,
    OutputSpec,
    PreprocessingConfig,
    APIPath,
    DEFAULT_TARGET_SIZE,
    DEFAULT_GRPC_PORT,
    DEFAULT_TIMEOUT_SECONDS,
    DEFAULT_INFERENCE_TIMEOUT_SECONDS,
    COMMON_CHANNEL_COUNTS,
)


class TestServerType:
    @pytest.mark.parametrize("member,value", [
        (ServerType.TRITON, "triton"),
        (ServerType.OPENVINO, "openvino"),
        (ServerType.UNKNOWN, "unknown"),
    ])
    def test_enum_values(self, member, value):
        assert member.value == value

    def test_legacy_constants_match_enum(self):
        assert SERVER_TYPE_TRITON == ServerType.TRITON.value
        assert SERVER_TYPE_OPENVINO == ServerType.OPENVINO.value
        assert SERVER_TYPE_UNKNOWN == ServerType.UNKNOWN.value


class TestInputSpec:
    def test_default_values(self):
        spec = InputSpec()
        assert spec.format in ("NCHW", "NHWC")
        assert isinstance(spec.shape, tuple)
        assert spec.datatype == "FP32"

    def test_frozen_immutability(self):
        spec = InputSpec()
        with pytest.raises(AttributeError):
            spec.name = "other"

    def test_to_dict_keys(self):
        d = InputSpec().to_dict()
        for key in ("name", "shape", "datatype", "format", "channels", "height", "width"):
            assert key in d


class TestOutputSpec:
    def test_default_values(self):
        spec = OutputSpec()
        assert spec.datatype == "FP32"

    def test_frozen_immutability(self):
        spec = OutputSpec()
        with pytest.raises(AttributeError):
            spec.name = "other"

    def test_to_dict_keys(self):
        d = OutputSpec().to_dict()
        for key in ("name", "shape", "datatype", "num_classes"):
            assert key in d


class TestPreprocessingConfig:
    def test_default_target_size(self):
        cfg = PreprocessingConfig()
        assert cfg.target_size == DEFAULT_TARGET_SIZE

    def test_default_normalize_enabled(self):
        assert PreprocessingConfig().normalize is True

    def test_to_dict_roundtrip(self):
        original = PreprocessingConfig()
        restored = PreprocessingConfig.from_dict(original.to_dict())
        assert restored.target_size == original.target_size
        assert restored.normalize == original.normalize
        assert restored.format == original.format

    def test_from_dict_partial_uses_defaults(self):
        cfg = PreprocessingConfig.from_dict({"normalize": False})
        assert cfg.normalize is False
        assert cfg.target_size == DEFAULT_TARGET_SIZE

    def test_mean_std_independence(self):
        a = PreprocessingConfig()
        b = PreprocessingConfig()
        a.mean[0] = 999.0
        assert b.mean[0] != 999.0


class TestAPIPath:
    def test_v2_model_contains_placeholder(self):
        assert "{model_name}" in APIPath.V2_MODEL

    def test_v2_infer_format(self):
        result = APIPath.V2_MODEL_INFER.format(model_name="resnet50")
        assert "resnet50" in result
        assert "/infer" in result


class TestConstants:
    def test_default_grpc_port_is_int(self):
        assert isinstance(DEFAULT_GRPC_PORT, int)

    def test_timeouts_positive(self):
        assert DEFAULT_TIMEOUT_SECONDS > 0
        assert DEFAULT_INFERENCE_TIMEOUT_SECONDS > 0

    def test_common_channel_counts(self):
        assert 3 in COMMON_CHANNEL_COUNTS
        assert 1 in COMMON_CHANNEL_COUNTS
