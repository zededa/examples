"""Tests for client/preprocessing.py — image loading, format conversion, normalization."""

import io
import numpy as np
import pytest
from PIL import Image

from client.preprocessing import ImagePreprocessor, PreprocessingParams
from client.config import PreprocessingConfig
from client.exceptions import ImagePreprocessingError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_image(mode: str = "RGB", size=(8, 8)):
    img = Image.new(mode, size, color=128)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# PreprocessingParams
# ---------------------------------------------------------------------------

class TestPreprocessingParams:
    def test_from_input_spec_nchw(self):
        spec = {"height": 640, "width": 640, "format": "NCHW"}
        p = PreprocessingParams.from_input_spec(spec)
        assert p.width == 640 and p.height == 640 and p.data_format == "NCHW"

    def test_from_input_spec_nhwc(self):
        spec = {"height": 320, "width": 320, "format": "NHWC"}
        p = PreprocessingParams.from_input_spec(spec)
        assert p.data_format == "NHWC"

    def test_from_input_spec_none_uses_defaults(self):
        p = PreprocessingParams.from_input_spec(None)
        assert p.width == 224 and p.height == 224

    def test_target_size_override(self):
        spec = {"height": 640, "width": 640, "format": "NCHW"}
        p = PreprocessingParams.from_input_spec(spec, target_size=(100, 100))
        assert p.height == 100 and p.width == 100


# ---------------------------------------------------------------------------
# ImagePreprocessor — shape and dtype contracts
# ---------------------------------------------------------------------------

class TestImagePreprocessor:
    @pytest.mark.parametrize("data_format,expected_shape_prefix", [
        ("NCHW", (1, 3)),
        ("NHWC", (1,)),
    ])
    def test_preprocess_bytes_shape(self, sample_image_bytes, data_format, expected_shape_prefix):
        cfg = PreprocessingConfig()
        cfg.format = data_format
        proc = ImagePreprocessor(cfg)
        spec = {"height": 32, "width": 32, "format": data_format}
        arr = proc.preprocess_bytes(sample_image_bytes, spec)
        assert arr.shape[:len(expected_shape_prefix)] == expected_shape_prefix
        if data_format == "NCHW":
            assert arr.shape == (1, 3, 32, 32)
        else:
            assert arr.shape == (1, 32, 32, 3)

    def test_preprocess_bytes_float32(self, sample_image_bytes):
        arr = ImagePreprocessor().preprocess_bytes(sample_image_bytes)
        assert arr.dtype == np.float32

    def test_preprocess_bytes_normalized_range(self, sample_image_bytes):
        cfg = PreprocessingConfig()
        cfg.normalize = True
        arr = ImagePreprocessor(cfg).preprocess_bytes(sample_image_bytes)
        # ImageNet normalization shifts values; they should be finite
        assert np.all(np.isfinite(arr))

    def test_preprocess_bytes_unnormalized_range(self, sample_image_bytes):
        cfg = PreprocessingConfig()
        cfg.normalize = False
        arr = ImagePreprocessor(cfg).preprocess_bytes(sample_image_bytes)
        assert arr.min() >= 0.0 and arr.max() <= 1.0

    def test_preprocess_file(self, sample_image_path):
        arr = ImagePreprocessor().preprocess_file(sample_image_path)
        assert arr.ndim == 4

    def test_preprocess_pil_image(self):
        img = Image.new("RGB", (8, 8), color=(100, 150, 200))
        arr = ImagePreprocessor().preprocess(img)
        assert arr.ndim == 4 and arr.dtype == np.float32

    def test_grayscale_converted_to_rgb(self):
        gray_bytes = _make_image("L")
        arr = ImagePreprocessor().preprocess_bytes(gray_bytes)
        # Must have 3 channels
        assert 3 in arr.shape

    def test_rgba_converted_to_rgb(self):
        rgba_bytes = _make_image("RGBA")
        arr = ImagePreprocessor().preprocess_bytes(rgba_bytes)
        assert 3 in arr.shape

    def test_invalid_bytes_raises(self):
        with pytest.raises(ImagePreprocessingError):
            ImagePreprocessor().preprocess_bytes(b"not_an_image")

    @pytest.mark.parametrize("target_size", [(64, 64), (224, 224), (640, 640)])
    def test_target_size_respected(self, sample_image_bytes, target_size):
        spec = {"height": target_size[0], "width": target_size[1], "format": "NCHW"}
        arr = ImagePreprocessor().preprocess_bytes(sample_image_bytes, spec)
        assert arr.shape[2] == target_size[0]
        assert arr.shape[3] == target_size[1]

    def test_update_config(self):
        proc = ImagePreprocessor()
        proc.update_config({"normalize": False})
        assert proc.config.normalize is False
