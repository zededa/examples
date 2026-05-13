"""
Image preprocessing module for Model Server Client.

This module handles all image loading and preprocessing operations
required before inference. Supports various input formats and
outputs properly formatted numpy arrays for inference servers.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from typing import Any, BinaryIO, Optional, Union

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from .config import (
    DEFAULT_DATA_FORMAT,
    DEFAULT_TARGET_SIZE,
    PIXEL_VALUE_MAX,
    PreprocessingConfig,
)
from .exceptions import ImagePreprocessingError

logger = logging.getLogger(__name__)


# =============================================================================
# Type Aliases
# =============================================================================

# Image array after preprocessing: float32 with shape [N, C, H, W] or [N, H, W, C]
ImageArray = NDArray[np.floating[Any]]

# Supported image input types
ImageInput = Union[bytes, BinaryIO, io.BytesIO, str, Image.Image]


# =============================================================================
# Data Classes
# =============================================================================

@dataclass(frozen=True)
class PreprocessingParams:
    """Immutable preprocessing parameters."""
    width: int
    height: int
    data_format: str
    
    @classmethod
    def from_input_spec(
        cls,
        input_spec: Optional[dict[str, Any]] = None,
        target_size: Optional[tuple[int, int]] = None,
        default_size: tuple[int, int] = DEFAULT_TARGET_SIZE,
        default_format: str = DEFAULT_DATA_FORMAT,
    ) -> "PreprocessingParams":
        """
        Create parameters from input spec and optional overrides.
        
        Args:
            input_spec: Model input specification dict.
            target_size: Optional (height, width) override.
            default_size: Default (height, width) if not specified.
            default_format: Default data format if not specified.
        
        Returns:
            PreprocessingParams instance.
        """
        if input_spec:
            height = input_spec.get("height", default_size[0])
            width = input_spec.get("width", default_size[1])
            data_format = input_spec.get("format", default_format)
        else:
            height, width = default_size
            data_format = default_format
        
        # Apply override if provided
        if target_size is not None:
            height, width = target_size
        
        return cls(width=width, height=height, data_format=data_format)


# =============================================================================
# Image Preprocessor
# =============================================================================

class ImagePreprocessor:
    """
    Handles image preprocessing for model inference.
    
    Supports various input formats (bytes, file paths, file objects, PIL Images)
    and outputs properly formatted numpy arrays for inference servers.
    
    Features:
        - Automatic format detection and conversion
        - Configurable normalization (ImageNet defaults)
        - NCHW/NHWC format conversion
        - High-quality LANCZOS resampling
    
    Example:
        >>> preprocessor = ImagePreprocessor()
        >>> image_array = preprocessor.preprocess(image_bytes, input_spec)
        >>> # image_array.shape: (1, 3, 224, 224) for NCHW format
    """
    
    __slots__ = ("_config",)
    
    def __init__(self, config: Optional[PreprocessingConfig] = None) -> None:
        """
        Initialize the preprocessor.
        
        Args:
            config: Preprocessing configuration. Uses defaults if not provided.
        """
        self._config = config or PreprocessingConfig()
    
    # =========================================================================
    # Properties
    # =========================================================================
    
    @property
    def config(self) -> PreprocessingConfig:
        """Get current preprocessing configuration."""
        return self._config
    
    @config.setter
    def config(self, value: PreprocessingConfig) -> None:
        """Set preprocessing configuration."""
        self._config = value
    
    # =========================================================================
    # Public API
    # =========================================================================
    
    def update_config(self, updates: dict[str, Any]) -> None:
        """
        Update preprocessing configuration with new values.
        
        Args:
            updates: Dictionary of config values to update.
        """
        current = self._config.to_dict()
        current.update(updates)
        self._config = PreprocessingConfig.from_dict(current)
        logger.info(f"Updated preprocessing config: {self._config.to_dict()}")
    
    def get_preprocessing_params(
        self,
        input_spec: Optional[dict[str, Any]] = None,
        target_size: Optional[tuple[int, int]] = None,
    ) -> tuple[int, int, str]:
        """
        Get preprocessing parameters (width, height, format).
        
        Args:
            input_spec: Model input specification, or None for defaults.
            target_size: Override (height, width), or None.
            
        Returns:
            Tuple of (width, height, data_format).
        """
        params = PreprocessingParams.from_input_spec(
            input_spec=input_spec,
            target_size=target_size,
            default_size=self._config.target_size,
            default_format=self._config.format,
        )
        logger.debug(f"Preprocessing params: {params.data_format} {params.height}x{params.width}")
        return params.width, params.height, params.data_format
    
    def preprocess_bytes(
        self,
        image_bytes: Union[bytes, BinaryIO, io.BytesIO],
        input_spec: Optional[dict[str, Any]] = None,
        target_size: Optional[tuple[int, int]] = None,
    ) -> ImageArray:
        """
        Preprocess image from bytes for model inference.
        
        Args:
            image_bytes: bytes, BytesIO, or file-like object containing image data.
            input_spec: Optional model input spec for auto-detecting dimensions.
            target_size: Optional (height, width) tuple to override auto-detection.
        
        Returns:
            Numpy array ready for inference [1, C, H, W] or [1, H, W, C].
            
        Raises:
            ImagePreprocessingError: If preprocessing fails.
        """
        try:
            params = PreprocessingParams.from_input_spec(
                input_spec, target_size, self._config.target_size, self._config.format
            )
            image = self._load_image_from_bytes(image_bytes)
            return self._preprocess_pil_image(image, params)
            
        except (OSError, ValueError) as e:
            raise ImagePreprocessingError(
                f"Failed to preprocess image from bytes: {e}",
                cause=e,
            ) from e
    
    def preprocess_file(
        self,
        image_path: str,
        input_spec: Optional[dict[str, Any]] = None,
        target_size: Optional[tuple[int, int]] = None,
    ) -> ImageArray:
        """
        Preprocess image from file path for model inference.
        
        Args:
            image_path: Path to image file.
            input_spec: Optional model input spec for auto-detecting dimensions.
            target_size: Optional (height, width) tuple to override auto-detection.
        
        Returns:
            Numpy array ready for inference [1, C, H, W] or [1, H, W, C].
            
        Raises:
            ImagePreprocessingError: If preprocessing fails.
        """
        try:
            params = PreprocessingParams.from_input_spec(
                input_spec, target_size, self._config.target_size, self._config.format
            )
            image = Image.open(image_path)
            return self._preprocess_pil_image(image, params)
            
        except (OSError, ValueError) as e:
            raise ImagePreprocessingError(
                f"Failed to preprocess image from file: {e}",
                image_source=image_path,
                cause=e,
            ) from e
    
    def preprocess(
        self,
        image_data: ImageInput,
        input_spec: Optional[dict[str, Any]] = None,
        target_size: Optional[tuple[int, int]] = None,
    ) -> ImageArray:
        """
        Preprocess image from any supported format.
        
        This is the recommended unified API for preprocessing. Automatically
        detects the input type and delegates to the appropriate handler.
        
        Args:
            image_data: Image bytes, file path, file object, or PIL Image.
            input_spec: Optional model input spec for auto-detecting dimensions.
            target_size: Optional (height, width) tuple to override.
            
        Returns:
            Numpy array ready for inference [1, C, H, W] or [1, H, W, C].
            
        Raises:
            ImagePreprocessingError: If preprocessing fails.
        """
        try:
            params = PreprocessingParams.from_input_spec(
                input_spec, target_size, self._config.target_size, self._config.format
            )
            image = self._load_image(image_data)
            return self._preprocess_pil_image(image, params)
            
        except (OSError, ValueError) as e:
            source = image_data if isinstance(image_data, str) else str(type(image_data))
            raise ImagePreprocessingError(
                f"Failed to preprocess image: {e}",
                image_source=source,
                cause=e,
            ) from e
    
    # =========================================================================
    # Private - Image Loading
    # =========================================================================
    
    def _load_image(self, image_data: ImageInput) -> Image.Image:
        """
        Load image from any supported format.
        
        Args:
            image_data: Image in any supported format.
            
        Returns:
            PIL Image object.
            
        Raises:
            ValueError: If image format is not supported.
        """
        if isinstance(image_data, str):
            return Image.open(image_data)
        
        if isinstance(image_data, Image.Image):
            return image_data
        
        if isinstance(image_data, (bytes, io.BytesIO)):
            return self._load_image_from_bytes(image_data)
        
        if hasattr(image_data, "read"):
            return self._load_image_from_bytes(image_data)
        
        raise ValueError(f"Unsupported image_data type: {type(image_data)}")
    
    def _load_image_from_bytes(
        self,
        image_bytes: Union[bytes, BinaryIO, io.BytesIO],
    ) -> Image.Image:
        """
        Load PIL Image from bytes or file-like object.
        
        Args:
            image_bytes: Image data as bytes or file-like object.
            
        Returns:
            PIL Image object.
        """
        if isinstance(image_bytes, bytes):
            return Image.open(io.BytesIO(image_bytes))
        
        if isinstance(image_bytes, io.BytesIO):
            return Image.open(image_bytes)
        
        # File-like object with read() method
        content = image_bytes.read()
        return Image.open(io.BytesIO(content))
    
    # =========================================================================
    # Private - Core Preprocessing
    # =========================================================================
    
    def _preprocess_pil_image(
        self,
        image: Image.Image,
        params: PreprocessingParams,
    ) -> ImageArray:
        """
        Core preprocessing logic for PIL images.
        
        Processing steps:
            1. Convert to RGB (handles grayscale, RGBA, etc.)
            2. Resize to target dimensions using LANCZOS
            3. Convert to float32 and normalize to [0, 1]
            4. Apply ImageNet normalization if configured
            5. Transpose to NCHW format if required
            6. Add batch dimension
        
        Args:
            image: PIL Image to preprocess.
            params: Preprocessing parameters.
            
        Returns:
            Preprocessed numpy array with shape [1, C, H, W] or [1, H, W, C].
        """
        # Step 1: Convert to RGB
        image = image.convert("RGB")
        
        # Step 2: Resize with high-quality resampling
        image = image.resize((params.width, params.height), Image.Resampling.LANCZOS)
        
        # Step 3: Convert to numpy and normalize to [0, 1]
        image_array = np.array(image, dtype=np.float32) / PIXEL_VALUE_MAX
        
        # Step 4: Apply ImageNet normalization if configured
        if self._config.normalize:
            image_array = self._apply_normalization(image_array)
        
        # Step 5: Convert to NCHW if required (default is HWC from PIL)
        if params.data_format == "NCHW":
            image_array = np.transpose(image_array, (2, 0, 1))  # HWC -> CHW
        
        # Step 6: Add batch dimension
        image_array = np.expand_dims(image_array, axis=0)
        
        logger.debug(f"Preprocessed image shape: {image_array.shape}")
        return image_array
    
    def _apply_normalization(self, image_array: NDArray[np.float32]) -> NDArray[np.float32]:
        """
        Apply ImageNet normalization: (x - mean) / std.
        
        Args:
            image_array: Image array in HWC format, values in [0, 1].
            
        Returns:
            Normalized image array.
        """
        mean = np.array(self._config.mean, dtype=np.float32)
        std = np.array(self._config.std, dtype=np.float32)
        return (image_array - mean) / std


__all__ = [
    "ImagePreprocessor",
    "PreprocessingParams",
    "ImageArray",
    "ImageInput",
]
