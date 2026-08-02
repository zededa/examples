"""
Get Model Input Requirements Tool

Provides detailed input requirements and preprocessing guidance for models.
"""

import logging
from typing import Dict, Any

from tools.base import ok, error_response, get_client
from tools.registry import register_tool

logger = logging.getLogger(__name__)


def _generate_preprocessing_code(width: int, height: int, data_format: str) -> str:
    """Generate example preprocessing code with configurable normalization."""
    return f'''# Python preprocessing example
import numpy as np
from PIL import Image
from enum import Enum

class NormalizationType(Enum):
    YOLO = "yolo"           # [0, 1] range - common for YOLO models
    IMAGENET = "imagenet"   # ImageNet mean/std - common for classification
    CENTERED = "centered"   # [-0.5, 0.5] range
    RAW = "raw"             # No normalization [0, 255]

def preprocess_image(image_path, normalization: NormalizationType = NormalizationType.YOLO):
    """
    Preprocess image for model inference.
    
    Args:
        image_path: Path to input image
        normalization: Type of normalization to apply (verify with model docs)
    """
    # Load and resize
    image = Image.open(image_path).convert('RGB')
    image = image.resize(({width}, {height}), Image.Resampling.LANCZOS)
    
    # Convert to numpy array
    img_array = np.array(image, dtype=np.float32)
    
    # Apply normalization based on model requirements
    if normalization == NormalizationType.YOLO:
        # YOLO-style: scale to [0, 1]
        img_array = img_array / 255.0
    elif normalization == NormalizationType.IMAGENET:
        # ImageNet normalization
        img_array = img_array / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_array = (img_array - mean) / std
    elif normalization == NormalizationType.CENTERED:
        # Centered [-0.5, 0.5]
        img_array = img_array / 255.0 - 0.5
    elif normalization == NormalizationType.RAW:
        # Keep as [0, 255]
        pass
    
    # Transpose to {data_format} format
    {"img_array = np.transpose(img_array, (2, 0, 1))  # HWC to CHW" if data_format == "NCHW" else "# Already in HWC format"}
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array.astype(np.float32)
'''


def get_model_input_requirements(model_name: str) -> Dict[str, Any]:
    """
    Get detailed input requirements for a model including preprocessing guidance.
    
    Explains what types of inputs the model expects (image formats, dimensions,
    normalization, camera feed properties, preprocessing requirements).
    
    Args:
        model_name: Name of the model to analyze
        
    Returns:
        Dict containing detailed input requirements and preprocessing guidance
    """
    try:
        client = get_client()
        input_spec = client.get_model_input_spec(model_name)
        
        # Extract dimensions
        shape = input_spec.get('shape', [])
        data_format = input_spec.get('format', 'NCHW')
        height = input_spec.get('height', 640)
        width = input_spec.get('width', 640)
        channels = input_spec.get('channels', 3)
        datatype = input_spec.get('datatype', 'FP32')
        
        # Determine color space
        if channels == 3:
            color_space = "RGB (3-channel color)"
        elif channels == 1:
            color_space = "Grayscale (single channel)"
        elif channels == 4:
            color_space = "RGBA (with alpha channel)"
        else:
            color_space = f"{channels} channels"
        
        # Build preprocessing guidance with conditional normalization
        preprocessing = {
            "resize": f"Resize images to {width}x{height} pixels",
            "color_conversion": "Convert to RGB color space (from BGR if using OpenCV)",
            "normalization": {
                "note": "Normalization depends on model training. Verify with model documentation or metadata.",
                "common_options": {
                    "imagenet": "mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] - Common for classification models",
                    "yolo_style": "Divide by 255.0 to [0,1] range - Common for YOLO detection models",
                    "centered": "Subtract 0.5 to [-0.5, 0.5] range - Some segmentation models",
                    "raw": "No normalization, use [0,255] integer values - Some TensorRT optimized models"
                },
                "recommendation": "Start with [0,1] normalization (divide by 255), test with sample images"
            },
            "format_conversion": f"Transpose to {data_format} format ({'CHW' if data_format == 'NCHW' else 'HWC'})",
            "batch_dimension": "Add batch dimension at axis 0"
        }
        
        # Camera/video feed recommendations
        camera_recommendations = {
            "minimum_resolution": f"{width}x{height} or higher (will be resized)",
            "aspect_ratio": f"Any (will be resized to {width}x{height})",
            "frame_rate": "10-30 FPS recommended for real-time inference",
            "color_format": "RGB or BGR (conversion handled in preprocessing)",
            "lighting": "Consistent lighting improves accuracy",
            "focus": "Ensure subjects are in focus for best results"
        }
        
        # Supported image formats
        supported_formats = {
            "file_formats": ["JPEG", "PNG", "BMP", "WebP", "GIF (first frame)"],
            "encoding": "Standard web image formats supported",
            "max_file_size": "Recommended under 10MB for performance"
        }
        
        return ok(
            model_name=model_name,
            input_tensor={
                "name": input_spec.get('name'),
                "shape": shape,
                "dimensions": {
                    "batch": "dynamic (-1)",
                    "channels": channels,
                    "height": height,
                    "width": width
                },
                "data_format": data_format,
                "datatype": datatype,
                "color_space": color_space
            },
            preprocessing_steps=preprocessing,
            camera_recommendations=camera_recommendations,
            supported_formats=supported_formats,
            code_example=_generate_preprocessing_code(width, height, data_format)
        )
    except Exception as e:
        logger.error(f"Error getting input requirements for {model_name}: {e}")
        return error_response(e, operation="get_input_requirements", model_name=model_name)


# Register the tool
register_tool(
    name="get_model_input_requirements",
    func=get_model_input_requirements,
    description="Get detailed input requirements for a model including image preprocessing guidance, camera feed recommendations, and supported formats. Use this when users ask about what images or inputs the model expects.",
    input_schema={
        "type": "object",
        "properties": {
            "model_name": {
                "type": "string",
                "description": "Name of the model to analyze"
            }
        },
        "required": ["model_name"]
    }
)
