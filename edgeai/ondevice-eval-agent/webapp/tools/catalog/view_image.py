"""
View Image Tool

Allows the agent to analyze images using vision capabilities.
Converts image to base64 and returns it for LLM vision analysis.
"""

import logging
import os
import base64
import mimetypes
from typing import Dict, Any, Optional

from tools.base import ok, error_response
from tools.registry import register_tool
from sessions.registry import SESSION_STORAGE_ROOT

logger = logging.getLogger(__name__)

# Maximum image size in bytes (5MB)
MAX_IMAGE_SIZE = 5 * 1024 * 1024


def view_image(
    image_path: str,
    max_dimension: int = 1024,
    description: Optional[str] = None
) -> Dict[str, Any]:
    """
    Load an image and prepare it for LLM vision analysis.
    
    This tool reads an image file and returns it as base64 data along with
    metadata. The calling LLM can then use this to analyze the image content.
    
    Args:
        image_path: Path to the image file
        max_dimension: Maximum width/height to resize to (default 1024 for efficiency)
        description: Optional context about what to look for in the image
    
    Returns:
        Dict containing image data, metadata, and analysis hints
    """
    try:
        # Validate path
        if not image_path:
            return error_response(
                ValueError("image_path is required"),
                operation="view_image"
            )
        
        # Security: Prevent path traversal attacks
        real_path = os.path.realpath(image_path)
        real_storage_root = os.path.realpath(SESSION_STORAGE_ROOT)
        if not real_path.startswith(real_storage_root + os.sep) and real_path != real_storage_root:
            return error_response(
                ValueError("Invalid file path - access denied"),
                operation="view_image"
            )
        
        if not os.path.exists(real_path):
            return error_response(
                FileNotFoundError(f"Image not found: {image_path}"),
                operation="view_image"
            )
        
        # Check file size
        file_size = os.path.getsize(real_path)
        if file_size > MAX_IMAGE_SIZE:
            return error_response(
                ValueError(f"Image too large: {file_size / 1024 / 1024:.1f}MB (max {MAX_IMAGE_SIZE / 1024 / 1024}MB)"),
                operation="view_image"
            )
        
        # Detect mime type
        mime_type, _ = mimetypes.guess_type(real_path)
        if not mime_type or not mime_type.startswith('image/'):
            mime_type = 'image/jpeg'  # Default to JPEG
        
        # Read and optionally resize image
        try:
            from PIL import Image
            import io
            
            with Image.open(real_path) as img:
                original_size = img.size
                original_mode = img.mode
                
                # Convert to RGB if necessary (for JPEG compatibility)
                if img.mode in ('RGBA', 'P'):
                    img = img.convert('RGB')
                
                # Resize if too large
                width, height = img.size
                if max(width, height) > max_dimension:
                    ratio = max_dimension / max(width, height)
                    new_size = (int(width * ratio), int(height * ratio))
                    img = img.resize(new_size, Image.Resampling.LANCZOS)
                    resized = True
                else:
                    resized = False
                
                # Convert to base64
                buffer = io.BytesIO()
                img.save(buffer, format='JPEG', quality=85)
                image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                
                final_size = img.size
        except ImportError:
            # PIL not available, read raw file
            with open(real_path, 'rb') as f:
                image_base64 = base64.b64encode(f.read()).decode('utf-8')
            original_size = None
            final_size = None
            original_mode = None
            resized = False
            # Use detected mime_type when not re-encoding
            detected_mime_type = mime_type
        else:
            # When PIL re-encodes to JPEG, use image/jpeg
            detected_mime_type = "image/jpeg"
        
        return ok(
            image_base64=image_base64,
            mime_type=detected_mime_type,
            original_path=image_path,
            original_size=original_size,
            final_size=final_size,
            original_mode=original_mode,
            resized=resized,
            file_size_kb=round(file_size / 1024, 1),
            description=description,
            message=f"Image loaded successfully ({final_size[0]}x{final_size[1]} pixels)" if final_size else "Image loaded successfully"
        )
        
    except Exception as e:
        logger.error(f"Error viewing image: {e}", exc_info=True)
        return error_response(
            e,
            operation="view_image",
            image_path=image_path
        )


def analyze_inference_result(
    inference_result: Optional[Dict[str, Any]] = None,
    include_visualization: bool = True
) -> Dict[str, Any]:
    """
    Extract and format inference results for detailed LLM analysis.
    
    This tool takes the raw inference result and prepares it for comprehensive
    explanation by the LLM, including the visualization image if available.
    
    Args:
        inference_result: The result dict from run_inference tool (optional - 
                         if not provided, returns helpful guidance)
        include_visualization: Whether to include the result image (default True)
    
    Returns:
        Dict containing structured results and optional visualization
    """
    try:
        # Handle missing inference_result
        if inference_result is None:
            return error_response(
                ValueError("No inference result provided. To analyze inference results, you need to first call the 'run_inference' tool on an image, then pass the results to this tool."),
                operation="analyze_inference_result",
                guidance="Look at the previous tool results in this conversation for the inference data to discuss."
            )
        
        if not isinstance(inference_result, dict):
            return error_response(
                ValueError("inference_result must be a dictionary"),
                operation="analyze_inference_result"
            )
        
        # Extract key information
        data = inference_result.get('data', inference_result)
        
        # Support both old key names (processing_type) and new processor keys
        # (task_type / detected_type).
        processing_type = (
            data.get('processing_type')
            or data.get('task_type')
            or data.get('detected_type')
            or 'unknown'
        )
        
        analysis = {
            "model_name": data.get('model_name', 'unknown'),
            "processing_type": processing_type,
            "auto_detected": data.get('auto_detected', False),
            "summary": data.get('summary', 'No summary available'),
        }
        
        # Add timing info
        if 'inference_time_ms' in data:
            analysis['inference_time_ms'] = data['inference_time_ms']
        
        # Add type-specific details, handling both old and new key names.
        if processing_type == 'segmentation':
            # Derive classes_found from either explicit field or class_stats, which may be
            # a dict (old shape) or a list of dicts (new shape from segmentation processor).
            classes_found = data.get('classes_found')
            if not classes_found:
                class_stats = data.get('class_stats')
                if isinstance(class_stats, dict):
                    classes_found = list(class_stats.keys())
                elif isinstance(class_stats, list):
                    derived_classes = []
                    for entry in class_stats:
                        if not isinstance(entry, dict):
                            continue
                        if 'class_name' in entry:
                            derived_classes.append(entry['class_name'])
                        elif 'id' in entry:
                            derived_classes.append(entry['id'])
                    classes_found = derived_classes
                else:
                    classes_found = []

            analysis['segmentation_details'] = {
                'num_classes': data.get('num_classes', 0),
                'classes_found': classes_found or [],
                'mask_shape': data.get('mask_shape', []),
                'explanation': (
                    f"The segmentation model identified {data.get('num_classes', 0)} distinct classes/regions in the image. "
                    f"Each pixel in the image has been assigned to one of these classes. "
                    f"The colored overlay shows which class each pixel belongs to."
                )
            }
        
        elif processing_type == 'detection':
            num_detections = data.get('total_detections', data.get('num_detections', 0))
            filtered = data.get('filtered_detections', num_detections)
            analysis['detection_details'] = {
                'total_detections': num_detections,
                'filtered_detections': filtered,
                'detections': data.get('detections', [])[:10],  # Limit to 10
                'class_summary': data.get('class_summary', {}),
                'explanation': (
                    f"The detection model found {num_detections} objects in the image. "
                    f"After applying confidence threshold, {filtered} detections remain. "
                    f"Each detection includes a bounding box and class label."
                )
            }
        
        elif processing_type == 'classification':
            predictions = data.get('predictions') if 'predictions' in data else data.get('top_predictions', [])
            analysis['classification_details'] = {
                'predictions': predictions[:5],
                'explanation': (
                    "The classification model assigned probabilities to different classes. "
                    "The top prediction indicates what the model thinks the image contains."
                )
            }
        
        elif processing_type in ['pose', 'keypoint']:
            num_people = data.get('num_people', data.get('num_poses', data.get('num_instances', 0)))
            analysis['pose_details'] = {
                'num_people': num_people,
                'keypoints_per_person': data.get('keypoints_per_person', 0),
                'explanation': (
                    f"The pose model detected {num_people} people in the image. "
                    f"For each person, it identified key body landmarks (joints) that show their pose."
                )
            }
        
        elif processing_type == 'ocr':
            text = data.get('text') if 'text' in data else data.get('recognized_text', '')
            analysis['ocr_details'] = {
                'text': text,
                'confidence': data.get('confidence', 0),
                'explanation': "The OCR model extracted text content from the image."
            }
        
        elif processing_type == 'panoptic':
            analysis['panoptic_details'] = {
                'num_segments': data.get('num_segments', 0),
                'segments': data.get('segments', []),
                'explanation': (
                    f"The panoptic model identified {data.get('num_segments', 0)} segments in the image, "
                    f"combining both stuff (amorphous regions) and things (countable objects)."
                )
            }
        
        # Include visualization if requested and available
        if include_visualization:
            viz_base64 = data.get('result_image_base64') or data.get('annotated_image')
            if viz_base64:
                analysis['visualization'] = {
                    'available': True,
                    'image_base64': viz_base64,
                    'mime_type': 'image/png',
                    'description': f"Visualization showing {processing_type} results overlaid on the original image"
                }
            else:
                analysis['visualization'] = {
                    'available': False,
                    'reason': 'No visualization was generated for this inference'
                }
        
        return ok(
            data=analysis,
            message=f"Analyzed {processing_type} inference results"
        )
        
    except Exception as e:
        logger.error(f"Error analyzing inference result: {e}", exc_info=True)
        return error_response(
            e,
            operation="analyze_inference_result"
        )


# Register the tools
register_tool(
    name="view_image",
    func=view_image,
    description="IMPORTANT: Use this tool to SEE and analyze images. Call this BEFORE running inference to describe what's in the uploaded image, and AFTER inference to see the visualization result. The image will be shown to you so you can describe objects, people, colors, and scene details.",
    input_schema={
        "type": "object",
        "properties": {
            "image_path": {
                "type": "string",
                "description": "Path to the image file to view and analyze"
            },
            "max_dimension": {
                "type": "integer",
                "default": 1024,
                "description": "Maximum dimension to resize image to (for efficiency)"
            },
            "description": {
                "type": "string",
                "description": "Optional context about what to look for in the image (e.g., 'looking for objects before detection' or 'analyzing segmentation result')"
            }
        },
        "required": ["image_path"]
    }
)

register_tool(
    name="analyze_inference_result", 
    func=analyze_inference_result,
    description="Analyze and explain inference results in detail. NOTE: This tool requires passing the full inference_result dictionary from a previous run_inference call. If you don't have the raw result data, use the information from the conversation history instead.",
    input_schema={
        "type": "object",
        "properties": {
            "inference_result": {
                "type": "object",
                "description": "The full result dictionary from a previous run_inference call. If not available, the tool will return guidance."
            },
            "include_visualization": {
                "type": "boolean",
                "default": True,
                "description": "Whether to include the visualization image"
            }
        },
        "required": []
    }
)
