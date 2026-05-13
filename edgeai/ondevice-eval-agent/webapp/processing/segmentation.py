"""Semantic segmentation processing."""

import logging
import time
import traceback
from typing import Any, Dict, List, Optional

import numpy as np

from utils.files import get_class_name
from observability.logging import log_processing_step
from utils.tensor import get_tensor_summary
from utils.visualization import draw_segmentation_mask

logger = logging.getLogger(__name__)


def process_segmentation(
    prediction: Dict[str, Any],
    response: Dict[str, Any],
    filepath: str,
    filename: str,
    model_name: str,
    inference_time: float,
    start_request_time: float,
    input_spec: Dict[str, Any],
    output_spec: Dict[str, Any],
    image_array: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Process semantic segmentation results.
    
    Supports formats:
    - Class probabilities: [batch, num_classes, H, W]
    - Class indices: [batch, H, W] or [batch, 1, H, W]
    """
    try:
        if 'outputs' not in response or len(response['outputs']) == 0:
            return {'success': False, 'error': 'No segmentation output found'}
        
        outputs = response['outputs']
        output_array = np.array(outputs[0]['data'])
        output_shape = outputs[0].get('shape', [])
        
        if output_shape:
            output_array = output_array.reshape(output_shape)
        
        logger.info(f"Segmentation output shape: {output_array.shape}")
        
        # Remove batch dimension
        if len(output_array.shape) >= 1 and output_array.shape[0] == 1:
            output_array = output_array[0]
        
        shape = output_array.shape
        
        # Determine format and get class indices
        if len(shape) == 3 and shape[0] > 1:
            # [num_classes, H, W] - take argmax
            num_classes = shape[0]
            class_map = np.argmax(output_array, axis=0)
        elif len(shape) == 3 and shape[0] == 1:
            # [1, H, W] - already class indices
            num_classes = int(output_array.max()) + 1
            class_map = output_array[0].astype(np.int32)
        elif len(shape) == 2:
            # [H, W] - already class indices
            num_classes = int(output_array.max()) + 1
            class_map = output_array.astype(np.int32)
        else:
            return {'success': False, 'error': f'Unsupported segmentation shape: {shape}'}
        
        # Calculate class statistics
        unique_classes, counts = np.unique(class_map, return_counts=True)
        total_pixels = class_map.size
        
        class_stats: List[Dict[str, Any]] = []
        for cls_id, count in zip(unique_classes, counts):
            percentage = (count / total_pixels) * 100
            class_stats.append({
                'class_id': int(cls_id),
                'class_name': get_class_name(int(cls_id), model_name),
                'pixel_count': int(count),
                'percentage': float(percentage)
            })
        
        class_stats.sort(key=lambda x: x['percentage'], reverse=True)
        
        log_processing_step("Segmentation", f"Found {len(unique_classes)} classes", "success")
        
        # Create colored segmentation mask
        annotated_image_base64 = draw_segmentation_mask(filepath, class_map)
        
        total_time = time.time() - start_request_time
        
        # Tensor info
        output_tensor_info = get_tensor_summary(output_array)
        output_tensor_info['shape'] = list(output_array.shape)
        output_tensor_info['name'] = outputs[0].get('name', 'output')
        
        input_tensor_info: Dict[str, Any] = {}
        if image_array is not None:
            input_tensor_info = get_tensor_summary(image_array)
            input_tensor_info['shape'] = list(image_array.shape)
            input_tensor_info['name'] = input_spec.get('name', 'input')
        
        return {
            'success': True,
            'task_type': 'segmentation',
            'detected_type': 'segmentation',
            'model_name': model_name,
            'latency': inference_time,
            'total_time': total_time,
            'num_classes': int(num_classes),
            'class_stats': class_stats,
            'mask_shape': list(class_map.shape),
            'annotated_image': annotated_image_base64,
            'image_filename': filename,
            'model_spec': {
                'input': {
                    'name': input_spec['name'],
                    'shape': input_spec['shape'],
                    'datatype': input_spec.get('datatype', 'FP32'),
                    'format': input_spec['format'],
                    'size': f"{input_spec['width']}x{input_spec['height']}"
                },
                'output': {
                    'name': output_spec['name'],
                    'shape': output_spec['shape'],
                    'datatype': output_spec.get('datatype', 'FP32')
                }
            },
            'tensor_info': {
                'input': input_tensor_info,
                'output': output_tensor_info
            }
        }
        
    except Exception as e:
        logger.error(f"Error processing segmentation: {e}")
        traceback.print_exc()
        return {'success': False, 'error': f'Segmentation failed: {str(e)}'}
