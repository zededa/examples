"""Panoptic segmentation processing."""

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


def process_panoptic_segmentation(
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
    Process panoptic segmentation results.
    
    Panoptic combines instance and semantic segmentation.
    Supports formats with separate semantic and instance outputs.
    """
    try:
        if 'outputs' not in response or len(response['outputs']) == 0:
            return {'success': False, 'error': 'No panoptic output found'}
        
        outputs = response['outputs']
        
        # Try to find semantic and instance outputs
        semantic_map = None
        instance_map = None
        panoptic_map = None
        
        for output in outputs:
            data = np.array(output['data'])
            shape = output.get('shape', [])
            name = output.get('name', '').lower()
            
            if shape:
                data = data.reshape(shape)
            
            # Remove batch dim
            if len(data.shape) >= 1 and data.shape[0] == 1:
                data = data[0]
            
            if 'semantic' in name or 'class' in name:
                if len(data.shape) == 3:
                    semantic_map = np.argmax(data, axis=0)
                else:
                    semantic_map = data.astype(np.int32)
            elif 'instance' in name:
                if len(data.shape) == 3 and data.shape[0] > 1:
                    instance_map = np.argmax(data, axis=0)
                else:
                    instance_map = data.astype(np.int32) if len(data.shape) == 2 else data[0].astype(np.int32)
            elif 'panoptic' in name:
                panoptic_map = data
        
        # If only one output, treat as semantic segmentation
        if len(outputs) == 1:
            output_array = np.array(outputs[0]['data'])
            output_shape = outputs[0].get('shape', [])
            if output_shape:
                output_array = output_array.reshape(output_shape)
            if len(output_array.shape) >= 1 and output_array.shape[0] == 1:
                output_array = output_array[0]
            
            if len(output_array.shape) == 3:
                semantic_map = np.argmax(output_array, axis=0)
            else:
                semantic_map = output_array.astype(np.int32)
        
        if semantic_map is None and panoptic_map is None:
            return {'success': False, 'error': 'Could not parse panoptic output'}
        
        # Use panoptic map if available, otherwise use semantic
        main_map = panoptic_map if panoptic_map is not None else semantic_map
        
        # Calculate statistics
        unique_segments = np.unique(main_map)
        
        segments: List[Dict[str, Any]] = []
        for seg_id in unique_segments:
            mask = main_map == seg_id
            pixel_count = int(np.sum(mask))
            
            # Find bounding box
            rows = np.any(mask, axis=1)
            cols = np.any(mask, axis=0)
            if rows.any() and cols.any():
                y1, y2 = np.where(rows)[0][[0, -1]]
                x1, x2 = np.where(cols)[0][[0, -1]]
                
                class_id = int(seg_id % 256) if panoptic_map is not None else int(seg_id)
                segments.append({
                    'segment_id': int(seg_id),
                    'class_id': class_id,
                    'instance_id': int(seg_id // 256) if panoptic_map is not None else 0,
                    'class_name': get_class_name(class_id, model_name),
                    'pixel_count': pixel_count,
                    'percentage': float(pixel_count / main_map.size * 100),
                    'bbox': [int(x1), int(y1), int(x2), int(y2)]
                })
        
        segments.sort(key=lambda x: x['percentage'], reverse=True)
        
        log_processing_step("Panoptic Segmentation", f"Found {len(segments)} segments", "success")
        
        # Draw panoptic visualization
        annotated_image_base64 = draw_segmentation_mask(filepath, main_map)
        
        total_time = time.time() - start_request_time
        
        # Tensor info
        output_shape = outputs[0].get('shape', [])
        first_output = np.array(outputs[0]['data'])
        if output_shape:
            first_output = first_output.reshape(output_shape)
        else:
            first_output = first_output.reshape((-1,))
        output_tensor_info = get_tensor_summary(first_output)
        output_tensor_info['shape'] = list(first_output.shape)
        output_tensor_info['name'] = outputs[0].get('name', 'output')
        
        input_tensor_info: Dict[str, Any] = {}
        if image_array is not None:
            input_tensor_info = get_tensor_summary(image_array)
            input_tensor_info['shape'] = list(image_array.shape)
            input_tensor_info['name'] = input_spec.get('name', 'input')
        
        return {
            'success': True,
            'task_type': 'panoptic',
            'detected_type': 'panoptic',
            'model_name': model_name,
            'latency': inference_time,
            'total_time': total_time,
            'num_segments': len(segments),
            'segments': segments,
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
        logger.error(f"Error processing panoptic segmentation: {e}")
        traceback.print_exc()
        return {'success': False, 'error': f'Panoptic segmentation failed: {str(e)}'}
