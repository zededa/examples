"""Generic keypoint detection processing (faces, hands, etc.)."""

import logging
import time
import traceback
from typing import Any, Dict, List, Optional

import numpy as np

from observability.logging import log_processing_step
from utils.tensor import get_tensor_summary
from utils.visualization import POSE_COLORS, draw_keypoints

logger = logging.getLogger(__name__)


def process_keypoint_detection(
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
    Process generic keypoint detection (facial landmarks, hand keypoints, etc.)
    
    Similar to pose but for non-body keypoints.
    """
    try:
        keypoint_results: List[Dict[str, Any]] = []
        confidence_threshold = 0.3
        
        input_width = input_spec.get('width', 640)
        input_height = input_spec.get('height', 640)
        
        if 'outputs' not in response or len(response['outputs']) == 0:
            return {'success': False, 'error': 'No keypoint output found'}
        
        outputs = response['outputs']
        output_array = np.array(outputs[0]['data'])
        output_shape = outputs[0].get('shape', [])
        
        if output_shape:
            output_array = output_array.reshape(output_shape)
        
        logger.info(f"Keypoint output shape: {output_array.shape}")
        
        # Remove batch dimension
        if len(output_array.shape) >= 1 and output_array.shape[0] == 1:
            output_array = output_array[0]
        
        shape = output_array.shape
        
        # Heatmap format: [num_keypoints, H, W]
        if len(shape) == 3 and shape[0] > 1 and shape[1] > 8 and shape[2] > 8:
            num_keypoints = shape[0]
            heatmap_h, heatmap_w = shape[1], shape[2]
            keypoints = []
            
            for kp_idx in range(num_keypoints):
                heatmap = output_array[kp_idx]
                max_idx = np.unravel_index(np.argmax(heatmap), heatmap.shape)
                conf = float(heatmap[max_idx])
                
                x = float(max_idx[1]) / heatmap_w
                y = float(max_idx[0]) / heatmap_h
                
                keypoints.append({
                    'id': kp_idx,
                    'name': f'keypoint_{kp_idx}',
                    'x': x,
                    'y': y,
                    'confidence': conf
                })
            
            avg_conf = np.mean([kp['confidence'] for kp in keypoints])
            
            if avg_conf > confidence_threshold:
                keypoint_results.append({
                    'instance_id': 0,
                    'keypoints': keypoints,
                    'confidence': float(avg_conf),
                    'num_keypoints': num_keypoints
                })
        
        # Coordinate format: [num_keypoints, 2/3]
        elif len(shape) == 2 and shape[-1] in [2, 3]:
            num_keypoints = shape[0]
            keypoints = []
            
            for kp_idx in range(num_keypoints):
                x = float(output_array[kp_idx, 0])
                y = float(output_array[kp_idx, 1])
                conf = float(output_array[kp_idx, 2]) if shape[-1] == 3 else 1.0
                
                if x > 1 or y > 1:
                    x = x / input_width
                    y = y / input_height
                
                keypoints.append({
                    'id': kp_idx,
                    'name': f'keypoint_{kp_idx}',
                    'x': x,
                    'y': y,
                    'confidence': conf
                })
            
            avg_conf = np.mean([kp['confidence'] for kp in keypoints])
            
            if avg_conf > confidence_threshold:
                keypoint_results.append({
                    'instance_id': 0,
                    'keypoints': keypoints,
                    'confidence': float(avg_conf),
                    'num_keypoints': num_keypoints
                })
        
        # Multi-instance format: [num_instances, num_keypoints, 2/3]
        elif len(shape) == 3 and shape[-1] in [2, 3]:
            num_instances = shape[0]
            num_keypoints = shape[1]
            
            for inst_idx in range(num_instances):
                keypoints = []
                inst_data = output_array[inst_idx]
                
                for kp_idx in range(num_keypoints):
                    x = float(inst_data[kp_idx, 0])
                    y = float(inst_data[kp_idx, 1])
                    conf = float(inst_data[kp_idx, 2]) if shape[-1] == 3 else 1.0
                    
                    if x > 1 or y > 1:
                        x = x / input_width
                        y = y / input_height
                    
                    keypoints.append({
                        'id': kp_idx,
                        'name': f'keypoint_{kp_idx}',
                        'x': x,
                        'y': y,
                        'confidence': conf
                    })
                
                avg_conf = np.mean([kp['confidence'] for kp in keypoints])
                
                if avg_conf > confidence_threshold:
                    keypoint_results.append({
                        'instance_id': inst_idx,
                        'keypoints': keypoints,
                        'confidence': float(avg_conf),
                        'num_keypoints': num_keypoints
                    })
        
        log_processing_step("Keypoint Detection", f"Found {len(keypoint_results)} instance(s)", "success")
        
        # Draw keypoints
        annotated_image_base64 = draw_keypoints(filepath, keypoint_results)
        
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
            'task_type': 'keypoint',
            'detected_type': 'keypoint',
            'model_name': model_name,
            'latency': inference_time,
            'total_time': total_time,
            'keypoint_results': keypoint_results,
            'num_instances': len(keypoint_results),
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
        logger.error(f"Error processing keypoint detection: {e}")
        traceback.print_exc()
        return {'success': False, 'error': f'Keypoint detection failed: {str(e)}'}
