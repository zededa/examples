"""Pose estimation processing."""

import logging
import time
import traceback
from typing import Any, Dict, List, Optional

import numpy as np

from observability.logging import log_processing_step
from utils.tensor import get_tensor_summary
from utils.visualization import draw_pose_keypoints

logger = logging.getLogger(__name__)

# Standard keypoint connections for different pose models
POSE_SKELETON_COCO = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # Head
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
    (5, 11), (6, 12), (11, 12),  # Torso
    (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
]

POSE_KEYPOINT_NAMES_COCO = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]


def process_pose_estimation(
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
    Process pose estimation results.
    
    Supports multiple pose estimation output formats:
    - COCO format: [batch, num_people, 17, 3] (x, y, confidence)
    - Heatmap format: [batch, num_keypoints, H, W]
    - Simple format: [batch, num_keypoints, 2/3]
    
    Note: YOLOv8-pose [batch, 56, 8400] format is not yet supported.
    Use the detection processor for YOLOv8-pose models.
    """
    try:
        poses: List[Dict[str, Any]] = []
        confidence_threshold = 0.3
        
        input_width = input_spec.get('width', 640)
        input_height = input_spec.get('height', 640)
        
        if 'outputs' not in response or len(response['outputs']) == 0:
            return {'success': False, 'error': 'No pose output found'}
        
        outputs = response['outputs']
        output_array = np.array(outputs[0]['data'])
        output_shape = outputs[0].get('shape', [])
        
        if output_shape:
            output_array = output_array.reshape(output_shape)
        
        logger.info(f"Pose output shape: {output_array.shape}")
        
        # Remove batch dimension
        if len(output_array.shape) >= 1 and output_array.shape[0] == 1:
            output_array = output_array[0]
        
        shape = output_array.shape
        
        # Detect pose output format
        if len(shape) == 3 and shape[-1] in [2, 3]:
            # Format: [num_people, num_keypoints, 2/3]
            num_people = shape[0]
            num_keypoints = shape[1]
            
            for person_idx in range(num_people):
                keypoints = []
                person_data = output_array[person_idx]
                avg_confidence = 0
                
                for kp_idx in range(num_keypoints):
                    x = float(person_data[kp_idx, 0])
                    y = float(person_data[kp_idx, 1])
                    conf = float(person_data[kp_idx, 2]) if shape[-1] == 3 else 1.0
                    
                    # Normalize if in pixel coordinates
                    if x > 1 or y > 1:
                        x = x / input_width
                        y = y / input_height
                    
                    keypoint_name = POSE_KEYPOINT_NAMES_COCO[kp_idx] if kp_idx < len(POSE_KEYPOINT_NAMES_COCO) else f'keypoint_{kp_idx}'
                    
                    keypoints.append({
                        'id': kp_idx,
                        'name': keypoint_name,
                        'x': x,
                        'y': y,
                        'confidence': conf
                    })
                    avg_confidence += conf
                
                avg_confidence = avg_confidence / num_keypoints if num_keypoints > 0 else 0
                
                if avg_confidence > confidence_threshold:
                    poses.append({
                        'person_id': person_idx,
                        'keypoints': keypoints,
                        'confidence': avg_confidence,
                        'num_keypoints': num_keypoints
                    })
        
        elif len(shape) == 2 and shape[-1] in [2, 3]:
            # Format: [num_keypoints, 2/3] - single person
            num_keypoints = shape[0]
            keypoints = []
            avg_confidence = 0
            
            for kp_idx in range(num_keypoints):
                x = float(output_array[kp_idx, 0])
                y = float(output_array[kp_idx, 1])
                conf = float(output_array[kp_idx, 2]) if shape[-1] == 3 else 1.0
                
                if x > 1 or y > 1:
                    x = x / input_width
                    y = y / input_height
                
                keypoint_name = POSE_KEYPOINT_NAMES_COCO[kp_idx] if kp_idx < len(POSE_KEYPOINT_NAMES_COCO) else f'keypoint_{kp_idx}'
                
                keypoints.append({
                    'id': kp_idx,
                    'name': keypoint_name,
                    'x': x,
                    'y': y,
                    'confidence': conf
                })
                avg_confidence += conf
            
            avg_confidence = avg_confidence / num_keypoints if num_keypoints > 0 else 0
            
            if avg_confidence > confidence_threshold:
                poses.append({
                    'person_id': 0,
                    'keypoints': keypoints,
                    'confidence': avg_confidence,
                    'num_keypoints': num_keypoints
                })
        
        elif len(shape) == 3 and shape[0] > 5 and shape[1] > 16 and shape[2] > 16:
            # Heatmap format: [num_keypoints, H, W]
            num_keypoints = shape[0]
            heatmap_h, heatmap_w = shape[1], shape[2]
            keypoints = []
            
            for kp_idx in range(num_keypoints):
                heatmap = output_array[kp_idx]
                max_idx = np.unravel_index(np.argmax(heatmap), heatmap.shape)
                conf = float(heatmap[max_idx])
                
                x = float(max_idx[1]) / heatmap_w
                y = float(max_idx[0]) / heatmap_h
                
                keypoint_name = POSE_KEYPOINT_NAMES_COCO[kp_idx] if kp_idx < len(POSE_KEYPOINT_NAMES_COCO) else f'keypoint_{kp_idx}'
                
                keypoints.append({
                    'id': kp_idx,
                    'name': keypoint_name,
                    'x': x,
                    'y': y,
                    'confidence': conf
                })
            
            avg_confidence = np.mean([kp['confidence'] for kp in keypoints])
            
            if avg_confidence > confidence_threshold:
                poses.append({
                    'person_id': 0,
                    'keypoints': keypoints,
                    'confidence': float(avg_confidence),
                    'num_keypoints': num_keypoints
                })
        
        log_processing_step("Pose Estimation", f"Found {len(poses)} person(s)", "success")
        
        # Draw pose on image
        annotated_image_base64 = draw_pose_keypoints(filepath, poses)
        
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
            'task_type': 'pose',
            'detected_type': 'pose',
            'model_name': model_name,
            'latency': inference_time,
            'total_time': total_time,
            'poses': poses,
            'num_poses': len(poses),
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
        logger.error(f"Error processing pose estimation: {e}")
        traceback.print_exc()
        return {'success': False, 'error': f'Pose estimation failed: {str(e)}'}
