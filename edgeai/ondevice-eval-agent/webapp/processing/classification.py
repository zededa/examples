"""Image classification processing."""

import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np

from utils.tensor import get_tensor_summary
from utils.visualization import draw_classification_result

logger = logging.getLogger(__name__)


def process_image_classification(
    prediction: Dict[str, Any],
    response: Optional[Dict[str, Any]],
    filepath: str,
    filename: str,
    model_name: str,
    inference_time: float,
    start_request_time: float,
    input_spec: Dict[str, Any],
    output_spec: Dict[str, Any],
    image_array: Optional[np.ndarray],
    model_check_time: float,
    preprocess_time: float,
    prediction_time: float
) -> Dict[str, Any]:
    """Process image classification results with raw tensor information.
    
    Args:
        prediction: Processed prediction from client
        response: Raw response from inference server
        filepath: Path to the input image file (for visualization)
        filename: Name of the input image file
        model_name: Name of the model used
        inference_time: Time taken for inference
        start_request_time: Start time of the request
        input_spec: Model input specification
        output_spec: Model output specification
        image_array: Preprocessed input image array
        model_check_time: Time taken to check model readiness
        preprocess_time: Time taken for preprocessing
        prediction_time: Time taken for post-processing prediction
    
    Returns:
        Dictionary with classification results and metadata
    """
    # Extract top predictions with class numbers
    top_predictions: List[Dict[str, Any]] = []
    if 'top_predictions' in prediction:
        for pred in prediction['top_predictions'][:5]:
            top_predictions.append({
                'class_id': pred['class_id'],
                'class_name': pred['class_name'],
                'confidence': pred['confidence']
            })
    
    total_time = time.time() - start_request_time
    
    # Extract raw tensor information
    output_tensor_info: Dict[str, Any] = {}
    if response and 'outputs' in response and len(response['outputs']) > 0:
        output_data = response['outputs'][0]
        raw_output_array = np.array(output_data.get('data', []))
        output_shape = output_data.get('shape', [])
        if output_shape:
            raw_output_array = raw_output_array.reshape(output_shape)
        output_tensor_info = get_tensor_summary(raw_output_array)
        output_tensor_info['shape'] = output_shape
        output_tensor_info['name'] = output_data.get('name', 'output')
    
    # Input tensor info
    input_tensor_info: Dict[str, Any] = {}
    if image_array is not None:
        input_tensor_info = get_tensor_summary(image_array)
        input_tensor_info['shape'] = list(image_array.shape)
        input_tensor_info['name'] = input_spec.get('name', 'input')
    
    # Generate annotated visualization with top predictions
    annotated_image_base64 = None
    try:
        if top_predictions and filepath:
            annotated_image_base64 = draw_classification_result(filepath, top_predictions)
    except Exception as vis_err:
        logger.warning(f"Failed to generate classification visualization: {vis_err}")

    result = {
        'success': True,
        'task_type': 'classification',
        'detected_type': 'classification',
        'model_name': model_name,
        'latency': inference_time,
        'total_time': total_time,
        'top_predictions': top_predictions,
        'annotated_image': annotated_image_base64,
        'image_filename': filename,
        'num_classes': prediction.get('num_classes'),
        'model_spec': {
            'input': {
                'name': input_spec.get('name', 'input'),
                'shape': input_spec.get('shape', []),
                'datatype': input_spec.get('datatype', 'FP32'),
                'format': input_spec.get('format', 'NCHW'),
                'size': f"{input_spec.get('width', 'unknown')}x{input_spec.get('height', 'unknown')}"
            },
            'output': {
                'name': output_spec['name'],
                'shape': output_spec['shape'],
                'datatype': output_spec.get('datatype', 'FP32'),
                'num_classes': output_spec.get('num_classes')
            }
        },
        'tensor_info': {
            'input': input_tensor_info,
            'output': output_tensor_info
        },
        'processing_times': {
            'model_check': model_check_time,
            'preprocessing': preprocess_time,
            'inference': inference_time,
            'postprocessing': prediction_time,
            'total': total_time
        }
    }
    
    return result
