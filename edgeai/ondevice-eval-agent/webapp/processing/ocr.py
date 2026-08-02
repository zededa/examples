"""OCR (Optical Character Recognition) processing."""

import logging
import time
import traceback
from typing import Any, Dict, Optional

import numpy as np

from observability.logging import log_processing_step
from utils.tensor import get_tensor_summary
from utils.visualization import draw_ocr_result

logger = logging.getLogger(__name__)

# Common OCR character sets
OCR_CHARSET_ALPHANUMERIC = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
OCR_CHARSET_EXTENDED = OCR_CHARSET_ALPHANUMERIC + ' !"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'


def process_ocr(
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
    Process OCR (text recognition) results.
    
    Supports formats:
    - CTC output: [sequence_length, vocab_size] - requires CTC decoding
    - Attention output: [sequence_length] or [sequence_length, vocab_size]
    - Direct text output: String or token IDs
    """
    try:
        if 'outputs' not in response or len(response['outputs']) == 0:
            return {'success': False, 'error': 'No OCR output found'}
        
        outputs = response['outputs']
        recognized_text = ""
        confidence = 0.0
        raw_output_info: Dict[str, Any] = {}
        
        # Check for multiple outputs (some models have text + confidence)
        text_output = None
        conf_output = None
        
        for output in outputs:
            data = np.array(output['data'])
            shape = output.get('shape', [])
            name = output.get('name', '').lower()
            
            if shape:
                data = data.reshape(shape)
            
            if 'text' in name or 'output' in name:
                text_output = data
            elif 'conf' in name or 'score' in name or 'prob' in name:
                conf_output = data
        
        if text_output is None:
            text_output = np.array(outputs[0]['data'])
            output_shape = outputs[0].get('shape', [])
            if output_shape:
                text_output = text_output.reshape(output_shape)
        
        logger.info(f"OCR output shape: {text_output.shape}")
        
        # Remove batch dimension
        if len(text_output.shape) >= 1 and text_output.shape[0] == 1:
            text_output = text_output[0]
        
        shape = text_output.shape
        
        # Determine output format and decode
        if len(shape) == 2:
            # [sequence_length, vocab_size] - CTC or attention logits
            seq_len, vocab_size = shape
            
            # Apply softmax
            exp_scores = np.exp(text_output - np.max(text_output, axis=1, keepdims=True))
            probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
            
            # Get best character at each position
            char_indices = np.argmax(probs, axis=1)
            char_confs = np.max(probs, axis=1)
            
            # CTC decode (merge repeated and remove blanks)
            # Assume blank token is at index 0 or last index
            blank_idx = 0 if vocab_size > len(OCR_CHARSET_ALPHANUMERIC) else vocab_size - 1
            
            decoded_chars = []
            prev_char = None
            
            for idx, conf in zip(char_indices, char_confs):
                if idx != blank_idx and idx != prev_char:
                    if idx < len(OCR_CHARSET_EXTENDED):
                        decoded_chars.append(OCR_CHARSET_EXTENDED[int(idx)])
                    else:
                        decoded_chars.append(f'[{idx}]')
                prev_char = idx
            
            recognized_text = ''.join(decoded_chars)
            confidence = float(np.mean(char_confs))
            
            raw_output_info = {
                'sequence_length': int(seq_len),
                'vocab_size': int(vocab_size),
                'decode_method': 'ctc_greedy'
            }
        
        elif len(shape) == 1:
            # [sequence_length] - token IDs
            token_ids = text_output.astype(np.int32)
            
            # Decode token IDs
            decoded_chars = []
            for idx in token_ids:
                if 0 <= idx < len(OCR_CHARSET_EXTENDED):
                    decoded_chars.append(OCR_CHARSET_EXTENDED[int(idx)])
                elif idx == 0:
                    break  # End of sequence
                else:
                    decoded_chars.append(f'[{idx}]')
            
            recognized_text = ''.join(decoded_chars)
            confidence = 1.0  # No confidence info available
            
            raw_output_info = {
                'sequence_length': len(token_ids),
                'decode_method': 'token_ids'
            }
        
        # Use confidence output if available
        if conf_output is not None:
            if len(conf_output.shape) >= 1 and conf_output.shape[0] == 1:
                conf_output = conf_output[0]
            confidence = float(np.mean(conf_output))
        
        log_processing_step("OCR", f"Recognized: '{recognized_text}'", "success")
        
        # Draw text on image
        annotated_image_base64 = draw_ocr_result(filepath, recognized_text, confidence)
        
        total_time = time.time() - start_request_time
        
        # Tensor info
        output_tensor_info = get_tensor_summary(text_output)
        output_tensor_info['shape'] = list(text_output.shape)
        output_tensor_info['name'] = outputs[0].get('name', 'output')
        
        input_tensor_info: Dict[str, Any] = {}
        if image_array is not None:
            input_tensor_info = get_tensor_summary(image_array)
            input_tensor_info['shape'] = list(image_array.shape)
            input_tensor_info['name'] = input_spec.get('name', 'input')
        
        return {
            'success': True,
            'task_type': 'ocr',
            'detected_type': 'ocr',
            'model_name': model_name,
            'latency': inference_time,
            'total_time': total_time,
            'recognized_text': recognized_text,
            'confidence': confidence,
            'raw_output_info': raw_output_info,
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
        logger.error(f"Error processing OCR: {e}")
        traceback.print_exc()
        return {'success': False, 'error': f'OCR processing failed: {str(e)}'}
