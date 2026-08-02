"""Core routes for the web application - main endpoints."""

import logging
import os
import time
import traceback
import uuid
from typing import Any, Dict

import requests
from flask import Blueprint, jsonify, request
from werkzeug.utils import secure_filename

from processing import (
    detect_model_type,
    process_image_classification,
    process_keypoint_detection,
    process_object_detection,
    process_ocr,
    process_panoptic_segmentation,
    process_pose_estimation,
    process_segmentation,
)
from utils.files import allowed_file
from observability.logging import (
    clear_all_logs,
    endpoint_logs,
    endpoint_logs_lock,
    log_endpoint_call,
    log_processing_step,
    processing_logs,
    processing_logs_lock,
)
from utils.tensor import format_tensor_shape
from utils.errors import (
    BadRequestError,
    ServiceUnavailableError,
    create_success_response,
    handle_exceptions,
)

logger = logging.getLogger(__name__)

# Create Blueprint
core_bp = Blueprint('core', __name__)

# These will be set by the main app when registering the blueprint
_app_config: Dict[str, Any] = {}
_client = None


def init_core_routes(app_config: Dict[str, Any], client) -> None:
    """Initialize core routes with app configuration and client."""
    global _app_config, _client
    _app_config = app_config
    _client = client


def execute_prediction(filepath: str, file_bytes: bytes, model_name: str, task_type: str = 'auto') -> dict:
    """
    Execute prediction on an image file - reusable core logic.
    Returns complete results including full tensor information.
    
    Args:
        filepath: Path to image file (for visualization functions)
        file_bytes: Image file bytes (for preprocessing)
        model_name: Name of the model to use
        task_type: Task type ('auto', 'detection', 'classification', etc.)
        
    Returns:
        Dict with complete results including tensor_info, model_spec, and task-specific results
    """
    start_request_time = time.time()
    filename = os.path.basename(filepath)
    
    # Check if model is ready
    model_check_start = time.time()
    model_ready = _client.check_model_ready(model_name)
    model_check_time = time.time() - model_check_start
    
    if not model_ready:
        return {
            'success': False,
            'error': f'Model {model_name} is not ready'
        }
    
    # Get model metadata to check for DETR or other multi-input models
    metadata = _client.get_model_metadata(model_name)
    
    # Check if this is a DETR model (requires special handling)
    from processing.detr import is_detr_model, run_detr_inference
    if is_detr_model(model_name, metadata):
        logger.info(f"Detected DETR model: {model_name}, using specialized inference")
        
        # Get server URL from client
        server_url = _client.server_url
        
        # Run DETR inference with specialized processing
        result = run_detr_inference(
            server_url=server_url,
            model_name=model_name,
            image_bytes=file_bytes,
            threshold=0.5,  # Lower threshold to get more detections
        )
        
        if not result.get("success"):
            return result
        
        # Format DETR results to match expected output format
        detections = result.get("detections", [])
        timing = result.get("timing", {})
        
        formatted_detections = [
            {
                'class': det['label'],
                'class_name': det['label'],
                'class_id': det.get('label_id', 0),
                'confidence': det['score'],
                'bbox': [
                    det['box']['xmin'],
                    det['box']['ymin'],
                    det['box']['xmax'],
                    det['box']['ymax']
                ]
            }
            for det in detections
        ]
        
        # Generate annotated visualization image
        annotated_image = None
        try:
            from utils.visualization import draw_bounding_boxes
            if filepath and os.path.exists(filepath) and formatted_detections:
                annotated_image = draw_bounding_boxes(filepath, formatted_detections)
                if annotated_image:
                    logger.info(f"Generated DETR visualization with {len(formatted_detections)} detections")
        except Exception as vis_err:
            logger.warning(f"Failed to generate DETR visualization: {vis_err}")
        
        return {
            'success': True,
            'model_name': model_name,
            'model_type': 'detection',
            'detected_type': 'detection',
            'auto_detected': True,
            'inference_time': timing.get('total_ms', 0) / 1000.0,
            'detections': formatted_detections,
            'annotated_image': annotated_image,
            'total_time': time.time() - start_request_time,
            'timing': timing,
            'original_size': result.get('original_size'),
        }
    
    # For other multi-input models that aren't DETR, return error
    if metadata:
        inputs = metadata.get('inputs', [])
        if len(inputs) > 1:
            input_names = [inp.get('name', 'unknown') for inp in inputs]
            return {
                'success': False,
                'error': f"Model '{model_name}' requires {len(inputs)} inputs ({', '.join(input_names)}). This multi-input model architecture is not yet supported."
            }
    
    # Get model input/output specs (auto-detected)
    input_spec = _client.get_model_input_spec(model_name)
    output_spec = _client.get_model_output_spec(model_name)
    
    # Preprocess image from bytes
    preprocess_start = time.time()
    image_array = _client.preprocess_image_bytes(file_bytes, model_name=model_name)
    preprocess_time = time.time() - preprocess_start
    
    if image_array is None:
        return {
            'success': False,
            'error': 'Failed to preprocess image'
        }
    
    # Send inference request
    inference_start = time.time()
    try:
        response = _client.send_inference_request(image_array, model_name, measure_latency=True)
    except Exception as e:
        return {
            'success': False,
            'error': f'Inference request failed: {str(e)}'
        }
    inference_time = time.time() - inference_start
    
    if response is None:
        return {
            'success': False,
            'error': 'Inference request failed - no response from server'
        }
    
    # Process prediction
    prediction_start = time.time()
    prediction = _client.process_prediction(response, model_name)
    prediction_time = time.time() - prediction_start
    
    if prediction is None:
        return {
            'success': False,
            'error': 'Failed to process prediction'
        }
    
    # Auto-detect model type if set to 'auto', otherwise use user selection
    actual_task_type = task_type
    if task_type == 'auto':
        # Get all output specs for better detection
        all_output_specs = None
        if response and 'outputs' in response:
            all_output_specs = [{'name': o.get('name', ''), 'shape': o.get('shape', [])} 
                               for o in response['outputs']]
        num_outputs = len(response.get('outputs', [])) if response else 1
        
        actual_task_type = detect_model_type(model_name, output_spec, num_outputs, all_output_specs)
    
    # Process based on task type
    if actual_task_type == 'detection':
        result = process_object_detection(
            prediction, response, filepath, filename, 
            model_name, inference_time, start_request_time,
            input_spec, output_spec, image_array
        )
    elif actual_task_type == 'pose':
        result = process_pose_estimation(
            prediction, response, filepath, filename,
            model_name, inference_time, start_request_time,
            input_spec, output_spec, image_array
        )
    elif actual_task_type == 'keypoint':
        result = process_keypoint_detection(
            prediction, response, filepath, filename,
            model_name, inference_time, start_request_time,
            input_spec, output_spec, image_array
        )
    elif actual_task_type == 'segmentation':
        result = process_segmentation(
            prediction, response, filepath, filename,
            model_name, inference_time, start_request_time,
            input_spec, output_spec, image_array
        )
    elif actual_task_type == 'panoptic':
        result = process_panoptic_segmentation(
            prediction, response, filepath, filename,
            model_name, inference_time, start_request_time,
            input_spec, output_spec, image_array
        )
    elif actual_task_type == 'ocr':
        result = process_ocr(
            prediction, response, filepath, filename,
            model_name, inference_time, start_request_time,
            input_spec, output_spec, image_array
        )
    else:
        # Image Classification processing (default)
        result = process_image_classification(
            prediction, response, filepath, filename, model_name, 
            inference_time, start_request_time,
            input_spec, output_spec, image_array,
            model_check_time, preprocess_time, prediction_time
        )
    
    # Add detected type info
    if task_type == 'auto':
        result['auto_detected'] = True
        result['detected_type'] = actual_task_type
    
    return result


# ------------------------------------------------------------------
# SPA (React) — single UI at `/`.
# The legacy Jinja chat + settings UI has been removed; all features
# are in the React SPA under webapp/spa/ (built by frontend/).
# ------------------------------------------------------------------
from flask import send_from_directory, abort

_SPA_DIST = os.environ.get(
    'SPA_DIST',
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'spa'),
)


@core_bp.route('/')
def index():
    """Root: serve the built React SPA."""
    idx = os.path.join(_SPA_DIST, 'index.html')
    if not os.path.isfile(idx):
        abort(
            500,
            description=(
                f"SPA not found at {_SPA_DIST}. "
                f"Build the frontend (cd frontend && pnpm build) or rebuild the Docker image."
            ),
        )
    # Vite asset filenames are content-hashed so /assets/* caches freely,
    # but index.html must revalidate so clients don't end up pinned to a
    # deleted bundle hash.
    resp = send_from_directory(_SPA_DIST, 'index.html')
    resp.headers['Cache-Control'] = 'no-cache, must-revalidate'
    return resp


@core_bp.route('/assets/<path:filename>')
def spa_assets(filename):
    """Serve SPA-built assets (JS/CSS/sourcemaps)."""
    assets_dir = os.path.join(_SPA_DIST, 'assets')
    if not os.path.isdir(assets_dir):
        abort(404)
    return send_from_directory(assets_dir, filename)


@core_bp.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests with proper resource management."""
    start_request_time = time.time()
    filepath = None  # Track filepath for cleanup in finally block
    
    try:
        log_processing_step("Request Received", "Starting prediction request", "info")
        
        if 'image' not in request.files:
            log_processing_step("Validation Failed", "No image file provided", "error")
            return jsonify({
                'success': False,
                'error': 'No image file provided',
                'error_code': 'MISSING_IMAGE'
            }), 400
        
        file = request.files['image']
        model_name = request.form.get('model')
        task_type = request.form.get('task_type', 'classification')
        
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No file selected',
                'error_code': 'EMPTY_FILENAME'
            }), 400
        
        if not model_name:
            return jsonify({
                'success': False,
                'error': 'No model selected',
                'error_code': 'MISSING_MODEL'
            }), 400
        
        log_processing_step("File Validation", f"Validating: {file.filename} (Task: {task_type})", "info")
        
        if not (file and allowed_file(file.filename, _app_config.get('allowed_extensions'))):
            return jsonify({
                'success': False,
                'error': 'Invalid file format',
                'error_code': 'INVALID_FILE_FORMAT'
            }), 400
        
        filename = secure_filename(file.filename)
        # Handle case where secure_filename returns empty string
        if not filename:
            filename = f"upload_{int(time.time())}.jpg"
        # Prefix with a unique ID to prevent concurrent upload collisions
        filename = f"{uuid.uuid4().hex[:8]}_{filename}"

        # Read image bytes directly from request
        file_bytes = file.read()

        # Save to disk for visualization functions that need a file path
        upload_folder = _app_config.get('upload_folder', '/tmp/uploads/')
        filepath = os.path.join(upload_folder, filename)
        with open(filepath, 'wb') as f:
            f.write(file_bytes)
        
        log_processing_step("File Upload", "File saved", "success")
        
        # Execute prediction using reusable function
        result = execute_prediction(filepath, file_bytes, model_name, task_type)
        
        # Check if prediction was successful
        if not result.get('success'):
            return jsonify(result), 500
        
        total_time = time.time() - start_request_time
        log_processing_step("Completion", f"Completed in {total_time:.3f}s", "success")
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': 'Internal server error during prediction',
            'error_code': 'INTERNAL_ERROR'
        }), 500
    
    finally:
        # Guaranteed cleanup of temporary file
        if filepath and os.path.exists(filepath):
            try:
                os.remove(filepath)
                logger.debug(f"Cleaned up temporary file: {filepath}")
            except Exception as cleanup_error:
                logger.warning(f"Failed to cleanup temporary file {filepath}: {cleanup_error}")


# Throttle backend health-check logs to avoid flooding on probe failures.
# Probe endpoints are called every ~10 seconds; we only want to log the
# backend state transitions, not every repeated failure.
_last_health_log: Dict[str, Any] = {
    'last_state': None,          # 'up' | 'down' | None
    'last_error_log_time': 0.0,  # epoch seconds of last ERROR log
    'log_interval_seconds': 300, # re-log persistent failures at most every 5 min
}


def _check_inference_backend() -> tuple:
    """
    Check inference backend (Triton/OpenVINO) status with log throttling.

    Returns a tuple of (server_healthy, server_type, server_info, models,
    health_message).  Suppresses repeated error logs when the backend is
    persistently unavailable.
    """
    import logging as _logging
    client_logger = _logging.getLogger('client')
    inference_logger = _logging.getLogger('inference')

    # Temporarily silence the noisy loggers for probe-driven checks.
    prev_levels = {}
    now = time.time()
    should_log = (
        _last_health_log['last_state'] != 'down'
        or now - _last_health_log['last_error_log_time']
            >= _last_health_log['log_interval_seconds']
    )
    if not should_log:
        for lg in (client_logger, inference_logger):
            prev_levels[lg] = lg.level
            lg.setLevel(_logging.CRITICAL)

    try:
        server_healthy, health_message = _client.check_server_health()
        server_type = _client.detect_server_type()
        server_info = _client.get_server_info()
        available_models = _client.get_available_models() or []
    finally:
        for lg, level in prev_levels.items():
            lg.setLevel(level)

    # Track state transitions for observability
    new_state = 'up' if server_healthy else 'down'
    if new_state != _last_health_log['last_state']:
        if new_state == 'up':
            logger.info(f"Inference backend is now available ({server_type})")
        else:
            logger.warning(
                f"Inference backend is unavailable: {health_message}. "
                f"Agent remains healthy; LLM and eval tools still work."
            )
        _last_health_log['last_state'] = new_state
        _last_health_log['last_error_log_time'] = now
    elif new_state == 'down' and should_log:
        logger.warning(
            f"Inference backend still unavailable after "
            f"{_last_health_log['log_interval_seconds']}s"
        )
        _last_health_log['last_error_log_time'] = now

    return server_healthy, server_type, server_info, available_models, health_message


_last_llm_log: Dict[str, Any] = {
    'last_state': None,
    'last_error_log_time': 0.0,
    'log_interval_seconds': 300,
}


def _check_llm_backend() -> Dict[str, Any]:
    """
    Check LLM backend (vLLM / llama.cpp) status with log throttling.

    Returns a dict with keys: ``available`` (bool), ``server_url`` (str),
    ``server_type`` (str), ``models`` (list[str]), ``error`` (Optional[str]).
    Returns ``available=False`` with no error if the LLM client is not
    configured (i.e. ``LLM_SERVER_URL`` is not set and no default reachable).
    """
    result: Dict[str, Any] = {
        'available': False,
        'server_url': None,
        'server_type': None,
        'models': [],
        'error': None,
    }
    try:
        from client.llm_client import get_llm_client
    except Exception as exc:
        result['error'] = f"LLM client unavailable: {exc}"
        return result

    try:
        llm_client = get_llm_client()
        result['server_url'] = llm_client.base_url
        result['server_type'] = llm_client.server_type.value
    except Exception as exc:
        result['error'] = f"Failed to init LLM client: {exc}"
        return result

    # Suppress repeated error logs during persistent down state
    import logging as _logging
    llm_logger = _logging.getLogger('client.llm_client')
    now = time.time()
    should_log = (
        _last_llm_log['last_state'] != 'down'
        or now - _last_llm_log['last_error_log_time']
            >= _last_llm_log['log_interval_seconds']
    )
    prev_level = llm_logger.level
    if not should_log:
        llm_logger.setLevel(_logging.CRITICAL)

    try:
        healthy = llm_client.is_healthy()
        if healthy:
            try:
                models_info = llm_client.list_models()
                result['models'] = [m.id for m in models_info]
            except Exception:
                result['models'] = []
        result['available'] = healthy
    except Exception as exc:
        result['error'] = str(exc)
    finally:
        llm_logger.setLevel(prev_level)

    # State transition logging
    new_state = 'up' if result['available'] else 'down'
    if new_state != _last_llm_log['last_state']:
        if new_state == 'up':
            logger.info(
                f"LLM backend is now available at {result['server_url']} "
                f"({len(result['models'])} model(s))"
            )
        else:
            logger.info(f"LLM backend not reachable at {result['server_url']}")
        _last_llm_log['last_state'] = new_state
        _last_llm_log['last_error_log_time'] = now

    return result


@core_bp.route('/health')
def health():
    """
    Liveness probe — always returns 200 OK as long as the Flask app
    is running.  This is the correct endpoint for Kubernetes liveness
    checks: it only signals "the application process is alive" and does
    NOT fail when optional dependencies (Triton/OpenVINO) are unreachable.

    For a stricter "is this ready to serve inference?" check, use
    ``/readiness``.
    """
    return jsonify(create_success_response({
        'status': 'ok',
        'service': 'ondevice-eval-agent',
    }))


@core_bp.route('/readiness')
@handle_exceptions("Readiness check failed")
def readiness():
    """
    Readiness probe — reports whether the agent can serve at least one
    type of inference request.

    The agent supports two independent backends:
      1. Triton/OpenVINO (discriminative models)
      2. vLLM / llama.cpp (LLMs)

    Returns 200 when EITHER backend is available with models. Returns
    200 (degraded) when a backend is reachable but has no models. Returns
    503 only when BOTH backends are unreachable.
    """
    # Check both backends
    (
        triton_healthy,
        triton_type,
        triton_info,
        triton_models,
        triton_msg,
    ) = _check_inference_backend()

    llm_status = _check_llm_backend()
    llm_available = llm_status.get('available', False)
    llm_models = llm_status.get('models', [])

    # Build consolidated status
    backends: Dict[str, Any] = {
        'inference': {
            'type': triton_type,
            'healthy': triton_healthy,
            'models': triton_models,
            'info': triton_info,
            'message': triton_msg,
        },
        'llm': {
            'type': llm_status.get('server_type'),
            'healthy': llm_available,
            'url': llm_status.get('server_url'),
            'models': llm_models,
            'error': llm_status.get('error'),
        },
    }

    # Ready if either backend has models loaded
    if triton_models or llm_models:
        return jsonify(create_success_response({
            'status': 'ready',
            'backends': backends,
            'message': (
                f"Ready — {len(triton_models)} inference model(s), "
                f"{len(llm_models)} LLM model(s)"
            ),
        }))

    # Degraded but usable — at least one backend is reachable
    if triton_healthy or llm_available:
        return jsonify(create_success_response({
            'status': 'degraded',
            'backends': backends,
            'message': 'Backend(s) reachable but no models are loaded',
        })), 200

    # Neither backend reachable — not ready
    raise ServiceUnavailableError(
        "No inference backend available (neither Triton/OpenVINO nor vLLM)",
        details={'backends': backends},
    )


@core_bp.route('/server-info')
@handle_exceptions("Failed to get server info")
def get_server_info():
    """Get inference server information"""
    server_type = _client.detect_server_type()
    server_info = _client.get_server_info()
    server_healthy, health_message = _client.check_server_health()
    
    return jsonify(create_success_response({
        'server_type': server_type,
        'server_info': server_info,
        'server_healthy': server_healthy,
        'health_message': health_message
    }))


@core_bp.route('/models')
@handle_exceptions("Failed to get available models")
def get_models():
    """Get available models"""
    models = _client.get_available_models()
    server_type = _client.detect_server_type()
    return jsonify(create_success_response({
        'models': models,
        'server_type': server_type
    }))


@core_bp.route('/debug/config')
@handle_exceptions("Failed to get debug config")
def debug_config():
    """Debug endpoint to check raw v1/config response.

    Only available when FLASK_DEBUG is enabled.
    """
    if os.environ.get("FLASK_DEBUG", "").lower() not in ("1", "true", "yes", "on"):
        return jsonify({
            'success': False,
            'error': 'Debug endpoints are only available when FLASK_DEBUG is enabled',
            'error_code': 'DEBUG_DISABLED'
        }), 403

    server_url = _client.server_url
    server_type = _client.detect_server_type()
    server_info = _client.get_server_info()
    known_models = _client._known_models
    
    # Try v1/config
    v1_config = None
    v1_config_error = None
    try:
        response = requests.get(f"{server_url}/v1/config", timeout=10)
        v1_config = {
            'status_code': response.status_code,
            'data': response.json() if response.status_code == 200 else response.text
        }
    except Exception as e:
        v1_config_error = str(e)
    
    # Try v2/repository/index
    v2_index = None
    v2_index_error = None
    try:
        response = requests.post(f"{server_url}/v2/repository/index", timeout=10)
        v2_index = {
            'status_code': response.status_code,
            'data': response.json() if response.status_code == 200 else response.text
        }
    except Exception as e:
        v2_index_error = str(e)
    
    return jsonify(create_success_response({
        'server_url': server_url,
        'server_type': server_type,
        'server_info': server_info,
        'known_models_from_env': known_models,
        'v1_config': v1_config,
        'v1_config_error': v1_config_error,
        'v2_repository_index': v2_index,
        'v2_repository_index_error': v2_index_error
    }))


@core_bp.route('/models/<model_name>/metadata')
@handle_exceptions("Failed to get model metadata")
def get_model_metadata(model_name):
    """Get detailed metadata for a specific model"""
    metadata = _client.get_model_metadata(model_name)
    input_spec = _client.get_model_input_spec(model_name)
    output_spec = _client.get_model_output_spec(model_name)
    
    detected_type = detect_model_type(model_name, output_spec)
    
    return jsonify(create_success_response({
        'model_name': model_name,
        'detected_type': detected_type,
        'metadata': metadata,
        'input_spec': input_spec,
        'output_spec': output_spec
    }))


@core_bp.route('/models/<model_name>/info')
@handle_exceptions("Failed to get model info")
def get_model_info(model_name):
    """Get comprehensive model information for display"""
    metadata = _client.get_model_metadata(model_name)
    input_spec = _client.get_model_input_spec(model_name)
    output_spec = _client.get_model_output_spec(model_name)
    all_output_specs = _client.get_all_output_specs(model_name)
    server_type = _client.detect_server_type()
    
    detected_type = detect_model_type(
        model_name, 
        output_spec, 
        num_outputs=len(all_output_specs),
        all_output_specs=all_output_specs
    )
    
    return jsonify(create_success_response({
        'model_name': model_name,
        'server_type': server_type,
        'detected_type': detected_type,
        'ready': _client.check_model_ready(model_name),
        'input': {
            'name': input_spec.get('name', 'input'),
            'shape': input_spec.get('shape', []),
            'shape_formatted': format_tensor_shape(input_spec.get('shape', [])),
            'datatype': input_spec.get('datatype', 'unknown'),
            'format': input_spec.get('format', 'unknown'),
            'width': input_spec.get('width'),
            'height': input_spec.get('height'),
            'channels': input_spec.get('channels', 3)
        },
        'output': {
            'name': output_spec.get('name', 'output'),
            'shape': output_spec.get('shape', []),
            'shape_formatted': format_tensor_shape(output_spec.get('shape', [])),
            'datatype': output_spec.get('datatype', 'unknown'),
            'num_classes': output_spec.get('num_classes')
        },
        'outputs': [
            {
                'name': spec.get('name', f'output_{i}'),
                'shape': spec.get('shape', []),
                'shape_formatted': format_tensor_shape(spec.get('shape', [])),
                'datatype': spec.get('datatype', 'unknown'),
                'num_classes': spec.get('num_classes')
            }
            for i, spec in enumerate(all_output_specs)
        ],
        'num_outputs': len(all_output_specs),
        'detection_disclaimer': 'Model type detection is based on heuristics and may be incorrect.'
    }))


@core_bp.route('/models/<model_name>/spec')
@handle_exceptions("Failed to get model spec")
def get_model_spec(model_name):
    """Get auto-detected input/output specifications for a model"""
    info = _client.get_full_model_info(model_name)
    
    return jsonify(create_success_response({
        'model_name': model_name,
        'ready': info['ready'],
        'input_spec': info['input_spec'],
        'output_spec': info['output_spec']
    }))


@core_bp.route('/models/<model_name>/endpoints')
@handle_exceptions("Failed to get model endpoints")
def get_model_endpoints(model_name):
    """Get API endpoint information for developers"""
    endpoints_info = _client.get_api_endpoints_info(model_name)
    
    return jsonify(create_success_response({
        'model_name': model_name,
        'endpoints': endpoints_info
    }))


@core_bp.route('/logs/endpoints')
def get_endpoint_logs():
    """Get recent endpoint call logs (thread-safe)"""
    with endpoint_logs_lock:
        return jsonify(create_success_response({'logs': list(endpoint_logs)}))


@core_bp.route('/logs/processing')
def get_processing_logs():
    """Get recent processing step logs (thread-safe)"""
    with processing_logs_lock:
        return jsonify(create_success_response({'logs': list(processing_logs)}))


@core_bp.route('/class_names', methods=['GET'])
def get_all_class_names():
    """Deprecated: Class names are now managed client-side."""
    return jsonify(create_success_response({
        'class_names': {},
        'message': 'Class names are now managed via client-side JSON upload.'
    }))


@core_bp.route('/class_names/<model_name>', methods=['GET'])
def get_model_class_names(model_name):
    """Deprecated: Class names are now managed client-side."""
    return jsonify(create_success_response({
        'model_name': model_name,
        'class_names': [],
        'message': 'Class names are now managed via client-side JSON upload.'
    }))


@core_bp.route('/class_names/<model_name>', methods=['POST'])
def update_model_class_names(model_name):
    """Deprecated: Class names are now managed client-side."""
    return jsonify({
        'success': False,
        'error': 'This endpoint is deprecated.',
        'error_code': 'ENDPOINT_DEPRECATED',
        'model_name': model_name
    }), 410


@core_bp.route('/logs/clear', methods=['POST'])
def clear_logs():
    """Clear all logs (thread-safe)"""
    clear_all_logs()
    return jsonify(create_success_response({'cleared': True}))


@core_bp.route('/config')
def get_config():
    """Get application configuration"""
    return jsonify(create_success_response(_app_config))
