#!/usr/bin/env python3
"""
Zededa Client Web Application
Flask web app that provides a web interface for ML model predictions.
"""

import os
import sys
import json
import time
import logging
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_from_directory, make_response
from werkzeug.utils import secure_filename
import numpy as np
from PIL import Image
from collections import deque

# Add parent directory to path to import client
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from client import ModelServerClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global stores for real-time logs and endpoint tracking
endpoint_logs = deque(maxlen=100)  # Store last 100 endpoint calls
processing_logs = deque(maxlen=100)  # Store last 100 processing steps

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = '/tmp/uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create upload folder
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'], mode=0o755)

# Initialize the model server client
client = ModelServerClient()

def log_endpoint_call(endpoint, method, status_code, response_time=None):
    """Log endpoint calls for real-time monitoring"""
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    log_entry = {
        'timestamp': timestamp,
        'endpoint': endpoint,
        'method': method,
        'status': status_code,
        'response_time': response_time
    }
    endpoint_logs.append(log_entry)
    
def log_processing_step(step, details, status="info"):
    """Log processing steps for real-time monitoring"""
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    log_entry = {
        'timestamp': timestamp,
        'step': step,
        'details': details,
        'status': status  # info, success, warning, error
    }
    processing_logs.append(log_entry)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def load_class_names():
    """Load class names from the class_names.json file"""
    try:
        class_names_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'class_names.json')
        with open(class_names_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading class names: {e}")
        return [f"Class_{i}" for i in range(196)]  # Fallback for Stanford Cars dataset

@app.route('/')
def index():
    """Main page with upload form"""
    log_processing_step("Page Load", "Loading main interface", "info")
    
    # Get available models from the server
    models_start = time.time()
    available_models = client.get_available_models()
    models_time = time.time() - models_start
    
    if not available_models:
        available_models = ['efb0_cars_onnx', 'efb3_car_onnx', 'efb3_car_openvino', 'efb5_cars_onnx', 'efv2m_cars_onnx']
        log_processing_step("Model Discovery", "Using fallback model list", "warning")
    else:
        log_endpoint_call("/v1/config", "GET", 200, models_time)
        log_processing_step("Model Discovery", f"Discovered {len(available_models)} models", "success")
    
    return render_template('index.html', models=available_models)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    start_request_time = time.time()
    
    try:
        log_processing_step("Request Received", "Starting image prediction request", "info")
        
        # Check if request has file
        if 'image' not in request.files:
            log_processing_step("Validation Failed", "No image file provided", "error")
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        model_name = request.form.get('model')
        
        if file.filename == '':
            log_processing_step("Validation Failed", "No file selected", "error")
            return jsonify({'error': 'No file selected'}), 400
        
        if not model_name:
            log_processing_step("Validation Failed", "No model selected", "error")
            return jsonify({'error': 'No model selected'}), 400
        
        log_processing_step("File Validation", f"Validating uploaded file: {file.filename}", "info")
        
        if file and allowed_file(file.filename):
            # Save uploaded file
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            log_processing_step("File Upload", f"File saved to {filepath}", "success")
            
            # Check if model is ready
            log_processing_step("Model Check", f"Checking if model {model_name} is ready", "info")
            model_check_start = time.time()
            
            model_ready = client.check_model_ready(model_name)
            model_check_time = time.time() - model_check_start
            
            # Log the endpoint call to check model readiness
            log_endpoint_call(f"/v1/models/{model_name}/versions/1/metadata", "GET", 
                            200 if model_ready else 503, model_check_time)
            
            if not model_ready:
                log_processing_step("Model Check", f"Model {model_name} is not ready", "error")
                return jsonify({'error': f'Model {model_name} is not ready'}), 503
            
            log_processing_step("Model Check", f"Model {model_name} is ready", "success")
            
            # Preprocess image
            log_processing_step("Image Processing", f"Preprocessing image: {filepath}", "info")
            logger.info(f"Preprocessing image: {filepath}")
            
            preprocess_start = time.time()
            image_array = client.preprocess_image(filepath, model_name=model_name)
            preprocess_time = time.time() - preprocess_start
            
            if image_array is None:
                logger.error("Failed to preprocess image")
                log_processing_step("Image Processing", "Failed to preprocess image", "error")
                return jsonify({'error': 'Failed to preprocess image'}), 400
            
            log_processing_step("Image Processing", 
                              f"Image preprocessed successfully, shape: {image_array.shape}, time: {preprocess_time:.3f}s", 
                              "success")
            logger.info(f"Image preprocessed successfully, shape: {image_array.shape}")
            
            # Send inference request
            log_processing_step("Inference", f"Sending inference request to model: {model_name}", "info")
            start_time = time.time()
            logger.info(f"Sending inference request to model: {model_name}")
            
            response = client.send_inference_request(image_array, model_name, measure_latency=True)
            end_time = time.time()
            inference_time = end_time - start_time
            
            # Log the inference endpoint call
            log_endpoint_call(f"/v1/models/{model_name}/versions/1:predict", "POST", 
                            200 if response else 500, inference_time)
            
            log_processing_step("Inference", 
                              f"Inference completed in {inference_time:.3f}s", 
                              "success" if response else "error")
            logger.info(f"Inference response received: {response}")
            
            if response is None:
                logger.error("Inference request failed - no response")
                log_processing_step("Inference", "Inference request failed - no response", "error")
                return jsonify({'error': 'Inference request failed'}), 500
            
            # Process prediction
            log_processing_step("Post-processing", f"Processing prediction for model: {model_name}", "info")
            logger.info(f"Processing prediction for model: {model_name}")
            
            prediction_start = time.time()
            prediction = client.process_prediction(response, model_name)
            prediction_time = time.time() - prediction_start
            
            log_processing_step("Post-processing", 
                              f"Prediction processed in {prediction_time:.3f}s", 
                              "success" if prediction else "error")
            logger.info(f"Processed prediction: {prediction}")
            
            if prediction is None:
                logger.error("Failed to process prediction")
                log_processing_step("Post-processing", "Failed to process prediction", "error")
                return jsonify({'error': 'Failed to process prediction'}), 500
            
            # Load class names and get top predictions
            class_names = load_class_names()
            
            # Extract top 5 predictions with class names
            top_predictions = []
            if 'top_predictions' in prediction:
                for pred in prediction['top_predictions'][:5]:
                    top_predictions.append({
                        'class_name': pred['class_name'],
                        'confidence': pred['confidence'],
                        'class_index': pred['class_id']
                    })
            
            total_time = time.time() - start_request_time
            log_processing_step("Completion", 
                              f"Request completed successfully in {total_time:.3f}s total", 
                              "success")
            
            # Get model input shape for processing information
            try:
                model_input_shape = client.get_model_input_shape(model_name)
            except Exception as e:
                logger.warning(f"Could not get input shape for {model_name}: {e}")
                model_input_shape = [224, 224]  # default
            
            result = {
                'success': True,
                'model_name': model_name,
                'latency': inference_time,
                'total_time': total_time,
                'top_predictions': top_predictions,
                'image_filename': filename,
                'model_input_shape': model_input_shape,
                'processing_times': {
                    'model_check': model_check_time,
                    'preprocessing': preprocess_time,
                    'inference': inference_time,
                    'postprocessing': prediction_time,
                    'total': total_time
                }
            }
            
            # Clean up uploaded file
            try:
                os.remove(filepath)
            except:
                pass
            
            return jsonify(result)
        
        else:
            log_processing_step("Validation Failed", "Invalid file format", "error")
            return jsonify({'error': 'Invalid file format. Please upload PNG, JPG, JPEG, or GIF files.'}), 400
    
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        log_processing_step("Error", f"Internal server error: {str(e)}", "error")
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

@app.route('/process_image', methods=['POST'])
def process_image():
    """Process image and return result with detection info in headers"""
    try:
        # Check if request has file
        if 'file' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['file']
        model_name = request.form.get('model')  # Get selected model from form
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not model_name:
            # Fallback to first available model if none selected
            available_models = client.get_available_models()
            if not available_models:
                available_models = ['efb3_car_onnx']  # fallback
            model_name = available_models[0]
        
        if file and allowed_file(file.filename):
            # Save uploaded file
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Check if model is ready
            if not client.check_model_ready(model_name):
                return jsonify({'error': f'Model {model_name} is not ready'}), 503
            
            # Preprocess image
            logger.info(f"Preprocessing image: {filepath}")
            image_array = client.preprocess_image(filepath, model_name=model_name)
            if image_array is None:
                logger.error("Failed to preprocess image")
                return jsonify({'error': 'Failed to preprocess image'}), 400
            
            logger.info(f"Image preprocessed successfully, shape: {image_array.shape}")
            
            # Send inference request
            start_time = time.time()
            logger.info(f"Sending inference request to model: {model_name}")
            response = client.send_inference_request(image_array, model_name, measure_latency=True)
            end_time = time.time()
            
            logger.info(f"Inference response received: {response}")
            
            if response is None:
                logger.error("Inference request failed - no response")
                return jsonify({'error': 'Inference request failed'}), 500
            
            # Process prediction
            logger.info(f"Processing prediction for model: {model_name}")
            prediction = client.process_prediction(response, model_name)
            logger.info(f"Processed prediction: {prediction}")
            
            if prediction is None:
                logger.error("Failed to process prediction")
                return jsonify({'error': 'Failed to process prediction'}), 500
            
            # Create detection info for header
            detection_info = f"Model: {model_name} | Runtime: ONNX | Latency: {(end_time - start_time)*1000:.0f}ms"
            
            # Get top prediction
            top_prediction = ""
            if 'top_predictions' in prediction and prediction['top_predictions']:
                top_pred = prediction['top_predictions'][0]
                top_prediction = f"Detections: {top_pred['class_name']} ({top_pred['confidence']:.2f})"
                detection_info += f" | {top_prediction}"
            
            # Return the original image with detection info in headers
            with open(filepath, 'rb') as f:
                img_data = f.read()
            
            # Clean up uploaded file
            try:
                os.remove(filepath)
            except:
                pass
            
            response = make_response(img_data)
            response.headers['Content-Type'] = 'image/jpeg'
            response.headers['X-Detection-Info'] = detection_info
            
            return response
        
        else:
            return jsonify({'error': 'Invalid file format. Please upload PNG, JPG, JPEG, or GIF files.'}), 400
    
    except Exception as e:
        logger.error(f"Error during image processing: {e}")
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    try:
        health_start = time.time()
        # Check if model server is accessible
        available_models = client.get_available_models()
        health_time = time.time() - health_start
        
        if available_models:
            log_endpoint_call("/v1/config", "GET", 200, health_time)
            log_processing_step("Health Check", f"System healthy - {len(available_models)} models available", "success")
            return jsonify({
                'status': 'healthy',
                'available_models': available_models
            })
        else:
            log_endpoint_call("/v1/config", "GET", 503, health_time)
            log_processing_step("Health Check", "System unhealthy - no models available", "warning")
            return jsonify({
                'status': 'unhealthy',
                'error': 'No models available'
            }), 503
    except Exception as e:
        log_processing_step("Health Check", f"System unhealthy - error: {str(e)}", "error")
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 503

@app.route('/models')
def get_models():
    """Get available models"""
    try:
        models_start = time.time()
        models = client.get_available_models()
        models_time = time.time() - models_start
        
        # Log the endpoint call
        log_endpoint_call("/v1/config", "GET", 200, models_time)
        log_processing_step("Model Discovery", f"Retrieved {len(models)} available models", "success")
        
        return jsonify({'models': models})
    except Exception as e:
        log_processing_step("Model Discovery", f"Failed to get models: {str(e)}", "error")
        return jsonify({'error': str(e)}), 500

@app.route('/models/<model_name>/metadata')
def get_model_metadata(model_name):
    """Get detailed metadata for a specific model"""
    try:
        metadata_start = time.time()
        metadata = client.get_model_metadata(model_name)
        metadata_time = time.time() - metadata_start
        
        if metadata:
            # Log the endpoint call
            log_endpoint_call(f"/v2/models/{model_name}/", "GET", 200, metadata_time)
            log_processing_step("Model Metadata", f"Retrieved metadata for {model_name}", "success")
            
            return jsonify({
                'success': True,
                'metadata': metadata,
                'model_name': model_name
            })
        else:
            log_endpoint_call(f"/v2/models/{model_name}/", "GET", 404, metadata_time)
            log_processing_step("Model Metadata", f"Failed to get metadata for {model_name}", "error")
            return jsonify({'error': f'Could not retrieve metadata for model {model_name}'}), 404
            
    except Exception as e:
        log_processing_step("Model Metadata", f"Error getting metadata for {model_name}: {str(e)}", "error")
        return jsonify({'error': str(e)}), 500

@app.route('/models/<model_name>/endpoints')
def get_model_endpoints(model_name):
    """Get API endpoint information for developers"""
    try:
        endpoints_info = client.get_api_endpoints_info(model_name)
        
        log_processing_step("Developer Info", f"Generated endpoint information for {model_name}", "success")
        
        return jsonify({
            'success': True,
            'model_name': model_name,
            'endpoints': endpoints_info
        })
        
    except Exception as e:
        log_processing_step("Developer Info", f"Error generating endpoints for {model_name}: {str(e)}", "error")
        return jsonify({'error': str(e)}), 500

@app.route('/logs/endpoints')
def get_endpoint_logs():
    """Get recent endpoint call logs"""
    return jsonify({'logs': list(endpoint_logs)})

@app.route('/logs/processing')
def get_processing_logs():
    """Get recent processing step logs"""
    return jsonify({'logs': list(processing_logs)})

@app.route('/logs/clear', methods=['POST'])
def clear_logs():
    """Clear all logs"""
    endpoint_logs.clear()
    processing_logs.clear()
    log_processing_step("System", "All logs cleared", "info")
    return jsonify({'success': True})

if __name__ == '__main__':
    logger.info("Starting Zededa Client Web Application...")
    app.run(host='0.0.0.0', port=8080, debug=True)
