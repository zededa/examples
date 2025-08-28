#!/usr/bin/env python3
"""
Streamlined client script focused on core model inference functionality.
Supports only the essential features needed by the webapp.
"""

import os
import time
import json
import requests
import numpy as np
from PIL import Image
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelServerClient:
    def __init__(self, server_url=None):
        """
        Initialize the client with server URL.
        
        Args:
            server_url (str): URL of the OpenVINO model server REST API
                             (can be overridden by MODEL_SERVER_URL env var)
        """
        # Read from environment variables with fallbacks
        self.server_url = server_url or os.environ.get('MODEL_SERVER_URL', 'http://localhost:8000')
        self.models_url = f"{self.server_url}/v1/config"
        
        # Load class names
        self.class_names = self.load_class_names()
        
        logger.info(f"Client initialized with server URL: {self.server_url}")
        
    def load_class_names(self):
        """
        Load class names from the class_names.json file.
        
        Returns:
            list: List of class names, indexed by class ID
        """
        try:
            class_names_file = os.path.join(os.path.dirname(__file__), 'class_names.json')
            with open(class_names_file, 'r') as f:
                class_names = json.load(f)
            logger.info(f"Loaded {len(class_names)} class names")
            return class_names
        except Exception as e:
            logger.error(f"Failed to load class names: {e}")
            return None
        
    def preprocess_image(self, image_path, model_name=None, target_size=None):
        """
        Preprocess image for model inference with dynamic sizing.
        
        Args:
            image_path (str): Path to the image file
            model_name (str): Name of the model (used to get optimal input shape)
            target_size (tuple): Target size for the image (width, height). 
                                If None and model_name provided, will get from model metadata
            
        Returns:
            np.ndarray: Preprocessed image array
        """
        try:
            # Determine target size
            if target_size is None and model_name:
                # Get optimal size from model metadata
                height, width = self.get_model_input_shape(model_name)
                target_size = (width, height)  # PIL expects (width, height)
                logger.info(f"🎯 Using model-specific input size for {model_name}: {width}x{height}")
            elif target_size is None:
                target_size = (224, 224)  # Default fallback
                logger.info(f"📏 Using default input size: 224x224")
            else:
                logger.info(f"📏 Using provided input size: {target_size[0]}x{target_size[1]}")
            
            # Load and resize image
            image = Image.open(image_path)
            image = image.convert('RGB')
            image = image.resize(target_size, Image.Resampling.LANCZOS)
            
            # Convert to numpy array and normalize
            image_array = np.array(image, dtype=np.float32)
            
            # Normalize to [0, 1] range
            image_array = image_array / 255.0
            
            # EfficientNet preprocessing (ImageNet normalization)
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            image_array = (image_array - mean) / std
            
            # Add batch dimension and convert to NCHW format (batch, channels, height, width)
            image_array = np.transpose(image_array, (2, 0, 1))  # HWC to CHW
            image_array = np.expand_dims(image_array, axis=0)   # Add batch dimension
            
            return image_array
            
        except Exception as e:
            logger.error(f"Error preprocessing image {image_path}: {e}")
            return None
    
    def send_inference_request(self, image_array, model_name, measure_latency=False):
        """
        Send inference request to the model server for a specific model.
        
        Args:
            image_array (np.ndarray): Preprocessed image array
            model_name (str): Name of the model to use for inference
            measure_latency (bool): Whether to measure and store latency
            
        Returns:
            dict: Response from the model server with optional timing info
        """
        try:
            inference_url = f"{self.server_url}/v1/models/{model_name}:predict"
            
            # Prepare the request payload
            payload = {
                "inputs": {
                    "input": image_array.tolist()
                }
            }
            
            # Send POST request with timing
            headers = {"Content-Type": "application/json"}
            
            start_time = time.time()
            response = requests.post(
                inference_url,
                json=payload,
                headers=headers,
                timeout=30
            )
            end_time = time.time()
            
            latency = end_time - start_time
            
            if response.status_code == 200:
                result = response.json()
                if measure_latency:
                    result['latency'] = latency
                logger.debug(f"Raw response from {model_name}: {json.dumps(result, indent=2)}")
                return result
            else:
                logger.error(f"Server returned status code {response.status_code} for model {model_name}: {response.text}")
                return None
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed for model {model_name}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error during inference for model {model_name}: {e}")
            return None
    
    def process_prediction(self, response, model_name=None):
        """
        Process the prediction response from the server.
        
        Args:
            response (dict): Response from the model server
            model_name (str): Name of the model (for logging purposes)
            
        Returns:
            dict: Processed prediction results
        """
        try:
            model_info = f" for model {model_name}" if model_name else ""
            logger.debug(f"Processing response{model_info}: {json.dumps(response, indent=2) if response else 'None'}")
            
            if not response or 'outputs' not in response:
                logger.error(f"Invalid response format{model_info}")
                return None
            
            # Get the output predictions - it's a nested list
            predictions = response['outputs']
            logger.debug(f"Raw predictions{model_info}: {predictions}")
            logger.debug(f"Predictions type: {type(predictions)}")
            
            # Handle the response format: {"outputs": [[values...]]}
            if isinstance(predictions, list) and len(predictions) > 0:
                # Get the first (and likely only) output array
                if isinstance(predictions[0], list):
                    scores = np.array(predictions[0])
                else:
                    scores = np.array(predictions)
            else:
                logger.error(f"Unexpected predictions format{model_info}: {type(predictions)}")
                return None
            
            # Get top 5 predictions
            top_indices = np.argsort(scores)[-5:][::-1]
            top_scores = scores[top_indices]
            
            results = {
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'model_name': model_name,
                'top_predictions': []
            }
            
            for i, (idx, score) in enumerate(zip(top_indices, top_scores)):
                class_name = "Unknown"
                if self.class_names and 0 <= idx < len(self.class_names):
                    class_name = self.class_names[idx]
                
                results['top_predictions'].append({
                    'rank': i + 1,
                    'class_id': int(idx),
                    'class_name': class_name,
                    'confidence': float(score),
                    'probability': float(score)  # Keep raw confidence value, don't multiply by 100
                })
            
            return results
            
        except Exception as e:
            model_info = f" for model {model_name}" if model_name else ""
            logger.error(f"Error processing prediction{model_info}: {e}")
            return None
    
    def get_available_models(self):
        """
        Get list of available models using multiple discovery methods.
        
        Returns:
            list: List of available model names
        """
        available_models = []
        
        # Method 1: Use the /v1/config endpoint to get all models and their status
        try:
            response = requests.get(self.models_url, timeout=10, headers={"Content-Type": "application/json"})
            
            if response.status_code == 200:
                models_data = response.json()
                logger.debug(f"Config response: {json.dumps(models_data, indent=2)}")
                
                # Parse models from OVMS config response
                # Format: {"model_name": {"model_version_status": [...]}, ...}
                for model_name, model_info in models_data.items():
                    if isinstance(model_info, dict) and 'model_version_status' in model_info:
                        # Check if any version is available
                        for version_status in model_info['model_version_status']:
                            if version_status.get('state') == 'AVAILABLE':
                                available_models.append(model_name)
                                logger.info(f"✅ Found available model via config: {model_name}")
                                break
                
                if available_models:
                    logger.info(f"🎯 Successfully discovered {len(available_models)} models via /v1/config: {available_models}")
                    return available_models
            else:
                logger.debug(f"Config endpoint returned: {response.status_code} - {response.text}")
                
        except Exception as e:
            logger.debug(f"Config API failed: {e}")
        
        # Method 2: Fallback to individual model testing if config doesn't work
        logger.info("Falling back to individual model endpoint testing...")
        
        # Known model names to test
        known_models = [
            "efb3_car_onnx",
            "car_classifier", 
            "resnet50",
            "mobilenet",
            "efficientnet"
        ]
        
        for model_name in known_models:
            try:
                # Test if model is available by checking its metadata
                metadata_url = f"{self.server_url}/v1/models/{model_name}/metadata"
                response = requests.get(metadata_url, timeout=5)
                
                if response.status_code == 200:
                    metadata = response.json()
                    # Check if model is actually available (not just loaded)
                    if 'model_version_status' in metadata:
                        for version_status in metadata['model_version_status']:
                            if version_status.get('state') == 'AVAILABLE':
                                available_models.append(model_name)
                                logger.info(f"✅ Found available model: {model_name}")
                                break
                    else:
                        # Some response formats might be different
                        available_models.append(model_name)
                        logger.info(f"✅ Found model: {model_name}")
                else:
                    logger.debug(f"❌ Model {model_name} not available: {response.status_code}")
                    
            except Exception as e:
                logger.debug(f"❌ Error checking model {model_name}: {e}")
        
        if available_models:
            logger.info(f"🎯 Discovered {len(available_models)} available models: {available_models}")
        else:
            logger.warning("⚠️ No models discovered via any method")
            
        return available_models

    def check_model_ready(self, model_name):
        """
        Check if a specific model is ready for inference.
        
        Args:
            model_name (str): Name of the model to check
            
        Returns:
            bool: True if model is ready, False otherwise
        """
        try:
            # Try metadata endpoint first
            metadata_url = f"{self.server_url}/v1/models/{model_name}/metadata"
            response = requests.get(metadata_url, timeout=10)
            
            if response.status_code == 200:
                logger.debug(f"Model {model_name} metadata available")
                
                # Also try ready endpoint if available
                ready_url = f"{self.server_url}/v1/models/{model_name}"
                ready_response = requests.get(ready_url, timeout=10)
                
                if ready_response.status_code == 200:
                    logger.debug(f"Model {model_name} is ready")
                    return True
                else:
                    # If ready endpoint doesn't exist, metadata success means ready
                    logger.debug(f"Model {model_name} ready endpoint not available, assuming ready from metadata")
                    return True
            else:
                logger.debug(f"Model {model_name} not ready: {response.status_code}")
                return False
                
        except Exception as e:
            logger.debug(f"Error checking if model {model_name} is ready: {e}")
            return False

    def get_model_metadata(self, model_name):
        """
        Get detailed model metadata using the v2 API endpoint.
        
        Args:
            model_name (str): Name of the model to get metadata for
            
        Returns:
            dict: Model metadata including input shape, data type, platform, etc.
        """
        try:
            # Use v2 API for detailed model information
            metadata_url = f"{self.server_url}/v2/models/{model_name}/"
            logger.info(f"🔍 Getting model metadata from: {metadata_url}")
            
            response = requests.get(metadata_url, timeout=30)
            response.raise_for_status()
            
            metadata = response.json()
            logger.info(f"📊 Model metadata retrieved for {model_name}: {metadata}")
            
            return metadata
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error getting model metadata for {model_name}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error getting model metadata for {model_name}: {e}")
            return None

    def get_model_input_shape(self, model_name):
        """
        Get the input shape for a specific model from its metadata.
        
        Args:
            model_name (str): Name of the model
            
        Returns:
            tuple: Input shape (height, width) or default (224, 224)
        """
        try:
            metadata = self.get_model_metadata(model_name)
            if metadata and 'inputs' in metadata and len(metadata['inputs']) > 0:
                input_info = metadata['inputs'][0]
                if 'shape' in input_info:
                    shape = input_info['shape']
                    # Assume shape is [-1, channels, height, width] or [-1, height, width, channels]
                    if len(shape) >= 4:
                        # For NCHW format: [-1, 3, 224, 224]
                        if shape[1] == 3:  # channels first
                            height, width = shape[2], shape[3]
                        # For NHWC format: [-1, 224, 224, 3]
                        elif shape[-1] == 3:  # channels last
                            height, width = shape[1], shape[2]
                        else:
                            # Default fallback
                            height, width = 224, 224
                        
                        logger.info(f"📐 Model {model_name} input shape: {height}x{width}")
                        return (height, width)
            
            # Default fallback
            logger.warning(f"⚠️ Could not determine input shape for {model_name}, using default 224x224")
            return (224, 224)
            
        except Exception as e:
            logger.error(f"Error determining input shape for {model_name}: {e}")
            return (224, 224)

    def get_api_endpoints_info(self, model_name):
        """
        Get information about API endpoints for developers.
        
        Args:
            model_name (str): Name of the model
            
        Returns:
            dict: Dictionary containing endpoint information and curl commands
        """
        base_url = self.server_url
        
        endpoints_info = {
            "model_metadata": {
                "endpoint": f"{base_url}/v2/models/{model_name}/",
                "method": "GET",
                "description": "Get detailed model metadata including input/output shapes",
                "curl_command": f"curl -v {base_url}/v2/models/{model_name}/ | jq",
                "example_response": "JSON with model inputs, outputs, platform info"
            },
            "model_ready": {
                "endpoint": f"{base_url}/v1/models/{model_name}",
                "method": "GET", 
                "description": "Check if model is ready for inference",
                "curl_command": f"curl -v {base_url}/v1/models/{model_name}",
                "example_response": "JSON with model version status"
            },
            "model_metadata_v1": {
                "endpoint": f"{base_url}/v1/models/{model_name}/metadata",
                "method": "GET",
                "description": "Get model metadata using TensorFlow Serving format",
                "curl_command": f"curl -v {base_url}/v1/models/{model_name}/metadata",
                "example_response": "JSON with signature_def and tensor shapes"
            },
            "inference_v1": {
                "endpoint": f"{base_url}/v1/models/{model_name}:predict",
                "method": "POST",
                "description": "Send inference request using v1 API (TensorFlow Serving format)",
                "curl_command": f"""curl -X POST {base_url}/v1/models/{model_name}:predict \\
  -H "Content-Type: application/json" \\
  -d '{{"inputs": {{"input": [[[[...]]]]}}}}'""",
                "example_response": "JSON with outputs in TensorFlow Serving format"
            },
            "model_ready_v2": {
                "endpoint": f"{base_url}/v2/models/{model_name}/ready",
                "method": "GET",
                "description": "Check if model is ready using v2 API",
                "curl_command": f"curl -v {base_url}/v2/models/{model_name}",
                "example_response": "Empty response with HTTP 200 if ready"
            },
            "inference_v2": {
                "endpoint": f"{base_url}/v2/models/{model_name}/infer",
                "method": "POST",
                "description": "Send inference request using v2 API (KServe format)",
                "curl_command": f"""curl -X POST {base_url}/v2/models/{model_name}/infer \\
  -H "Content-Type: application/json" \\
  -d '{{"inputs": [{{"name": "input", "shape": [1,3,224,224], "datatype": "FP32", "data": [...]}}]}}'""",
                "example_response": "JSON with outputs in KServe format"
            },
            "all_models": {
                "endpoint": f"{base_url}/v1/config",
                "method": "GET",
                "description": "Get configuration and status of all models",
                "curl_command": f"curl -v {base_url}/v1/config | jq",
                "example_response": "JSON with all model configurations and status"
            }
        }
        
        return endpoints_info

if __name__ == "__main__":
    # Simple test functionality
    import argparse
    
    parser = argparse.ArgumentParser(description='Simple OpenVINO Model Server Client')
    parser.add_argument('--server-url', default=os.getenv('MODEL_SERVER_URL', 'http://localhost:8000'),
                       help='Model server URL')
    parser.add_argument('--image-path', required=True, help='Path to test image')
    parser.add_argument('--model', help='Specific model to use (default: first available)')
    
    args = parser.parse_args()
    
    # Create client
    client = ModelServerClient(server_url=args.server_url)
    
    # Get available models
    models = client.get_available_models()
    if not models:
        logger.error("No models available")
        exit(1)
    
    model_name = args.model if args.model and args.model in models else models[0]
    logger.info(f"Using model: {model_name}")
    
    # Check if image exists
    if not os.path.exists(args.image_path):
        logger.error(f"Image file not found: {args.image_path}")
        exit(1)
    
    # Run inference
    image_array = client.preprocess_image(args.image_path, model_name)
    if image_array is None:
        logger.error("Failed to preprocess image")
        exit(1)
    
    response = client.send_inference_request(image_array, model_name, measure_latency=True)
    if response is None:
        logger.error("Inference failed")
        exit(1)
    
    results = client.process_prediction(response, model_name)
    if results is None:
        logger.error("Failed to process prediction")
        exit(1)
    
    # Display results
    print(f"\nPrediction Results for {model_name}:")
    print(f"Timestamp: {results['timestamp']}")
    if 'latency' in response:
        print(f"Latency: {response['latency']*1000:.2f} ms")
    
    print("\nTop predictions:")
    for pred in results['top_predictions']:
        print(f"  {pred['rank']}. {pred['class_name']} "
              f"(Class {pred['class_id']}) - "
              f"Confidence: {pred['confidence']:.4f} "
              f"({pred['probability']*100:.2f}%)")
