# EfficientNet-B3 Stanford Cars Model - Triton Deployment

## Model Overview
- **Model Name**: efficientnet_b3_cars
- **Architecture**: EfficientNet-B3  
- **Task**: Classification
- **Dataset**: Stanford Cars
- **Classes**: 196
- **Format**: ONNX
- **Size**: 44.19 MB

## Triton Inference Server Configuration

This model is configured for deployment with NVIDIA Triton Inference Server:

### Key Features
- **Multi-GPU Support**: Configured for both CPU and GPU inference
- **Dynamic Batching**: Optimized batch sizes for classification
- **Performance Optimization**: OpenVINO acceleration enabled
- **Flexible Scaling**: Instance groups for different hardware

### Configuration Details
```
Max Batch Size: 8
Input Shape: [3, 224, 224]
Output Classes: 196
Batching Strategy: Dynamic with preferred sizes [1, 2, 4, 8]
```

## Deployment Instructions

### 1. Setup Triton Server
```bash
# Pull Triton container
docker pull nvcr.io/nvidia/tritonserver:23.04-py3

# Create model repository
mkdir -p model_repository/efficientnet_b3_cars/1
cp model.onnx model_repository/efficientnet_b3_cars/1/
cp config.pbtxt model_repository/efficientnet_b3_cars/
```

### 2. Start Triton Server
```bash
docker run --gpus=all -it --rm \
  -p8000:8000 -p8001:8001 -p8002:8002 \
  -v$(pwd)/model_repository:/models \
  nvcr.io/nvidia/tritonserver:23.04-py3 \
  tritonserver --model-repository=/models
```

### 3. Client Usage
```python
import tritonclient.http as httpclient
import numpy as np

# Create client
client = httpclient.InferenceServerClient("localhost:8000")

# Prepare input
input_data = np.random.random((1, 3, 224, 224)).astype(np.float32)

# Create input object
inputs = [httpclient.InferInput("images", input_data.shape, "FP32")]
inputs[0].set_data_from_numpy(input_data)

# Create output object  
outputs = [httpclient.InferRequestedOutput("predictions")]

# Run inference
results = client.infer("efficientnet_b3_cars", inputs, outputs=outputs)

# Get predictions
predictions = results.as_numpy("predictions")
predicted_class = np.argmax(predictions[0])
print(f"Predicted class: {predicted_class}")
```

## Performance Optimization

### GPU Acceleration
- Model supports both CPU and GPU inference
- Automatic GPU selection with fallback to CPU
- OpenVINO optimization for Intel CPUs

### Batch Processing
- Optimal batch sizes: 1, 2, 4, 8
- Dynamic batching reduces latency
- Queue delay: 500μs for classification

### Monitoring
```bash
# Check model status
curl localhost:8000/v2/models/efficientnet_b3_cars

# Get server metrics  
curl localhost:8000/metrics
```

## Production Considerations

1. **Resource Requirements**
   - CPU: 4+ cores recommended
   - RAM: 8GB+ for model + batches
   - GPU: 6GB+ VRAM for optimal performance

2. **Scaling**
   - Horizontal: Multiple Triton instances
   - Vertical: Increase instance count in config

3. **Monitoring**
   - Use Prometheus metrics endpoint
   - Monitor batch queue depth
   - Track inference latency

## Model-Specific Notes
- **Input**: RGB images, normalized [0,1]
- **Preprocessing**: Resize to 224x224, center crop
- **Output**: Class probabilities for 196 categories
- **Postprocessing**: Apply softmax, get argmax for prediction

This configuration is optimized for production deployment of EfficientNet-B3 models.
