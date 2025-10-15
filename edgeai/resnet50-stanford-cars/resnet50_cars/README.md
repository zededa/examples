# ResNet50 Stanford Cars Model - Triton Deployment

## Model Overview
- **Model Name**: resnet50_cars
- **Architecture**: ResNet50  
- **Task**: Classification
- **Dataset**: Stanford Cars
- **Classes**: 196
- **Format**: ONNX
- **Size**: 94.00 MB

## Triton Inference Server Configuration

This model is configured for deployment with NVIDIA Triton Inference Server:

### Key Features
- **Multi-GPU Support**: Configured for both CPU and GPU inference
- **Dynamic Batching**: Optimized batch sizes for classification
- **Performance Optimization**: OpenVINO acceleration enabled
- **Flexible Scaling**: Instance groups for different hardware

### Configuration Details
```
Max Batch Size: 32
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
mkdir -p model_repository/resnet50_cars/1
cp model.onnx model_repository/resnet50_cars/1/
cp config.pbtxt model_repository/resnet50_cars/
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
results = client.infer("resnet50_cars", inputs, outputs=outputs)

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
- Dynamic batching with queue optimization
- Configurable timeout settings

### Hardware Requirements
- **Minimum**: 4GB RAM, 2 CPU cores  
- **Recommended**: 8GB RAM, 4 CPU cores, NVIDIA GPU
- **Storage**: ~188 MB for model + dependencies

## Model Performance
- **Accuracy**: 89.39% on Stanford Cars validation set (measured)
- **Inference Time**: 398.45ms per image (CPU/ONNX Runtime)
- **Model Size**: 94.0 MB (FP32 baseline)
- **Throughput**: ~2.5 images/second (single image inference)

## Class Labels
This model predicts among 196 car classes from the Stanford Cars dataset.

## Troubleshooting

### Common Issues
1. **GPU Memory**: Reduce batch size if OOM errors occur
2. **Dependencies**: Install all requirements from requirements.txt
3. **Model Loading**: Ensure ONNX model is valid and accessible

### Performance Tuning
- Adjust `max_batch_size` in config.pbtxt
- Enable TensorRT optimization for NVIDIA GPUs
- Use multiple instance groups for high throughput

## Support
For deployment issues, check:
- Model path and permissions
- GPU availability and drivers
- Triton server logs
- Network connectivity (ports 8000-8002)
