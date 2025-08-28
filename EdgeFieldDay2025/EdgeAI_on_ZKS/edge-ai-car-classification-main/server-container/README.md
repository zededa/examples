# Server Container

This container packages an OpenVINO Model Server with the EfficientNet B3 car classifier model.

## Model Loading Options

The container supports two modes for loading models:

### 1. External Volume (Recommended for Production)
Mount your models from an external volume:

```bash
docker run -d --rm \
  -p 9001:9000 -p 8001:8000 \
  -v /path/to/your/models:/models \
  car-classifier-server
```

### 2. Embedded Models (Fallback)
If no external volume is mounted, the container will use embedded models:

```bash
docker run -d --rm -p 9001:9000 -p 8001:8000 car-classifier-server
```

## Building the Container

```bash
docker build -t car-classifier-server .
```

## Model Directory Structure

The container expects models to be organized as:
```
/models/
└── efb3_car_onnx/
    └── 1/
        └── efficientnet_b3_car_classifier.onnx
```

## Kubernetes Deployment

When deployed in Kubernetes with persistent volumes:
- Models are mounted from PVC to `/models`
- The entrypoint script automatically detects and uses external models
- Fallback to embedded models if volume mounting fails

## Ports

- **8001**: REST API port (host) → 8000 (container)
- **9001**: gRPC port (host) → 9000 (container)

## Model Details

- **Model Name**: efb3_car_onnx
- **Model Path**: /models/efb3_car_onnx/
- **Model File**: efficientnet_b3_car_classifier.onnx

## API Endpoints

Once running, you can access:

- REST API: `http://localhost:8001/v1/models/efb3_car_onnx`
- Model metadata: `http://localhost:8001/v1/models/efb3_car_onnx/metadata`
- Health check: `http://localhost:8001/v1/models/efb3_car_onnx/ready`

## Testing the Model

You can test the model by sending POST requests to:
```
http://localhost:8001/v1/models/efb3_car_onnx:predict
```

Example using curl:
```bash
curl -X POST http://localhost:8001/v1/models/efb3_car_onnx:predict \
  -H "Content-Type: application/json" \
  -d '{"inputs": [{"name": "input", "shape": [1, 3, 224, 224], "datatype": "FP32", "data": [...]}]}'
```
