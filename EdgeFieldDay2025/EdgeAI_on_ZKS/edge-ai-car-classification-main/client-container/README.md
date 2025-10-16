# OpenVINO Model Server Client

This client sends images to an OpenVINO Model Server running an EfficientNet-B3 car classification model every 15 seconds and displays the prediction results.

## Prerequisites

- Docker installed on your system
- OpenVINO Model Server running with the EfficientNet-B3 car model
- Stanford Cars dataset (included in this repository)

## Your Model Server

Make sure your OpenVINO Model Server is running:

```bash
docker run -d --rm -v ${PWD}/models:/models -p 9000:9000 -p 8000:8000 openvino/model_server:latest \
--model_path /models/efb3_car_onnx/ --model_name efb3_car_onnx --port 9000 --rest_port 8000 --log_level DEBUG
```

## Running the Client

### Option 1: Using the shell script (recommended)

```bash
./run_client.sh
```

### Option 2: Using Docker directly

Build the client image:
```bash
docker build -t openvino-client .
```

Run the client container:
```bash
docker run -d --rm \
  --name model_client \
  --network host \
  -e MODEL_SERVER_URL=http://localhost:8000 \
  -e MODEL_NAME=efb3_car_onnx \
  -e IMAGE_PATH="/app/stanford-cars/test/Acura TSX Sedan 2012/000366.jpg" \
  -e INTERVAL=15 \
  openvino-client
```

### Option 3: Using Docker Compose

```bash
docker-compose up -d
```

## Environment Variables

- `MODEL_SERVER_URL`: URL of the OpenVINO model server REST API (default: http://localhost:8000)
- `MODEL_NAME`: Name of the model (default: efb3_car_onnx)
- `IMAGE_PATH`: Path to the image file inside the container (default: /app/stanford-cars/test/Acura TSX Sedan 2012/000366.jpg)
- `INTERVAL`: Interval between requests in seconds (default: 15)

## Monitoring

To view the client logs:
```bash
docker logs -f model_client
```

To stop the client:
```bash
docker stop model_client
```

## Running Locally (without Docker)

If you prefer to run the client locally:

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set environment variables (optional):
```bash
export MODEL_SERVER_URL=http://localhost:8000
export MODEL_NAME=efb3_car_onnx
export IMAGE_PATH="/home/zedtel/Developer/client-container/stanford-cars/test/Acura TSX Sedan 2012/000366.jpg"
export INTERVAL=15
```

3. Run the client:
```bash
python client.py
```

## What the Client Does

1. **Health Check**: Verifies that the model server is running and the model is ready
2. **Image Preprocessing**: Loads and preprocesses the image for EfficientNet-B3:
   - Resizes to 224x224 pixels
   - Normalizes pixel values
   - Applies ImageNet normalization
   - Converts to NCHW format (batch, channels, height, width)
3. **Inference**: Sends the preprocessed image to the model server via REST API
4. **Results Processing**: Displays the top 5 predictions with confidence scores
5. **Continuous Operation**: Repeats every 15 seconds (configurable)

## Expected Output

The client will display logs like this:

```
2025-07-30 10:30:00 - INFO - Starting OpenVINO Model Server Client
2025-07-30 10:30:00 - INFO - Server URL: http://localhost:8000
2025-07-30 10:30:00 - INFO - Model Name: efb3_car_onnx
2025-07-30 10:30:00 - INFO - Image Path: /app/stanford-cars/test/Acura TSX Sedan 2012/000366.jpg
2025-07-30 10:30:00 - INFO - Interval: 15 seconds
2025-07-30 10:30:00 - INFO - Model server is healthy and model is ready
2025-07-30 10:30:00 - INFO - Image preprocessed successfully. Shape: (1, 3, 224, 224)
2025-07-30 10:30:00 - INFO - --- Request #1 ---
2025-07-30 10:30:01 - INFO - Prediction results:
2025-07-30 10:30:01 - INFO - Timestamp: 2025-07-30 10:30:01
2025-07-30 10:30:01 - INFO -   Rank 1: Class 123 - Confidence: 0.8542 (85.42%)
2025-07-30 10:30:01 - INFO -   Rank 2: Class 45 - Confidence: 0.1123 (11.23%)
2025-07-30 10:30:01 - INFO -   Rank 3: Class 67 - Confidence: 0.0234 (2.34%)
...
```

## Accessing the Services

### Client Web UI Access

When deployed via Helm chart, you can access the client web UI in several ways:

#### Option 1: NodePort (Direct Access)
```bash
# Access directly via NodePort
http://localhost:30080
```

#### Option 2: Port Forward (Recommended)
```bash
# Port forward to the client service
kubectl port-forward svc/ml-platform-edge-ai-car-classification-client 8080:8080

# Then access via browser
http://localhost:8080
```

#### Option 3: Get Pod and Port Forward
```bash
# Get the client pod name
export POD_NAME=$(kubectl get pods -l "app.kubernetes.io/name=edge-ai-car-classification,app.kubernetes.io/component=client" -o jsonpath="{.items[0].metadata.name}")

# Port forward to the pod
kubectl port-forward $POD_NAME 8080:8080

# Access via browser
http://localhost:8080
```

### MinIO Bucket Access

Your MinIO object storage can be accessed for uploading and managing ML models:

#### Web Console Access
```bash
# Access MinIO web console directly (Docker containers)
http://localhost:9001

# Login credentials:
# Username: minioadmin
# Password: minioadmin123
```

#### MinIO Client (mc) via Docker
```bash
# Configure MinIO client to connect to your server
docker exec -it minio-client mc alias set local http://minio-server:9000 minioadmin minioadmin123

# List buckets
docker exec -it minio-client mc ls local

# Create the ml-models bucket (if it doesn't exist)
docker exec -it minio-client mc mb local/ml-models

# Upload a model (example)
docker exec -it minio-client mc cp /path/to/your/model.onnx local/ml-models/your-model/1/

# List bucket contents
docker exec -it minio-client mc ls local/ml-models --recursive
```

#### MinIO Client (mc) - Local Installation
```bash
# Install MinIO client locally (optional)
wget https://dl.min.io/client/mc/release/linux-amd64/mc
chmod +x mc
sudo mv mc /usr/local/bin/

# Configure alias
mc alias set local http://localhost:9000 minioadmin minioadmin123

# Use mc commands directly
mc ls local
mc mb local/ml-models
mc cp your-model.onnx local/ml-models/your-model/1/
```

### Model Upload Structure

When uploading models to MinIO, follow this directory structure:
```
ml-models/
├── model-name-1/
│   └── 1/
│       └── model.onnx (or model.xml + model.bin for OpenVINO)
├── model-name-2/
│   └── 1/
│       └── model.onnx
└── model-name-3/
    ├── 1/
    │   └── model.onnx
    └── 2/
        └── model.onnx  # Version 2 of the model
```

The sync sidecar will automatically detect new models and reload the OVMS configuration.

## Troubleshooting

- If the client can't connect to the server, make sure the model server is running and accessible
- Check that the ports (8000, 9000) are not blocked by firewall
- Verify that the image file exists and is readable
- Check Docker logs for detailed error messages
- For Kubernetes deployment issues, check pod logs: `kubectl logs -l app.kubernetes.io/name=edge-ai-car-classification`
- For MinIO connection issues, verify the endpoint configuration in your Helm values
