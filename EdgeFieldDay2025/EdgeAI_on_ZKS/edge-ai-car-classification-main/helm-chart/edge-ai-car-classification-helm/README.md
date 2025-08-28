# Edge AI Car Classification Helm Chart

A comprehensive Helm chart for deploying the Edge AI Car Classification, a dynamic machine learning inference platform on Kubernetes. This platform features automatic model loading from external MinIO storage, OpenVINO Model Server for inference, and a client web application for testing.

## Architecture

This Helm chart deploys the following components:

### 1. External MinIO (Object Storage)
- **Purpose**: Stores ML models and serves as the central model repository
- **Deployment**: External Docker containers (not managed by this chart)
- **Features**: 
  - S3-compatible API for programmatic access
  - Web console for easy model management at `http://localhost:9001`
  - Automatic model synchronization to Kubernetes pods

### 2. Server Pod (Dual Container)
- **OpenVINO Model Server (OVMS)**: Primary inference container
  - Serves ML models via gRPC and REST APIs
  - Automatically detects configuration changes
  - Supports multiple model formats (ONNX, OpenVINO IR, etc.)
  
- **Model Sync Sidecar**: Automated model management
  - Continuously monitors external MinIO bucket for new models
  - Downloads new models to shared storage
  - Updates OVMS configuration automatically
  - Provides health checks and logging

### 3. Client Pod
- **Purpose**: Web application for testing inference with image uploads
- **Features**:
  - Web UI accessible via NodePort (port 30080) or port-forward
  - Upload images for car classification
  - Real-time inference results display
  - Connects to OVMS REST API

### 4. Shared Storage
- **PersistentVolumeClaim**: Shared between OVMS and sync sidecar
- **Purpose**: Stores downloaded models and configuration files
- **Mount Point**: `/models` in both containers

## Prerequisites

- Kubernetes cluster (v1.21+)
- Helm 3.8+
- kubectl configured to access your cluster
- **External MinIO running** (Docker containers already set up)
- Persistent volume support in your cluster

## Quick Start

### Current Setup (Working Configuration)

Your deployment is already configured with the correct external MinIO connection:

```bash
# The chart is currently deployed with external MinIO
helm list

# Check pod status
kubectl get pods -l app.kubernetes.io/name=edge-ai-car-classification

# Access client web UI (Option 1: NodePort)
http://localhost:30080

# Access client web UI (Option 2: Port Forward)
You can now access the application in your local browser:

```bash
kubectl port-forward svc/edgeai-platform-edge-ai-car-classification-client 8080:8080
# Then visit: http://localhost:8080
```

### Access MinIO for Model Management

Your external MinIO is accessible at:
- **Web Console**: `http://localhost:9001`
- **API Endpoint**: `http://localhost:9000`
- **Credentials**: `minioadmin` / `minioadmin123`

### Deploying from Scratch

If you need to redeploy or install on a new cluster:

```bash
# From the project root directory (/home/zedtel/Developer/edge-ai-car-classification)
# Install the chart with your external MinIO configuration
helm install edgeai-platform ./helm-chart/edge-ai-car-classification/ -f ./helm-chart/edge-ai-car-classification/values.yaml

# Or upgrade existing deployment
helm upgrade edgeai-platform ./helm-chart/edge-ai-car-classification/ -f ./helm-chart/edge-ai-car-classification/values.yaml
```

**Alternative (from inside the chart directory):**
```bash
# Change to the chart directory first
cd helm-chart/edge-ai-car-classification/

# Then run the simpler commands
helm install edgeai-platform . -f values.yaml
# or
helm upgrade edgeai-platform . -f values.yaml
```

## Configuration

The main configuration is in `values.yaml`. Key sections:

### External MinIO Configuration
```yaml
minio:
  enabled: false  # We use external MinIO
  external:
    endpoint: "172.16.8.136:9000"  # Your host IP
    accessKey: "minioadmin"
    secretKey: "minioadmin123"
    secure: false
    bucketName: "ml-models"
```

### Server Configuration
```yaml
server:
  enabled: true
  ovms:
    image: "edgeai-ovms-server:latest"
  sidecar:
    image: "edgeai-model-sync-sidecar:latest"
```

### Client Configuration
```yaml
client:
  enabled: true
  image: "edgeai-client-app:latest"
  service:
    type: NodePort
    nodePort: 30080
```

## Key Configuration Parameters

| Parameter | Description | Current Value |
|-----------|-------------|---------------|
| `minio.enabled` | Deploy MinIO as part of chart | `false` (external) |
| `minio.external.endpoint` | External MinIO endpoint | `172.16.8.136:9000` |
| `minio.external.accessKey` | MinIO access key | `minioadmin` |
| `minio.external.secretKey` | MinIO secret key | `minioadmin123` |
| `minio.external.bucketName` | Bucket for models | `ml-models` |
| `server.enabled` | Enable server deployment | `true` |
| `server.ovms.image` | OVMS container image | `edgeai-ovms-server:latest` |
| `server.sidecar.image` | Sync sidecar image | `edgeai-model-sync-sidecar:latest` |
| `client.enabled` | Enable client deployment | `true` |
| `client.image` | Client container image | `edgeai-client-app:latest` |
| `client.service.type` | Client service type | `NodePort` |
| `client.service.nodePort` | NodePort for client access | `30080` |

## Usage and Verification

### Step 1: Verify Deployment Status

```bash
# Check all pods are running
kubectl get pods -l app.kubernetes.io/name=edge-ai-car-classification

# Expected output:
# NAME                                                        READY   STATUS    RESTARTS   AGE
# edgeai-platform-edge-ai-car-classification-client-7b6f4b4f68-5qnpv   1/1     Running   0          5m
# edgeai-platform-edge-ai-car-classification-server-64c8d7df5b-fnz7z   2/2     Running   0          5m

# Check services
kubectl get svc -l app.kubernetes.io/name=edge-ai-car-classification

# Check PVCs
kubectl get pvc
```

### Step 2: Access the Client Web UI

#### Option 1: Direct NodePort Access

**For Minikube:**
```bash
# Get the correct URL for minikube
minikube service edgeai-platform-edge-ai-car-classification-client --url
# Output example: http://192.168.58.2:30080

# Open in browser or use the URL directly
minikube service edgeai-platform-edge-ai-car-classification-client
```

**For other Kubernetes clusters:**
```bash
# Access directly via NodePort (if cluster supports it)
http://localhost:30080
```

#### Option 2: Port Forward (Works for all cluster types)
```bash
# Port forward to the client service
kubectl port-forward svc/edgeai-platform-edge-ai-car-classification-client 8080:8080

# Then access via browser
http://localhost:8080
```

#### Option 3: SSH Tunnel (for remote access)
If you're accessing a remote Kubernetes cluster via SSH:

```bash
# Step 1: On the remote system (SSH session), start port forwarding that binds to all interfaces
kubectl port-forward --address 0.0.0.0 svc/edgeai-platform-edge-ai-car-classification-client 8080:8080

# Step 2: From your LOCAL machine, create SSH tunnel
ssh -L 8080:localhost:8080 username@remote-system-ip

# Step 3: Access from your local browser
http://localhost:8080
```

### Step 3: Access MinIO for Model Management

#### Web Console Access
```bash
# MinIO web console (external Docker container)
http://localhost:9001

# Login credentials:
# Username: minioadmin
# Password: minioadmin123
```

#### Command Line Access
```bash
# Configure MinIO client to connect to your server
docker exec -it minio-client mc alias set local http://minio-server:9000 minioadmin minioadmin123

# List buckets
docker exec -it minio-client mc ls local

# List models in the ml-models bucket
docker exec -it minio-client mc ls local/ml-models --recursive
```

### Step 4: Upload a New Model

1. **Access MinIO Console**: Navigate to http://localhost:9001
2. **Navigate to Bucket**: Go to the `ml-models` bucket  
3. **Upload Model**: Create directory structure and upload your model:
   ```
   ml-models/
   └── your-new-model/
       └── 1/
           └── model.onnx  (or other supported format)
   ```

### Step 5: Monitor Automatic Model Loading

```bash
# Watch sync sidecar logs to see model detection and download
kubectl logs -l app.kubernetes.io/component=server -c model-sync -f

# Sample output:
# 2025-08-25 20:25:03,986 - __main__ - INFO - Discovered 6 models in MinIO
# 2025-08-25 20:25:03,986 - __main__ - INFO - Successfully synced 6/6 models

# Watch OVMS logs to see model loading
kubectl logs -l app.kubernetes.io/component=server -c ovms -f

# Check current OVMS configuration
kubectl exec deployment/edgeai-platform-edge-ai-car-classification-server -c ovms -- cat /models/config.json
```

### Step 6: Test Inference

#### Via Web UI
1. Access the client web UI at http://localhost:30080 or http://localhost:8080 (if port-forwarded)
2. Upload an image file
3. Select a model from the dropdown
4. Click "Classify" to get predictions

#### Via API (Direct)
```bash
# Port forward to OVMS service
kubectl port-forward svc/edgeai-platform-edge-ai-car-classification-ovms 8000:8000

# Test health endpoint
curl http://localhost:8000/v1/config

# Test model status
curl http://localhost:8000/v1/models/efb3_car_onnx/ready
```

## Management Operations
```bash
CMD ["--config_path", "/models/config.json", \
     "--port", "9000", \
     "--rest_port", "8000", \
     "--log_level", "DEBUG", \
     "--file_system_poll_wait_seconds", "5"]
```

#### 2. Update entrypoint.sh

Modify your `entrypoint.sh` to handle the new configuration approach:

```bash
#!/bin/bash
set -e

echo "🚀 Starting Car Classifier Server with config-based loading..."

# Wait for config file to be created by sidecar
while [ ! -f "/models/config.json" ]; do
    echo "⏳ Waiting for config.json to be created by sidecar..."
    sleep 2
done

echo "✅ Config file found, starting OVMS..."

# Start OVMS with the provided arguments
exec /ovms/bin/ovms "$@"
```

### Client Container Modifications

Your client application needs to read the server URL from environment variables instead of hardcoded values.

#### 1. Update client.py

Modify your `client.py` to use environment variables:

```python
# OLD:
# def __init__(self, server_url="http://localhost:8000", model_name="efb3_car_onnx"):

# NEW:
def __init__(self, 
             server_url=None, 
             model_name=None):
    # Read from environment variables with fallbacks
    self.server_url = server_url or os.environ.get('MODEL_SERVER_URL', 'http://localhost:8000')
    self.model_name = model_name or os.environ.get('MODEL_NAME', 'efb3_car_onnx')
    self.inference_url = f"{self.server_url}/v1/models/{self.model_name}:predict"
    
    # Also read other config from environment
    self.image_path = os.environ.get('IMAGE_PATH', '/app/stanford-cars/test/Acura TSX Sedan 2012/000366.jpg')
    self.interval = int(os.environ.get('INTERVAL', '15'))
```

#### 2. Update main execution loop

```python
def main():
    """Main execution function."""
    # Initialize client with environment variables
    client = ModelServerClient()
    
    while True:
        try:
            # Your existing inference logic here
            result = client.predict_image(client.image_path)
            # ... rest of your code
            
            time.sleep(client.interval)
        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.error(f"Error in main loop: {str(e)}")
            time.sleep(client.interval)
```

## Management Operations

### Upgrading the Deployment

```bash
# Upgrade with new values
helm upgrade ml-platform ./edge-ai-car-classification -f my-values.yaml

# Upgrade specific image versions
helm upgrade ml-platform ./edge-ai-car-classification \
  --set server.ovms.image=your-registry/my-ovms-server:v2.0.0 \
  --reuse-values
```

### Scaling Components

```bash
# Scale server replicas
helm upgrade ml-platform ./edge-ai-car-classification \
  --set server.replicaCount=3 \
  --reuse-values

# Scale client replicas
helm upgrade ml-platform ./edge-ai-car-classification \
  --set client.replicaCount=2 \
  --reuse-values
```

### Monitoring and Debugging

```bash
# View all resources
kubectl get all -l app.kubernetes.io/instance=ml-platform

# Describe problematic pods
kubectl describe pod <pod-name>

# Check events
kubectl get events --sort-by=.metadata.creationTimestamp

# Access pod shell for debugging
kubectl exec -it deployment/ml-platform-server -c ovms -- /bin/bash
kubectl exec -it deployment/ml-platform-server -c model-sync -- /bin/bash
```
### Accessing the Client using UI
Using Direct NodePort Access
```bash
http://localhost:30080
```

Using Port forwarding
```bash
kubectl port-forward svc/edgeai-platform-edge-ai-car-classification-client 8080:8080
# Then access: http://localhost:8080
```

### Accessing the MinIO Bucket

Using Web-console:
```bash
http://localhost:9001
# Login: minioadmin / minioadmin123
```
Using MinIO client on Docker:
```bash
# Configure client
docker exec -it minio-client mc alias set local http://minio-server:9000 minioadmin minioadmin123

# List buckets
docker exec -it minio-client mc ls local

# Upload models
docker exec -it minio-client mc cp /path/to/model.onnx local/ml-models/your-model/1/
```



### Backup and Restore

```bash
# Backup models from MinIO
kubectl exec -it deployment/ml-platform-minio -- mc cp --recursive /data/ml-models /backup/

# Backup Helm configuration
helm get values ml-platform > ml-platform-backup-values.yaml
```

### Uninstalling

```bash
# Uninstall the release (keeps PVCs by default)
helm uninstall ml-platform

# To also delete PVCs
kubectl delete pvc -l app.kubernetes.io/instance=ml-platform
```

## Troubleshooting

### Common Issues

1. **Pods stuck in Init state**
   - Check init container logs: `kubectl logs <pod-name> -c init-minio-wait` or `kubectl logs <pod-name> -c init-ovms-wait`
   - Verify external MinIO connectivity from cluster
   - Check MinIO endpoint configuration in values.yaml

2. **Models not loading**
   - Check MinIO connectivity: `kubectl logs -l app.kubernetes.io/component=server -c model-sync`
   - Verify bucket permissions and model file structure  
  - Check OVMS config.json: `kubectl exec deployment/edgeai-platform-edge-ai-car-classification-server -c ovms -- cat /models/config.json`

3. **Client cannot connect to server**
  - Verify service connectivity: `kubectl get svc edgeai-platform-edge-ai-car-classification-ovms`
  - Test OVMS health: `kubectl port-forward svc/edgeai-platform-edge-ai-car-classification-ovms 8000:8000` then `curl http://localhost:8000/v1/config`

4. **Persistent volume issues**
   - Check storage class availability: `kubectl get storageclass`
   - Verify PVC status: `kubectl get pvc`
   - Check node disk space and permissions

### Useful Commands

```bash
# View all logs
kubectl logs -l app.kubernetes.io/name=edge-ai-car-classification --all-containers=true

# Follow specific component logs
kubectl logs -l app.kubernetes.io/component=server -c ovms -f
kubectl logs -l app.kubernetes.io/component=server -c model-sync -f
kubectl logs -l app.kubernetes.io/component=client -f

# Check MinIO connection from cluster
kubectl run debug --image=busybox --rm -it --restart=Never -- nc -v 172.16.8.136 9000
```

## Files Structure

The chart contains:
- `values.yaml` - Main configuration file (currently in use)
- `examples/values-dev.yaml` - Example development configuration
- `templates/` - Kubernetes resource templates
- `Chart.yaml` - Chart metadata

## Contributing

To contribute to this chart:
1. Test changes thoroughly with your MinIO setup
2. Update this README with any configuration changes
3. Follow Kubernetes and Helm best practices

## License

This project is licensed under the MIT License.
