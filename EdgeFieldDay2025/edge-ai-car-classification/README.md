# Edge AI ML Inference Platform

A complete, production-ready machine learning inference platform with dynamic model management on Kubernetes.

> **🆕 NEW in v5.6**: Model Cleanup Management - Remove orphaned models directly from the web interface! [Learn more](MODEL_CLEANUP_FEATURE.md)

## 🚀 Quick Start

### Deploy with Docker Hub (Recommended)
```bash
# Build any local changes
./build.sh

# Deploy using pre-built Docker Hub images
cd helm-chart/edge-ai-car-classification
helm dependency update
helm install edge-ai-car-classification . --namespace edge-ai-car-classification --create-namespace \
  --set server.ovms.image=adithyazededa/edgeai-ovms-server:latest \
  --set server.sidecar.image=adithyazededa/model-sync-sidecar:latest \
  --set client.image=adithyazededa/edgeai-client-app:latest
```

### Deploy with Local Build
```bash
# Build all images locally
./build.sh

# Deploy with locally built images
cd helm-chart/edge-ai-car-classification
helm dependency update
helm install ml-platform . --namespace ml-platform --create-namespace
```

### Cleanup
```bash
./cleanup.sh
```

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Kubernetes Cluster                       │
│                                                             │
│  ┌─────────────────┐    ┌─────────────────┐                │
│  │   MinIO Pod     │    │   Server Pod    │                │
│  │                 │    │                 │                │
│  │  ┌───────────┐  │    │ ┌─────────────┐ │                │
│  │  │   MinIO   │  │    │ │    OVMS     │ │                │
│  │  │  Storage  │  │◄───┤ │  Container  │ │                │
│  │  │           │  │    │ └─────────────┘ │                │
│  │  └───────────┘  │    │                 │                │
│  │                 │    │ ┌─────────────┐ │                │
│  └─────────────────┘    │ │    Sync     │ │                │
│                         │ │   Sidecar   │ │                │
│                         │ └─────────────┘ │                │
│  ┌─────────────────┐    └─────────────────┘                │
│  │   Client Pod    │                                       │
│  │                 │    ┌─────────────────┐                │
│  │ ┌─────────────┐ │    │ Shared Storage  │                │
│  │ │   Client    │ │    │     (PVC)       │                │
│  │ │Application  │ │◄───┤   /models/      │                │
│  │ └─────────────┘ │    │                 │                │
│  └─────────────────┘    └─────────────────┘                │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## ✨ Features

- **Dynamic Model Loading** - Upload models to MinIO, they're automatically loaded
- **Multi-Model Support** - Client discovers and uses all available models
- **Model Cleanup Management** - Remove orphaned models with web interface (NEW in v5.6)
- **Zero Downtime** - Add new models without restarting services
- **Production Ready** - Complete Kubernetes deployment with monitoring
- **Auto-scaling** - Horizontal pod autoscaling based on demand
- **Security** - RBAC integration and configurable secrets management

## 📁 Project Structure

```
edge-ai-car-classification/
├── client-container/          # Multi-model client application
├── server-container/          # OpenVINO Model Server setup
├── volume-for-model/          # Model files
├── sync-sidecar/              # Dynamic model loading component
├── helm-chart/                # Kubernetes deployment charts
├── jupyter-notebook/          # ML model tracking notebook
├── build.sh                   # Build automation script
├── cleanup.sh                 # Cleanup automation script
└── README.md                  # This file
```

## 🔧 Prerequisites

- Kubernetes cluster (minikube, kind, or any K8s cluster)
- kubectl configured to access your cluster
- Helm 3.x installed
- Docker for building images

## 📦 Deployment Options

### Option 1: Docker Hub Deployment (Easiest)
Uses pre-built images from Docker Hub - no building required:

```bash
cd helm-chart/edge-ai-car-classification
helm dependency update
helm install ml-platform . --namespace ml-platform --create-namespace \
  --set server.ovms.image=adithyazededa/edgeai-ovms-server:latest \
  --set server.sidecar.image=adithyazededa/model-sync-sidecar:latest \
  --set client.image=adithyazededa/edgeai-client-app:latest
```

### Option 2: Local Development Deployment
Builds images locally and deploys:

```bash
# Build all container images
./build.sh

# Deploy the platform with local images
cd helm-chart/edge-ai-car-classification
helm dependency update
helm install ml-platform . --namespace ml-platform --create-namespace
```

### Option 3: Custom Registry Deployment
Build and push to your own registry:

```bash
# Build and push to custom registry
REGISTRY=your-username PUSH=true ./build.sh

# Deploy with custom images
cd helm-chart/edge-ai-car-classification
helm dependency update
helm install ml-platform . --namespace ml-platform --create-namespace \
  --set server.ovms.image=your-username/edgeai-ovms-server:latest \
  --set server.sidecar.image=your-username/model-sync-sidecar:latest \
  --set client.image=your-username/edgeai-client-app:latest
```

## 🔄 How Dynamic Model Loading Works

1. **Upload Model**: Add new model to MinIO bucket with correct structure:
   ```
   ml-models/
   └── model_name/          # e.g., efb3_car_onnx
       └── version/         # e.g., 1, 2, latest  
           └── model_files  # e.g., model.onnx
   ```

2. **Sync Detection**: Sidecar detects new model via polling (every 30s)
3. **Download**: Sidecar downloads model to shared volume at `/models/`
4. **Config Update**: Sidecar updates `/models/config.json` with new model
5. **OVMS Reload**: OVMS automatically detects config change and loads model
6. **Ready**: New model is available for inference at `/v1/models/model_name:predict`

## 📊 Monitoring and Management

### Check Deployment Status
```bash
# Check all pods status
kubectl get pods -n ml-platform

# Check services and endpoints
kubectl get svc,endpoints -n ml-platform
```

### View Logs
```bash
# View OVMS Server logs
kubectl logs -l app.kubernetes.io/component=server -c ovms -n ml-platform -f

# View Model Sync Sidecar logs  
kubectl logs -l app.kubernetes.io/component=server -c model-sync -n ml-platform -f

# View Client Application logs
kubectl logs -l app.kubernetes.io/component=client -n ml-platform -f
```

### Upload Models
```bash
# Access MinIO console
kubectl port-forward svc/ml-platform-minio-console 9001:9001 -n ml-platform

# Upload models via web UI at http://localhost:9001
# Login with: minioadmin / minioadmin123
```

### Scale Components
```bash
# Scale server pods
helm upgrade ml-platform ./helm-chart/edge-ai-car-classification --set server.replicaCount=3 --reuse-values -n ml-platform

# Scale client pods  
helm upgrade ml-platform ./helm-chart/edge-ai-car-classification --set client.replicaCount=2 --reuse-values -n ml-platform
```

## 🔒 Security Considerations

1. **Change default MinIO credentials** in production
2. **Use image pull secrets** for private registries
3. **Enable RBAC** with minimal permissions
4. **Use security contexts** to run as non-root
5. **Enable network policies** to restrict traffic

## 🆘 Troubleshooting

### Common Issues

#### Models not loading
- Check sync sidecar logs: `kubectl logs -l app.kubernetes.io/component=server -c model-sync -n ml-platform`
- Verify MinIO connectivity and bucket exists
- Ensure correct model directory structure

#### Client connection errors
- Verify service names and ports
- Check OVMS server is running: `kubectl get pods -l app.kubernetes.io/component=server -n ml-platform`

#### Image pull failures
- Check registry credentials and image names
- For minikube: ensure images are loaded with `minikube image load`

#### Storage issues
- Verify PVC creation: `kubectl get pvc -n ml-platform`
- Check storage class availability: `kubectl get storageclass`

### Debug Commands
```bash
# Check all platform resources
kubectl get all -l app.kubernetes.io/instance=ml-platform -n ml-platform

# Get detailed resource information
kubectl describe all -l app.kubernetes.io/instance=ml-platform -n ml-platform

# Check pod events
kubectl get events --sort-by=.metadata.creationTimestamp -n ml-platform

# Access container shell
kubectl exec -it deployment/ml-platform-edge-ai-car-classification-server -c ovms -n ml-platform -- /bin/bash
```

### Missing MinIO Bucket
```bash
# Create bucket manually using MinIO client
kubectl port-forward svc/ml-platform-minio 9000:9000 -n ml-platform
mc alias set local http://localhost:9000 minioadmin minioadmin123
mc mb local/ml-models
```

## 🎯 Production Deployment

For production environments:

1. Use the production values template:
   ```bash
   cp helm-chart/edge-ai-car-classification/examples/values-prod.yaml production-values.yaml
   # Edit with production settings
   helm install ml-platform ./helm-chart/edge-ai-car-classification -f production-values.yaml
   ```

2. Key production settings:
   - Multiple replicas for high availability
   - Resource limits and requests
   - Ingress configuration
   - Auto-scaling enabled
   - Monitoring integration
   - Security hardening

## 📚 Additional Documentation

- **[Client Container](client-container/README.md)** - Multi-model client application details
- **[Server Container](server-container/README.md)** - OpenVINO Model Server setup
- **[Sync Sidecar](sync-sidecar/README.md)** - Dynamic model loading component
- **[Helm Chart](helm-chart/edge-ai-car-classification/README.md)** - Kubernetes deployment details


## 📞 Support

For issues or questions:
1. Check the troubleshooting section above
2. Review logs using the provided kubectl commands
3. Consult the detailed documentation in component README files

---

**Ready to deploy?** Start with `./build.sh` followed by the Helm deployment commands above for the quickest setup!

## 📦 Docker Images (Docker Hub)

- `adithyazededa/edgeai-ovms-server:latest`
- `adithyazededa/edgeai-model-sync-sidecar:latest` 
- `adithyazededa/edgeai-client-app:latest`

Deploy anywhere with: `./deploy.sh --dockerhub`
