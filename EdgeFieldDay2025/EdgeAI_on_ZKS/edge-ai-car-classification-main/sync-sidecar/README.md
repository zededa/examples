# Model Sync Sidecar

A Python-based sidecar container that automatically synchronizes machine learning models from MinIO object storage to a shared volume, enabling dynamic model loading for OpenVINO Model Server (OVMS).

## Features

- **Continuous Monitoring**: Watches MinIO bucket for new models
- **Automatic Download**: Downloads new/updated models to shared storage
- **OVMS Integration**: Updates OVMS config.json automatically
- **Health Checks**: Built-in health monitoring and status reporting
- **Error Handling**: Robust error handling and retry mechanisms
- **Configurable**: YAML-based configuration for all parameters

## How It Works

1. **Monitor**: Continuously polls MinIO bucket for model changes
2. **Detect**: Identifies new or updated models using ETag comparison
3. **Download**: Downloads model files to shared volume
4. **Configure**: Updates OVMS config.json with new model definitions
5. **Notify**: OVMS automatically detects config changes and loads models

## Configuration

The sidecar is configured via a YAML file mounted at `/config/sync-config.yaml`:

```yaml
minio:
  endpoint: "http://minio:9000"
  access_key: "minioadmin"
  secret_key: "minioadmin123"
  bucket_name: "ml-models"
  secure: false

sync:
  interval: 30                    # Check interval in seconds
  models_path: "/models"          # Local path for models
  log_level: "INFO"               # Log level

ovms:
  model_name: "efb3_car_onnx"    # Default model name
  config_path: "/models/config.json"  # OVMS config file path
```

## Model Directory Structure

Models in MinIO should follow this structure:

```
ml-models/                    # Bucket name
├── model-name-1/            # Model name
│   └── 1/                   # Version number
│       ├── model.onnx       # Model files
│       └── metadata.json    # Optional metadata
├── model-name-2/
│   ├── 1/
│   └── 2/                   # Multiple versions supported
└── another-model/
    └── 1/
```

The sidecar will:
1. Download models to `/models/model-name/version/`
2. Update `/models/config.json` with model configurations
3. OVMS will automatically detect and load the models

## Building the Image

### Using the build script (recommended):

```bash
# Build only
./build.sh

# Build and push
REGISTRY=your-registry.com PUSH=true ./build.sh

# Custom image name and tag
REGISTRY=your-registry.com IMAGE_NAME=my-sync-sidecar TAG=v1.0.0 ./build.sh
```

### Manual build:

```bash
docker build -t your-registry/model-sync-sidecar:latest .
docker push your-registry/model-sync-sidecar:latest
```

## Running Standalone (for testing)

```bash
# Create test config
mkdir -p /tmp/sync-config /tmp/models
cat > /tmp/sync-config/sync-config.yaml << EOF
minio:
  endpoint: "http://localhost:9000"
  access_key: "minioadmin"
  secret_key: "minioadmin123"
  bucket_name: "ml-models"
  secure: false
sync:
  interval: 10
  models_path: "/tmp/models"
  log_level: "DEBUG"
ovms:
  model_name: "test-model"
  config_path: "/tmp/models/config.json"
EOF

# Run the container
docker run --rm -it \
  -v /tmp/sync-config:/config:ro \
  -v /tmp/models:/tmp/models \
  -e CONFIG_PATH=/config/sync-config.yaml \
  --network host \
  your-registry/model-sync-sidecar:latest
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `CONFIG_PATH` | Path to YAML configuration file | `/config/sync-config.yaml` |

## Health Checks

The sidecar includes built-in health checks that verify:
- MinIO connectivity
- Models directory accessibility
- OVMS config file accessibility

Health check endpoint is available via the container's health check mechanism.

## Logging

The sidecar provides detailed logging with configurable levels:
- `DEBUG`: Detailed sync operations
- `INFO`: Model sync events and status
- `WARNING`: Non-critical issues
- `ERROR`: Critical errors

Example log output:
```
2024-01-15 10:30:00 - model_sync - INFO - Connected to MinIO at http://minio:9000
2024-01-15 10:30:01 - model_sync - INFO - Discovered 2 models in MinIO
2024-01-15 10:30:05 - model_sync - INFO - Updated model.onnx for model my-model:1
2024-01-15 10:30:06 - model_sync - INFO - Successfully synced model my-model:1
2024-01-15 10:30:07 - model_sync - INFO - Saved OVMS config with 2 models
```

## Integration with Helm Chart

When deployed via the Helm chart, the sidecar:
1. Receives configuration via ConfigMap
2. Shares volume with OVMS container
3. Automatically starts monitoring after MinIO becomes ready
4. Updates shared OVMS configuration file

## Development

### Requirements

- Python 3.11+
- Dependencies listed in `requirements.txt`

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run with local config
CONFIG_PATH=./test-config.yaml python model_sync.py
```

### Testing

The sidecar includes comprehensive error handling and logging for debugging:

1. **MinIO Connection Issues**: Detailed connection error messages
2. **Model Download Failures**: Retry logic with exponential backoff
3. **Configuration Errors**: Validation and clear error messages
4. **File System Issues**: Permission and space checks

## Security Considerations

- Runs as non-root user (syncuser)
- Minimal file system permissions
- Secure credential handling via configuration
- Network traffic only to MinIO endpoints

## Performance

- **Memory Usage**: ~50-100MB typical usage
- **CPU Usage**: Low, periodic polling only
- **Network**: Efficient delta downloads using ETags
- **Storage**: Minimal overhead, only stores active models

## Troubleshooting

### Common Issues

1. **Cannot connect to MinIO**
   ```
   ERROR - Failed to connect to MinIO: [Errno -2] Name or service not known
   ```
   - Check MinIO service name and port
   - Verify network connectivity
   - Check DNS resolution

2. **Permission denied on models directory**
   ```
   ERROR - Permission denied: '/models'
   ```
   - Check volume mount permissions
   - Ensure container runs with correct user ID
   - Verify shared volume configuration

3. **Models not downloading**
   ```
   WARNING - No models found in MinIO bucket
   ```
   - Check bucket name in configuration
   - Verify models exist in MinIO
   - Check MinIO credentials

### Debug Mode

Enable debug logging for detailed troubleshooting:

```yaml
sync:
  log_level: "DEBUG"
```

This will show:
- Detailed MinIO operations
- File system operations
- Configuration updates
- Network requests and responses

## Architecture Integration

The sync sidecar is designed to work seamlessly with:
- **OpenVINO Model Server**: Automatic config management
- **MinIO**: S3-compatible object storage
- **Kubernetes**: Health checks and lifecycle management
- **Helm Charts**: Configuration management and deployment

This enables a fully automated ML inference pipeline where models can be updated simply by uploading to MinIO storage.
