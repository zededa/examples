# Model Cleanup Feature

## Overview

The EdgeAI Car Classification platform now includes a comprehensive model cleanup feature that addresses the issue where models deleted from MinIO storage may still appear as available in the system.

## Problem Statement

When models are deleted from MinIO storage, the OpenVINO Model Server (OVMS) and the web application may continue to show these models as available. This happens because:

1. **Local Model Cache**: Models are downloaded and cached locally in the OVMS container
2. **Configuration Persistence**: The OVMS configuration file still references the deleted models
3. **Stale State**: The system doesn't automatically detect when models are removed from MinIO

## Solution

The new cleanup functionality provides:

1. **Automatic Detection**: Identifies models that exist locally but are no longer in MinIO
2. **Safe Removal**: Removes orphaned models from both local storage and OVMS configuration
3. **User Control**: Web interface option with proper warnings about irreversibility
4. **Real-time Updates**: Immediate refresh of available models after cleanup

## Features

### 1. Web Interface Management

- **New "Models" Tab**: Added to the web interface for model management
- **Status Monitoring**: Shows current model count, sync status, and last updated time
- **Model List**: Displays all currently available models with their status
- **Cleanup Controls**: Safe cleanup with confirmation dialog and warnings

### 2. Cleanup Functionality

- **Orphaned Model Detection**: Compares MinIO storage with local model cache
- **Selective Removal**: Only removes models that are no longer in MinIO
- **Configuration Updates**: Updates OVMS config.json automatically
- **File System Cleanup**: Removes local model files to free up disk space

### 3. Safety Features

- **Confirmation Required**: User must explicitly confirm cleanup action
- **Warning Messages**: Clear warnings about irreversibility
- **Logging**: Comprehensive logging of all cleanup operations
- **Error Handling**: Graceful error handling with user feedback

## Usage

### Accessing the Model Management

1. Open the web interface
2. Navigate to the **"Models"** tab in the right panel
3. Click **"Refresh Status"** to get current model information

### Performing Model Cleanup

1. In the Models tab, review the current status and model list
2. Click **"Clean Up Orphaned Models"** button
3. Read the warning dialog carefully
4. Confirm by clicking **"Yes, Clean Up Models"**
5. Monitor the results in the cleanup results section

### Understanding the Warning

⚠️ **Important**: Model cleanup is **irreversible**. Once a model is removed:
- It will no longer be available for inference
- The only way to restore it is to re-upload it to MinIO storage
- All local cached data for that model will be deleted

## API Endpoints

### Check Models Status
```bash
GET /admin/models-status
```
Returns detailed information about available models and sync status.

### Perform Cleanup
```bash
POST /admin/cleanup-models
Content-Type: application/json

{
  "confirm": "yes"
}
```
Triggers the cleanup process for orphaned models.

## Implementation Details

### Sidecar Enhancements

The model sync sidecar now includes:

1. **Signal File Monitoring**: Watches for cleanup signal files
2. **MinIO Comparison**: Compares local models with MinIO storage
3. **Cleanup Operations**: Removes orphaned models and updates configuration
4. **Status Reporting**: Maintains models.json with current status

### Signal Files

The cleanup process uses signal files for communication:
- `.cleanup_models`: Triggers orphaned model cleanup
- `.force_cleanup_models`: Removes all models (for testing/reset)

### Configuration Updates

The system automatically updates:
- OVMS `config.json` file
- Models status file (`models.json`)
- Real-time logging and monitoring

## Monitoring and Logging

All cleanup operations are logged with:
- Timestamps for all operations
- Success/failure status
- Detailed error messages
- User action tracking

Monitor logs through:
- Web interface System Logs tab
- Container logs: `kubectl logs <pod-name> -c model-sync`

## Best Practices

1. **Regular Status Checks**: Periodically check the Models tab for sync status
2. **Backup Important Models**: Ensure important models are backed up in MinIO
3. **Monitor Disk Space**: Use cleanup to free up disk space when needed
4. **Verify After Cleanup**: Always refresh status after cleanup operations

## Troubleshooting

### Common Issues

1. **Cleanup Button Disabled**: Check that models are loaded and status is refreshed
2. **Network Errors**: Ensure the sidecar container is running and accessible
3. **Permission Issues**: Verify shared volume permissions between containers

### Error Messages

- **"Confirmation required"**: Must send `{"confirm": "yes"}` in request body
- **"Failed to signal cleanup"**: Check file system permissions on models directory
- **"Sidecar not responding"**: Verify sidecar container is healthy and running

## Version Information

- **Introduced in**: v5.6
- **Client Image**: `adithyazededa/edgeai-client-app:v5.6`
- **Sidecar Image**: `adithyazededa/edgeai-model-sync-sidecar:v5.6`
- **Helm Chart**: `edge-ai-car-classification-2.5.0`

## Future Enhancements

Planned improvements include:
1. **Automated Cleanup**: Option for automatic cleanup on a schedule
2. **Model Versioning**: Support for model version management
3. **Backup Integration**: Automatic backup before cleanup
4. **Advanced Filtering**: Cleanup specific models or versions
