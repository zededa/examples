#!/bin/bash

# Entrypoint script for car classifier server
# Handles both config-based (dynamic) and static model loading

set -e

echo "🚀 Starting Car Classifier Server..."
echo "📁 Model directory: /models"
echo "👤 Running as user: $(id)"

# Check if this is config-based mode (used by Helm chart)
if [ "$1" = "--config_path" ] || [ -f "/models/config.json" ]; then
    echo "🔄 Using config-based model loading (dynamic mode)"
    
    # Wait for config file to be created by sidecar
    echo "⏳ Waiting for config.json to be created by sidecar..."
    timeout=60  # 60 seconds timeout
    counter=0
    while [ ! -f "/models/config.json" ] && [ $counter -lt $timeout ]; do
        echo "   Waiting for config.json... ($counter/$timeout)"
        sleep 2
        counter=$((counter + 2))
    done
    
    if [ ! -f "/models/config.json" ]; then
        echo "❌ Timeout waiting for config.json, creating basic config..."
        mkdir -p /models
        echo '{"model_config_list": []}' > /models/config.json
    fi
    
    echo "✅ Config file found, checking content..."
    cat /models/config.json
    
    echo "� Starting OVMS in config mode..."
    exec /ovms/bin/ovms \
        --config_path /models/config.json \
        --port 9000 \
        --rest_port 8000 \
        --log_level DEBUG \
        --file_system_poll_wait_seconds 5
fi

# Fallback to static model loading (original behavior)
echo "� Using static model loading (legacy mode)"

# Check if external models are mounted
if [ -d "/models/efb3_car_onnx" ] && find "/models/efb3_car_onnx" -name "*.onnx" -type f | grep -q .; then
    echo "✅ External models detected at /models"
    echo "📋 Model structure:"
    ls -la /models/efb3_car_onnx/
    if [ -d "/models/efb3_car_onnx/1" ]; then
        ls -la /models/efb3_car_onnx/1/
    fi
    MODEL_PATH="/models/efb3_car_onnx/"
elif [ -d "/models-fallback/efb3_car_onnx" ]; then
    echo "⚠️  No external models found, using fallback models"
    echo "📋 Copying fallback models to /models..."
    
    # Check if we can write to /models directory
    if [ -w "/models" ]; then
        cp -r /models-fallback/* /models/
        echo "✅ Fallback models copied successfully"
    else
        echo "⚠️  Cannot write to /models, using fallback path directly"
        MODEL_PATH="/models-fallback/efb3_car_onnx/"
    fi
    
    if [ -z "$MODEL_PATH" ]; then
        MODEL_PATH="/models/efb3_car_onnx/"
        echo "📋 Model structure:"
        ls -la /models/efb3_car_onnx/
        if [ -d "/models/efb3_car_onnx/1" ]; then
            ls -la /models/efb3_car_onnx/1/
        fi
    fi
else
    echo "❌ No models found in /models or /models-fallback"
    echo "📋 Available directories:"
    ls -la /
    echo "🔍 Checking /models contents:"
    ls -la /models/ || echo "  /models is empty or doesn't exist"
    echo "🔍 Checking /models-fallback contents:"
    ls -la /models-fallback/ || echo "  /models-fallback is empty or doesn't exist"
    exit 1
fi

# Verify model file exists
MODEL_FILE=$(find ${MODEL_PATH:-/models/efb3_car_onnx} -name "*.onnx" -type f 2>/dev/null | head -1)
if [ -z "$MODEL_FILE" ]; then
    echo "❌ No .onnx model file found in ${MODEL_PATH:-/models/efb3_car_onnx}"
    exit 1
fi

echo "✅ Model file found: $MODEL_FILE"
echo "🔧 Model path set to: $MODEL_PATH"

# Set environment variables for the OpenVINO Model Server
export MODEL_PATH="$MODEL_PATH"

# Start the OpenVINO Model Server with provided arguments
echo "🚀 Starting OpenVINO Model Server..."
echo "📋 Command: /ovms/bin/ovms $@"

# Execute the model server with all passed arguments
exec /ovms/bin/ovms "$@"
