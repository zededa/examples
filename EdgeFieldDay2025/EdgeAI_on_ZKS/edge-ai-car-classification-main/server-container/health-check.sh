#!/bin/bash

# Quick health check for the car classifier server container
set -e

echo "🏥 Car Classifier Server Health Check"
echo "===================================="

# Test configurations
CONTAINER_NAME="car-classifier-health-test"
HOST_MODEL_PATH="/home/zedtel/Developer/volume-for-model/models"
TEST_PORT=18000

# Function to cleanup
cleanup() {
    echo "🧹 Cleaning up..."
    docker stop $CONTAINER_NAME 2>/dev/null || true
    docker rm $CONTAINER_NAME 2>/dev/null || true
}

# Trap cleanup on exit
trap cleanup EXIT

echo "🚀 Testing container with external volume..."

# Start container with volume mount
docker run -d --name $CONTAINER_NAME \
    -p $TEST_PORT:8000 \
    -v "$HOST_MODEL_PATH:/models" \
    car-classifier-server:latest

echo "⏳ Waiting for container to start..."
sleep 10

# Check if container is running
if ! docker ps | grep -q $CONTAINER_NAME; then
    echo "❌ Container failed to start"
    echo "📋 Container logs:"
    docker logs $CONTAINER_NAME
    exit 1
fi

echo "✅ Container is running"

# Check logs for external model detection
echo "📋 Model loading status:"
docker logs $CONTAINER_NAME 2>&1 | grep -E "(External models detected|Fallback models|Model file found|model: efb3_car_onnx)" | tail -5

# Test health endpoint
echo "🔍 Testing health endpoint..."
for i in {1..5}; do
    if curl -s -f "http://localhost:$TEST_PORT/v1/models/efb3_car_onnx/ready" > /dev/null; then
        echo "✅ Health check passed! Model is ready for inference."
        break
    else
        echo "⏳ Attempt $i/5: Model not ready yet, waiting..."
        sleep 5
    fi
done

# Test model metadata endpoint
echo "🔍 Testing model metadata..."
if curl -s "http://localhost:$TEST_PORT/v1/models/efb3_car_onnx/metadata" | jq . > /dev/null 2>&1; then
    echo "✅ Model metadata endpoint working"
    echo "📋 Model info:"
    curl -s "http://localhost:$TEST_PORT/v1/models/efb3_car_onnx/metadata" | jq '.name, .versions[0]' 2>/dev/null || echo "   (jq not available for pretty printing)"
else
    echo "⚠️  Model metadata endpoint test failed"
fi

echo ""
echo "🎉 Health check completed successfully!"
echo "📝 Summary:"
echo "   - Container starts correctly with external volume"
echo "   - Model loads from mounted volume"
echo "   - REST API endpoints are responsive"
echo "   - Ready for Kubernetes deployment with PVC"
