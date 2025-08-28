#!/bin/bash

# ML Inference Platform - Build Script
# This script builds all container images and prepares the platform for deployment

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default configuration
REGISTRY=${REGISTRY:-"adithyazededa"}
PUSH=${PUSH:-"false"}
MINIKUBE=${MINIKUBE:-"true"}
BUILD_ALL=${BUILD_ALL:-"true"}

echo -e "${BLUE}🚀 ML Inference Platform Build Script${NC}"
echo "======================================"
echo -e "Registry: ${YELLOW}${REGISTRY}${NC}"
echo -e "Push images: ${YELLOW}${PUSH}${NC}"
echo -e "Load to minikube: ${YELLOW}${MINIKUBE}${NC}"
echo -e "Build all images: ${YELLOW}${BUILD_ALL}${NC}"
echo ""

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to build and optionally push an image
build_image() {
    local name=$1
    local path=$2
    local version=$3
    local tag_latest="${REGISTRY}/${name}:latest"
    local tag_versioned="${REGISTRY}/${name}:${version}"
    
    echo -e "${BLUE}📦 Building ${name}...${NC}"
    
    if [ ! -d "$path" ]; then
        echo -e "${RED}❌ Directory $path not found${NC}"
        return 1
    fi
    
    cd "$path"
    
    # Build the image with latest tag
    docker build -t "${tag_latest}" .
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ Built ${tag_latest}${NC}"
        
        # Tag with version
        docker tag "${tag_latest}" "${tag_versioned}"
        echo -e "${GREEN}✅ Tagged as ${tag_versioned}${NC}"
    else
        echo -e "${RED}❌ Failed to build ${tag_latest}${NC}"
        return 1
    fi
    
    # Push to registry if requested
    if [ "$PUSH" = "true" ]; then
        echo -e "${BLUE}📤 Pushing ${tag_latest}...${NC}"
        docker push "${tag_latest}"
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}✅ Pushed ${tag_latest}${NC}"
        else
            echo -e "${RED}❌ Failed to push ${tag_latest}${NC}"
            return 1
        fi
        
        echo -e "${BLUE}📤 Pushing ${tag_versioned}...${NC}"
        docker push "${tag_versioned}"
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}✅ Pushed ${tag_versioned}${NC}"
        else
            echo -e "${RED}❌ Failed to push ${tag_versioned}${NC}"
            return 1
        fi
    fi
    
    # Load to minikube if requested
    if [ "$MINIKUBE" = "true" ] && command_exists minikube; then
        echo -e "${BLUE}📥 Loading ${tag} to minikube...${NC}"
        minikube image load "${tag}"
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}✅ Loaded ${tag} to minikube${NC}"
        else
            echo -e "${YELLOW}⚠️  Failed to load ${tag} to minikube${NC}"
        fi
    fi
    
    cd - > /dev/null
    echo ""
}

# Function to start local registry if needed
start_local_registry() {
    if [ "$REGISTRY" = "localhost:5000" ]; then
        echo -e "${BLUE}🏗️  Managing local Docker registry...${NC}"
        
        # Check if registry container exists
        if docker ps -a --format "table {{.Names}}" | grep -q "^registry$"; then
            # Container exists, check if it's running
            if docker ps --format "table {{.Names}}" | grep -q "^registry$"; then
                echo -e "${GREEN}✅ Local registry already running${NC}"
            else
                # Container exists but is stopped, start it
                echo -e "${YELLOW}🔄 Starting existing registry container...${NC}"
                docker start registry
                if [ $? -eq 0 ]; then
                    echo -e "${GREEN}✅ Local registry started on port 5000${NC}"
                else
                    echo -e "${RED}❌ Failed to start existing registry container${NC}"
                    return 1
                fi
            fi
        else
            # Container doesn't exist, create and start it
            echo -e "${BLUE}🆕 Creating new registry container...${NC}"
            docker run -d -p 5000:5000 --name registry --restart=unless-stopped registry:2
            if [ $? -eq 0 ]; then
                echo -e "${GREEN}✅ Local registry created and started on port 5000${NC}"
            else
                echo -e "${RED}❌ Failed to create registry container${NC}"
                return 1
            fi
        fi
        
        # Verify registry is accessible
        sleep 2
        if curl -sf http://localhost:5000/v2/ >/dev/null 2>&1; then
            echo -e "${GREEN}✅ Registry health check passed${NC}"
        else
            echo -e "${YELLOW}⚠️  Registry health check failed, but continuing...${NC}"
        fi
        echo ""
    elif [ "$REGISTRY" != "localhost:5000" ] && [ "$PUSH" = "true" ]; then
        echo -e "${BLUE}🔐 Docker Hub Push Mode${NC}"
        echo -e "${YELLOW}⚠️  Make sure you're logged in to Docker Hub: docker login${NC}"
        
        # Check if user is logged in to Docker Hub
        if ! docker info | grep -q "Username"; then
            echo -e "${RED}❌ Not logged in to Docker Hub. Please run: docker login${NC}"
            exit 1
        fi
        echo ""
    fi
}

# Function to check prerequisites
check_prerequisites() {
    echo -e "${BLUE}🔍 Checking prerequisites...${NC}"
    
    # Check Docker
    if ! command_exists docker; then
        echo -e "${RED}❌ Docker is not installed${NC}"
        exit 1
    fi
    
    # Check if Docker daemon is running
    if ! docker info >/dev/null 2>&1; then
        echo -e "${RED}❌ Docker daemon is not running${NC}"
        exit 1
    fi
    
    # Check minikube if needed
    if [ "$MINIKUBE" = "true" ] && ! command_exists minikube; then
        echo -e "${YELLOW}⚠️  Minikube not found, skipping image loading${NC}"
        MINIKUBE="false"
    fi
    
    # Check if minikube is running
    if [ "$MINIKUBE" = "true" ] && ! minikube status >/dev/null 2>&1; then
        echo -e "${YELLOW}⚠️  Minikube is not running, skipping image loading${NC}"
        MINIKUBE="false"
    fi
    
    echo -e "${GREEN}✅ Prerequisites check passed${NC}"
    echo ""
}

# Function to validate Dockerfiles
validate_dockerfiles() {
    echo -e "${BLUE}🔍 Validating Dockerfiles...${NC}"
    
    local dockerfiles=(
        "server-container/Dockerfile"
        "client-container/Dockerfile"
        "sync-sidecar/Dockerfile"
    )
    
    for dockerfile in "${dockerfiles[@]}"; do
        if [ ! -f "$dockerfile" ]; then
            echo -e "${RED}❌ Missing Dockerfile: $dockerfile${NC}"
            exit 1
        fi
    done
    
    echo -e "${GREEN}✅ All Dockerfiles found${NC}"
    echo ""
}

# Function to show build summary
show_summary() {
    echo -e "${GREEN}🎉 Build Summary${NC}"
    echo "================"
    echo "Built images:"
    docker images | grep -E "(edgeai-ovms-server|edgeai-client-app|edgeai-model-sync-sidecar)" | grep -v "<none>"
    echo ""
    
    if [ "$MINIKUBE" = "true" ]; then
        echo "Images in minikube:"
        minikube image ls | grep -E "(edgeai-ovms-server|edgeai-client-app|edgeai-model-sync-sidecar)" || echo "No matching images found in minikube"
        echo ""
    fi
    
    if [ "$REGISTRY" = "localhost:5000" ]; then
        echo "Images in local registry:"
        curl -s http://localhost:5000/v2/_catalog | jq -r '.repositories[]' 2>/dev/null || echo "Could not query registry"
        echo ""
    fi
}

# Main execution
main() {
    # Save original directory
    ORIGINAL_DIR=$(pwd)
    
    # Check prerequisites
    check_prerequisites
    
    # Validate Dockerfiles
    validate_dockerfiles
    
    # Start local registry if needed
    start_local_registry
    
    # Build images
    if [ "$BUILD_ALL" = "true" ]; then
        echo -e "${BLUE}🏗️  Building all images...${NC}"
        echo ""
        
        # Build sync sidecar (most likely to change)
        build_image "edgeai-model-sync-sidecar" "sync-sidecar" "v5"
        
        # Build server container
        build_image "edgeai-ovms-server" "server-container" "v5"
        
        # Build client container
        build_image "edgeai-client-app" "client-container" "v5.0"
    else
        echo -e "${YELLOW}⚠️  BUILD_ALL=false, skipping image builds${NC}"
    fi
    
    # Show summary
    show_summary
    
    # Return to original directory
    cd "$ORIGINAL_DIR"
    
    echo -e "${GREEN}🎉 Build completed successfully!${NC}"
    echo ""
    echo -e "${BLUE}Next steps:${NC}"
    echo "1. Deploy with Helm: cd helm-chart/edge-ai-car-classification && helm install ml-platform . -f local-values.yaml"
    echo "2. Check deployment: kubectl get pods -n ml-platform"
    echo "3. Upload models to MinIO to test dynamic loading"
}

# Help function
show_help() {
    echo "ML Inference Platform Build Script"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -h, --help              Show this help message"
    echo "  --registry REGISTRY     Set container registry (default: adithyazededa)"
    echo "  --push                  Push images to registry"
    echo "  --no-minikube          Skip loading images to minikube"
    echo "  --sidecar-only         Build only the sync sidecar"
    echo "  --server-only          Build only the server container"
    echo "  --client-only          Build only the client container"
    echo ""
    echo "Environment variables:"
    echo "  REGISTRY               Container registry (default: adithyazededa)"
    echo "  PUSH                   Push images (default: false)"
    echo "  MINIKUBE              Load to minikube (default: true)"
    echo "  BUILD_ALL             Build all images (default: true)"
    echo ""
    echo "Examples:"
    echo "  $0                                        # Build all images for Docker Hub"
    echo "  $0 --push                                 # Build and push to Docker Hub (adithyazededa)"
    echo "  $0 --push --registry my-user             # Build and push to Docker Hub (my-user)"
    echo "  $0 --registry localhost:5000 --push      # Use local registry"
    echo "  $0 --sidecar-only                        # Build only sync sidecar"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        --registry)
            REGISTRY="$2"
            shift 2
            ;;
        --push)
            PUSH="true"
            shift
            ;;
        --no-minikube)
            MINIKUBE="false"
            shift
            ;;
        --sidecar-only)
            BUILD_ALL="false"
            echo -e "${BLUE}Building sync sidecar only...${NC}"
            build_image "edgeai-model-sync-sidecar" "sync-sidecar" "v5"
            exit 0
            ;;
        --server-only)
            BUILD_ALL="false"
            echo -e "${BLUE}Building server container only...${NC}"
            build_image "edgeai-ovms-server" "server-container" "v5"
            exit 0
            ;;
        --client-only)
            BUILD_ALL="false"
            echo -e "${BLUE}Building client container only...${NC}"
            build_image "edgeai-client-app" "client-container" "v5.0"
            exit 0
            ;;
        *)
            echo -e "${RED}❌ Unknown option: $1${NC}"
            show_help
            exit 1
            ;;
    esac
done

# Run main function
main
