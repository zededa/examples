#!/bin/bash

# ML Inference Platform - Cleanup Script
# This script cleans up deployments, images, and resources

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default configuration
NAMESPACE=${NAMESPACE:-"ml-platform"}
RELEASE_NAME=${RELEASE_NAME:-"ml-platform"}
CLEAN_IMAGES=${CLEAN_IMAGES:-"false"}
CLEAN_REGISTRY=${CLEAN_REGISTRY:-"false"}
CLEAN_MINIKUBE_IMAGES=${CLEAN_MINIKUBE_IMAGES:-"false"}
CLEAN_VOLUMES=${CLEAN_VOLUMES:-"false"}
FORCE=${FORCE:-"false"}

echo -e "${BLUE}🧹 ML Inference Platform Cleanup Script${NC}"
echo "========================================"
echo -e "Namespace: ${YELLOW}${NAMESPACE}${NC}"
echo -e "Release name: ${YELLOW}${RELEASE_NAME}${NC}"
echo -e "Clean images: ${YELLOW}${CLEAN_IMAGES}${NC}"
echo -e "Clean registry: ${YELLOW}${CLEAN_REGISTRY}${NC}"
echo -e "Clean minikube images: ${YELLOW}${CLEAN_MINIKUBE_IMAGES}${NC}"
echo -e "Clean volumes: ${YELLOW}${CLEAN_VOLUMES}${NC}"
echo -e "Force cleanup: ${YELLOW}${FORCE}${NC}"
echo ""

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to confirm action
confirm_action() {
    local message="$1"
    if [ "$FORCE" = "true" ]; then
        return 0
    fi
    
    echo -e "${YELLOW}⚠️  $message${NC}"
    read -p "Do you want to continue? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${BLUE}ℹ️  Operation cancelled${NC}"
        return 1
    fi
    return 0
}

# Function to uninstall Helm release
uninstall_helm_release() {
    if ! command_exists helm; then
        echo -e "${YELLOW}⚠️  Helm not found, skipping Helm cleanup${NC}"
        return
    fi
    
    echo -e "${BLUE}🔍 Checking for Helm release...${NC}"
    
    if helm list -n "$NAMESPACE" | grep -q "$RELEASE_NAME"; then
        if confirm_action "This will uninstall the Helm release '$RELEASE_NAME' in namespace '$NAMESPACE'"; then
            echo -e "${BLUE}🗑️  Uninstalling Helm release...${NC}"
            helm uninstall "$RELEASE_NAME" -n "$NAMESPACE"
            if [ $? -eq 0 ]; then
                echo -e "${GREEN}✅ Helm release uninstalled${NC}"
            else
                echo -e "${RED}❌ Failed to uninstall Helm release${NC}"
            fi
        fi
    else
        echo -e "${BLUE}ℹ️  No Helm release found with name '$RELEASE_NAME' in namespace '$NAMESPACE'${NC}"
    fi
    echo ""
}

# Function to clean up Kubernetes resources
cleanup_k8s_resources() {
    if ! command_exists kubectl; then
        echo -e "${YELLOW}⚠️  kubectl not found, skipping Kubernetes cleanup${NC}"
        return
    fi
    
    echo -e "${BLUE}🔍 Checking for Kubernetes resources...${NC}"
    
    # Check if namespace exists
    if ! kubectl get namespace "$NAMESPACE" >/dev/null 2>&1; then
        echo -e "${BLUE}ℹ️  Namespace '$NAMESPACE' does not exist${NC}"
        return
    fi
    
    # List resources in namespace
    local resources=$(kubectl get all -n "$NAMESPACE" 2>/dev/null | tail -n +2)
    if [ -n "$resources" ]; then
        echo -e "${YELLOW}Found resources in namespace '$NAMESPACE':${NC}"
        kubectl get all -n "$NAMESPACE"
        echo ""
        
        if confirm_action "This will delete all resources in namespace '$NAMESPACE'"; then
            echo -e "${BLUE}🗑️  Deleting all resources in namespace...${NC}"
            kubectl delete all --all -n "$NAMESPACE"
            
            # Delete configmaps and secrets
            kubectl delete configmaps --all -n "$NAMESPACE" 2>/dev/null || true
            kubectl delete secrets --all -n "$NAMESPACE" 2>/dev/null || true
            
            echo -e "${GREEN}✅ Kubernetes resources deleted${NC}"
        fi
    else
        echo -e "${BLUE}ℹ️  No resources found in namespace '$NAMESPACE'${NC}"
    fi
    
    # Optionally delete PVCs
    if [ "$CLEAN_VOLUMES" = "true" ]; then
        local pvcs=$(kubectl get pvc -n "$NAMESPACE" 2>/dev/null | tail -n +2)
        if [ -n "$pvcs" ]; then
            echo -e "${YELLOW}Found PVCs in namespace '$NAMESPACE':${NC}"
            kubectl get pvc -n "$NAMESPACE"
            echo ""
            
            if confirm_action "This will delete all Persistent Volume Claims (data will be lost!)"; then
                echo -e "${BLUE}🗑️  Deleting PVCs...${NC}"
                kubectl delete pvc --all -n "$NAMESPACE"
                echo -e "${GREEN}✅ PVCs deleted${NC}"
            fi
        fi
    fi
    
    # Optionally delete namespace
    if confirm_action "Delete the entire namespace '$NAMESPACE'?"; then
        echo -e "${BLUE}🗑️  Deleting namespace...${NC}"
        kubectl delete namespace "$NAMESPACE"
        echo -e "${GREEN}✅ Namespace deleted${NC}"
    fi
    
    echo ""
}

# Function to clean up Docker images
cleanup_docker_images() {
    if ! command_exists docker; then
        echo -e "${YELLOW}⚠️  Docker not found, skipping image cleanup${NC}"
        return
    fi
    
    echo -e "${BLUE}🔍 Checking for Docker images...${NC}"
    
    local images=(
        "my-ovms-server"
        "my-client-app"
        "model-sync-sidecar"
    )
    
    local found_images=()
    for image in "${images[@]}"; do
        if docker images | grep -q "$image"; then
            found_images+=("$image")
        fi
    done
    
    if [ ${#found_images[@]} -gt 0 ]; then
        echo -e "${YELLOW}Found ML platform images:${NC}"
        for image in "${found_images[@]}"; do
            docker images | grep "$image"
        done
        echo ""
        
        if confirm_action "This will delete all ML platform Docker images"; then
            echo -e "${BLUE}🗑️  Deleting Docker images...${NC}"
            for image in "${found_images[@]}"; do
                docker rmi $(docker images | grep "$image" | awk '{print $3}') 2>/dev/null || true
            done
            echo -e "${GREEN}✅ Docker images deleted${NC}"
        fi
    else
        echo -e "${BLUE}ℹ️  No ML platform images found${NC}"
    fi
    echo ""
}

# Function to clean up local registry
cleanup_local_registry() {
    if ! command_exists docker; then
        echo -e "${YELLOW}⚠️  Docker not found, skipping registry cleanup${NC}"
        return
    fi
    
    echo -e "${BLUE}🔍 Checking for local registry...${NC}"
    
    if docker ps | grep -q "registry.*5000"; then
        if confirm_action "This will stop and remove the local Docker registry container"; then
            echo -e "${BLUE}🗑️  Stopping local registry...${NC}"
            docker stop registry 2>/dev/null || true
            docker rm registry 2>/dev/null || true
            echo -e "${GREEN}✅ Local registry removed${NC}"
        fi
    else
        echo -e "${BLUE}ℹ️  No local registry container found${NC}"
    fi
    echo ""
}

# Function to clean up minikube images
cleanup_minikube_images() {
    if ! command_exists minikube; then
        echo -e "${YELLOW}⚠️  Minikube not found, skipping minikube image cleanup${NC}"
        return
    fi
    
    if ! minikube status >/dev/null 2>&1; then
        echo -e "${YELLOW}⚠️  Minikube is not running, skipping image cleanup${NC}"
        return
    fi
    
    echo -e "${BLUE}🔍 Checking for images in minikube...${NC}"
    
    local minikube_images=$(minikube image ls | grep -E "(my-ovms-server|my-client-app|model-sync-sidecar)")
    if [ -n "$minikube_images" ]; then
        echo -e "${YELLOW}Found ML platform images in minikube:${NC}"
        echo "$minikube_images"
        echo ""
        
        if confirm_action "This will delete ML platform images from minikube"; then
            echo -e "${BLUE}🗑️  Deleting minikube images...${NC}"
            echo "$minikube_images" | while read -r image; do
                if [ -n "$image" ]; then
                    minikube image rm "$image" 2>/dev/null || true
                fi
            done
            echo -e "${GREEN}✅ Minikube images deleted${NC}"
        fi
    else
        echo -e "${BLUE}ℹ️  No ML platform images found in minikube${NC}"
    fi
    echo ""
}

# Function to show cleanup summary
show_summary() {
    echo -e "${GREEN}🎉 Cleanup Summary${NC}"
    echo "=================="
    
    # Check remaining resources
    echo "Remaining resources:"
    
    if command_exists kubectl && kubectl get namespace "$NAMESPACE" >/dev/null 2>&1; then
        echo "  - Namespace '$NAMESPACE': EXISTS"
        local remaining_resources=$(kubectl get all -n "$NAMESPACE" 2>/dev/null | tail -n +2)
        if [ -n "$remaining_resources" ]; then
            echo "  - Resources in namespace: $(echo "$remaining_resources" | wc -l) items"
        else
            echo "  - Resources in namespace: NONE"
        fi
    else
        echo "  - Namespace '$NAMESPACE': DELETED"
    fi
    
    if command_exists docker; then
        local remaining_images=$(docker images | grep -E "(my-ovms-server|my-client-app|model-sync-sidecar)" | wc -l)
        echo "  - Docker images: $remaining_images remaining"
        
        if docker ps | grep -q "registry.*5000"; then
            echo "  - Local registry: RUNNING"
        else
            echo "  - Local registry: STOPPED"
        fi
    fi
    
    echo ""
}

# Function to show help
show_help() {
    echo "ML Inference Platform Cleanup Script"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -h, --help              Show this help message"
    echo "  --namespace NAMESPACE   Kubernetes namespace (default: ml-platform)"
    echo "  --release RELEASE       Helm release name (default: ml-platform)"
    echo "  --clean-images          Delete Docker images"
    echo "  --clean-registry        Stop and remove local Docker registry"
    echo "  --clean-minikube        Delete images from minikube"
    echo "  --clean-volumes         Delete Persistent Volume Claims (DATA LOSS!)"
    echo "  --force                 Skip confirmation prompts"
    echo "  --all                   Clean everything (images, registry, minikube, volumes)"
    echo ""
    echo "Environment variables:"
    echo "  NAMESPACE              Kubernetes namespace (default: ml-platform)"
    echo "  RELEASE_NAME           Helm release name (default: ml-platform)"
    echo "  CLEAN_IMAGES           Delete Docker images (default: false)"
    echo "  CLEAN_REGISTRY         Clean local registry (default: false)"
    echo "  CLEAN_MINIKUBE_IMAGES  Clean minikube images (default: false)"
    echo "  CLEAN_VOLUMES          Clean volumes (default: false)"
    echo "  FORCE                  Skip confirmations (default: false)"
    echo ""
    echo "Examples:"
    echo "  $0                                    # Basic cleanup (Helm + K8s resources)"
    echo "  $0 --clean-images --clean-registry   # Also clean Docker images and registry"
    echo "  $0 --all --force                     # Clean everything without prompts"
    echo "  $0 --namespace my-ns --release my-app # Clean specific deployment"
    echo ""
    echo "⚠️  Warning: Use --clean-volumes with caution as it will delete all data!"
}

# Main execution
main() {
    # Ensure we're in the project root
    if [ ! -f "SOLUTION-SUMMARY.md" ]; then
        echo -e "${RED}❌ Please run this script from the project root directory${NC}"
        exit 1
    fi
    
    echo -e "${BLUE}🧹 Starting cleanup process...${NC}"
    echo ""
    
    # Uninstall Helm release
    uninstall_helm_release
    
    # Clean up Kubernetes resources
    cleanup_k8s_resources
    
    # Clean up Docker images if requested
    if [ "$CLEAN_IMAGES" = "true" ]; then
        cleanup_docker_images
    fi
    
    # Clean up local registry if requested
    if [ "$CLEAN_REGISTRY" = "true" ]; then
        cleanup_local_registry
    fi
    
    # Clean up minikube images if requested
    if [ "$CLEAN_MINIKUBE_IMAGES" = "true" ]; then
        cleanup_minikube_images
    fi
    
    # Show summary
    show_summary
    
    echo -e "${GREEN}🎉 Cleanup completed!${NC}"
    echo ""
    echo -e "${BLUE}Note:${NC} If you want to clean Docker images or registry, run:"
    echo "  $0 --clean-images --clean-registry"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        --namespace)
            NAMESPACE="$2"
            shift 2
            ;;
        --release)
            RELEASE_NAME="$2"
            shift 2
            ;;
        --clean-images)
            CLEAN_IMAGES="true"
            shift
            ;;
        --clean-registry)
            CLEAN_REGISTRY="true"
            shift
            ;;
        --clean-minikube)
            CLEAN_MINIKUBE_IMAGES="true"
            shift
            ;;
        --clean-volumes)
            CLEAN_VOLUMES="true"
            shift
            ;;
        --force)
            FORCE="true"
            shift
            ;;
        --all)
            CLEAN_IMAGES="true"
            CLEAN_REGISTRY="true"
            CLEAN_MINIKUBE_IMAGES="true"
            CLEAN_VOLUMES="true"
            shift
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
