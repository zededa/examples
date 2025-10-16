#!/bin/bash

# Release Script for EdgeAI Car Classification v5.3
# This script will:
# 1. Build and push updated container images
# 2. Update Helm chart with new image versions
# 3. Package the Helm chart as v2.2.0

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
REGISTRY="adithyazededa"
NEW_VERSION="v5.4"
NEW_HELM_VERSION="2.2.0"
NEW_APP_VERSION="v5.4"

echo -e "${BLUE}🚀 EdgeAI Car Classification Release ${NEW_VERSION}${NC}"
echo "=================================================="
echo -e "Registry: ${YELLOW}${REGISTRY}${NC}"
echo -e "Image Version: ${YELLOW}${NEW_VERSION}${NC}"
echo -e "Helm Chart Version: ${YELLOW}${NEW_HELM_VERSION}${NC}"
echo -e "App Version: ${YELLOW}${NEW_APP_VERSION}${NC}"
echo ""

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
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
    
    # Check Helm
    if ! command_exists helm; then
        echo -e "${RED}❌ Helm is not installed${NC}"
        exit 1
    fi
    
    # Check if logged into Docker Hub
    if ! docker info | grep -q "Username"; then
        echo -e "${YELLOW}⚠️  Please login to Docker Hub first: docker login${NC}"
        read -p "Press enter to continue if you're already logged in..."
    fi
    
    echo -e "${GREEN}✅ Prerequisites check passed${NC}"
    echo ""
}

# Function to build and push an image
build_and_push_image() {
    local name=$1
    local path=$2
    local version=$3
    local tag_latest="${REGISTRY}/${name}:latest"
    local tag_versioned="${REGISTRY}/${name}:${version}"
    
    echo -e "${BLUE}📦 Building and pushing ${name}...${NC}"
    
    if [ ! -d "$path" ]; then
        echo -e "${RED}❌ Directory $path not found${NC}"
        return 1
    fi
    
    cd "$path"
    
    # Build the image with latest tag
    echo -e "${BLUE}🔨 Building ${tag_latest}...${NC}"
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
    
    # Push latest tag
    echo -e "${BLUE}📤 Pushing ${tag_latest}...${NC}"
    docker push "${tag_latest}"
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ Pushed ${tag_latest}${NC}"
    else
        echo -e "${RED}❌ Failed to push ${tag_latest}${NC}"
        return 1
    fi
    
    # Push versioned tag
    echo -e "${BLUE}📤 Pushing ${tag_versioned}...${NC}"
    docker push "${tag_versioned}"
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ Pushed ${tag_versioned}${NC}"
    else
        echo -e "${RED}❌ Failed to push ${tag_versioned}${NC}"
        return 1
    fi
    
    cd - > /dev/null
    echo ""
}

# Function to update Helm chart versions
update_helm_chart() {
    echo -e "${BLUE}🔧 Updating Helm chart versions...${NC}"
    
    # Update Chart.yaml
    local chart_file="helm-chart/edge-ai-car-classification-helm/Chart.yaml"
    echo -e "${BLUE}📝 Updating ${chart_file}...${NC}"
    
    # Backup the original file
    cp "${chart_file}" "${chart_file}.backup"
    
    # Update version and appVersion
    sed -i "s/^version: .*/version: ${NEW_HELM_VERSION}/" "${chart_file}"
    sed -i "s/^appVersion: .*/appVersion: \"${NEW_APP_VERSION}\"/" "${chart_file}"
    
    echo -e "${GREEN}✅ Updated Chart.yaml${NC}"
    
    # Update values.yaml with new image versions
    local values_file="helm-chart/edge-ai-car-classification-helm/values.yaml"
    echo -e "${BLUE}📝 Updating ${values_file}...${NC}"
    
    # Backup the original file
    cp "${values_file}" "${values_file}.backup"
    
    # Update image versions in values.yaml
    sed -i "s/edgeai-ovms-server:v[0-9.]\+/edgeai-ovms-server:${NEW_VERSION}/" "${values_file}"
    sed -i "s/edgeai-model-sync-sidecar:v[0-9.]\+/edgeai-model-sync-sidecar:${NEW_VERSION}/" "${values_file}"
    sed -i "s/edgeai-client-app:v[0-9.]\+/edgeai-client-app:${NEW_VERSION}/" "${values_file}"
    
    echo -e "${GREEN}✅ Updated values.yaml${NC}"
    echo ""
}

# Function to package Helm chart
package_helm_chart() {
    echo -e "${BLUE}📦 Packaging Helm chart...${NC}"
    
    cd helm-chart
    
    # Package the chart
    helm package edge-ai-car-classification-helm/
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ Packaged Helm chart${NC}"
        
        # Generate SHA256 checksum
        local chart_package="edgeai-alpha-5-${NEW_HELM_VERSION}.tgz"
        if [ -f "${chart_package}" ]; then
            sha256sum "${chart_package}" > "${chart_package}.sha256"
            echo -e "${GREEN}✅ Generated SHA256 checksum${NC}"
        fi
    else
        echo -e "${RED}❌ Failed to package Helm chart${NC}"
        return 1
    fi
    
    cd - > /dev/null
    echo ""
}

# Function to show summary
show_summary() {
    echo -e "${GREEN}🎉 Release Summary${NC}"
    echo "=================="
    echo -e "Released version: ${YELLOW}${NEW_VERSION}${NC}"
    echo -e "Helm chart version: ${YELLOW}${NEW_HELM_VERSION}${NC}"
    echo ""
    echo "Built and pushed images:"
    echo "- ${REGISTRY}/edgeai-ovms-server:${NEW_VERSION}"
    echo "- ${REGISTRY}/edgeai-model-sync-sidecar:${NEW_VERSION}"  
    echo "- ${REGISTRY}/edgeai-client-app:${NEW_VERSION}"
    echo ""
    echo "Packaged Helm chart:"
    echo "- helm-chart/edgeai-alpha-5-${NEW_HELM_VERSION}.tgz"
    echo "- helm-chart/edgeai-alpha-5-${NEW_HELM_VERSION}.tgz.sha256"
    echo ""
    echo -e "${BLUE}Next steps:${NC}"
    echo "1. Deploy with Helm:"
    echo "   helm upgrade --install edgeai-car-classification helm-chart/edgeai-alpha-5-${NEW_HELM_VERSION}.tgz"
    echo "2. Or update existing deployment:"
    echo "   helm upgrade edgeai-car-classification helm-chart/edge-ai-car-classification-helm/"
    echo ""
}

# Main execution
main() {
    # Save original directory
    ORIGINAL_DIR=$(pwd)
    
    # Check prerequisites
    check_prerequisites
    
    # Build and push all images
    echo -e "${BLUE}🏗️  Building and pushing all images...${NC}"
    echo ""
    
    # Build sync sidecar
    build_and_push_image "edgeai-model-sync-sidecar" "sync-sidecar" "${NEW_VERSION}"
    
    # Build server container
    build_and_push_image "edgeai-ovms-server" "server-container" "${NEW_VERSION}"
    
    # Build client container (with updated UI)
    build_and_push_image "edgeai-client-app" "client-container" "${NEW_VERSION}"
    
    # Update Helm chart versions
    update_helm_chart
    
    # Package Helm chart
    package_helm_chart
    
    # Show summary
    show_summary
    
    # Return to original directory
    cd "$ORIGINAL_DIR"
    
    echo -e "${GREEN}🎉 Release ${NEW_VERSION} completed successfully!${NC}"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            echo "EdgeAI Car Classification Release Script"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -h, --help              Show this help message"
            echo "  --registry REGISTRY     Set container registry (default: adithyazededa)"
            echo "  --version VERSION       Set version tag (default: v5.3)"
            echo "  --helm-version VERSION  Set Helm chart version (default: 2.2.0)"
            echo ""
            exit 0
            ;;
        --registry)
            REGISTRY="$2"
            shift 2
            ;;
        --version)
            NEW_VERSION="$2"
            NEW_APP_VERSION="$2"
            shift 2
            ;;
        --helm-version)
            NEW_HELM_VERSION="$2"
            shift 2
            ;;
        *)
            echo -e "${RED}❌ Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Run main function
main
