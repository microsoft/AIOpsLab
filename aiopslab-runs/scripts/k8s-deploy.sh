#!/bin/bash

# AIOpsLab Viewer - Kubernetes Deployment Script
# This script helps deploy AIOpsLab Viewer to Kubernetes using Helm

set -e

# Default values
NAMESPACE="aiopslab"
RELEASE_NAME="aiopslab-viewer"
ENVIRONMENT="dev"
HELM_CHART_PATH="./helm/aiopslab-viewer"
IMAGE_TAG="latest"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Print colored output
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Show usage
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Deploy AIOpsLab Viewer to Kubernetes using Helm

OPTIONS:
    -n, --namespace NAMESPACE     Kubernetes namespace (default: aiopslab)
    -r, --release RELEASE_NAME    Helm release name (default: aiopslab-viewer)
    -e, --environment ENV         Environment: dev|prod (default: dev)
    -t, --tag IMAGE_TAG          Docker image tag (default: latest)
    -u, --upgrade                Upgrade existing deployment
    -d, --dry-run                Perform a dry run
    -h, --help                   Show this help message

EXAMPLES:
    $0                           # Deploy to dev environment
    $0 -e prod -t v1.0.0        # Deploy to production with specific tag
    $0 -u -e prod               # Upgrade production deployment
    $0 -d -e prod               # Dry run for production

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -n|--namespace)
            NAMESPACE="$2"
            shift 2
            ;;
        -r|--release)
            RELEASE_NAME="$2"
            shift 2
            ;;
        -e|--environment)
            ENVIRONMENT="$2"
            shift 2
            ;;
        -t|--tag)
            IMAGE_TAG="$2"
            shift 2
            ;;
        -u|--upgrade)
            UPGRADE=true
            shift
            ;;
        -d|--dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            print_error "Unknown option $1"
            usage
            exit 1
            ;;
    esac
done

# Validate environment
if [[ ! "$ENVIRONMENT" =~ ^(dev|prod)$ ]]; then
    print_error "Environment must be 'dev' or 'prod'"
    exit 1
fi

# Check if required tools are installed
check_requirements() {
    print_info "Checking requirements..."
    
    if ! command -v kubectl &> /dev/null; then
        print_error "kubectl is not installed"
        exit 1
    fi
    
    if ! command -v helm &> /dev/null; then
        print_error "helm is not installed"
        exit 1
    fi
    
    if ! command -v docker &> /dev/null; then
        print_warn "docker is not installed - image building will not be available"
    fi
    
    print_info "Requirements check passed"
}

# Create namespace if it doesn't exist
create_namespace() {
    print_info "Ensuring namespace '$NAMESPACE' exists..."
    kubectl create namespace "$NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -
}

# Build Docker image if requested
build_image() {
    if [[ -n "$BUILD_IMAGE" ]]; then
        print_info "Building Docker image..."
        docker build -t "aiopslab-viewer:$IMAGE_TAG" .
        print_info "Image built successfully"
    fi
}

# Deploy using Helm
deploy() {
    local values_file="${HELM_CHART_PATH}/values-${ENVIRONMENT}.yaml"
    local helm_command="helm"
    local action="install"
    
    if [[ "$UPGRADE" == "true" ]]; then
        action="upgrade"
    fi
    
    if [[ "$DRY_RUN" == "true" ]]; then
        helm_command="$helm_command --dry-run"
        print_info "Performing dry run..."
    fi
    
    print_info "Deploying AIOpsLab Viewer..."
    print_info "Namespace: $NAMESPACE"
    print_info "Release: $RELEASE_NAME"
    print_info "Environment: $ENVIRONMENT"
    print_info "Image Tag: $IMAGE_TAG"
    
    # Check if values file exists
    if [[ ! -f "$values_file" ]]; then
        print_warn "Values file $values_file not found, using default values"
        values_file="${HELM_CHART_PATH}/values.yaml"
    fi
    
    # Prepare Helm command
    local cmd="$helm_command $action $RELEASE_NAME $HELM_CHART_PATH"
    cmd="$cmd --namespace $NAMESPACE"
    cmd="$cmd --values $values_file"
    cmd="$cmd --set image.tag=$IMAGE_TAG"
    cmd="$cmd --wait --timeout=300s"
    
    if [[ "$action" == "install" ]]; then
        cmd="$cmd --create-namespace"
    fi
    
    print_info "Executing: $cmd"
    eval $cmd
    
    if [[ "$DRY_RUN" != "true" ]]; then
        print_info "Deployment completed successfully!"
        
        # Show deployment status
        print_info "Checking deployment status..."
        kubectl get pods -n "$NAMESPACE" -l "app.kubernetes.io/name=aiopslab-viewer"
        
        # Show service information
        print_info "Service information:"
        kubectl get svc -n "$NAMESPACE" -l "app.kubernetes.io/name=aiopslab-viewer"
        
        # Show ingress if enabled
        if kubectl get ingress -n "$NAMESPACE" "$RELEASE_NAME" &> /dev/null; then
            print_info "Ingress information:"
            kubectl get ingress -n "$NAMESPACE" "$RELEASE_NAME"
        fi
        
        # Show access instructions
        print_info "Access instructions:"
        echo "  kubectl port-forward -n $NAMESPACE svc/$RELEASE_NAME 3000:3000"
        echo "  Then visit: http://localhost:3000"
    fi
}

# Main execution
main() {
    print_info "Starting AIOpsLab Viewer deployment..."
    
    check_requirements
    create_namespace
    build_image
    deploy
    
    print_info "Deployment script completed!"
}

# Run main function
main
