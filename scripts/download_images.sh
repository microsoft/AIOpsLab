#!/bin/bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

#
# Download all required images for AIOpsLab offline deployment
#
# Usage: ./download_images.sh [output_dir]
#
# This script downloads all Docker images required by AIOpsLab applications
# and saves them as tar files for offline deployment.
#

set -e

OUTPUT_DIR="${1:-./images}"
mkdir -p "$OUTPUT_DIR"

echo "=============================================="
echo "AIOpsLab Image Downloader"
echo "=============================================="
echo "Output directory: $OUTPUT_DIR"
echo ""

# List of required images for AIOpsLab applications
# Format: registry/image:tag
IMAGES=(
    # OpenTelemetry Astronomy Shop
    "ghcr.io/open-telemetry/demo:1.11.1-accountingservice"
    "ghcr.io/open-telemetry/demo:1.11.1-adservice"
    "ghcr.io/open-telemetry/demo:1.11.1-cartservice"
    "ghcr.io/open-telemetry/demo:1.11.1-checkoutservice"
    "ghcr.io/open-telemetry/demo:1.11.1-currencyservice"
    "ghcr.io/open-telemetry/demo:1.11.1-emailservice"
    "ghcr.io/open-telemetry/demo:1.11.1-flagd"
    "ghcr.io/open-telemetry/demo:1.11.1-frauddetectionservice"
    "ghcr.io/open-telemetry/demo:1.11.1-frontend"
    "ghcr.io/open-telemetry/demo:1.11.1-frontendproxy"
    "ghcr.io/open-telemetry/demo:1.11.1-imageprovider"
    "ghcr.io/open-telemetry/demo:1.11.1-kafka"
    "ghcr.io/open-telemetry/demo:1.11.1-loadgenerator"
    "ghcr.io/open-telemetry/demo:1.11.1-paymentservice"
    "ghcr.io/open-telemetry/demo:1.11.1-productcatalogservice"
    "ghcr.io/open-telemetry/demo:1.11.1-quoteservice"
    "ghcr.io/open-telemetry/demo:1.11.1-recommendationservice"
    "ghcr.io/open-telemetry/demo:1.11.1-shippingservice"
    "ghcr.io/open-telemetry/demo:1.11.1-valkey"
    
    # Observability stack
    "quay.io/prometheus/prometheus:v2.47.2"
    "grafana/grafana:10.2.0"
    "jaegertracing/all-in-one:1.50"
    "otel/opentelemetry-collector-contrib:0.88.0"
    
    # Infrastructure
    "docker.io/library/redis:7.2-alpine"
    "docker.io/library/postgres:16"
    "bitnami/kafka:3.6"
    
    # OpenEBS for local storage
    "openebs/provisioner-localpv:3.4.0"
    "openebs/linux-utils:3.4.0"
    "openebs/node-disk-manager:2.1.0"
    "openebs/node-disk-operator:2.1.0"
)

# Function to convert image name to tar filename
image_to_filename() {
    local image="$1"
    # Replace / with _ and : with _
    echo "${image//\//_}" | sed 's/:/_/g'
}

# Download and save images
total=${#IMAGES[@]}
count=0
success=0

for image in "${IMAGES[@]}"; do
    ((count++))
    filename=$(image_to_filename "$image")
    tar_path="$OUTPUT_DIR/${filename}.tar"
    
    echo ""
    echo "[$count/$total] $image"
    
    # Skip if already downloaded
    if [ -f "$tar_path" ]; then
        echo "  ✓ Already exists: $tar_path"
        ((success++))
        continue
    fi
    
    # Pull image
    echo "  Pulling..."
    if docker pull "$image" > /dev/null 2>&1; then
        # Save to tar
        echo "  Saving to $tar_path..."
        if docker save -o "$tar_path" "$image"; then
            echo "  ✓ Success"
            ((success++))
        else
            echo "  ✗ Failed to save"
        fi
    else
        echo "  ✗ Failed to pull"
    fi
done

echo ""
echo "=============================================="
echo "Download complete: $success/$total images"
echo "Output directory: $OUTPUT_DIR"
echo "=============================================="

# Print disk usage
du -sh "$OUTPUT_DIR"
