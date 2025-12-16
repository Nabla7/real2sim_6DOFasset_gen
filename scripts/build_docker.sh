#!/bin/bash
# Build script for production Docker image
# This builds a fully self-contained image with all dependencies

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

echo "=== Building Spatial Memory Service Docker Image ==="
echo "Project root: $PROJECT_ROOT"
echo ""

# Check for NVIDIA Docker support
if ! docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi &>/dev/null; then
    echo "ERROR: NVIDIA Docker support not available"
    echo "Install nvidia-container-toolkit: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
    exit 1
fi

echo "✓ NVIDIA Docker support detected"
echo ""

# Build the image
echo "Building Docker image (this will take 30-60 minutes)..."
echo ""

docker build \
    -f Dockerfile.production \
    -t spatial-memory-service:latest \
    -t spatial-memory-service:$(date +%Y%m%d) \
    --build-arg BUILDKIT_INLINE_CACHE=1 \
    .

echo ""
echo "=== Build complete ==="
echo ""
echo "Image tags:"
echo "  - spatial-memory-service:latest"
echo "  - spatial-memory-service:$(date +%Y%m%d)"
echo ""
echo "To run:"
echo "  docker run --gpus all -p 8080:8080 -v /path/to/weights:/weights spatial-memory-service:latest"
echo ""
echo "Or use docker-compose:"
echo "  WEIGHTS_PATH=/path/to/weights docker-compose -f docker-compose.production.yml up"

