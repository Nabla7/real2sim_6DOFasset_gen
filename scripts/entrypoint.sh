#!/bin/bash
set -e

echo "=== Spatial Memory Service Entrypoint ==="
echo "WEIGHTS_DIR: ${WEIGHTS_DIR:-/weights}"
echo "MESH_OUTPUT_DIR: ${MESH_OUTPUT_DIR:-/tmp/spatial_memory/meshes}"

# Create necessary directories
mkdir -p /tmp/spatial_memory/meshes
mkdir -p /tmp/spatial_memory/logs

# Check if weights directory exists
if [ ! -d "${WEIGHTS_DIR:-/weights}" ]; then
    echo "WARNING: Weights directory not found at ${WEIGHTS_DIR:-/weights}"
    echo "Make sure to mount your weights volume."
fi

# Verify weights structure
if [ -d "${WEIGHTS_DIR:-/weights}/sam3" ]; then
    echo "Found SAM3 weights"
else
    echo "WARNING: SAM3 weights not found at ${WEIGHTS_DIR:-/weights}/sam3"
fi

if [ -d "${WEIGHTS_DIR:-/weights}/sam3d-objects" ]; then
    echo "Found SAM3D weights"
else
    echo "WARNING: SAM3D weights not found at ${WEIGHTS_DIR:-/weights}/sam3d-objects"
fi

if [ -d "${WEIGHTS_DIR:-/weights}/foundationpose" ]; then
    echo "Found FoundationPose weights"
else
    echo "WARNING: FoundationPose weights not found at ${WEIGHTS_DIR:-/weights}/foundationpose"
fi

echo "=== Starting services ==="

# Start supervisor
exec /usr/bin/supervisord -c /etc/supervisor/conf.d/supervisord.conf


