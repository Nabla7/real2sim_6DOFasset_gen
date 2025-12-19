#!/bin/bash
set -e

echo "=== Spatial Memory Service Entrypoint ==="

# Copy FoundationPose weights to expected location
if [ -d "/weights/foundationpose" ]; then
    echo "Copying FoundationPose weights to /app/third_party/FoundationPose/weights/..."
    mkdir -p /app/third_party/FoundationPose/weights
    cp -r /weights/foundationpose/* /app/third_party/FoundationPose/weights/
    echo "FoundationPose weights copied successfully"
else
    echo "WARNING: /weights/foundationpose not found - FoundationPose may fail to initialize"
fi

# Clean mesh directory
echo "Cleaning mesh output directory..."
rm -rf /tmp/spatial_memory/meshes/*
mkdir -p /tmp/spatial_memory/meshes

# Clean log directory
echo "Cleaning log directory..."
rm -f /tmp/spatial_memory/logs/*.log /tmp/spatial_memory/logs/*.pid
mkdir -p /tmp/spatial_memory/logs

echo "=== Starting services ==="
exec "$@"

