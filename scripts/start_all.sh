#!/bin/bash
# Start all spatial memory services
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

# Defaults are repo-local; override via env if desired.
export WEIGHTS_DIR="${WEIGHTS_DIR:-$REPO_DIR/weights}"
export THIRD_PARTY_DIR="${THIRD_PARTY_DIR:-$REPO_DIR/third_party}"
export LOG_DIR="${LOG_DIR:-$REPO_DIR/logs}"
export MESH_OUTPUT_DIR="${MESH_OUTPUT_DIR:-$REPO_DIR/spatial_memory/meshes}"

LOG_DIR="$LOG_DIR"
MESH_DIR="$MESH_OUTPUT_DIR"

# Create directories and clear old data
mkdir -p "$LOG_DIR" "$MESH_DIR"
rm -f "$LOG_DIR"/*.log "$LOG_DIR"/*.pid
rm -rf "$MESH_DIR"/*

cd "$REPO_DIR"

echo "=== Starting Spatial Memory Services ==="
echo "Logs will be written to: $LOG_DIR"
echo "Meshes will be saved to: $MESH_DIR"
echo ""

# Gateway (port 8080)
echo "Starting Gateway..."
conda run -n gateway --no-capture-output \
  env PYTHONPATH="$REPO_DIR" \
      WEIGHTS_DIR=$WEIGHTS_DIR \
      MESH_OUTPUT_DIR=$MESH_DIR \
  python -m gateway.main >> "$LOG_DIR/gateway.log" 2>&1 &
echo $! > "$LOG_DIR/gateway.pid"

# SAM3 (port 8091)
echo "Starting SAM3..."
conda run -n sam3 --no-capture-output \
  env PYTHONPATH="$REPO_DIR:$THIRD_PARTY_DIR/sam3" \
      WEIGHTS_DIR=$WEIGHTS_DIR \
      LOG_DIR=$LOG_DIR \
  python -m sam3_service.main >> "$LOG_DIR/sam3.log" 2>&1 &
echo $! > "$LOG_DIR/sam3.pid"

# SAM3D (port 8092)
echo "Starting SAM3D..."
conda run -n sam3d-objects --no-capture-output \
  env PYTHONPATH="$REPO_DIR:$THIRD_PARTY_DIR/sam-3d-objects" \
      WEIGHTS_DIR=$WEIGHTS_DIR \
      SAM3D_PATH="$THIRD_PARTY_DIR/sam-3d-objects" \
      MESH_OUTPUT_DIR=$MESH_DIR \
      LOG_DIR=$LOG_DIR \
      PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python -m sam3d_service.main >> "$LOG_DIR/sam3d.log" 2>&1 &
echo $! > "$LOG_DIR/sam3d.pid"

# FoundationPose (port 8093)
echo "Starting FoundationPose..."
conda run -n foundationpose --no-capture-output \
  env PYTHONPATH="$REPO_DIR:$THIRD_PARTY_DIR/FoundationPose" \
      WEIGHTS_DIR=$WEIGHTS_DIR \
      FOUNDATIONPOSE_PATH="$THIRD_PARTY_DIR/FoundationPose" \
      LOG_DIR=$LOG_DIR \
  python -m foundationpose_service.main >> "$LOG_DIR/foundationpose.log" 2>&1 &
echo $! > "$LOG_DIR/foundationpose.pid"

# GraspGen (port 8094)
echo "Starting GraspGen..."
conda run -n GraspGen --no-capture-output \
  env PYTHONPATH="$REPO_DIR:$THIRD_PARTY_DIR/GraspGen" \
      GRASPGEN_PATH="$THIRD_PARTY_DIR/GraspGen" \
      LOG_DIR=$LOG_DIR \
      DEBUG_DIR="${DEBUG_DIR:-$REPO_DIR/debug}" \
      PYOPENGL_PLATFORM=egl \
  python -m graspgen_service.main >> "$LOG_DIR/graspgen.log" 2>&1 &
echo $! > "$LOG_DIR/graspgen.pid"

echo "Starting Scene Capture..."
conda run -n gateway --no-capture-output \
  env PYTHONPATH="$REPO_DIR" \
      LOG_DIR=$LOG_DIR \
  python -m scene_capture_service.main >> "$LOG_DIR/scene_capture.log" 2>&1 &
echo $! > "$LOG_DIR/scene_capture.pid"

echo "Starting COLMAP Service..."
conda run -n gateway --no-capture-output \
  env PYTHONPATH="$REPO_DIR" \
      LOG_DIR=$LOG_DIR \
  python -m colmap_service.main >> "$LOG_DIR/colmap.log" 2>&1 &
echo $! > "$LOG_DIR/colmap.pid"

echo "Starting Neural Recon Service..."
conda run -n gateway --no-capture-output \
  env PYTHONPATH="$REPO_DIR" \
      LOG_DIR=$LOG_DIR \
  python -m neural_recon_service.main >> "$LOG_DIR/neural_recon.log" 2>&1 &
echo $! > "$LOG_DIR/neural_recon.pid"

echo ""
echo "All services started. Waiting for initialization..."
sleep 20

# Health check
echo ""
echo "=== Health Check ==="
declare -A SERVICES
SERVICES[8080]="Gateway"
SERVICES[8091]="SAM3"
SERVICES[8092]="SAM3D"
SERVICES[8093]="FoundationPose"
SERVICES[8094]="GraspGen"
SERVICES[8095]="SceneCapture"
SERVICES[8096]="COLMAP"
SERVICES[8097]="NeuralRecon"

ALL_OK=true
for port in 8080 8091 8092 8093 8094 8095 8096 8097; do
  if curl -sf http://localhost:$port/health > /dev/null 2>&1; then
    echo "✓ ${SERVICES[$port]} (port $port) - OK"
  else
    echo "✗ ${SERVICES[$port]} (port $port) - FAILED"
    ALL_OK=false
  fi
done

echo ""
if [ "$ALL_OK" = true ]; then
  echo "=== All services running! ==="
else
  echo "=== Some services failed to start. Check logs: ==="
  echo "  tail -f $LOG_DIR/gateway.log"
  echo "  tail -f $LOG_DIR/sam3.log"
  echo "  tail -f $LOG_DIR/sam3d.log"
  echo "  tail -f $LOG_DIR/foundationpose.log"
fi

