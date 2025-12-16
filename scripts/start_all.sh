#!/bin/bash
# Start all spatial memory services
set -e

export WEIGHTS_DIR=/workspace/weights
LOG_DIR=/workspace/logs

# Create log directory and clear old logs
mkdir -p "$LOG_DIR"
rm -f "$LOG_DIR"/*.log "$LOG_DIR"/*.pid

cd /workspace/dimos_hosted_services

echo "=== Starting Spatial Memory Services ==="
echo "Logs will be written to: $LOG_DIR"
echo ""

# Gateway (port 8080)
echo "Starting Gateway..."
conda run -n gateway --no-capture-output \
  env PYTHONPATH=/workspace/dimos_hosted_services \
      WEIGHTS_DIR=$WEIGHTS_DIR \
  python -m gateway.main >> "$LOG_DIR/gateway.log" 2>&1 &
echo $! > "$LOG_DIR/gateway.pid"

# SAM3 (port 8091)
echo "Starting SAM3..."
conda run -n sam3 --no-capture-output \
  env PYTHONPATH=/workspace/dimos_hosted_services:/workspace/third_party/sam3 \
      WEIGHTS_DIR=$WEIGHTS_DIR \
  python -m sam3_service.main >> "$LOG_DIR/sam3.log" 2>&1 &
echo $! > "$LOG_DIR/sam3.pid"

# SAM3D (port 8092)
echo "Starting SAM3D..."
conda run -n sam3d-objects --no-capture-output \
  env PYTHONPATH=/workspace/dimos_hosted_services:/workspace/third_party/sam-3d-objects \
      WEIGHTS_DIR=$WEIGHTS_DIR \
      PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python -m sam3d_service.main >> "$LOG_DIR/sam3d.log" 2>&1 &
echo $! > "$LOG_DIR/sam3d.pid"

# FoundationPose (port 8093)
echo "Starting FoundationPose..."
conda run -n foundationpose --no-capture-output \
  env PYTHONPATH=/workspace/dimos_hosted_services:/workspace/third_party/FoundationPose \
      WEIGHTS_DIR=$WEIGHTS_DIR \
  python -m foundationpose_service.main >> "$LOG_DIR/foundationpose.log" 2>&1 &
echo $! > "$LOG_DIR/foundationpose.pid"

echo ""
echo "All services started. Waiting for initialization..."
sleep 15

# Health check
echo ""
echo "=== Health Check ==="
declare -A SERVICES
SERVICES[8080]="Gateway"
SERVICES[8091]="SAM3"
SERVICES[8092]="SAM3D"
SERVICES[8093]="FoundationPose"

ALL_OK=true
for port in 8080 8091 8092 8093; do
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

