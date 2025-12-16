#!/bin/bash
# Start all spatial memory services
export WEIGHTS_DIR=/workspace/weights

cd /workspace/dimos_hosted_services

# Gateway
conda run -n gateway --no-capture-output \
  env PYTHONPATH=/workspace/dimos_hosted_services \
  python -m gateway.main > /tmp/gateway.log 2>&1 &

# SAM3
conda run -n sam3 --no-capture-output \
  env PYTHONPATH=/workspace/dimos_hosted_services:/workspace/third_party/sam3 \
  python -m sam3_service.main > /tmp/sam3.log 2>&1 &

# SAM3D
conda run -n sam3d-objects --no-capture-output \
  env PYTHONPATH=/workspace/dimos_hosted_services:/workspace/third_party/sam-3d-objects \
      PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python -m sam3d_service.main > /tmp/sam3d.log 2>&1 &

# FoundationPose
conda run -n foundationpose --no-capture-output \
  env PYTHONPATH=/workspace/dimos_hosted_services:/workspace/third_party/FoundationPose \
  python -m foundationpose_service.main > /tmp/foundationpose.log 2>&1 &

echo "Started all services. Waiting for startup..."
sleep 10

# Health check
for port in 8080 8091 8092 8093; do
  if curl -sf http://localhost:$port/health > /dev/null; then
    echo "✓ Port $port OK"
  else
    echo "✗ Port $port FAILED"
  fi
done

