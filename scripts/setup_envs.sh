#!/bin/bash
# Setup script for local development (without Docker)
# Creates the conda environments with all dependencies

set -e

echo "=== Setting up Spatial Memory Service environments ==="

# Check conda is available
if ! command -v conda &> /dev/null; then
    echo "ERROR: conda not found. Please install Miniconda or Anaconda."
    exit 1
fi

# Source conda
source "$(conda info --base)/etc/profile.d/conda.sh"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVICE_DIR="$(dirname "$SCRIPT_DIR")"

echo "Service directory: $SERVICE_DIR"

# ============================================
# Gateway environment
# ============================================
echo ""
echo "=== Creating gateway environment ==="
if conda env list | grep -q "^sm-gateway "; then
    echo "Environment sm-gateway already exists. Skipping."
else
    conda create -n sm-gateway python=3.11 -y
    conda activate sm-gateway
    pip install -r "$SERVICE_DIR/requirements/gateway.txt"
    conda deactivate
    echo "Created sm-gateway environment"
fi

# ============================================
# SAM3 environment
# ============================================
echo ""
echo "=== Creating SAM3 environment ==="
if conda env list | grep -q "^sm-sam3 "; then
    echo "Environment sm-sam3 already exists. Skipping."
else
    conda create -n sm-sam3 python=3.11 -y
    conda activate sm-sam3
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    pip install -r "$SERVICE_DIR/requirements/sam3.txt"
    conda deactivate
    echo "Created sm-sam3 environment"
fi

# ============================================
# SAM3D environment
# ============================================
echo ""
echo "=== Creating SAM3D environment ==="
if conda env list | grep -q "^sm-sam3d "; then
    echo "Environment sm-sam3d already exists. Skipping."
else
    conda create -n sm-sam3d python=3.11 -y
    conda activate sm-sam3d
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    pip install -r "$SERVICE_DIR/requirements/sam3d.txt"
    conda deactivate
    echo "Created sm-sam3d environment"
fi

# ============================================
# FoundationPose environment
# ============================================
echo ""
echo "=== Creating FoundationPose environment ==="
if conda env list | grep -q "^sm-foundationpose "; then
    echo "Environment sm-foundationpose already exists. Skipping."
else
    conda create -n sm-foundationpose python=3.9 -y
    conda activate sm-foundationpose
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    pip install -r "$SERVICE_DIR/requirements/foundationpose.txt"
    conda deactivate
    echo "Created sm-foundationpose environment"
fi

echo ""
echo "=== Setup complete ==="
echo ""
echo "To run services locally:"
echo "  Gateway:        conda activate sm-gateway && python -m gateway.main"
echo "  SAM3:           conda activate sm-sam3 && python -m sam3_service.main"
echo "  SAM3D:          conda activate sm-sam3d && python -m sam3d_service.main"
echo "  FoundationPose: conda activate sm-foundationpose && python -m foundationpose_service.main"


