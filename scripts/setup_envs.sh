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
WORKSPACE_DIR="$(dirname "$SERVICE_DIR")"
THIRD_PARTY_DIR="$WORKSPACE_DIR/third_party"

echo "Service directory: $SERVICE_DIR"
echo "Third-party directory: $THIRD_PARTY_DIR"

# ============================================
# Clone third-party repositories if needed
# ============================================
echo ""
echo "=== Checking third-party repositories ==="
mkdir -p "$THIRD_PARTY_DIR"

if [ ! -d "$THIRD_PARTY_DIR/sam3" ]; then
    echo "Cloning SAM3..."
    git clone https://github.com/facebookresearch/sam3.git "$THIRD_PARTY_DIR/sam3"
fi

if [ ! -d "$THIRD_PARTY_DIR/sam-3d-objects" ]; then
    echo "Cloning SAM3D..."
    git clone https://github.com/facebookresearch/sam-3d-objects.git "$THIRD_PARTY_DIR/sam-3d-objects"
fi

if [ ! -d "$THIRD_PARTY_DIR/FoundationPose" ]; then
    echo "Cloning FoundationPose..."
    git clone https://github.com/NVlabs/FoundationPose.git "$THIRD_PARTY_DIR/FoundationPose"
fi

# ============================================
# Gateway environment
# ============================================
echo ""
echo "=== Creating gateway environment ==="
if conda env list | grep -q "^gateway "; then
    echo "Environment 'gateway' already exists. Skipping."
else
    conda create -n gateway python=3.11 -y
    conda activate gateway
    pip install -r "$SERVICE_DIR/requirements/gateway.txt"
    conda deactivate
    echo "✅ Created gateway environment"
fi

# ============================================
# SAM3 environment
# ============================================
echo ""
echo "=== Creating SAM3 environment ==="
if conda env list | grep -q "^sam3 "; then
    echo "Environment 'sam3' already exists. Skipping."
else
    conda create -n sam3 python=3.11 -y
    conda activate sam3
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    pip install -r "$SERVICE_DIR/requirements/sam3.txt"
    
    # Install SAM3 package
    cd "$THIRD_PARTY_DIR/sam3"
    pip install -e .
    
    conda deactivate
    echo "✅ Created sam3 environment"
fi

# ============================================
# SAM3D environment (COMPLEX!)
# ============================================
echo ""
echo "=== Creating SAM3D environment ==="
if conda env list | grep -q "^sam3d-objects "; then
    echo "Environment 'sam3d-objects' already exists. Skipping."
else
    echo "This will take 10-15 minutes due to complex dependencies..."
    
    cd "$THIRD_PARTY_DIR/sam-3d-objects"
    
    # Create from their environment file
    conda env create -f environments/default.yml
    
    conda activate sam3d-objects
    
    # Install with special pip indexes
    export PIP_EXTRA_INDEX_URL="https://pypi.ngc.nvidia.com https://download.pytorch.org/whl/cu121"
    pip install -e '.[dev]'
    pip install -e '.[p3d]'
    
    # Install Kaolin
    export PIP_FIND_LINKS="https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html"
    pip install -e '.[inference]'
    
    # Apply hydra patch
    chmod +x ./patching/hydra
    ./patching/hydra
    
    # Install service dependencies
    pip install fastapi uvicorn pydantic
    
    conda deactivate
    echo "✅ Created sam3d-objects environment"
fi

# ============================================
# FoundationPose environment
# ============================================
echo ""
echo "=== Creating FoundationPose environment ==="
if conda env list | grep -q "^foundationpose "; then
    echo "Environment 'foundationpose' already exists. Skipping."
else
    echo "This will take 5-10 minutes due to C++ compilation..."
    
    conda create -n foundationpose python=3.9 -y
    conda activate foundationpose
    
    # Install Eigen (required for C++)
    conda install -c conda-forge eigen=3.4.0 -y
    
    # Install PyTorch (CUDA 11.8 for FoundationPose)
    pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 --index-url https://download.pytorch.org/whl/cu118
    
    # Install FoundationPose requirements
    cd "$THIRD_PARTY_DIR/FoundationPose"
    pip install -r requirements.txt
    pip install fastapi uvicorn pydantic
    
    # Build C++ extensions
    echo "Building C++ extensions..."
    cd mycpp
    mkdir -p build
    cd build
    cmake ..
    make
    
    cd "$THIRD_PARTY_DIR/FoundationPose/bundlesdf/mycuda"
    python setup.py install
    
    conda deactivate
    echo "✅ Created foundationpose environment"
fi

# ============================================
# Copy weights for FoundationPose
# ============================================
echo ""
echo "=== Checking FoundationPose weights ==="
if [ -d "$WORKSPACE_DIR/weights/foundationpose" ]; then
    echo "Copying FoundationPose weights..."
    mkdir -p "$THIRD_PARTY_DIR/FoundationPose/weights"
    cp -r "$WORKSPACE_DIR/weights/foundationpose/"* "$THIRD_PARTY_DIR/FoundationPose/weights/"
    echo "✅ Weights copied"
else
    echo "⚠️  WARNING: $WORKSPACE_DIR/weights/foundationpose not found"
    echo "   FoundationPose will fail without weights!"
fi

echo ""
echo "=== Setup complete ==="
echo ""
echo "Environment names (match Docker):"
echo "  • gateway"
echo "  • sam3"
echo "  • sam3d-objects"
echo "  • foundationpose"
echo ""
echo "To run services locally, use scripts/start_all.sh"
echo "Or manually:"
echo "  Gateway:        conda activate gateway && python -m gateway.main"
echo "  SAM3:           conda activate sam3 && python -m sam3_service.main"
echo "  SAM3D:          conda activate sam3d-objects && python -m sam3d_service.main"
echo "  FoundationPose: conda activate foundationpose && python -m foundationpose_service.main"
echo ""
echo "Make sure to set environment variables:"
echo "  export WEIGHTS_DIR=$WORKSPACE_DIR/weights"
echo "  export SAM3D_PATH=$THIRD_PARTY_DIR/sam-3d-objects"
echo "  export FOUNDATIONPOSE_PATH=$THIRD_PARTY_DIR/FoundationPose"