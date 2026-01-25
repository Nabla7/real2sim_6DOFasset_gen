#!/bin/bash
# Setup script for local development (without Docker)
# Creates the conda environments with all dependencies
#
# Architecture support:
#   - x86_64 (linux-64): Full support (all environments)
#   - aarch64 (linux-aarch64): Partial support
#       • gateway: ✅ Works
#       • sam3: ✅ Works
#       • sam3d-objects: ❌ Skipped (x86_64-only dependency spec)
#       • foundationpose: ⚠️  May work with adjustments
#       • GraspGen: ⚠️  May work with adjustments
#
# For ARM/aarch64 deployments:
#   Recommended: Use Docker on x86_64 hardware, or run SAM3D remotely and set SAM3D_URL

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
REPO_DIR="$SERVICE_DIR"

# Default locations are repo-local to keep the workspace self-contained.
# You can override them by exporting THIRD_PARTY_DIR / WEIGHTS_DIR before running.
THIRD_PARTY_DIR_DEFAULT="$REPO_DIR/third_party"
WEIGHTS_DIR_DEFAULT="$REPO_DIR/weights"

# Detect whether caller explicitly set THIRD_PARTY_DIR (even if empty)
THIRD_PARTY_DIR_ENV_SET=0
if [ -n "${THIRD_PARTY_DIR+x}" ]; then
    THIRD_PARTY_DIR_ENV_SET=1
fi

THIRD_PARTY_DIR="${THIRD_PARTY_DIR:-$THIRD_PARTY_DIR_DEFAULT}"
WEIGHTS_DIR="${WEIGHTS_DIR:-$WEIGHTS_DIR_DEFAULT}"

# Backwards-compat migration: older versions used a sibling directory (../third_party).
# If the user didn't override THIRD_PARTY_DIR and the old folder exists, move it in-repo.
OLD_THIRD_PARTY_DIR="$(dirname "$REPO_DIR")/third_party"
if [ "$THIRD_PARTY_DIR_ENV_SET" -eq 0 ] && [ -d "$OLD_THIRD_PARTY_DIR" ] && [ ! -d "$THIRD_PARTY_DIR" ]; then
    echo ""
    echo "=== Migrating third-party directory into repo ==="
    echo "Moving: $OLD_THIRD_PARTY_DIR -> $THIRD_PARTY_DIR"
    mv "$OLD_THIRD_PARTY_DIR" "$THIRD_PARTY_DIR"
fi

# Detect architecture
ARCH="$(uname -m)"

echo "Service directory: $SERVICE_DIR"
echo "Third-party directory: $THIRD_PARTY_DIR"
echo "Architecture: $ARCH"

# Warn about ARM limitations
if [ "$ARCH" != "x86_64" ]; then
    echo ""
    echo "⚠️  WARNING: Detected non-x86_64 architecture ($ARCH)"
    echo "   Some environments (SAM3D) are not available for $ARCH and will be skipped."
    echo "   For full functionality, use Docker on an x86_64 machine or run SAM3D remotely."
    echo ""
fi

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

if [ ! -d "$THIRD_PARTY_DIR/GraspGen" ]; then
    echo "Cloning GraspGen..."
    git clone https://github.com/NVlabs/GraspGen.git "$THIRD_PARTY_DIR/GraspGen"
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
    
    # Install SAM3 package (brings all ML dependencies)
    cd "$THIRD_PARTY_DIR/sam3"
    pip install -e .
    
    # Add server wrapper dependencies
    pip install -r "$SERVICE_DIR/requirements/sam3.txt"
    
    conda deactivate
    echo "✅ Created sam3 environment"
fi

# ============================================
# SAM3D environment (COMPLEX!)
# ============================================
echo ""
echo "=== Creating SAM3D environment ==="

# SAM3D's environment spec (environments/default.yml) pins linux-64 (x86_64) packages
# and cannot be installed on linux-aarch64. Skip on ARM architectures.
if [ "$ARCH" != "x86_64" ]; then
    echo "⚠️  SKIPPING SAM3D on $ARCH architecture"
    echo "   SAM3D environment is x86_64-only (contains linux-64 package pins)."
    echo "   Options:"
    echo "     1. Run SAM3D in Docker on an x86_64 machine and point SAM3D_URL to it"
    echo "     2. Use the full Docker deployment (all services together)"
    echo ""
elif conda env list | grep -q "^sam3d-objects "; then
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
    
    # Install FoundationPose requirements (all ML dependencies)
    cd "$THIRD_PARTY_DIR/FoundationPose"
    pip install -r requirements.txt
    
    # Add server wrapper dependencies
    pip install -r "$SERVICE_DIR/requirements/foundationpose.txt"
    
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
# GraspGen environment
# ============================================
echo ""
echo "=== Creating GraspGen environment ==="
if conda env list | grep -q "^GraspGen "; then
    echo "Environment 'GraspGen' already exists. Skipping."
else
    echo "This will take 5-10 minutes due to C++ compilation..."
    
    conda create -n GraspGen python=3.10 -y
    conda activate GraspGen
    
    # Set CUDA paths for compilation
    export CUDA_HOME=/usr/local/cuda-12.1
    export PATH=/usr/local/cuda-12.1/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH
    
    # Install PyTorch with CUDA 12.1
    pip install torch==2.1.0 torchvision==0.16.0 torch-cluster \
        -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
    
    # Install GraspGen package (brings all ML dependencies from requirements.txt)
    cd "$THIRD_PARTY_DIR/GraspGen"
    pip install -e .
    
    # Build PointNet++ C++ extensions
    echo "Building PointNet++ C++ extensions..."
    cd pointnet2_ops
    pip install --no-build-isolation .
    
    # Install torch-scatter (not in GraspGen requirements.txt)
    pip install torch-scatter -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
    
    # Add server wrapper dependencies
    pip install -r "$SERVICE_DIR/requirements/graspgen.txt"
    
    # Set environment variable for offscreen rendering
    echo "export PYOPENGL_PLATFORM=egl" >> ~/.bashrc
    
    conda deactivate
    echo "✅ Created GraspGen environment"
fi

# ============================================
# Copy weights for FoundationPose
# ============================================
echo ""
echo "=== Checking FoundationPose weights ==="
if [ -d "$WEIGHTS_DIR/foundationpose" ]; then
    echo "Copying FoundationPose weights..."
    mkdir -p "$THIRD_PARTY_DIR/FoundationPose/weights"
    cp -r "$WEIGHTS_DIR/foundationpose/"* "$THIRD_PARTY_DIR/FoundationPose/weights/"
    echo "✅ Weights copied"
else
    echo "⚠️  WARNING: $WEIGHTS_DIR/foundationpose not found"
    echo "   FoundationPose will fail without weights!"
fi

echo ""
echo "=== Setup complete ==="
echo ""
echo "Created environments:"
echo "  ✅ gateway"
echo "  ✅ sam3"
if conda env list | grep -q "^sam3d-objects "; then
    echo "  ✅ sam3d-objects"
else
    echo "  ⚠️  sam3d-objects (skipped on $ARCH)"
fi
if conda env list | grep -q "^foundationpose "; then
    echo "  ✅ foundationpose"
else
    echo "  ⚠️  foundationpose (not created)"
fi
if conda env list | grep -q "^GraspGen "; then
    echo "  ✅ GraspGen"
else
    echo "  ⚠️  GraspGen (not created)"
fi
echo ""

if [ "$ARCH" != "x86_64" ]; then
    echo "⚠️  Architecture: $ARCH"
    echo "   This architecture has limited support. For full functionality:"
    echo "     • Use Docker on x86_64 hardware (recommended)"
    echo "     • Run SAM3D service separately on x86_64 and set SAM3D_URL"
    echo ""
fi

echo "To run services locally, use scripts/start_all.sh"
echo "Or manually:"
echo "  Gateway:        conda activate gateway && python -m gateway.main"
echo "  SAM3:           conda activate sam3 && python -m sam3_service.main"
if conda env list | grep -q "^sam3d-objects "; then
    echo "  SAM3D:          conda activate sam3d-objects && python -m sam3d_service.main"
fi
if conda env list | grep -q "^foundationpose "; then
    echo "  FoundationPose: conda activate foundationpose && python -m foundationpose_service.main"
fi
if conda env list | grep -q "^GraspGen "; then
    echo "  GraspGen:       conda activate GraspGen && cd $THIRD_PARTY_DIR/GraspGen && python scripts/demo_object_mesh.py --help"
fi
echo ""
echo "Make sure to set environment variables:"
echo "  export WEIGHTS_DIR=$WEIGHTS_DIR"
echo "  export SAM3D_PATH=$THIRD_PARTY_DIR/sam-3d-objects"
echo "  export FOUNDATIONPOSE_PATH=$THIRD_PARTY_DIR/FoundationPose"
echo "  export PYOPENGL_PLATFORM=egl  # For GraspGen offscreen rendering"