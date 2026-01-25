#!/bin/bash
#
# Build COLMAP from source (Ubuntu 22.04-friendly) with optional CUDA support.
#
# This script is intentionally conservative:
# - It does NOT auto-install system deps unless INSTALL_DEPS=1 is set.
# - It installs to a repo-local prefix by default (no sudo needed).
#
# Usage:
#   ./scripts/build_colmap.sh
#
# Optional env vars:
#   INSTALL_DEPS=1                  # run apt-get install for dependencies (requires sudo)
#   USE_GCC10=1                     # export CC/CXX/CUDAHOSTCXX to GCC-10 (recommended on Ubuntu 22.04)
#   ENABLE_CUDA=1                   # force CUDA ON (requires nvcc); default auto
#   CMAKE_CUDA_ARCHITECTURES=native # or "all", "all-major", "75", etc.
#   COLMAP_INSTALL_PREFIX=...       # install prefix (default: ./.local/colmap)
#   COLMAP_BUILD_DIR=...            # build dir (default: third_party/colmap/build)
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
COLMAP_SRC_DIR="${COLMAP_SRC_DIR:-$REPO_DIR/third_party/colmap}"
COLMAP_BUILD_DIR="${COLMAP_BUILD_DIR:-$COLMAP_SRC_DIR/build}"
COLMAP_INSTALL_PREFIX="${COLMAP_INSTALL_PREFIX:-$REPO_DIR/.local/colmap}"

INSTALL_DEPS="${INSTALL_DEPS:-0}"
USE_GCC10="${USE_GCC10:-1}"
ENABLE_CUDA="${ENABLE_CUDA:-auto}"
CMAKE_CUDA_ARCHITECTURES="${CMAKE_CUDA_ARCHITECTURES:-native}"

echo "=== Building COLMAP from source ==="
echo "Repo:          $REPO_DIR"
echo "Source:        $COLMAP_SRC_DIR"
echo "Build dir:     $COLMAP_BUILD_DIR"
echo "Install prefix:$COLMAP_INSTALL_PREFIX"

if [ ! -d "$COLMAP_SRC_DIR/.git" ]; then
  echo "ERROR: COLMAP source not found at $COLMAP_SRC_DIR"
  echo "Run: git clone https://github.com/colmap/colmap.git $COLMAP_SRC_DIR"
  exit 1
fi

if [ "$INSTALL_DEPS" = "1" ]; then
  echo ""
  echo "=== Installing system dependencies (Ubuntu 22.04) ==="
  sudo apt-get update
  sudo apt-get install -y \
    git cmake ninja-build build-essential \
    libboost-program-options-dev libboost-graph-dev libboost-system-dev \
    libeigen3-dev libfreeimage-dev libmetis-dev \
    libgoogle-glog-dev libgtest-dev libgmock-dev libsqlite3-dev \
    libglew-dev qt6-base-dev libqt6opengl6-dev libqt6openglwidgets6 \
    libcgal-dev libceres-dev \
    libcurl4-openssl-dev libssl-dev \
    libopenimageio-dev openimageio-tools

  if [ "${USE_GCC10:-1}" = "1" ]; then
    echo ""
    echo "Installing GCC-10 toolchain (recommended on Ubuntu 22.04 for CUDA builds)..."
    sudo apt-get install -y gcc-10 g++-10
  fi

  echo ""
  echo "NOTE: If you want MKL BLAS, install it separately (optional):"
  echo "  sudo apt-get install -y libmkl-full-dev"
  echo ""
  echo "NOTE: For CUDA support, prefer a proper CUDA toolkit (nvcc) from NVIDIA."
  echo "If you rely on Ubuntu's toolkit:"
  echo "  sudo apt-get install -y nvidia-cuda-toolkit nvidia-cuda-toolkit-gcc"
fi

if [ "$USE_GCC10" = "1" ]; then
  echo ""
  echo "=== Using GCC-10 toolchain (recommended on Ubuntu 22.04 when building with CUDA) ==="
  if ! command -v gcc-10 >/dev/null 2>&1; then
    echo "GCC-10 not found. Install it with:"
    echo "  sudo apt-get install -y gcc-10 g++-10"
    exit 1
  fi
  export CC=/usr/bin/gcc-10
  export CXX=/usr/bin/g++-10
  export CUDAHOSTCXX=/usr/bin/g++-10
  echo "CC=$CC"
  echo "CXX=$CXX"
  echo "CUDAHOSTCXX=$CUDAHOSTCXX"
fi

# --- CUDA toolkit discovery (prefer system-installed /usr/local/cuda*) ---
CUDA_HOME="${CUDA_HOME:-}"
if [ -z "$CUDA_HOME" ]; then
  if [ -d "/usr/local/cuda" ]; then
    CUDA_HOME="/usr/local/cuda"
  elif [ -d "/usr/local/cuda-12.8" ]; then
    CUDA_HOME="/usr/local/cuda-12.8"
  fi
fi

NVCC_CANDIDATE=""
if command -v nvcc >/dev/null 2>&1; then
  NVCC_CANDIDATE="$(command -v nvcc)"
elif [ -n "$CUDA_HOME" ] && [ -x "$CUDA_HOME/bin/nvcc" ]; then
  NVCC_CANDIDATE="$CUDA_HOME/bin/nvcc"
  export PATH="$CUDA_HOME/bin:$PATH"
fi

if [ -n "$CUDA_HOME" ] && [ -d "$CUDA_HOME/lib64" ]; then
  export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
fi

CUDA_ON=OFF
if [ "$ENABLE_CUDA" = "1" ]; then
  if [ -z "$NVCC_CANDIDATE" ]; then
    echo "ERROR: ENABLE_CUDA=1 but nvcc was not found."
    echo "Hint: nvcc is typically at /usr/local/cuda/bin/nvcc and requires CUDA toolkit installed."
    exit 1
  fi
  CUDA_ON=ON
elif [ "$ENABLE_CUDA" = "0" ]; then
  CUDA_ON=OFF
else
  if [ -n "$NVCC_CANDIDATE" ]; then
    CUDA_ON=ON
  fi
fi

# If user left CMAKE_CUDA_ARCHITECTURES= "native" but the GPU is very new,
# older CMake versions can produce an empty arch ("compute_") and nvcc fails.
# We set a sane default for known GPUs unless the user explicitly overrides.
if [ "$CUDA_ON" = "ON" ] && [ "${CMAKE_CUDA_ARCHITECTURES}" = "native" ]; then
  GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n 1 || true)"
  case "$GPU_NAME" in
    *"RTX 5090"*|*"GeForce RTX 5090"*|*"5090"*)
      CMAKE_CUDA_ARCHITECTURES="120"
      ;;
    *"RTX 4090"*|*"GeForce RTX 4090"*|*"4090"*|*"RTX 4080"*|*"4080"*)
      CMAKE_CUDA_ARCHITECTURES="89"
      ;;
    *"A100"*|*"A800"*)
      CMAKE_CUDA_ARCHITECTURES="80"
      ;;
    *"H100"*|*"H800"*)
      CMAKE_CUDA_ARCHITECTURES="90"
      ;;
    *)
      # keep "native"
      ;;
  esac
fi

echo ""
echo "=== Configuring (CUDA=$CUDA_ON) ==="
echo "CMAKE_CUDA_ARCHITECTURES=$CMAKE_CUDA_ARCHITECTURES"
mkdir -p "$COLMAP_BUILD_DIR"
cd "$COLMAP_BUILD_DIR"

cmake .. -GNinja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$COLMAP_INSTALL_PREFIX" \
  -DCUDA_ENABLED="$CUDA_ON" \
  -DCMAKE_CUDA_ARCHITECTURES="$CMAKE_CUDA_ARCHITECTURES" \
  ${NVCC_CANDIDATE:+-DCMAKE_CUDA_COMPILER="$NVCC_CANDIDATE"} \
  ${CUDA_HOME:+-DCUDAToolkit_ROOT="$CUDA_HOME"}

echo ""
echo "=== Building ==="
ninja

echo ""
echo "=== Installing to $COLMAP_INSTALL_PREFIX ==="
ninja install

echo ""
echo "=== Done ==="
echo "Try:"
echo "  $COLMAP_INSTALL_PREFIX/bin/colmap -h"
echo "  $COLMAP_INSTALL_PREFIX/bin/colmap gui"

