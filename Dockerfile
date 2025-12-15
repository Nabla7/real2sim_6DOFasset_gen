# Spatial Memory Service Docker Image
# Contains SAM3, SAM3D, and FoundationPose with separate conda environments
#
# Build:
#   docker build -t spatial-memory-service .
#
# Run:
#   docker run --gpus all -p 8080:8080 -v /path/to/weights:/weights spatial-memory-service

FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    git \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    supervisor \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
ENV CONDA_DIR=/opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p $CONDA_DIR && \
    rm ~/miniconda.sh
ENV PATH=$CONDA_DIR/bin:$PATH

# Initialize conda for shell
RUN conda init bash

# Create app directory
WORKDIR /app

# Copy requirements
COPY requirements/ /app/requirements/

# ============================================
# Environment 1: Gateway (Python 3.11, CPU)
# ============================================
RUN conda create -n gateway python=3.11 -y && \
    conda run -n gateway pip install --no-cache-dir -r /app/requirements/gateway.txt

# ============================================
# Environment 2: SAM3 (Python 3.11, CUDA)
# ============================================
RUN conda create -n sam3 python=3.11 -y && \
    conda run -n sam3 pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu121 && \
    conda run -n sam3 pip install --no-cache-dir -r /app/requirements/sam3.txt

# ============================================
# Environment 3: SAM3D (Python 3.11, CUDA)
# ============================================
RUN conda create -n sam3d python=3.11 -y && \
    conda run -n sam3d pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu121 && \
    conda run -n sam3d pip install --no-cache-dir -r /app/requirements/sam3d.txt

# ============================================
# Environment 4: FoundationPose (Python 3.9, CUDA)
# ============================================
RUN conda create -n foundationpose python=3.9 -y && \
    conda run -n foundationpose pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu121 && \
    conda run -n foundationpose pip install --no-cache-dir -r /app/requirements/foundationpose.txt

# Copy application code
COPY gateway/ /app/gateway/
COPY sam3_service/ /app/sam3_service/
COPY sam3d_service/ /app/sam3d_service/
COPY foundationpose_service/ /app/foundationpose_service/
COPY scripts/ /app/scripts/

# Copy third-party code (will be mounted or baked in)
# These paths should match where the weights expect to find the code
RUN mkdir -p /app/third_party

# Make scripts executable
RUN chmod +x /app/scripts/*.sh

# Create directories for outputs and logs
RUN mkdir -p /tmp/spatial_memory/meshes /tmp/spatial_memory/logs

# Environment variables
ENV WEIGHTS_DIR=/weights
ENV MESH_OUTPUT_DIR=/tmp/spatial_memory/meshes
ENV PYTHONPATH=/app:/app/third_party

# Supervisor configuration
COPY scripts/supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Expose ports
EXPOSE 8080 8091 8092 8093

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Start all services
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]


