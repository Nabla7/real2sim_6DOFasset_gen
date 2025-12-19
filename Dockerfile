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

# ============================================
# Clone Third-Party Repositories
# ============================================
WORKDIR /app/third_party

RUN git clone https://github.com/facebookresearch/sam3.git && \
    cd sam3 && \
    git checkout main

RUN git clone https://github.com/facebookresearch/sam-3d-objects.git && \
    cd sam-3d-objects && \
    git checkout main

RUN git clone https://github.com/NVlabs/FoundationPose.git && \
    cd FoundationPose && \
    git checkout main

RUN git clone https://github.com/NVlabs/GraspGen.git && \
    cd GraspGen && \
    git checkout main

# ============================================
# Copy application code and requirements
# ============================================
WORKDIR /app
COPY requirements/ /app/requirements/
COPY gateway/ /app/gateway/
COPY sam3_service/ /app/sam3_service/
COPY sam3d_service/ /app/sam3d_service/
COPY foundationpose_service/ /app/foundationpose_service/
COPY scripts/ /app/scripts/

RUN chmod +x /app/scripts/*.sh

# ============================================
# Environment 1: Gateway (Python 3.11, CPU)
# ============================================
RUN conda create -n gateway python=3.11 -y && \
    conda run -n gateway pip install --no-cache-dir -r /app/requirements/gateway.txt

# ============================================
# Environment 2: SAM3 (Python 3.11, CUDA)
# ============================================
RUN conda create -n sam3 python=3.11 -y && \
    conda run -n sam3 pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu121

WORKDIR /app/third_party/sam3
RUN conda run -n sam3 pip install --no-cache-dir -e .

# Add server wrapper dependencies
RUN conda run -n sam3 pip install --no-cache-dir -r /app/requirements/sam3.txt

# ============================================
# Environment 3: SAM3D (Python 3.11, Complex)
# ============================================
WORKDIR /app/third_party/sam-3d-objects
RUN conda env create -f environments/default.yml && \
    conda clean -afy

RUN conda run -n sam3d-objects bash -c "\
    export PIP_EXTRA_INDEX_URL='https://pypi.ngc.nvidia.com https://download.pytorch.org/whl/cu121' && \
    pip install --no-cache-dir -e '.[dev]' && \
    pip install --no-cache-dir -e '.[p3d]' && \
    export PIP_FIND_LINKS='https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html' && \
    pip install --no-cache-dir -e '.[inference]' && \
    chmod +x ./patching/hydra && ./patching/hydra && \
    pip install --no-cache-dir fastapi uvicorn pydantic"

# ============================================
# Environment 4: FoundationPose (Python 3.9)
# ============================================
RUN conda create -n foundationpose python=3.9 -y && \
    conda install -n foundationpose -c conda-forge eigen=3.4.0 -y && \
    conda clean -afy

WORKDIR /app/third_party/FoundationPose
RUN conda run -n foundationpose pip install --no-cache-dir \
    torch==2.0.0+cu118 torchvision==0.15.1+cu118 --index-url https://download.pytorch.org/whl/cu118

RUN conda run -n foundationpose pip install --no-cache-dir -r requirements.txt && \
    conda run -n foundationpose pip install --no-cache-dir fastapi uvicorn pydantic

# Build C++ extensions
RUN conda run -n foundationpose bash -c "cd mycpp && mkdir -p build && cd build && cmake .. && make"
RUN conda run -n foundationpose bash -c "cd bundlesdf/mycuda && python setup.py install"

# ============================================
# Environment 5: GraspGen (Python 3.10, CUDA 12.1)
# ============================================
RUN conda create -n GraspGen python=3.10 -y && \
    conda clean -afy

WORKDIR /app/third_party/GraspGen

# Install PyTorch with CUDA 12.1
RUN conda run -n GraspGen pip install --no-cache-dir \
    torch==2.1.0 torchvision==0.16.0 torch-cluster \
    -f https://data.pyg.org/whl/torch-2.1.0+cu121.html

# Install GraspGen package (includes all ML dependencies from requirements.txt)
RUN conda run -n GraspGen pip install --no-cache-dir -e .

# Build PointNet++ C++ extensions
RUN conda run -n GraspGen bash -c "cd pointnet2_ops && pip install --no-build-isolation ."

# Install torch-scatter (not in GraspGen requirements.txt)
RUN conda run -n GraspGen pip install --no-cache-dir \
    torch-scatter -f https://data.pyg.org/whl/torch-2.1.0+cu121.html

# Add server wrapper dependencies
RUN conda run -n GraspGen pip install --no-cache-dir -r /app/requirements/graspgen.txt

# Install system dependencies for offscreen rendering
RUN apt-get update && apt-get install -y \
    libglu1-mesa libglu1-mesa-dev libegl1-mesa-dev \
    && rm -rf /var/lib/apt/lists/*

# ============================================
# Final Setup
# ============================================
WORKDIR /app

# Create directories for outputs and logs
RUN mkdir -p /tmp/spatial_memory/meshes /tmp/spatial_memory/logs

# Environment variables
ENV WEIGHTS_DIR=/weights
ENV MESH_OUTPUT_DIR=/tmp/spatial_memory/meshes
ENV PYTHONPATH=/app:/app/third_party

# Supervisor configuration
COPY scripts/supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Expose ports
EXPOSE 8080 8091 8092 8093 8094

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Entrypoint to copy weights on startup
COPY scripts/docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

ENTRYPOINT ["/docker-entrypoint.sh"]
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]


