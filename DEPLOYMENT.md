# Spatial Memory Service - Deployment Guide

Production deployment guide for the SAM3 + SAM3D + FoundationPose mesh/pose estimation service.

## Quick Start

```bash
# 1. Build the Docker image (takes 60-90 minutes due to C++ compilation)
docker build -t spatial-memory-service:latest .

# 2. Set weights path
export WEIGHTS_PATH=/path/to/perception_model_weights

# 3. Start the service
docker run --gpus all \
  -p 8080:8080 \
  -v $WEIGHTS_PATH:/weights:ro \
  --name spatial-memory \
  spatial-memory-service:latest

# 4. Check health
curl http://localhost:8080/health
```

**Note:** The build process clones third-party repositories, installs complex dependencies, and compiles C++ extensions. This is a one-time process.

---

## Requirements

### Hardware
- **GPU**: NVIDIA GPU with ≥48GB VRAM (A100-80GB recommended)
- **RAM**: ≥64GB system RAM
- **Disk**: ≥100GB free space (for Docker image + weights)
- **Network**: ≥100 Mbps for weights download

### Software
- **Docker**: ≥20.10
- **nvidia-container-toolkit**: Latest
- **NVIDIA Driver**: ≥525 (supports CUDA 12.1+)

### Weights Structure

The `WEIGHTS_PATH` should contain:

```
perception_model_weights/
├── foundationpose/
│   ├── 2023-10-28-18-33-37/   # Refiner
│   │   ├── config.yml
│   │   └── model_best.pth
│   └── 2024-01-11-20-02-45/   # Scorer
│       ├── config.yml
│       └── model_best.pth
├── sam3/
│   ├── sam3.pt                # 3.3GB
│   └── config.json
└── sam3d-objects/
    ├── pipeline.yaml
    ├── ss_generator.ckpt      # 6.3GB
    ├── slat_generator.ckpt    # 4.6GB
    └── ... (other checkpoints)
```

### GraspGen Models

GraspGen models must be cloned separately to `third_party/GraspGen/`:

```bash
cd /workspace/third_party/GraspGen
git clone https://huggingface.co/adithyamurali/GraspGenModels

# This contains:
# - checkpoints/*.yml (gripper configurations)
# - checkpoints/*_gen.pth (generator models)
# - checkpoints/*_dis.pth (discriminator models)
```

**Note:** In Docker, this is handled automatically during entrypoint.

---

## Building from Source

### Option 1: Docker (Recommended)

```bash
# Clone the repository
git clone git@github.com:dimensionalOS/dimos_hosted_services.git
cd dimos_hosted_services

# Build (this will take 60-90 minutes)
docker build -t spatial-memory-service:latest .

# What happens during build:
# 1. Clones third-party repos (SAM3, SAM3D, FoundationPose)
# 2. Creates 4 separate conda environments
# 3. Installs PyTorch with CUDA support for each
# 4. Installs SAM3D with complex dependencies (Kaolin, etc.)
# 5. Compiles FoundationPose C++ extensions (mycpp, bundlesdf)
```

### Option 2: Local Development

See `scripts/start_all.sh` for setting up conda environments locally.

**Important:** Local development requires:
- Conda/Mamba installed
- CUDA 12.1+ toolkit (and CUDA 11.8 for FoundationPose)
- Third-party repos cloned to `/workspace/third_party/`
- Weights in `/workspace/weights/`
- System packages: `libglu1-mesa-dev`, `libegl1-mesa-dev` (for GraspGen)

---

## Running the Service

### Production (Docker Compose)

```bash
# Start in detached mode
export WEIGHTS_PATH=/data/perception_model_weights
docker-compose -f docker-compose.production.yml up -d

# View logs
docker-compose -f docker-compose.production.yml logs -f

# Stop
docker-compose -f docker-compose.production.yml down
```

### Development (Manual Docker)

```bash
docker run --gpus all \
  -p 8080:8080 \
  -v /path/to/weights:/weights:ro \
  -v $(pwd)/logs:/tmp/spatial_memory/logs \
  -e LOG_LEVEL=DEBUG \
  spatial-memory-service:latest
```

### Exposing Individual Services

By default, only the gateway (port 8080) is exposed. To debug individual services:

```yaml
# In docker-compose.production.yml, uncomment:
ports:
  - "8080:8080"   # Gateway
  - "8091:8091"   # SAM3
  - "8092:8092"   # SAM3D
  - "8093:8093"   # FoundationPose
```

---

## API Usage

### Health Check

```bash
curl http://localhost:8080/health
```

**Response:**
```json
{
  "status": "ok",
  "service": "spatial_memory_gateway"
}
```

### Process Detection

```bash
curl -X POST http://localhost:8080/process \
  -H "Content-Type: application/json" \
  -d @request.json
```

**Request format** (`request.json`):
```json
{
  "request_id": "req_12345",
  "label": "bottle",
  "bbox": [120, 80, 200, 280],
  "image_rgb_b64": "base64_encoded_rgb_png...",
  "depth_b64": "base64_encoded_depth_float32...",
  "K": [
    [615.0, 0.0, 320.0],
    [0.0, 615.0, 240.0],
    [0.0, 0.0, 1.0]
  ]
}
```

**Response:**
```json
{
  "request_id": "req_12345",
  "success": true,
  "label": "bottle",
  "mesh_b64": "base64_encoded_obj_file...",
  "pose": {
    "position": {"x": 0.15, "y": -0.08, "z": 0.72},
    "orientation": {"x": 0.0, "y": 0.0, "z": 0.1, "w": 0.995}
  },
  "bbox_3d": {"sx": 0.08, "sy": 0.25, "sz": 0.08},
  "confidence": 0.95
}
```

---

## Monitoring

### Logs

```bash
# All services
docker-compose -f docker-compose.production.yml logs -f

# Specific service
docker exec -it spatial-memory-production tail -f /tmp/spatial_memory/logs/gateway.log
docker exec -it spatial-memory-production tail -f /tmp/spatial_memory/logs/sam3.log
docker exec -it spatial-memory-production tail -f /tmp/spatial_memory/logs/sam3d.log
docker exec -it spatial-memory-production tail -f /tmp/spatial_memory/logs/foundationpose.log
```

### Metrics

Health check endpoints:
- Gateway: `http://localhost:8080/health`
- SAM3: `http://localhost:8091/health`
- SAM3D: `http://localhost:8092/health`
- FoundationPose: `http://localhost:8093/health`
- GraspGen: `http://localhost:8094/health`

### Performance

Expected processing times (A100-80GB):
- **SAM3**: ~1-2 seconds
- **SAM3D**: ~10-30 seconds (depends on mesh complexity)
- **FoundationPose**: ~5-10 seconds (parallel with GraspGen)
- **GraspGen**: ~1-5 seconds (collision filtering adds ~1-2s)
- **Total (full pipeline)**: ~20-45 seconds per object
- **Total (grasp-only)**: ~5-10 seconds per object

---

## Troubleshooting

### Container won't start

**Check NVIDIA Docker:**
```bash
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

**Check logs:**
```bash
docker logs spatial-memory
```

**Common issues:**
- Missing weights: Ensure `/weights` volume is mounted correctly
- GPU not available: Check `--gpus all` flag and nvidia-container-toolkit
- Port conflicts: Check if ports 8080-8093 are already in use

### Out of Memory (OOM)

**Symptoms:** Container crashes or service stops responding.

**Solutions:**
1. Use a GPU with more VRAM (≥48GB minimum, 80GB recommended)
2. Restart services to clear GPU memory: `docker restart spatial-memory`
3. Check for orphaned processes: `docker exec spatial-memory nvidia-smi`

**Note:** Worker processes are properly cleaned up on restart (fixed in this version).

### Slow inference

**Check GPU utilization:**
```bash
docker exec -it spatial-memory nvidia-smi
```

**Warm-up:** First request is slow due to model loading (30-60s). Subsequent requests are faster.

### Build failures

**SAM3D build fails:**
- Ensure good internet connection (downloads from NVIDIA NGC, PyPI)
- Check CUDA version compatibility (needs 12.1+)
- Retry: `docker build --no-cache -t spatial-memory-service:latest .`

**FoundationPose C++ compilation fails:**
- Ensure build-essential and cmake are installed (already in Dockerfile)
- Check for sufficient disk space (≥20GB free)
- Check eigen library installation

### Service-specific issues

**SAM3D fails with "No module named 'notebook.inference'":**
- This is fixed in the current version
- Ensure you're using the latest Dockerfile

**FoundationPose fails with "No such file or directory: weights/...":**
- This is fixed in the current version via docker-entrypoint.sh
- Weights are automatically copied on container startup

**Too many detections returned:**
- Default is now 3 per keyword (was 30)
- Override with `SAM3_MAX_DETECTIONS` environment variable

### Individual service debugging

**Test SAM3:**
```bash
curl -X POST http://localhost:8091/segment \
  -H "Content-Type: application/json" \
  -d '{"image_b64": "...", "text_prompts": ["bottle"]}'
```

**Test SAM3D:**
```bash
curl -X POST http://localhost:8092/reconstruct \
  -H "Content-Type: application/json" \
  -d '{"image_b64": "...", "mask_b64": "..."}'
```

**Test FoundationPose:**
```bash
curl -X POST http://localhost:8093/register \
  -H "Content-Type: application/json" \
  -d '{"object_id": "test", "mesh_path": "/tmp/test.obj", ...}'
```

**Test GraspGen:**
```bash
curl -X POST http://localhost:8094/generate \
  -H "Content-Type: application/json" \
  -d '{"depth_b64": "...", "mask_b64": "...", "K": [[...]], ...}'
```

**Test Grasp-Only Pipeline:**
```bash
curl -X POST http://localhost:8080/grasp \
  -H "Content-Type: application/json" \
  -d '{"image_rgb_b64": "...", "depth_b64": "...", "K": [[...]], "label": "bottle", "bbox": [...]}'
```

---

## Updating the Service

### Pull latest code

```bash
cd dimos_hosted_services
git pull origin main
```

### Rebuild

```bash
./scripts/build_docker.sh
```

### Update running container

```bash
docker-compose -f docker-compose.production.yml down
docker-compose -f docker-compose.production.yml up -d
```

---

## Security

### Network

- Only expose port 8080 publicly (gateway)
- Keep ports 8091-8093 internal for debugging only

### Volumes

- Mount weights as read-only (`:ro`)
- Mesh outputs are ephemeral (can be cleared periodically)

### Resource Limits

Add to `docker-compose.production.yml`:
```yaml
deploy:
  resources:
    limits:
      cpus: '16'
      memory: 64G
```

---

## Production Checklist

- [ ] GPU with ≥48GB VRAM available
- [ ] NVIDIA drivers ≥525 installed
- [ ] nvidia-container-toolkit configured
- [ ] Weights directory prepared and accessible
- [ ] Docker image built successfully
- [ ] Health check returns `{"status": "ok"}`
- [ ] Test request completes successfully
- [ ] Logs are being written to persistent volume
- [ ] Auto-restart configured (`restart: unless-stopped`)
- [ ] Monitoring/alerting set up (if applicable)

---

## Support

For issues or questions:
1. Check logs: `docker logs spatial-memory`
2. Verify health: `curl http://localhost:8080/health`
3. Review this guide's troubleshooting section
4. Open an issue on GitHub

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│              Docker Container                            │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌─────────────────┐                                    │
│  │  Gateway :8080  │ ← API Endpoint                     │
│  └────────┬────────┘                                    │
│           │ Pipeline Orchestration                       │
│           ▼                                              │
│  ┌────────────┐  ┌────────────┐  ┌──────────────┐     │
│  │ SAM3 :8091 │→│SAM3D :8092 │→│FP :8093      │     │
│  │  Segment   │  │   Mesh     │  │Pose Estimate │     │
│  └────────────┘  └────────────┘  └──────────────┘     │
│                                                          │
│  Conda Environments:                                     │
│  • gateway (Python 3.11, CPU)                           │
│  • sam3 (Python 3.11, CUDA 12.1)                        │
│  • sam3d-objects (Python 3.11, CUDA 12.1 + Kaolin)     │
│  • foundationpose (Python 3.9, CUDA 11.8)               │
│  • GraspGen (Python 3.10, CUDA 12.1)                    │
│                                                          │
│  Third-party repos:                                      │
│  • /app/third_party/sam3                                │
│  • /app/third_party/sam-3d-objects                      │
│  • /app/third_party/FoundationPose                      │
│  • /app/third_party/GraspGen                            │
│                                                          │
│  Volumes:                                                │
│  • /weights (read-only mount) → Model weights           │
│  • /tmp/spatial_memory/meshes → Generated meshes        │
│  • /tmp/spatial_memory/logs → Service logs              │
└─────────────────────────────────────────────────────────┘
```

**Pipeline Parallelism:** Each stage processes different requests concurrently (assembly-line style).

