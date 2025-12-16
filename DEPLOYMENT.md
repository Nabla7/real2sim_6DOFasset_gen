# Spatial Memory Service - Deployment Guide

Production deployment guide for the SAM3 + SAM3D + FoundationPose mesh/pose estimation service.

## Quick Start

```bash
# 1. Build the Docker image (takes 30-60 minutes)
./scripts/build_docker.sh

# 2. Set weights path
export WEIGHTS_PATH=/path/to/perception_model_weights

# 3. Start the service
docker-compose -f docker-compose.production.yml up -d

# 4. Check health
curl http://localhost:8080/health
```

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

---

## Building from Source

### Option 1: Docker (Recommended)

```bash
# Clone the repository
git clone git@github.com:dimensionalOS/dimos_hosted_services.git
cd dimos_hosted_services

# Build
./scripts/build_docker.sh

# This creates:
#   - spatial-memory-service:latest
#   - spatial-memory-service:YYYYMMDD
```

### Option 2: Local Development

See `scripts/setup_envs.sh` for setting up conda environments locally.

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

### Performance

Expected processing times (A100-80GB):
- **SAM3**: ~1-2 seconds
- **SAM3D**: ~10-30 seconds (depends on mesh complexity)
- **FoundationPose**: ~5-10 seconds
- **Total**: ~20-45 seconds per object

---

## Troubleshooting

### Container won't start

**Check NVIDIA Docker:**
```bash
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

**Check logs:**
```bash
docker-compose -f docker-compose.production.yml logs
```

### Out of Memory (OOM)

**Symptoms:** Container crashes or service stops responding.

**Solutions:**
1. Use a GPU with more VRAM (≥48GB minimum, 80GB recommended)
2. Reduce batch size (set `MAX_JOBS=1` in environment)
3. Process detections sequentially (already default)

### Slow inference

**Check GPU utilization:**
```bash
docker exec -it spatial-memory-production nvidia-smi
```

**Warm-up:** First request is slow due to model loading (30-60s). Subsequent requests are faster.

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
1. Check logs: `docker-compose logs -f`
2. Verify health: `curl http://localhost:8080/health`
3. Review this guide's troubleshooting section
4. Open an issue on GitHub

---

## Architecture

```
┌─────────────────────────────────────────┐
│         Docker Container                 │
├─────────────────────────────────────────┤
│                                          │
│  ┌─────────────────┐                    │
│  │  Gateway :8080  │ ← API Endpoint     │
│  └────────┬────────┘                    │
│           │ Pipeline Orchestration       │
│           ▼                              │
│  ┌────────────┐  ┌────────────┐  ┌────┐│
│  │ SAM3 :8091 │→│SAM3D :8092 │→│FP  ││
│  │  Segment   │  │   Mesh     │  │Pose││
│  └────────────┘  └────────────┘  └────┘│
│                                          │
│  Weights: /weights (read-only mount)    │
│  Output:  /tmp/spatial_memory/meshes    │
│  Logs:    /tmp/spatial_memory/logs      │
└─────────────────────────────────────────┘
```

**Pipeline Parallelism:** Each stage processes different requests concurrently (assembly-line style).

