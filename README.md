# Spatial Memory Service

Unified service for 3D mesh reconstruction and 6D pose estimation, designed to integrate with DIMOS's perception stack.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                 Docker Container                         │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌─────────────────┐                                    │
│  │  Gateway :8080  │ ← DIMOS sends requests here        │
│  │  (FastAPI)      │                                    │
│  └────────┬────────┘                                    │
│           │ pipeline orchestration                      │
│           ▼                                             │
│  ┌────────────────┐  ┌────────────────┐  ┌───────────┐ │
│  │  SAM3 :8091    │→│  SAM3D :8092   │→│ FP :8093  │ │
│  │  (segmentation)│  │  (mesh + scale)│  │ (6D pose) │ │
│  └────────────────┘  └────────────────┘  └───────────┘ │
│                                                          │
│  Shared Volume: /weights                                │
└─────────────────────────────────────────────────────────┘
```

## Pipeline Parallelism

Each stage processes different requests concurrently (assembly-line style):

| Time | SAM3 | SAM3D | FoundationPose |
|------|------|-------|----------------|
| T1 | Request A | idle | idle |
| T2 | Request B | Request A | idle |
| T3 | Request C | Request B | Request A |
| T4 | Request D | Request C | Request B |

## API

### POST /process

Process an object through the full pipeline.

**Request:**
```json
{
  "image_rgb_b64": "base64...",
  "depth_b64": "base64...",
  "K": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
  "label": "bottle",
  "bbox": [x1, y1, x2, y2]
}
```

**Response:**
```json
{
  "label": "bottle",
  "mesh_b64": "base64 .obj...",
  "pose": {
    "position": {"x": 0.1, "y": 0.2, "z": 0.5},
    "orientation": {"x": 0, "y": 0, "z": 0, "w": 1}
  },
  "bbox_3d": {"sx": 0.05, "sy": 0.05, "sz": 0.12},
  "confidence": 0.95
}
```

## Weights Structure

Expected structure at `/weights`:

```
/weights/
├── foundationpose/
│   ├── 2023-10-28-18-33-37/        # Refiner
│   │   ├── config.yml
│   │   └── model_best.pth
│   └── 2024-01-11-20-02-45/        # Scorer
│       ├── config.yml
│       └── model_best.pth
├── sam3/
│   ├── sam3.pt                      # 3.3GB
│   └── config.json
└── sam3d-objects/
    ├── pipeline.yaml
    ├── ss_generator.ckpt            # 6.3GB
    ├── slat_generator.ckpt          # 4.6GB
    ├── slat_decoder_mesh.ckpt
    ├── slat_decoder_mesh.pt
    ├── slat_decoder_gs.ckpt
    ├── slat_decoder_gs_4.ckpt
    ├── ss_decoder.ckpt
    └── *.yaml configs
```

## Quick Start (Production)

```bash
# 1. Build the production image (30-60 minutes)
./scripts/build_docker.sh

# 2. Set weights path
export WEIGHTS_PATH=/path/to/perception_model_weights

# 3. Start the service
docker-compose -f docker-compose.production.yml up -d

# 4. Test
./scripts/test_deployment.sh
```

**For full production deployment guide, see [DEPLOYMENT.md](DEPLOYMENT.md)**

## Running with Docker (Development)

```bash
# Set path to your weights
export WEIGHTS_PATH=/path/to/perception_model_weights

# Build and run (development mode with code mounting)
docker-compose up --build

# Or with docker directly
docker build -t spatial-memory-service .
docker run --gpus all -p 8080:8080 -v $WEIGHTS_PATH:/weights spatial-memory-service
```

## Running Locally (Development)

```bash
# Setup conda environments
./scripts/setup_envs.sh

# Run each service in separate terminals:

# Terminal 1: Gateway
conda activate sm-gateway
cd /path/to/services/spatial_memory_service
python -m gateway.main

# Terminal 2: SAM3
conda activate sm-sam3
python -m sam3_service.main

# Terminal 3: SAM3D
conda activate sm-sam3d
python -m sam3d_service.main

# Terminal 4: FoundationPose
conda activate sm-foundationpose
python -m foundationpose_service.main
```

## DIMOS Integration

From DIMOS, use the `SpatialMemoryClient`:

```python
from dimos.perception.spatial_memory import SpatialMemoryClient

# Create client
client = SpatialMemoryClient(service_url="http://localhost:8080")

# Process a detection
response = client.process(
    image_rgb=rgb_image,      # numpy array (H, W, 3)
    depth=depth_image,        # numpy array (H, W) in meters
    K=camera_intrinsics,      # 3x3 matrix
    label="bottle",           # from YOLO-E
    bbox=[100, 150, 200, 300] # from YOLO-E
)

if response.success:
    print(f"Pose: {response.pose.position}")
    print(f"Size: {response.bbox_3d.dimensions}")
    response.save_mesh("/tmp/object.obj")
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `WEIGHTS_DIR` | `/weights` | Path to model weights |
| `MESH_OUTPUT_DIR` | `/tmp/spatial_memory/meshes` | Mesh output directory |
| `GATEWAY_PORT` | `8080` | Gateway service port |
| `SAM3_PORT` | `8091` | SAM3 service port |
| `SAM3D_PORT` | `8092` | SAM3D service port |
| `FOUNDATIONPOSE_PORT` | `8093` | FoundationPose service port |
| `SAM3_TIMEOUT` | `60` | SAM3 request timeout (seconds) |
| `SAM3D_TIMEOUT` | `300` | SAM3D request timeout (seconds) |
| `FOUNDATIONPOSE_TIMEOUT` | `120` | FoundationPose request timeout |


