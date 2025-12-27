# Spatial Memory Service

**Unified microservice for 3D object understanding and robot manipulation.**

Combines state-of-the-art models from Meta and NVIDIA into a single REST API for robotics:
- Segment objects from images (SAM3)
- Reconstruct 3D meshes (SAM3D)
- Estimate 6D poses (FoundationPose)
- Generate collision-free grasps (GraspGen)

---

## Two Pipeline Modes

### Full Pipeline: Complete Scene Understanding
```
POST /process?include_grasps=true

RGB + Depth + Label → Mesh ID + Pose + Grasps
Time: ~20-45 seconds (response immediate, mesh download <1s)
```

**Use when you need:**
- 3D mesh for visualization/simulation (download via GET /mesh/{id})
- Object pose for tracking/manipulation
- Collision-free grasp candidates

**Note:** Meshes are automatically decimated to 1-5MB (down from 25-60MB) and downloaded separately.

### Fast Pipeline: Grasp-Only
```
POST /grasp

RGB + Depth + Label → Grasps
Time: ~5-10 seconds
```

**Use when you only need:**
- Grasp poses for immediate picking
- Fast response time (no mesh reconstruction)

---

## Architecture

```
┌───────────────────────────────────────────────────────────────┐
│                    Gateway Service :8080                       │
│              (Pipeline Orchestration + REST API)               │
└────┬──────────────────────────────────────────────┬───────────┘
     │                                               │
     │ Full Pipeline (/process)       Fast Pipeline (/grasp)
     │                                               │
     ▼                                               ▼
┌─────────────────────────────────────┐    ┌──────────────┐
│  SAM3 :8091 (Text Segmentation)     │    │  SAM3 :8091  │
└────┬────────────────────────────────┘    └──────┬───────┘
     │                                             │
     ▼                                             ▼
┌─────────────────────────────────────┐    ┌──────────────────┐
│  SAM3D :8092 (Mesh Reconstruction)  │    │ GraspGen :8094   │
└────┬────────────────────────────────┘    │ (Grasp Gen)      │
     │                                      └──────────────────┘
     ├─────────────┬──────────────┐
     │ (parallel)  │              │
     ▼             ▼              ▼
┌────────────┐ ┌─────────────┐ ┌─────────────┐
│ FP :8093   │ │ GG :8094    │ │  Result:    │
│ (Pose Est) │ │ (Grasps)    │ │  Mesh +     │
└────────────┘ └─────────────┘ │  Pose +     │
                                │  100 Grasps │
                                └─────────────┘
```

**5 Microservices:**
- **Gateway** (8080) - Public API, orchestration
- **SAM3** (8091) - Segmentation via text prompts
- **SAM3D** (8092) - 3D mesh from 2D mask + depth
- **FoundationPose** (8093) - 6D pose estimation
- **GraspGen** (8094) - Collision-aware grasp generation

---

## Quick Start

### Docker (Recommended)

```bash
# 1. Build (60-90 minutes - includes C++ compilation)
docker build -t spatial-memory:latest .

# 2. Run
docker run --gpus all \
  -p 8080:8080 \
  -v /path/to/weights:/weights:ro \
  spatial-memory:latest

# 3. Test
curl http://localhost:8080/health
```

### Local Development

```bash
# 1. Setup environments (one-time, ~30 minutes)
./scripts/setup_envs.sh

# 2. Start all services
./scripts/start_all.sh

# 3. Check logs
tail -f logs/*.log
```

---

## API Reference

### POST /process - Full Pipeline

**Complete object understanding: mesh ID + pose + grasps**

Returns metadata immediately (~500 bytes). Download mesh separately via `GET /mesh/{id}`.

```bash
curl -X POST http://localhost:8080/process \
  -H "Content-Type: application/json" \
  -d '{
    "image_rgb_b64": "iVBORw0KGgoAAAANS...",
    "depth_b64": "AAAAAAAAAAAAAAAA...",
    "K": [[615.0, 0, 320], [0, 615, 240], [0, 0, 1]],
    "label": "bottle",
    "bbox": [120, 80, 200, 280],
    "include_grasps": true,
    "filter_collisions": true,
    "gripper_type": "robotiq_2f_140"
  }'
```

**Response:**
```json
{
  "label": "bottle",
  "mesh_id": "a3f2c91b-4e7d-4a1c-8b3e-123456789abc",
  "pose": {
    "position": {"x": 0.15, "y": -0.08, "z": 0.72},
    "orientation": {"x": 0.0, "y": 0.0, "z": 0.1, "w": 0.995}
  },
  "bbox_3d": {"sx": 0.08, "sy": 0.25, "sz": 0.08},
  "confidence": 0.95,
  "grasps": [
    {
      "transform": [0.998, -0.052, 0.034, 0.15, 0.053, 0.998, -0.028, -0.08, -0.033, 0.029, 0.999, 0.72, 0, 0, 0, 1],
      "score": 0.947,
      "collision_free": true
    },
    {
      "transform": [0.982, 0.174, -0.067, 0.14, -0.178, 0.984, -0.028, -0.09, 0.061, 0.042, 0.997, 0.73, 0, 0, 0, 1],
      "score": 0.932,
      "collision_free": true
    },
    {
      "transform": [0.967, -0.254, 0.012, 0.16, 0.255, 0.966, 0.034, -0.07, -0.023, -0.029, 0.999, 0.71, 0, 0, 0, 1],
      "score": 0.918,
      "collision_free": true
    }
    ... 97 more grasps
  ]
}
```

**Parameters:**
| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `image_rgb_b64` | string | required | Base64-encoded RGB PNG |
| `depth_b64` | string | required | Base64-encoded float32 depth array (meters) |
| `K` | float[][] | required | 3x3 camera intrinsics matrix |
| `label` | string | "" | Object label (e.g. "bottle", "cup") |
| `bbox` | float[] | required | 2D bbox [x1, y1, x2, y2] |
| `use_box_prompt` | bool | false | Use bbox as geometric prompt (ignore label) |
| `include_grasps` | bool | false | Include grasp generation |
| `filter_collisions` | bool | true | Filter colliding grasps (requires depth) |
| `gripper_type` | string | "robotiq_2f_140" | "robotiq_2f_140" \| "franka_panda" \| "single_suction_cup_30mm" |

**Segmentation Modes:**
- **Text-based** (`use_box_prompt=false`): Uses `label` as text prompt (e.g. "red coffee mug")
- **Box-based** (`use_box_prompt=true`): Uses `bbox` as geometric prompt, ignores `label`
  - **Use this when YOLO-E labels are garbage** but bounding boxes are accurate

**SAM3D scaling (default):**
- Mesh reconstruction uses **internal scaling by default** (server ignores `depth_b64`/`K` for mesh scaling).
- To enable depth+intrinsics pointmap scaling, run SAM3D with `SAM3D_DISABLE_POINTMAP_SCALING=0`.

---

### GET /mesh/{mesh_id} - Download Mesh Artifact

Download a generated 3D mesh by ID. Meshes are automatically decimated to 1-5MB.

```bash
curl http://localhost:8080/mesh/a3f2c91b-4e7d-4a1c-8b3e-123456789abc \
  --output bottle.obj
```

- **Format:** OBJ (compatible with all 3D tools)
- **Size:** 1-5MB (decimated from 25-60MB original)

---

### POST /grasp - Fast Grasp-Only Pipeline

**Direct grasp generation without mesh/pose reconstruction**

```bash
curl -X POST http://localhost:8080/grasp \
  -H "Content-Type: application/json" \
  -d '{
    "image_rgb_b64": "iVBORw0KGgoAAAANS...",
    "depth_b64": "AAAAAAAAAAAAAAAA...",
    "K": [[615.0, 0, 320], [0, 615, 240], [0, 0, 1]],
    "label": "bottle",
    "bbox": [120, 80, 200, 280],
    "use_box_prompt": false,
    "filter_collisions": true,
    "gripper_type": "robotiq_2f_140"
  }'
```

Use `use_box_prompt: true` to ignore the label and segment by bounding box only.

**Response:**
```json
{
  "label": "bottle",
  "grasps": [
    {
      "transform": [0.998, -0.052, 0.034, 0.15, ...],
      "score": 0.947,
      "collision_free": true
    },
    ...
  ],
  "gripper_type": "robotiq_2f_140",
  "inference_time_ms": 850,
  "debug_image_path": "/workspace/debug/grasp_abc123_20251227_120000.png"
}
```

**Grasp Transform Format:**
Each `transform` is a flattened 4x4 homogeneous matrix (16 floats, row-major):
```python
T = np.array(grasp["transform"]).reshape(4, 4)
# T = [[R R R tx],
#      [R R R ty],
#      [R R R tz],
#      [0 0 0 1 ]]
```

**Debug Visualization:**
GraspGen saves a debug image showing grasps overlaid on the RGB image. Check `debug_image_path` in the response or browse `/workspace/debug/`.

---

## Performance

Expected timings on NVIDIA A100-80GB:

| Pipeline | Time | Breakdown |
|----------|------|-----------|
| **Full** (mesh + pose + grasps) | 20-45s | SAM3 (2s) + SAM3D (15-30s) + FP\\|\\|GG (5-10s) |
| **Mesh download** | <1s | Separate GET request (1-5MB decimated mesh) |
| **Grasp-only** | 5-10s | SAM3 (2s) + GraspGen (3-8s) |

*FoundationPose and GraspGen run in parallel after SAM3D completes.*

**Response Sizes:**
- `/process` response: ~500 bytes (metadata only, no mesh)
- `/mesh/{id}` download: 1-5MB (decimated from 25-60MB original)
- `/grasp` response: ~50KB (100 grasp poses)

---

## When to Use Box vs Text Prompts

| Scenario | Mode | Why |
|----------|------|-----|
| **YOLO-E / generic detectors** | `use_box_prompt=true` | Labels are often generic ("object_27"), but bboxes are accurate |
| **Known objects** | `use_box_prompt=false` | Descriptive labels ("red coffee mug") help SAM3 segment precisely |
| **Unknown objects** | `use_box_prompt=true` | Segment whatever is in the bounding box |

---

## Python Integration

```python
import requests
import numpy as np
import base64
import io

# Prepare request
rgb_png = encode_image_to_base64(rgb_image)  # Your PNG encoder
depth_bytes = depth_array.astype(np.float32).tobytes()
depth_b64 = base64.b64encode(depth_bytes).decode()

# Step 1: Full pipeline with grasps (returns metadata + mesh_id)
response = requests.post("http://localhost:8080/process", json={
    "image_rgb_b64": rgb_png,
    "depth_b64": depth_b64,
    "K": [[615, 0, 320], [0, 615, 240], [0, 0, 1]],
    "label": "bottle",
    "bbox": [120, 80, 200, 280],
    "include_grasps": True,
    "filter_collisions": True,
    "gripper_type": "robotiq_2f_140",
})

result = response.json()

# Extract metadata (fast response, ~500 bytes)
mesh_id = result["mesh_id"]
pose = result["pose"]  # Single object pose
grasps = result["grasps"]  # List of ~100 grasp candidates

print(f"Found {len(grasps)} collision-free grasps")
print(f"Best grasp score: {max(g['score'] for g in grasps):.3f}")

# Step 2: Download mesh separately if needed (1-5MB)
if mesh_id:
    mesh_response = requests.get(f"http://localhost:8080/mesh/{mesh_id}")
    mesh_obj = mesh_response.content  # Raw OBJ bytes
    
    # Save to file
    with open(f"bottle_{mesh_id}.obj", "wb") as f:
        f.write(mesh_obj)
    print(f"Mesh saved: {len(mesh_obj)} bytes")

# Use top grasp
best_grasp = max(grasps, key=lambda g: g["score"])
grasp_transform = np.array(best_grasp["transform"]).reshape(4, 4)
```

---

## Deployment

### Requirements

**Hardware:**
- NVIDIA GPU with ≥48GB VRAM (A100-80GB recommended)
- 64GB+ system RAM
- 100GB+ disk space

**Software:**
- Docker ≥20.10 with nvidia-container-toolkit
- NVIDIA Driver ≥525

**Weights:**
Download model weights and mount at `/weights`:
```
/weights/
├── sam3/
│   ├── sam3.pt (3.3GB)
│   └── config.json
├── sam3d-objects/
│   └── *.ckpt files (15GB total)
└── foundationpose/
    └── 2023-*/2024-*/ (scorer/refiner)
```

**GraspGen Models:**
Clone to `third_party/GraspGen/GraspGenModels/`:
```bash
cd third_party/GraspGen
git clone https://huggingface.co/adithyamurali/GraspGenModels
```

See [`DEPLOYMENT.md`](DEPLOYMENT.md) for complete deployment guide.

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `WEIGHTS_DIR` | `/weights` | Model weights directory |
| `MESH_OUTPUT_DIR` | `/tmp/spatial_memory/meshes` | Mesh output directory |
| `GATEWAY_PORT` | `8080` | Gateway API port |
| `SAM3_PORT` | `8091` | SAM3 service port |
| `SAM3D_PORT` | `8092` | SAM3D service port |
| `FOUNDATIONPOSE_PORT` | `8093` | FoundationPose service port |
| `GRASPGEN_PORT` | `8094` | GraspGen service port |
| `SAM3_MAX_DETECTIONS` | `3` | Max detections per keyword |
| `PYOPENGL_PLATFORM` | `egl` | OpenGL platform (for GraspGen) |

---

## Development

### Local Setup

```bash
# 1. Clone third-party repos and setup environments
./scripts/setup_envs.sh

# 2. Download GraspGen models
cd /workspace/third_party/GraspGen
git clone https://huggingface.co/adithyamurali/GraspGenModels

# 3. Start all services
./scripts/start_all.sh

# 4. Check status
curl http://localhost:8080/health
```

### Individual Services

Run in separate terminals:

```bash
# Gateway
conda activate gateway
python -m gateway.main

# SAM3
conda activate sam3
python -m sam3_service.main

# SAM3D
conda activate sam3d-objects
python -m sam3d_service.main

# FoundationPose
conda activate foundationpose
python -m foundationpose_service.main

# GraspGen
conda activate GraspGen
python -m graspgen_service.main
```

---

## Third-Party Components

| Component | Source | Version | Purpose |
|-----------|--------|---------|---------|
| SAM3 | Meta AI | [GitHub](https://github.com/facebookresearch/sam3) | Text-prompted segmentation |
| SAM3D | Meta AI | [GitHub](https://github.com/facebookresearch/sam-3d-objects) | 3D reconstruction |
| FoundationPose | NVIDIA | [GitHub](https://github.com/NVlabs/FoundationPose) | 6D pose estimation |
| GraspGen | NVIDIA | [GitHub](https://github.com/NVlabs/GraspGen) | Grasp generation |

All repositories are automatically cloned during Docker build or via `setup_envs.sh`.

---

## Supported Grippers

GraspGen supports three gripper types:

| Gripper | ID | Use Case |
|---------|----|----|
| **Robotiq 2F-140** | `robotiq_2f_140` | Industrial parallel-jaw gripper |
| **Franka Panda** | `franka_panda` | Research/collaborative robot |
| **Suction Cup 30mm** | `single_suction_cup_30mm` | Single-cup vacuum gripper |

Specify via `gripper_type` parameter in requests.

---

## Health Checks

```bash
# All services healthy
curl http://localhost:8080/health  # {"status": "ok"}
curl http://localhost:8091/health  # SAM3
curl http://localhost:8092/health  # SAM3D
curl http://localhost:8093/health  # FoundationPose
curl http://localhost:8094/health  # GraspGen
```