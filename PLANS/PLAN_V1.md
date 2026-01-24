# Real2Sim Pipeline Expansion Plan

## Overview

Expand the existing object processing pipeline into a full **real2sim** system that:
1. Captures images from robot cameras (Booster K1 + ZED Mini, Unitree Go2 + RealSense)
2. Reconstructs photorealistic 3D scenes using COLMAP + 3DGUT/NuRec
3. Composes scenes with positioned objects for Isaac Sim

**Hardware:** NVIDIA DGX Spark
**Timeline:** 1-2 days
**Camera:** RealSense on Unitree Go2 (pyrealsense2, no ROS)
**ZED:** Deferred (requires manual SDK install)

---

## How This Repo Works (Important!)

```
workspace/
├── dimos_hosted_services/    # THIS REPO - thin FastAPI wrappers
│   ├── gateway/
│   ├── sam3_service/
│   ├── sam3d_service/
│   ├── foundationpose_service/
│   ├── graspgen_service/
│   ├── requirements/         # Minimal deps (fastapi, uvicorn, pydantic)
│   └── scripts/
│       └── setup_envs.sh     # Clones third-party repos + creates conda envs
│
├── third_party/              # CLONED BY setup_envs.sh (NOT in git)
│   ├── sam3/                 # facebook/sam3
│   ├── sam-3d-objects/       # facebook/sam-3d-objects
│   ├── FoundationPose/       # NVlabs/FoundationPose
│   └── GraspGen/             # NVlabs/GraspGen
│
└── weights/                  # Model weights (mounted, NOT in git)
```

**Key pattern:** Service wrappers import from `third_party/` repos. The heavy ML code lives there.

---

## Current Architecture

```
gateway/          :8080  - REST API orchestrator
sam3_service/     :8091  - 2D segmentation
sam3d_service/    :8092  - 3D mesh reconstruction
foundationpose_service/ :8093  - 6D pose estimation
graspgen_service/ :8094  - Grasp generation
```

**Key patterns to follow:**
- FastAPI + Pydantic models
- Worker process architecture for GPU memory (see `sam3d_service/main.py`)
- Environment-based config (see `gateway/config.py`)
- Base64-encoded images over HTTP/JSON

---

## New Services to Add

### 1. Scene Capture Service (:8095)
**Purpose:** Interface with robot cameras, capture images for COLMAP

```
POST /sessions/create        - Create capture session
POST /sessions/{id}/capture  - Capture single frame
POST /sessions/{id}/auto     - Auto-capture with interval
GET  /sessions/{id}/status   - Check progress
POST /sessions/{id}/finalize - Package for COLMAP
```

### 2. COLMAP Service (:8096)
**Purpose:** Structure-from-Motion, compute camera poses

```
POST /reconstruct          - Start SfM reconstruction
GET  /jobs/{id}/status     - Check progress
GET  /jobs/{id}/result     - Get transforms.json + sparse cloud
```

### 3. Neural Reconstruction Service (:8097)
**Purpose:** 3DGUT/NuRec training, export USDZ for Isaac Sim

```
POST /train                - Start neural reconstruction
GET  /jobs/{id}/status     - Check training progress
POST /jobs/{id}/export     - Export to USDZ
```

### 4. Scene Composer Service (:8098)
**Purpose:** Combine scene + objects at 6DOF poses

```
POST /compose              - Compose scene with objects
POST /scenes/{id}/add_object - Add object to scene
GET  /scenes/{id}/export   - Export final USDZ
```

---

## Data Flow

```
Robot Camera (K1/Go2)
       │
       ▼
Scene Capture :8095
       │ images/ + metadata.json
       ▼
COLMAP :8096
       │ transforms.json + sparse cloud
       ▼
Neural Recon :8097
       │ scene.usdz
       ▼
Scene Composer :8098  ◄── Objects from existing pipeline
       │
       ▼
composed_scene.usdz → Isaac Sim
```

---

## Team Work Breakdown

### Person A: Robot Camera Integration (RealSense + Go2)
**Files to create:**
- `scene_capture_service/main.py`
- `scene_capture_service/drivers/base.py`
- `scene_capture_service/drivers/realsense.py`
- `requirements/scene_capture.txt`

**Tasks:**
- Install pyrealsense2 (`pip install pyrealsense2`)
- Implement RealSense D455 capture (RGB frames for COLMAP)
- Session management (create, capture, finalize)
- Export metadata.json with camera intrinsics
- Auto-capture with configurable interval

**Dependencies:** None - can start immediately

---

### Person B: COLMAP + Neural Reconstruction
**Files to create:**
- `colmap_service/main.py`
- `neural_recon_service/main.py`
- `requirements/colmap.txt`
- `requirements/neural_recon.txt`

**Day 1:** Install COLMAP, implement service wrapper, transforms.json export
**Day 2:** Set up 3DGUT, implement training service, USDZ export
**Day 3:** Integration, optimization, demo support

**Dependencies:** Needs sample images (use placeholder dataset initially)

---

### Person C: Scene Composer + Gateway
**Files to create:**
- `scene_composer_service/main.py`
- `gateway/scene_pipeline.py`
- `requirements/scene_composer.txt`

**Tasks:**
- Set up OpenUSD (`pip install usd-core`)
- Implement USD scene composition (load scene + place objects)
- Extend gateway with scene reconstruction routes
- Scene pipeline orchestration (like existing `pipeline.py`)

**Dependencies:** Needs scene USDZ (use placeholder initially)

---

## Gateway Config Additions (gateway/config.py)

```python
# New service URLs
SCENE_CAPTURE_URL = os.getenv("SCENE_CAPTURE_URL", "http://localhost:8095")
COLMAP_URL = os.getenv("COLMAP_URL", "http://localhost:8096")
NEURAL_RECON_URL = os.getenv("NEURAL_RECON_URL", "http://localhost:8097")
SCENE_COMPOSER_URL = os.getenv("SCENE_COMPOSER_URL", "http://localhost:8098")

# Timeouts
COLMAP_TIMEOUT = int(os.getenv("COLMAP_TIMEOUT", "1800"))  # 30 min
NEURAL_RECON_TIMEOUT = int(os.getenv("NEURAL_RECON_TIMEOUT", "3600"))  # 1 hour

# Data directories
DATA_DIR = os.getenv("DATA_DIR", "/data")
```

---

## Integration Points (Sync Needed)

| When | What | Who |
|------|------|-----|
| Day 1 | Agree on JSON API contracts | All |
| Day 1 | Test capture → COLMAP chain | A + B |
| Day 2 | Test composer with mock scene | B + C |
| Day 2 | Full pipeline dry run | All |

---

## Verification Plan

1. **Scene Capture:** Run `curl http://localhost:8095/health`, capture 10 test images
2. **COLMAP:** Feed test images, verify transforms.json output
3. **Neural Recon:** Train on small dataset (~50 images), verify USDZ export
4. **Scene Composer:** Load scene + 1 object, verify combined USDZ
5. **Isaac Sim:** Load composed_scene.usdz, add robot, press Play

---

## Quick Start Commands

```bash
# Create new service directories
mkdir -p scene_capture_service/drivers
mkdir -p colmap_service
mkdir -p neural_recon_service
mkdir -p scene_composer_service

# Create requirements files
touch requirements/scene_capture.txt
touch requirements/colmap.txt
touch requirements/neural_recon.txt
touch requirements/scene_composer.txt
```

---

## Key Design Decisions

1. **No ROS:** Use native camera SDKs (pyzed, pyrealsense2) for simplicity
2. **Follow the existing pattern:** Service wrappers + third_party repos + setup_envs.sh
3. **Async jobs:** COLMAP and neural training are long-running, use job-based API
4. **DGX Spark:** All services run locally, cameras connect via USB

---

## Third-Party Dependencies to Add

New repos to clone in `setup_envs.sh`:

| Service | Third-Party | How to Get |
|---------|-------------|------------|
| Scene Capture | pyrealsense2 | `pip install pyrealsense2` |
| COLMAP | colmap binary | `apt-get install colmap` or conda |
| Neural Recon | 3dgrut | `git clone https://github.com/nv-tlabs/3dgrut.git` |
| Scene Composer | usd-core | `pip install usd-core` |

---

## Files to Create/Modify

### 1. New Service Directories
```
scene_capture_service/
├── __init__.py
├── main.py            # FastAPI service
└── drivers/
    ├── __init__.py
    ├── base.py        # Abstract camera interface
    └── realsense.py   # pyrealsense2 wrapper (ZED deferred)

colmap_service/
├── __init__.py
└── main.py            # Wraps COLMAP CLI

neural_recon_service/
├── __init__.py
└── main.py            # Wraps 3dgrut training

scene_composer_service/
├── __init__.py
└── main.py            # USD composition (defines its own models)
```

### 2. Requirements Files (minimal, following existing pattern)
```
requirements/scene_capture.txt   # fastapi, uvicorn, pydantic, pyrealsense2
requirements/colmap.txt          # fastapi, uvicorn, pydantic, numpy
requirements/neural_recon.txt    # fastapi, uvicorn, pydantic
requirements/scene_composer.txt  # fastapi, uvicorn, pydantic, usd-core, trimesh
```

### 3. Modify scripts/setup_envs.sh
Add sections for:
- Clone 3dgrut to third_party/
- Create conda envs: scene_capture, colmap, neural_recon, scene_composer
- pyrealsense2 installs via pip (no manual download needed)

### 4. Modify scripts/supervisord.conf
Add program entries for new services (:8095-8098)

### 5. Modify gateway/config.py
Add URLs and timeouts for new services

### 6. Add gateway/scene_pipeline.py
Scene reconstruction orchestration (like existing pipeline.py for objects)
