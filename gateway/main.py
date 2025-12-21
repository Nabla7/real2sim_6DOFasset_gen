"""
Spatial Memory Gateway Service

Main FastAPI application that exposes the /process endpoint
and orchestrates the SAM3 -> SAM3D -> FoundationPose pipeline.
"""

import logging
import os
from contextlib import asynccontextmanager
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .config import (
    GATEWAY_HOST,
    GATEWAY_PORT,
    MESH_OUTPUT_DIR,
    SAM3_URL,
    SAM3_TIMEOUT,
    GRASPGEN_URL,
    GRASPGEN_TIMEOUT,
)
from .pipeline import get_pipeline, compute_iou

# Configure logging
LOG_DIR = Path(os.getenv("LOG_DIR", "/workspace/logs"))
LOG_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("spatial_memory.gateway")
logger.setLevel(logging.INFO)
if not logger.handlers:
    file_handler = RotatingFileHandler(
        LOG_DIR / "gateway.log", maxBytes=10_000_000, backupCount=5
    )
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)


# --- Pydantic Models ---


class ProcessRequest(BaseModel):
    """Request to process an object detection through the spatial memory pipeline."""

    image_rgb_b64: str = Field(..., description="Base64-encoded RGB image")
    depth_b64: str = Field(..., description="Base64-encoded depth image (meters, float32)")
    K: List[List[float]] = Field(
        ..., description="Camera intrinsics matrix 3x3: [[fx,0,cx],[0,fy,cy],[0,0,1]]"
    )
    label: str = Field("", description="Object label (optional if use_box_prompt=True)")
    bbox: List[float] = Field(
        ..., description="2D bounding box [x1, y1, x2, y2]"
    )
    # Segmentation mode
    use_box_prompt: bool = Field(False, description="Use bbox as geometric prompt (ignore label)")
    # Optional grasp generation
    include_grasps: bool = Field(False, description="Include grasp pose generation")
    filter_collisions: bool = Field(True, description="Filter collision grasps (if include_grasps=True)")
    gripper_type: str = Field("robotiq_2f_140", description="Gripper type for grasping")


class Position(BaseModel):
    x: float
    y: float
    z: float


class Orientation(BaseModel):
    x: float
    y: float
    z: float
    w: float


class Pose(BaseModel):
    position: Position
    orientation: Orientation


class BBox3D(BaseModel):
    sx: float
    sy: float
    sz: float


class GraspPose(BaseModel):
    transform: List[float] = Field(..., description="4x4 transform matrix (16 floats)")
    score: float = Field(..., description="Grasp quality score")
    collision_free: Optional[bool] = Field(None, description="Collision-free status")


class ProcessResponse(BaseModel):
    """Response from the spatial memory pipeline."""

    label: str
    mesh_id: Optional[str] = Field(None, description="ID for mesh download via GET /mesh/{id}")
    pose: Optional[Pose] = Field(None, description="6D pose in camera frame")
    bbox_3d: Optional[BBox3D] = Field(None, description="3D bounding box dimensions")
    confidence: float = Field(1.0, description="Detection confidence")
    grasps: Optional[List[GraspPose]] = Field(None, description="Grasp poses (if requested)")


class GraspRequest(BaseModel):
    """Request for grasp-only generation (faster, no mesh/pose)."""

    image_rgb_b64: str = Field(..., description="Base64-encoded RGB image")
    depth_b64: str = Field(..., description="Base64-encoded depth image (meters, float32)")
    K: List[List[float]] = Field(
        ..., description="Camera intrinsics matrix 3x3"
    )
    label: str = Field("", description="Object label (optional if use_box_prompt=True)")
    bbox: List[float] = Field(..., description="2D bounding box [x1, y1, x2, y2]")
    use_box_prompt: bool = Field(False, description="Use bbox as geometric prompt (ignore label)")
    filter_collisions: bool = Field(True, description="Filter colliding grasps")
    gripper_type: str = Field("robotiq_2f_140", description="Gripper type")
    num_grasps: int = Field(400, description="Number of grasps to generate")
    topk_num_grasps: int = Field(100, description="Return top K grasps")


class GraspResponse(BaseModel):
    """Response from grasp-only pipeline."""

    label: str
    grasps: List[GraspPose]
    gripper_type: str
    inference_time_ms: float


# --- FastAPI App ---


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle."""
    logger.info("Starting Spatial Memory Gateway...")
    pipeline = await get_pipeline()
    logger.info("Pipeline workers ready")
    yield
    logger.info("Shutting down Spatial Memory Gateway...")
    await pipeline.stop()


app = FastAPI(
    title="Spatial Memory Gateway",
    description="Orchestrates SAM3, SAM3D, and FoundationPose for object mesh reconstruction and pose estimation",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health() -> Dict[str, Any]:
    """Health check endpoint."""
    return {
        "status": "ok",
        "service": "spatial_memory_gateway",
    }


@app.get("/mesh/{mesh_id}")
async def download_mesh(mesh_id: str):
    """Download a generated mesh by ID."""
    mesh_path = Path(MESH_OUTPUT_DIR) / f"{mesh_id}.obj"
    if not mesh_path.exists():
        raise HTTPException(status_code=404, detail=f"Mesh {mesh_id} not found")
    return FileResponse(
        path=mesh_path,
        media_type="application/octet-stream",
        filename=f"{mesh_id}.obj"
    )


@app.post("/process", response_model=ProcessResponse)
async def process(request: ProcessRequest) -> ProcessResponse:
    """
    Process an object through the full spatial memory pipeline.

    1. SAM3: Segment the object using the label as text prompt
    2. SAM3D: Reconstruct a scaled 3D mesh using depth
    3. FoundationPose: Estimate 6D pose

    Returns the mesh (base64 .obj) and pose.
    """
    pipeline = await get_pipeline()

    try:
        result = await pipeline.submit(request.model_dump())

        # Convert nested dicts to Pydantic models
        pose = None
        if result.get("pose"):
            p = result["pose"]
            pose = Pose(
                position=Position(**p.get("position", {"x": 0, "y": 0, "z": 0})),
                orientation=Orientation(**p.get("orientation", {"x": 0, "y": 0, "z": 0, "w": 1})),
            )

        bbox_3d = None
        if result.get("bbox_3d"):
            bbox_3d = BBox3D(**result["bbox_3d"])

        grasps = None
        if result.get("grasps"):
            grasps = [GraspPose(**g) for g in result["grasps"]]

        return ProcessResponse(
            label=result.get("label", request.label),
            mesh_id=result.get("mesh_id"),
            pose=pose,
            bbox_3d=bbox_3d,
            confidence=result.get("confidence", 1.0),
            grasps=grasps,
        )

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/grasp", response_model=GraspResponse)
async def grasp(request: GraspRequest) -> GraspResponse:
    """
    Direct grasp generation without mesh/pose estimation (faster path).
    
    Flow: SAM3 segmentation -> GraspGen
    Skips mesh reconstruction and pose estimation for speed.
    """
    pipeline = await get_pipeline()
    
    try:
        # Call SAM3 first to get segmentation mask
        if request.use_box_prompt:
            sam3_payload = {
                "image_b64": request.image_rgb_b64,
                "box_prompts": [request.bbox],
                "use_box_prompt": True,
            }
        else:
            sam3_payload = {
                "image_b64": request.image_rgb_b64,
                "text_prompts": [request.label] if request.label else ["object"],
                "use_box_prompt": False,
            }
        
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{SAM3_URL}/segment",
                json=sam3_payload,
                timeout=aiohttp.ClientTimeout(total=SAM3_TIMEOUT),
            ) as resp:
                if resp.status != 200:
                    error = await resp.text()
                    raise RuntimeError(f"SAM3 error: {error}")
                sam3_result = await resp.json()

            detections = sam3_result.get("detections", [])
            if not detections:
                raise RuntimeError(f"SAM3 found no objects for label '{request.label}'")

            # Match detection to input bbox
            best_idx = 0
            best_iou = 0.0
            for i, det in enumerate(detections):
                iou = compute_iou(request.bbox, det["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_idx = i
            matched_det = detections[best_idx]
            mask_b64 = matched_det["mask"]

            logger.info(f"SAM3 completed for grasp request, IoU={best_iou:.3f}")

            # Call GraspGen with depth + mask
            # First decode depth to get shape
            depth_bytes = base64.b64decode(request.depth_b64)
            num_pixels = len(depth_bytes) // 4
            if num_pixels == 480 * 640:
                depth_shape = [480, 640]
            elif num_pixels == 720 * 1280:
                depth_shape = [720, 1280]
            else:
                depth_shape = [480, 640]  # fallback

            graspgen_payload = {
                "depth_b64": request.depth_b64,
                "depth_shape": depth_shape,
                "mask_b64": mask_b64,
                "rgb_b64": request.image_rgb_b64,
                "K": request.K,
                "filter_collisions": request.filter_collisions,
                "gripper_type": request.gripper_type,
                "num_grasps": request.num_grasps,
                "topk_num_grasps": request.topk_num_grasps,
            }

            async with session.post(
                f"{GRASPGEN_URL}/generate",
                json=graspgen_payload,
                timeout=aiohttp.ClientTimeout(total=GRASPGEN_TIMEOUT),
            ) as resp:
                if resp.status != 200:
                    error = await resp.text()
                    raise RuntimeError(f"GraspGen error: {error}")
                graspgen_result = await resp.json()

            logger.info(f"GraspGen completed, {len(graspgen_result.get('grasps', []))} grasps generated")

            # Convert grasps format
            grasps = [GraspPose(**g) for g in graspgen_result.get("grasps", [])]

            return GraspResponse(
                label=request.label,
                grasps=grasps,
                gripper_type=request.gripper_type,
                inference_time_ms=graspgen_result.get("inference_time_ms", 0),
            )

    except Exception as e:
        logger.error(f"Grasp pipeline failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host=GATEWAY_HOST, port=GATEWAY_PORT)


