"""
Spatial Memory Gateway Service

Main FastAPI application that exposes the /process endpoint
and orchestrates the SAM3 -> SAM3D -> FoundationPose pipeline.
"""

import logging
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .config import GATEWAY_HOST, GATEWAY_PORT
from .pipeline import get_pipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("spatial_memory.gateway")


# --- Pydantic Models ---


class ProcessRequest(BaseModel):
    """Request to process an object detection through the spatial memory pipeline."""

    image_rgb_b64: str = Field(..., description="Base64-encoded RGB image")
    depth_b64: str = Field(..., description="Base64-encoded depth image (meters, float32)")
    K: List[List[float]] = Field(
        ..., description="Camera intrinsics matrix 3x3: [[fx,0,cx],[0,fy,cy],[0,0,1]]"
    )
    label: str = Field(..., description="Object label from YOLO-E detection")
    bbox: List[float] = Field(
        ..., description="2D bounding box [x1, y1, x2, y2] from YOLO-E"
    )


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


class ProcessResponse(BaseModel):
    """Response from the spatial memory pipeline."""

    label: str
    mesh_b64: Optional[str] = Field(None, description="Base64-encoded .obj mesh file")
    pose: Optional[Pose] = Field(None, description="6D pose in camera frame")
    bbox_3d: Optional[BBox3D] = Field(None, description="3D bounding box dimensions")
    confidence: float = Field(1.0, description="Detection confidence")


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

        return ProcessResponse(
            label=result.get("label", request.label),
            mesh_b64=result.get("mesh_b64"),
            pose=pose,
            bbox_3d=bbox_3d,
            confidence=result.get("confidence", 1.0),
        )

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host=GATEWAY_HOST, port=GATEWAY_PORT)


