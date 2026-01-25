"""
Scene Capture Service

Manages capture sessions for robot cameras (RealSense, ZED).
Captures images for COLMAP reconstruction pipeline.
"""

import json
import uuid
import time
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from utils import get_token
import uvicorn
import numpy as np
import cv2
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from drivers.realsense import RealSenseDriver

# Global driver instance
driver: Optional[RealSenseDriver] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup resources."""
    global driver
    # Startup: Connect to camera
    print("Connecting to RealSense camera...")
    driver = RealSenseDriver()
    driver.connect()
    print(f"Camera connected: {driver.get_intrinsics()}")

    yield

    # Shutdown: Disconnect camera
    if driver:
        print("Disconnecting camera...")
        driver.disconnect()


app = FastAPI(
    title="Scene Capture Service",
    description="Capture images from robot cameras for COLMAP reconstruction",
    version="1.0.0",
    lifespan=lifespan,
)

# Session storage
_sessions: Dict[str, dict] = {}
CAPTURE_DIR = Path("captures")


# Pydantic models
class SessionCreate(BaseModel):
    name: Optional[str] = Field(None, description="Optional session name")


class SessionResponse(BaseModel):
    session_id: str
    name: Optional[str]
    status: str
    created_at: str
    capture_count: int
    output_dir: str


class CaptureRequest(BaseModel):
    pose: Optional[Dict[str, float]] = Field(
        None, description="Optional robot pose (x, y, z, rx, ry, rz)"
    )


class CaptureResponse(BaseModel):
    capture_id: int
    timestamp: str
    image_path: str


class AutoCaptureRequest(BaseModel):
    num_captures: int = Field(10, description="Number of frames to capture", ge=1)
    interval: float = Field(0.5, description="Interval between captures in seconds", ge=0.1)


class StatusResponse(BaseModel):
    session_id: str
    status: str
    capture_count: int
    created_at: str
    finalized_at: Optional[str]


def get_driver() -> RealSenseDriver:
    """Get the camera driver or raise error."""
    if driver is None:
        raise HTTPException(status_code=503, detail="Camera not initialized")
    return driver


def get_session(session_id: str) -> dict:
    """Get a session by ID or raise 404."""
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")
    return _sessions[session_id]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/sessions/create", response_model=SessionResponse)
def create_session(request: SessionCreate):
    """Create a new capture session."""
    session_id = str(uuid.uuid4())
    session_dir = CAPTURE_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    session_data = {
        "session_id": session_id,
        "name": request.name,
        "status": "active",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "finalized_at": None,
        "capture_count": 0,
        "output_dir": str(session_dir),
        "captures": [],
    }

    _sessions[session_id] = session_data

    # Save camera intrinsics for COLMAP
    intrinsics = get_driver().get_intrinsics()
    intrinsics_data = {
        "width": intrinsics.width,
        "height": intrinsics.height,
        "fx": intrinsics.fx,
        "fy": intrinsics.fy,
        "ppx": intrinsics.ppx,
        "ppy": intrinsics.ppy,
        "model": str(intrinsics.model),
        "coeffs": intrinsics.coeffs,
    }

    with open(session_dir / "camera_intrinsics.json", "w") as f:
        json.dump(intrinsics_data, f, indent=2)

    return SessionResponse(**session_data)


@app.post("/sessions/{session_id}/capture", response_model=CaptureResponse)
def capture_frame(session_id: str, request: CaptureRequest):
    """Capture a single frame in the session."""
    session = get_session(session_id)

    if session["status"] != "active":
        raise HTTPException(status_code=400, detail="Session is not active")

    # Capture frame from camera
    frame_data = get_driver().capture_frame()

    # Convert bytes to numpy array and save as image
    capture_id = session["capture_count"]
    timestamp = datetime.now(timezone.utc).isoformat()
    image_filename = f"frame_{capture_id:04d}.png"
    image_path = Path(session["output_dir"]) / image_filename

    # Save the frame (assuming BGR format from RealSense)
    frame_array = np.frombuffer(frame_data, dtype=np.uint8).reshape((480, 640, 3))
    cv2.imwrite(str(image_path), frame_array)

    # Store capture metadata
    capture_info = {
        "capture_id": capture_id,
        "timestamp": timestamp,
        "image_path": str(image_path),
        "pose": request.pose,
    }

    session["captures"].append(capture_info)
    session["capture_count"] += 1

    return CaptureResponse(
        capture_id=capture_id,
        timestamp=timestamp,
        image_path=str(image_path),
    )


@app.post("/sessions/{session_id}/auto")
def auto_capture(session_id: str, request: AutoCaptureRequest):
    """Automatically capture multiple frames at regular intervals."""
    session = get_session(session_id)

    if session["status"] != "active":
        raise HTTPException(status_code=400, detail="Session is not active")

    captured = []

    for i in range(request.num_captures):
        # Capture frame
        frame_data = get_driver().capture_frame()

        capture_id = session["capture_count"]
        timestamp = datetime.now(timezone.utc).isoformat()
        image_filename = f"frame_{capture_id:04d}.png"
        image_path = Path(session["output_dir"]) / image_filename

        # Save the frame
        frame_array = np.frombuffer(frame_data, dtype=np.uint8).reshape((480, 640, 3))
        cv2.imwrite(str(image_path), frame_array)

        capture_info = {
            "capture_id": capture_id,
            "timestamp": timestamp,
            "image_path": str(image_path),
            "pose": None,
        }

        session["captures"].append(capture_info)
        session["capture_count"] += 1

        captured.append({
            "capture_id": capture_id,
            "timestamp": timestamp,
            "image_path": str(image_path),
        })

        # Wait before next capture (except for the last one)
        if i < request.num_captures - 1:
            time.sleep(request.interval)

    return {
        "session_id": session_id,
        "captures": captured,
        "total_captured": len(captured),
    }


@app.get("/sessions/{session_id}/status", response_model=StatusResponse)
def get_status(session_id: str):
    """Get the status of a capture session."""
    session = get_session(session_id)

    return StatusResponse(
        session_id=session["session_id"],
        status=session["status"],
        capture_count=session["capture_count"],
        created_at=session["created_at"],
        finalized_at=session["finalized_at"],
    )


@app.post("/sessions/{session_id}/finalize")
def finalize_session(session_id: str):
    """Finalize a session and prepare data for COLMAP processing."""
    session = get_session(session_id)

    if session["status"] != "active":
        raise HTTPException(status_code=400, detail="Session is not active")

    # Mark session as finalized
    session["status"] = "finalized"
    session["finalized_at"] = datetime.now(timezone.utc).isoformat()

    # Save session metadata
    session_dir = Path(session["output_dir"])
    metadata = {
        "session_id": session["session_id"],
        "name": session["name"],
        "created_at": session["created_at"],
        "finalized_at": session["finalized_at"],
        "capture_count": session["capture_count"],
        "captures": session["captures"],
    }

    with open(session_dir / "session_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    return {
        "session_id": session_id,
        "status": "finalized",
        "capture_count": session["capture_count"],
        "output_dir": session["output_dir"],
        "message": "Session finalized successfully. Ready for COLMAP processing.",
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(get_token("PORT")))
