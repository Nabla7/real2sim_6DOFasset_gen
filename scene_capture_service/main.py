"""Scene capture service - port 8095."""
import asyncio
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, Optional
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from shared.jobs import ensure_run_dirs, get_repo_root, read_status, write_status

app = FastAPI(title="Scene Capture Service", version="0.1.0")
logger = logging.getLogger("scene_capture_service")


class CreateRunRequest(BaseModel):
    run_id: Optional[str] = Field(None, description="Optional run_id to reuse")


class ExtractFramesRequest(BaseModel):
    video_path: Optional[str] = Field(
        None, description="Override input video path (defaults to run input/video.mov)"
    )
    fps: float = Field(3.0, ge=0.1, description="Target extraction FPS")
    width: int = Field(1920, ge=0, description="Resize width; 0 keeps original")
    format: str = Field("jpg", description="Output format: jpg|png")
    jpeg_quality: int = Field(2, ge=1, le=31, description="ffmpeg JPEG quality (1-31)")
    dedupe: bool = Field(False, description="Enable near-duplicate removal")
    scene: Optional[float] = Field(None, ge=0.0, le=1.0, description="Scene threshold")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/runs/create")
def create_run(request: Optional[CreateRunRequest] = None) -> Dict[str, Any]:
    run_id = request.run_id if request and request.run_id else uuid4().hex
    ensure_run_dirs(run_id)
    write_status(run_id, state="queued", stage="extract_frames", progress=0.0)
    return {"run_id": run_id}


@app.post("/runs/{run_id}/upload_video")
def upload_video(run_id: str, file: UploadFile = File(...)) -> Dict[str, Any]:
    paths = ensure_run_dirs(run_id)
    video_path = paths.input_dir / "video.mov"
    try:
        with video_path.open("wb") as f:
            shutil.copyfileobj(file.file, f)
    finally:
        file.file.close()

    write_status(run_id, message="video uploaded")
    return {"run_id": run_id, "video_path": str(video_path)}


@app.post("/runs/{run_id}/extract_frames")
async def extract_frames(run_id: str, request: ExtractFramesRequest) -> Dict[str, Any]:
    paths = ensure_run_dirs(run_id)
    script_path = get_repo_root() / "scripts" / "preprocess" / "extract_frames.sh"
    if not script_path.exists():
        raise HTTPException(status_code=500, detail="extract_frames.sh not found")

    video_path = Path(request.video_path) if request.video_path else paths.input_dir / "video.mov"
    if not video_path.exists():
        raise HTTPException(status_code=404, detail=f"Video not found: {video_path}")

    if request.format not in {"jpg", "jpeg", "png"}:
        raise HTTPException(status_code=400, detail="format must be jpg or png")

    log_path = paths.logs_dir / "extract_frames.log"
    write_status(run_id, state="running", stage="extract_frames", message="starting")

    cmd = [
        str(script_path),
        str(video_path),
        "--out_dir",
        str(paths.root),
        "--name",
        "frames",
        "--fps",
        str(request.fps),
        "--width",
        str(request.width),
        "--format",
        request.format,
        "--jpeg_quality",
        str(request.jpeg_quality),
    ]
    if request.dedupe:
        cmd.append("--dedupe")
    if request.scene is not None:
        cmd.extend(["--scene", str(request.scene)])

    with log_path.open("wb") as log_f:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=log_f,
            stderr=log_f,
        )
        write_status(run_id, pid=proc.pid, message="extracting frames")
        rc = await proc.wait()

    if rc != 0:
        write_status(run_id, state="failed", error=f"extract_frames failed (rc={rc})")
        raise HTTPException(status_code=500, detail="Frame extraction failed")

    artifacts = read_status(run_id).get("artifacts", {})
    artifacts["frames_dir"] = str(paths.images_dir)
    write_status(
        run_id,
        state="succeeded",
        progress=1.0,
        message="frames extracted",
        artifacts=artifacts,
    )
    return {"run_id": run_id, "frames_dir": str(paths.images_dir)}


@app.get("/runs/{run_id}/status")
def run_status(run_id: str) -> Dict[str, Any]:
    return read_status(run_id)
