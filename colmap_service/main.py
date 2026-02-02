"""COLMAP service - port 8096."""
import asyncio
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from shared.jobs import ensure_run_dirs, read_status, write_status

app = FastAPI(title="COLMAP Service", version="0.1.0")
logger = logging.getLogger("colmap_service")


class ReconstructSparseRequest(BaseModel):
    run_id: str
    images_dir: Optional[str] = Field(
        None, description="Override images dir (defaults to run frames/images)"
    )
    use_gpu: bool = Field(True, description="Use GPU for SIFT extraction/matching")
    max_image_size: int = Field(2000, ge=200, description="Max image size for SIFT")
    overlap: int = Field(10, ge=1, description="Sequential matcher overlap")
    single_camera: bool = Field(True, description="Assume a single camera intrinsics")


class SelectModelRequest(BaseModel):
    model_id: int = Field(..., ge=0, description="Sparse model index to select")


@app.get("/health")
def health():
    return {"status": "ok"}


async def _run_cmd(cmd: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("wb") as log_f:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=log_f,
            stderr=log_f,
        )
        return await proc.wait()


@app.post("/reconstruct_sparse")
async def reconstruct_sparse(request: ReconstructSparseRequest) -> Dict[str, Any]:
    paths = ensure_run_dirs(request.run_id)
    images_dir = Path(request.images_dir) if request.images_dir else paths.images_dir
    if not images_dir.exists():
        raise HTTPException(status_code=404, detail=f"Images not found: {images_dir}")

    colmap_bin = shutil.which("colmap")
    if not colmap_bin:
        raise HTTPException(status_code=500, detail="COLMAP binary not found in PATH")

    database_path = paths.colmap_dir / "database.db"
    sparse_dir = paths.colmap_dir / "sparse"
    sparse_dir.mkdir(parents=True, exist_ok=True)

    write_status(request.run_id, state="running", stage="colmap_sparse", message="starting")

    feature_cmd = [
        colmap_bin,
        "feature_extractor",
        "--database_path",
        str(database_path),
        "--image_path",
        str(images_dir),
        "--ImageReader.single_camera",
        "1" if request.single_camera else "0",
        "--SiftExtraction.max_image_size",
        str(request.max_image_size),
        "--FeatureExtraction.use_gpu",
        "1" if request.use_gpu else "0",
    ]

    match_cmd = [
        colmap_bin,
        "sequential_matcher",
        "--database_path",
        str(database_path),
        "--SequentialMatching.overlap",
        str(request.overlap),
        "--SiftMatching.use_gpu",
        "1" if request.use_gpu else "0",
    ]

    mapper_cmd = [
        colmap_bin,
        "mapper",
        "--database_path",
        str(database_path),
        "--image_path",
        str(images_dir),
        "--output_path",
        str(sparse_dir),
    ]

    rc = await _run_cmd(feature_cmd, paths.logs_dir / "colmap_feature.log")
    if rc != 0:
        write_status(request.run_id, state="failed", error=f"feature_extractor failed (rc={rc})")
        raise HTTPException(status_code=500, detail="COLMAP feature extraction failed")

    rc = await _run_cmd(match_cmd, paths.logs_dir / "colmap_match.log")
    if rc != 0:
        write_status(request.run_id, state="failed", error=f"sequential_matcher failed (rc={rc})")
        raise HTTPException(status_code=500, detail="COLMAP matching failed")

    rc = await _run_cmd(mapper_cmd, paths.logs_dir / "colmap_mapper.log")
    if rc != 0:
        write_status(request.run_id, state="failed", error=f"mapper failed (rc={rc})")
        raise HTTPException(status_code=500, detail="COLMAP mapping failed")

    models = sorted([p.name for p in sparse_dir.iterdir() if p.is_dir()])
    artifacts = read_status(request.run_id).get("artifacts", {})
    artifacts["colmap_sparse_dir"] = str(sparse_dir)
    artifacts["colmap_models"] = models
    write_status(
        request.run_id,
        state="succeeded",
        progress=1.0,
        message="colmap sparse complete",
        artifacts=artifacts,
    )
    return {"run_id": request.run_id, "sparse_dir": str(sparse_dir), "models": models}


@app.post("/runs/{run_id}/select_model")
def select_model(run_id: str, request: SelectModelRequest) -> Dict[str, Any]:
    paths = ensure_run_dirs(run_id)
    source = paths.colmap_dir / "sparse" / str(request.model_id)
    if not source.exists():
        raise HTTPException(status_code=404, detail=f"Model not found: {source}")

    target = paths.colmap_dir / "selected"
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(source, target)

    artifacts = read_status(run_id).get("artifacts", {})
    artifacts["selected_sparse_dir"] = str(target)
    write_status(run_id, message="sparse model selected", artifacts=artifacts)
    return {"run_id": run_id, "selected_sparse_dir": str(target)}


@app.get("/runs/{run_id}/status")
def run_status(run_id: str) -> Dict[str, Any]:
    return read_status(run_id)
