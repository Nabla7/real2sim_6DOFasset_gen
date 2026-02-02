"""Neural reconstruction service - port 8097."""
import asyncio
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from shared.jobs import ensure_run_dirs, read_status, write_status

app = FastAPI(title="Neural Reconstruction Service", version="0.1.0")
logger = logging.getLogger("neural_recon_service")


class PrepareDatasetRequest(BaseModel):
    run_id: str
    images_dir: Optional[str] = Field(
        None, description="Override images dir (defaults to run frames/images)"
    )
    sparse_model_dir: Optional[str] = Field(
        None, description="Override sparse model dir (defaults to run colmap/selected)"
    )


class TrainRequest(BaseModel):
    run_id: str
    config_name: str = Field("apps/colmap_3dgut_mcmc.yaml")
    experiment_name: Optional[str] = Field(None)
    overrides: List[str] = Field(default_factory=list)


class ExportRequest(BaseModel):
    run_id: str
    ply_path: Optional[str] = Field(None, description="Explicit PLY path to export")
    export_usdz: bool = Field(True, description="Export USDZ from PLY")
    render: bool = Field(False, description="Render images from checkpoint")
    checkpoint_path: Optional[str] = Field(None, description="Checkpoint for rendering")


@app.get("/health")
def health():
    return {"status": "ok"}


def _conda_env() -> str:
    return os.getenv("THREEDGRUT_ENV", "3dgrut_cuda12")


def _build_env() -> Dict[str, str]:
    env = os.environ.copy()
    env.setdefault("PYTHONNOUSERSITE", "1")
    if "TORCH_CUDA_ARCH_LIST" in os.environ:
        env["TORCH_CUDA_ARCH_LIST"] = os.environ["TORCH_CUDA_ARCH_LIST"]
    if "LD_LIBRARY_PATH" in os.environ:
        env["LD_LIBRARY_PATH"] = os.environ["LD_LIBRARY_PATH"]
    return env


async def _run_conda(cmd: List[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("wb") as log_f:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=log_f,
            stderr=log_f,
            env=_build_env(),
        )
        return await proc.wait()


def _find_latest(path: Path, pattern: str) -> Optional[Path]:
    matches = list(path.rglob(pattern))
    if not matches:
        return None
    return max(matches, key=lambda p: p.stat().st_mtime)


@app.post("/prepare_dataset")
def prepare_dataset(request: PrepareDatasetRequest) -> Dict[str, Any]:
    paths = ensure_run_dirs(request.run_id)
    images_dir = Path(request.images_dir) if request.images_dir else paths.images_dir
    if not images_dir.exists():
        raise HTTPException(status_code=404, detail=f"Images not found: {images_dir}")

    sparse_model_dir = (
        Path(request.sparse_model_dir)
        if request.sparse_model_dir
        else paths.colmap_dir / "selected"
    )
    if not sparse_model_dir.exists():
        raise HTTPException(status_code=404, detail=f"Sparse model not found: {sparse_model_dir}")

    dataset_images = paths.dataset_dir / "images"
    dataset_sparse = paths.dataset_dir / "sparse" / "0"
    dataset_sparse.parent.mkdir(parents=True, exist_ok=True)

    if dataset_images.exists() or dataset_images.is_symlink():
        dataset_images.unlink()
    dataset_images.symlink_to(images_dir)

    if dataset_sparse.exists():
        shutil.rmtree(dataset_sparse)
    shutil.copytree(sparse_model_dir, dataset_sparse)

    artifacts = read_status(request.run_id).get("artifacts", {})
    artifacts["dataset_dir"] = str(paths.dataset_dir)
    write_status(
        request.run_id,
        stage="dataset_prep",
        progress=1.0,
        message="dataset prepared",
        artifacts=artifacts,
    )
    return {"run_id": request.run_id, "dataset_dir": str(paths.dataset_dir)}


@app.post("/train_3dgrut")
async def train_3dgrut(request: TrainRequest) -> Dict[str, Any]:
    paths = ensure_run_dirs(request.run_id)
    dataset_dir = paths.dataset_dir
    if not (dataset_dir / "images").exists():
        raise HTTPException(status_code=400, detail="Dataset not prepared")

    conda_bin = shutil.which("conda")
    if not conda_bin:
        raise HTTPException(status_code=500, detail="conda not found in PATH")

    repo_root = Path(__file__).resolve().parents[1]
    grut_root = repo_root / "third_party" / "3dgrut"

    experiment_name = request.experiment_name or request.run_id
    out_dir = paths.recon_dir

    cmd = [
        conda_bin,
        "run",
        "-n",
        _conda_env(),
        "python",
        str(grut_root / "train.py"),
        "--config-name",
        request.config_name,
        f"path={dataset_dir}",
        f"out_dir={out_dir}",
        f"experiment_name={experiment_name}",
    ]
    cmd.extend(request.overrides)

    write_status(request.run_id, state="running", stage="3dgrut_train", message="training")
    rc = await _run_conda(cmd, paths.logs_dir / "3dgrut_train.log")
    if rc != 0:
        write_status(request.run_id, state="failed", error=f"3dgrut train failed (rc={rc})")
        raise HTTPException(status_code=500, detail="3DGUT training failed")

    artifacts = read_status(request.run_id).get("artifacts", {})
    artifacts["3dgrut_out_dir"] = str(out_dir)
    write_status(
        request.run_id,
        state="succeeded",
        progress=1.0,
        message="3dgrut training complete",
        artifacts=artifacts,
    )
    return {"run_id": request.run_id, "out_dir": str(out_dir)}


@app.post("/export")
async def export_assets(request: ExportRequest) -> Dict[str, Any]:
    paths = ensure_run_dirs(request.run_id)
    conda_bin = shutil.which("conda")
    if not conda_bin:
        raise HTTPException(status_code=500, detail="conda not found in PATH")

    ply_path = Path(request.ply_path) if request.ply_path else _find_latest(paths.recon_dir, "*.ply")
    if request.export_usdz and not ply_path:
        raise HTTPException(status_code=404, detail="PLY file not found for USDZ export")

    artifacts = read_status(request.run_id).get("artifacts", {})

    if request.export_usdz and ply_path:
        export_script = (
            Path(__file__).resolve().parents[1]
            / "third_party"
            / "3dgrut"
            / "threedgrut"
            / "export"
            / "scripts"
            / "ply_to_usd.py"
        )
        usdz_path = ply_path.with_suffix(".usdz")
        cmd = [
            conda_bin,
            "run",
            "-n",
            _conda_env(),
            "python",
            str(export_script),
            str(ply_path),
            "--output_file",
            str(usdz_path),
        ]
        write_status(request.run_id, stage="export", message="exporting usdz")
        rc = await _run_conda(cmd, paths.logs_dir / "3dgrut_export_usdz.log")
        if rc != 0:
            write_status(request.run_id, state="failed", error=f"usdz export failed (rc={rc})")
            raise HTTPException(status_code=500, detail="USDZ export failed")
        artifacts["export_usdz"] = str(usdz_path)

    if request.render:
        checkpoint = Path(request.checkpoint_path) if request.checkpoint_path else _find_latest(
            paths.recon_dir, "*.pt"
        )
        if not checkpoint:
            raise HTTPException(status_code=404, detail="Checkpoint not found for rendering")
        renders_dir = paths.recon_dir / "renders"
        renders_dir.mkdir(parents=True, exist_ok=True)
        render_cmd = [
            conda_bin,
            "run",
            "-n",
            _conda_env(),
            "python",
            str(Path(__file__).resolve().parents[1] / "third_party" / "3dgrut" / "render.py"),
            "--checkpoint",
            str(checkpoint),
            "--out-dir",
            str(renders_dir),
        ]
        write_status(request.run_id, stage="export", message="rendering")
        rc = await _run_conda(render_cmd, paths.logs_dir / "3dgrut_render.log")
        if rc != 0:
            write_status(request.run_id, state="failed", error=f"render failed (rc={rc})")
            raise HTTPException(status_code=500, detail="Render failed")
        artifacts["renders_dir"] = str(renders_dir)

    write_status(request.run_id, message="export complete", artifacts=artifacts)
    return {"run_id": request.run_id, "artifacts": artifacts}


@app.get("/runs/{run_id}/status")
def run_status(run_id: str) -> Dict[str, Any]:
    return read_status(run_id)
