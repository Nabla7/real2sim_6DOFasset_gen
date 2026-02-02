"""Shared run directory and status helpers for scene reconstruction jobs."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


RUN_STATES = {"queued", "running", "succeeded", "failed", "cancelled"}
RUN_STAGES = {
    "extract_frames",
    "colmap_sparse",
    "dataset_prep",
    "3dgrut_train",
    "export",
}


@dataclass
class RunPaths:
    run_id: str
    root: Path
    input_dir: Path
    frames_dir: Path
    images_dir: Path
    colmap_dir: Path
    dataset_dir: Path
    recon_dir: Path
    logs_dir: Path
    status_file: Path


def get_repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def get_runs_root() -> Path:
    return get_repo_root() / "data" / "runs"


def get_run_paths(run_id: str) -> RunPaths:
    root = get_runs_root() / run_id
    return RunPaths(
        run_id=run_id,
        root=root,
        input_dir=root / "input",
        frames_dir=root / "frames",
        images_dir=root / "frames" / "images",
        colmap_dir=root / "colmap",
        dataset_dir=root / "dataset",
        recon_dir=root / "3dgrut",
        logs_dir=root / "logs",
        status_file=root / "status.json",
    )


def ensure_run_dirs(run_id: str) -> RunPaths:
    paths = get_run_paths(run_id)
    paths.input_dir.mkdir(parents=True, exist_ok=True)
    paths.frames_dir.mkdir(parents=True, exist_ok=True)
    paths.images_dir.mkdir(parents=True, exist_ok=True)
    paths.colmap_dir.mkdir(parents=True, exist_ok=True)
    paths.dataset_dir.mkdir(parents=True, exist_ok=True)
    paths.recon_dir.mkdir(parents=True, exist_ok=True)
    paths.logs_dir.mkdir(parents=True, exist_ok=True)
    return paths


def init_status(run_id: str) -> Dict[str, Any]:
    return {
        "run_id": run_id,
        "state": "queued",
        "stage": "extract_frames",
        "progress": 0.0,
        "message": "queued",
        "artifacts": {},
        "error": None,
        "pid": None,
        "created_at": time.time(),
        "updated_at": time.time(),
    }


def read_status(run_id: str) -> Dict[str, Any]:
    paths = get_run_paths(run_id)
    if not paths.status_file.exists():
        return init_status(run_id)
    return json.loads(paths.status_file.read_text())


def write_status(
    run_id: str,
    *,
    state: Optional[str] = None,
    stage: Optional[str] = None,
    progress: Optional[float] = None,
    message: Optional[str] = None,
    artifacts: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None,
    pid: Optional[int] = None,
) -> Dict[str, Any]:
    paths = ensure_run_dirs(run_id)
    status = read_status(run_id)

    if state is not None:
        if state not in RUN_STATES:
            raise ValueError(f"Invalid state: {state}")
        status["state"] = state
    if stage is not None:
        if stage not in RUN_STAGES:
            raise ValueError(f"Invalid stage: {stage}")
        status["stage"] = stage
    if progress is not None:
        status["progress"] = float(progress)
    if message is not None:
        status["message"] = message
    if artifacts is not None:
        status["artifacts"] = artifacts
    if error is not None:
        status["error"] = error
    if pid is not None:
        status["pid"] = pid

    status["updated_at"] = time.time()
    paths.status_file.write_text(json.dumps(status, indent=2))
    return status
