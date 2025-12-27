#!/usr/bin/env python3
"""
SAM 3D Objects Reconstruction Service (Worker Process Architecture)

Runs the SAM 3D Objects pipeline in a separate process to allow
complete GPU memory reclamation on unload.

Enhanced with depth-based pointmap scaling using RealSense depth.
"""

import base64
import io
import logging
import multiprocessing as mp
import os
import sys
import uuid
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel

# Configure logging
LOG_DIR = Path(os.getenv("LOG_DIR", "/workspace/logs"))
LOG_DIR.mkdir(parents=True, exist_ok=True)


def _configure_logging(name: str = "sam3d_service") -> logging.Logger:
    logger = logging.getLogger(f"spatial_memory.{name}")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        logfile = LOG_DIR / "sam3d_service.log"
        file_handler = RotatingFileHandler(
            logfile, maxBytes=10_000_000, backupCount=5
        )
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    return logger


logger = _configure_logging()

# Configuration
WEIGHTS_DIR = os.getenv("WEIGHTS_DIR", "/weights")
SAM3D_WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, "sam3d-objects")
SAM3D_CONFIG = os.path.join(SAM3D_WEIGHTS_PATH, "pipeline.yaml")
OUTPUT_DIR = Path(os.getenv("MESH_OUTPUT_DIR", "/tmp/spatial_memory/meshes"))

# If set, ignore depth_b64/K pointmap scaling and let SAM3D use its internal scaling.
# Default: disabled (internal scaling) unless explicitly enabled.
DISABLE_POINTMAP_SCALING = os.getenv("SAM3D_DISABLE_POINTMAP_SCALING", "1").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
    "on",
}

# Pointmap validity thresholds (mask-aware)
# If too few pixels inside the mask have valid depth, passing a pointmap tends to
# produce badly scaled meshes and cascades into FoundationPose failures.
POINTMAP_MIN_VALID_RATIO = float(os.getenv("SAM3D_POINTMAP_MIN_VALID_RATIO", "0.30"))
POINTMAP_MIN_VALID_PIXELS = int(os.getenv("SAM3D_POINTMAP_MIN_VALID_PIXELS", "2000"))


def compute_pointmap_from_depth(depth: np.ndarray, K: np.ndarray) -> np.ndarray:
    """
    Convert depth image to 3D pointmap using camera intrinsics.

    Args:
        depth: Depth image in meters, shape (H, W)
        K: Camera intrinsics 3x3 matrix [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]

    Returns:
        Pointmap of shape (H, W, 3) with XYZ coordinates in meters
    """
    h, w = depth.shape[:2]
    K = np.array(K)

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # Create pixel coordinate grids
    u, v = np.meshgrid(np.arange(w), np.arange(h))

    # Convert invalid depth values (0 or very small) to NaN
    # RealSense and other depth sensors use 0 for invalid measurements
    z = depth.copy()
    z[z < 0.01] = np.nan  # Less than 1cm is likely invalid

    # Project to 3D
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    pointmap = np.stack([x, y, z], axis=-1).astype(np.float32)
    return pointmap


# --- Worker Process for Inference ---


class InferenceWorker(mp.Process):
    """Worker process that loads SAM3D model and processes requests."""

    def __init__(self, task_queue, result_queue, config_path, output_dir):
        super().__init__()
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.config_path = str(config_path)
        self.output_dir = str(output_dir)
        self._inference = None

    def run(self):
        # Re-configure logging in worker process
        worker_logger = _configure_logging("sam3d_worker")
        worker_logger.info("Worker process started. Initializing SAM 3D pipeline...")

        # Add SAM3D to path (following official demo.py pattern)
        sam3d_path = os.getenv("SAM3D_PATH", "/workspace/third_party/sam-3d-objects")
        notebook_path = os.path.join(sam3d_path, "notebook")
        if os.path.exists(notebook_path) and notebook_path not in sys.path:
            sys.path.insert(0, notebook_path)
        if os.path.exists(sam3d_path) and sam3d_path not in sys.path:
            sys.path.insert(0, sam3d_path)

        try:
            # Import following official demo.py pattern: from inference import Inference
            from inference import Inference

            self._inference = Inference(self.config_path, compile=False)
            worker_logger.info("SAM 3D pipeline initialized successfully in worker.")
        except Exception as e:
            worker_logger.error(f"Failed to initialize inference: {e}")
            # Signal failure for any pending tasks
            while True:
                task = self.task_queue.get()
                if task is None:
                    break
                self.result_queue.put({"error": str(e)})
            return

        while True:
            task = self.task_queue.get()
            if task is None:  # Sentinel to stop
                break

            req_id, image_np, mask_np, seed, pointmap = task
            worker_logger.info(f"Processing request {req_id}")

            try:
                # Call inference with optional pointmap for depth-based scaling
                if pointmap is not None:
                    worker_logger.info(f"Using provided pointmap for scaling: {pointmap.shape}")
                    # Convert numpy pointmap to torch tensor (expected by SAM3D pipeline)
                    pointmap_tensor = torch.from_numpy(pointmap).float()
                    output = self._inference(image_np, mask_np, seed=seed, pointmap=pointmap_tensor)
                else:
                    worker_logger.info("No pointmap provided, using internal depth estimation")
                    output = self._inference(image_np, mask_np, seed=seed)

                # Save outputs
                ply_output_path = Path(self.output_dir) / f"{req_id}.ply"
                mesh_output_path = Path(self.output_dir) / f"{req_id}.obj"

                result = {"ply_path": None, "mesh_path": None, "mesh_id": None}

                if "gs" in output:
                    output["gs"].save_ply(str(ply_output_path))
                    result["ply_path"] = str(ply_output_path)
                    worker_logger.info(f"Saved PLY to {ply_output_path}")

                if "glb" in output and output["glb"] is not None:
                    try:
                        mesh_glb = output["glb"]
                        
                        # Apply scale from pointmap normalization to transform from
                        # canonical space to real-world metric scale
                        if "scale" in output and output["scale"] is not None:
                            scale_tensor = output["scale"]
                            if hasattr(scale_tensor, 'squeeze'):
                                # Tensor: scale is typically (1, 3) or (3,), use first value for uniform
                                scale_val = float(scale_tensor.squeeze()[0].item())
                            else:
                                scale_val = float(scale_tensor)
                            
                            if scale_val > 0 and scale_val != 1.0:
                                mesh_glb.vertices *= scale_val
                                worker_logger.info(f"Applied scale factor {scale_val:.4f} to mesh")
                            else:
                                worker_logger.info(f"Scale factor is {scale_val:.4f}, skipping scaling")
                        else:
                            worker_logger.warning("No scale in output, mesh will be in canonical space")
                        
                        mesh_glb.export(str(mesh_output_path))
                        result["mesh_path"] = str(mesh_output_path)
                        result["mesh_id"] = req_id
                        worker_logger.info(f"Saved OBJ to {mesh_output_path}")

                        # Decimate mesh to reduce file size
                        try:
                            import open3d as o3d
                            mesh = o3d.io.read_triangle_mesh(str(mesh_output_path))
                            original_triangles = len(mesh.triangles)
                            
                            # Target: reduce to ~50k triangles or 10% of original, whichever is smaller
                            target_triangles = min(50000, int(original_triangles * 0.1))
                            if original_triangles > target_triangles:
                                mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=target_triangles)
                                o3d.io.write_triangle_mesh(str(mesh_output_path), mesh)
                                worker_logger.info(
                                    f"Decimated mesh: {original_triangles} -> {len(mesh.triangles)} triangles"
                                )
                            else:
                                worker_logger.info(
                                    f"Mesh already small ({original_triangles} triangles), skipping decimation"
                                )
                        except Exception as e:
                            worker_logger.warning(f"Mesh decimation failed (using original): {e}")

                    except Exception as e:
                        worker_logger.warning(f"Failed to export mesh: {e}")

                self.result_queue.put(result)

            except Exception as e:
                worker_logger.error(f"Inference failed: {e}")
                self.result_queue.put({"error": str(e)})

        worker_logger.info("Worker process stopping.")


# --- Global State Manager ---


class ServiceState:
    def __init__(self):
        self.worker: Optional[InferenceWorker] = None
        # Use 'spawn' context for queues
        ctx = mp.get_context("spawn")
        self.task_queue = ctx.Queue()
        self.result_queue = ctx.Queue()
        self.config_path: Optional[Path] = None
        self.output_dir = OUTPUT_DIR

    def ensure_worker(self):
        if self.worker is not None and self.worker.is_alive():
            return

        if self.config_path is None:
            self.config_path = Path(SAM3D_CONFIG)

        if not self.config_path.exists():
            raise RuntimeError(f"Config not found at {self.config_path}")

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create new queues for the new worker
        ctx = mp.get_context("spawn")
        self.task_queue = ctx.Queue()
        self.result_queue = ctx.Queue()

        logger.info("Spawning new inference worker...")
        self.worker = InferenceWorker(
            self.task_queue,
            self.result_queue,
            self.config_path,
            self.output_dir,
        )
        self.worker.start()

    def unload(self):
        if self.worker is not None:
            logger.info("Terminating worker process...")
            self.worker.terminate()
            self.worker.join()
            self.worker = None

            # Clean queues
            while not self.task_queue.empty():
                self.task_queue.get()
            while not self.result_queue.empty():
                self.result_queue.get()

            return True
        return False


_state = ServiceState()


# --- API Models ---


class ReconstructRequest(BaseModel):
    image_b64: str
    mask_b64: str
    depth_b64: Optional[str] = None  # RealSense depth for pointmap scaling
    K: Optional[List[List[float]]] = None  # Camera intrinsics 3x3


class ReconstructResponse(BaseModel):
    ply_path: Optional[str] = None
    mesh_path: Optional[str] = None
    mesh_id: Optional[str] = None  # Mesh ID for download via GET /mesh/{id}


# --- FastAPI App ---

app = FastAPI(title="SAM 3D Objects Service (with Depth Scaling)", version="2.0.0")


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "worker_alive": _state.worker is not None and _state.worker.is_alive(),
        "output_dir": str(_state.output_dir),
        "weights_path": SAM3D_WEIGHTS_PATH,
    }


@app.post("/reconstruct", response_model=ReconstructResponse)
def reconstruct(req: ReconstructRequest) -> ReconstructResponse:
    """
    Reconstruct a 3D mesh from an image and mask.

    If depth_b64 and K are provided, uses RealSense depth to compute
    a pointmap for accurate real-world scaling of the mesh.
    """
    try:
        _state.ensure_worker()

        # Decode RGB image
        img_b64 = req.image_b64.split(",", 1)[1] if "," in req.image_b64 else req.image_b64
        img_bytes = base64.b64decode(img_b64)
        pil_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        image_np = np.array(pil_image)

        # Decode mask
        mask_b64 = req.mask_b64.split(",", 1)[1] if "," in req.mask_b64 else req.mask_b64
        mask_bytes = base64.b64decode(mask_b64)
        mask_img = Image.open(io.BytesIO(mask_bytes)).convert("L")
        mask_np = np.array(mask_img) > 0

        # Compute pointmap from depth if provided (unless disabled)
        pointmap = None
        if DISABLE_POINTMAP_SCALING:
            if req.depth_b64 or req.K:
                logger.warning(
                    "SAM3D pointmap scaling is DISABLED (SAM3D_DISABLE_POINTMAP_SCALING=1). "
                    "Ignoring depth_b64/K and using internal SAM3D scaling."
                )
        elif req.depth_b64 and req.K:
            logger.info("Computing pointmap from RealSense depth...")
            depth_b64 = req.depth_b64.split(",", 1)[1] if "," in req.depth_b64 else req.depth_b64
            depth_bytes = base64.b64decode(depth_b64)

            # Depth is expected as float32 raw bytes or PNG
            try:
                # Try as raw float32
                depth_np = np.frombuffer(depth_bytes, dtype=np.float32).reshape(
                    image_np.shape[0], image_np.shape[1]
                )
            except ValueError:
                # Try as PNG (uint16 in mm, convert to meters)
                depth_img = Image.open(io.BytesIO(depth_bytes))
                depth_np = np.array(depth_img).astype(np.float32) / 1000.0

            # Mask-aware validation: only trust depth scaling if enough of the mask
            # has valid (finite, >1cm) depth. Otherwise, fall back to internal scaling.
            mask_area = int(mask_np.sum())
            depth_masked = depth_np.astype(np.float32, copy=False).copy()
            depth_masked[~mask_np] = np.nan

            valid = np.isfinite(depth_masked) & (depth_masked >= 0.01)
            valid_pixels = int(valid.sum())
            valid_ratio = (valid_pixels / mask_area) if mask_area > 0 else 0.0

            # Log meaningful ranges (nan-safe)
            try:
                dmin = float(np.nanmin(depth_masked))
                dmax = float(np.nanmax(depth_masked))
                depth_range_str = f"[{dmin:.3f}, {dmax:.3f}]"
            except Exception:
                depth_range_str = "[nan, nan]"

            logger.info(
                "Depth(masked) stats: "
                f"mask_area={mask_area}, valid_pixels={valid_pixels}, "
                f"valid_ratio={valid_ratio:.3f}, range={depth_range_str}"
            )

            if (
                mask_area >= 1
                and valid_pixels >= POINTMAP_MIN_VALID_PIXELS
                and valid_ratio >= POINTMAP_MIN_VALID_RATIO
            ):
                pointmap = compute_pointmap_from_depth(depth_masked, np.array(req.K))
                try:
                    pmin = float(np.nanmin(pointmap))
                    pmax = float(np.nanmax(pointmap))
                    prange = f"[{pmin:.3f}, {pmax:.3f}]"
                except Exception:
                    prange = "[nan, nan]"
                logger.info(f"Pointmap computed: shape={pointmap.shape}, range={prange}")
            else:
                logger.warning(
                    "Insufficient valid in-mask depth for pointmap scaling; "
                    "falling back to internal SAM3D scaling. "
                    f"(min_valid_ratio={POINTMAP_MIN_VALID_RATIO:.2f}, "
                    f"min_valid_pixels={POINTMAP_MIN_VALID_PIXELS})"
                )

        req_id = str(uuid.uuid4())
        logger.info(f"Queueing reconstruction request {req_id}")

        _state.task_queue.put((req_id, image_np, mask_np, 42, pointmap))

        # Wait for result (5 min timeout for complex reconstructions)
        result = _state.result_queue.get(timeout=300)

        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])

        return ReconstructResponse(
            ply_path=result.get("ply_path"),
            mesh_path=result.get("mesh_path"),
            mesh_id=result.get("mesh_id"),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Reconstruction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/unload")
def unload() -> Dict[str, Any]:
    if _state.unload():
        return {"ok": True, "message": "Worker terminated, GPU freed"}
    return {"ok": True, "message": "Worker was not running"}


if __name__ == "__main__":
    # Set start method to 'spawn' to avoid CUDA initialization issues
    mp.set_start_method("spawn", force=True)
    port = int(os.getenv("SAM3D_PORT", "8092"))
    uvicorn.run(app, host="0.0.0.0", port=port)


