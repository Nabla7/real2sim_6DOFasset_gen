#!/usr/bin/env python3
"""
FoundationPose 6D Pose Estimation Service (Worker Process Architecture)

Runs FoundationPose in a separate process to allow complete GPU memory
reclamation on unload.

Provides:
- POST /register: Initial pose estimation for a new object
- POST /track: Track an already-registered object (for future use)
"""

import base64
import io
import logging
import multiprocessing as mp
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

import cv2
import numpy as np
import trimesh
import uvicorn
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel

# --- Paths (repo-local by default; Docker still works) ---
REPO_DIR = Path(__file__).resolve().parents[1]
WORKSPACE_DIR = Path(os.getenv("WORKSPACE_DIR", str(REPO_DIR)))
THIRD_PARTY_DIR = Path(os.getenv("THIRD_PARTY_DIR", str(WORKSPACE_DIR / "third_party")))
DEFAULT_LOG_DIR = Path("/tmp/spatial_memory/logs")
DEFAULT_DEBUG_DIR = Path("/tmp/spatial_memory/debug")

# Configure logging
LOG_DIR = Path(
    os.getenv(
        "LOG_DIR",
        str(DEFAULT_LOG_DIR if DEFAULT_LOG_DIR.exists() else WORKSPACE_DIR / "logs"),
    )
)
LOG_DIR.mkdir(parents=True, exist_ok=True)


def _configure_logging(name: str = "foundationpose_service") -> logging.Logger:
    logger = logging.getLogger(f"spatial_memory.{name}")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        logfile = LOG_DIR / "foundationpose_service.log"
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
DEFAULT_WEIGHTS_DIR = "/weights" if Path("/weights").exists() else str(WORKSPACE_DIR / "weights")
WEIGHTS_DIR = os.getenv("WEIGHTS_DIR", DEFAULT_WEIGHTS_DIR)
FP_WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, "foundationpose")
FP_REFINER_DIR = os.path.join(FP_WEIGHTS_PATH, "2023-10-28-18-33-37")
FP_SCORER_DIR = os.path.join(FP_WEIGHTS_PATH, "2024-01-11-20-02-45")

DEBUG_DIR = Path(
    os.getenv(
        "DEBUG_DIR",
        str(DEFAULT_DEBUG_DIR if DEFAULT_DEBUG_DIR.exists() else WORKSPACE_DIR / "debug"),
    )
)
DEBUG_DIR.mkdir(parents=True, exist_ok=True)


# --- Worker Process ---


class FoundationPoseWorker(mp.Process):
    """Worker process that loads FoundationPose and handles pose estimation."""

    def __init__(self, task_queue, result_queue):
        super().__init__()
        self.task_queue = task_queue
        self.result_queue = result_queue
        self._estimator = None
        self._scorer = None
        self._refiner = None
        self._registered_objects = {}

    def run(self):
        worker_logger = _configure_logging("foundationpose_worker")
        worker_logger.info("Worker started. Initializing FoundationPose predictors...")

        # Add FoundationPose to path
        fp_path = os.getenv("FOUNDATIONPOSE_PATH", str(THIRD_PARTY_DIR / "FoundationPose"))
        if os.path.exists(fp_path) and fp_path not in sys.path:
            sys.path.insert(0, fp_path)

        try:
            import torch
            from learning.training.predict_score import ScorePredictor
            from learning.training.predict_pose_refine import PoseRefinePredictor

            # Initialize predictor models (FoundationPose will be created lazily with first mesh)
            worker_logger.info("Loading ScorePredictor...")
            self._scorer = ScorePredictor()
            worker_logger.info("Loading PoseRefinePredictor...")
            self._refiner = PoseRefinePredictor()
            worker_logger.info("FoundationPose predictors loaded. Ready to accept requests.")

        except Exception as e:
            worker_logger.error(f"Failed to initialize FoundationPose: {e}")
            while True:
                task = self.task_queue.get()
                if task is None:
                    break
                self.result_queue.put({"error": str(e)})
            return

        while True:
            task = self.task_queue.get()
            if task is None:
                break

            task_type = task.get("type")
            req_id = task.get("request_id")

            try:
                if task_type == "register":
                    result = self._handle_register(task, worker_logger)
                elif task_type == "track":
                    result = self._handle_track(task, worker_logger)
                else:
                    result = {"error": f"Unknown task type: {task_type}"}

                self.result_queue.put(result)

            except Exception as e:
                worker_logger.error(f"Task {req_id} failed: {e}")
                self.result_queue.put({"error": str(e)})

        worker_logger.info("Worker stopping.")

    def _handle_register(self, task: Dict, worker_logger) -> Dict:
        """Handle object registration with initial pose estimation."""
        import torch
        from estimater import FoundationPose

        object_id = task["object_id"]
        mesh_path = task["mesh_path"]
        rgb = task["rgb"]
        depth = task["depth"]
        mask = task["mask"]
        K = np.array(task["K"])
        iterations = task.get("iteration", 5)

        worker_logger.info(f"Registering object {object_id} with mesh {mesh_path}")

        # Load mesh
        mesh = trimesh.load(mesh_path)
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)

        # Note: Mesh should already be scaled by SAM3D using depth pointmap
        # We don't apply additional scaling here

        # Get mesh points and normals
        pts = mesh.vertices.astype(np.float32)
        normals = mesh.vertex_normals.astype(np.float32)

        # Create or update FoundationPose estimator with mesh
        if self._estimator is None:
            worker_logger.info("Creating FoundationPose estimator with first mesh...")
            self._estimator = FoundationPose(
                model_pts=pts,
                model_normals=normals,
                mesh=mesh,
                scorer=self._scorer,
                refiner=self._refiner,
            )
        else:
            self._estimator.reset_object(pts, normals, mesh=mesh)

        # Run pose estimation
        with torch.no_grad():
            pose = self._estimator.register(
                rgb=rgb,
                depth=depth,
                K=K,
                ob_mask=mask,
                iteration=iterations,
            )

        if pose is None:
            return {"error": "Pose estimation failed"}

        # Extract pose components
        position = pose[:3, 3].tolist()
        rotation_matrix = pose[:3, :3]

        # Convert rotation matrix to quaternion
        quat = self._rotation_matrix_to_quaternion(rotation_matrix)

        # Compute bounding box from mesh
        extents = mesh.bounding_box.extents.tolist()

        # Store for tracking
        self._registered_objects[object_id] = {
            "mesh": mesh,
            "pts": pts,
            "normals": normals,
            "last_pose": pose,
        }

        # Generate debug visualization
        self._save_debug_visualization(
            object_id=object_id,
            rgb=rgb,
            depth=depth,
            mask=mask,
            mesh=mesh,
            pose=pose,
            K=K,
            worker_logger=worker_logger,
        )

        return {
            "object_id": object_id,
            "pose": {
                "position": {"x": position[0], "y": position[1], "z": position[2]},
                "orientation": {"x": quat[0], "y": quat[1], "z": quat[2], "w": quat[3]},
            },
            "size": {"sx": extents[0], "sy": extents[1], "sz": extents[2]},
            "confidence": 1.0,
        }

    def _handle_track(self, task: Dict, worker_logger) -> Dict:
        """Handle pose tracking for already-registered object."""
        import torch

        object_id = task["object_id"]
        rgb = task["rgb"]
        depth = task["depth"]
        K = np.array(task["K"])
        iterations = task.get("iteration", 2)

        if object_id not in self._registered_objects:
            return {"error": f"Object {object_id} not registered"}

        obj_data = self._registered_objects[object_id]
        worker_logger.info(f"Tracking object {object_id}")

        # Set mesh points for estimator
        self._estimator.reset_object(obj_data["pts"], obj_data["normals"], mesh=obj_data["mesh"])

        # Run tracking
        with torch.no_grad():
            pose = self._estimator.track(
                rgb=rgb,
                depth=depth,
                K=K,
                ob_in_cam=obj_data["last_pose"],
                iteration=iterations,
            )

        if pose is None:
            return {"error": "Tracking failed"}

        # Update last pose
        obj_data["last_pose"] = pose

        # Extract pose components
        position = pose[:3, 3].tolist()
        rotation_matrix = pose[:3, :3]
        quat = self._rotation_matrix_to_quaternion(rotation_matrix)

        return {
            "object_id": object_id,
            "pose": {
                "position": {"x": position[0], "y": position[1], "z": position[2]},
                "orientation": {"x": quat[0], "y": quat[1], "z": quat[2], "w": quat[3]},
            },
            "confidence": 1.0,
        }

    def _rotation_matrix_to_quaternion(self, R: np.ndarray) -> List[float]:
        """Convert 3x3 rotation matrix to quaternion [x, y, z, w]."""
        trace = np.trace(R)
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (R[2, 1] - R[1, 2]) * s
            y = (R[0, 2] - R[2, 0]) * s
            z = (R[1, 0] - R[0, 1]) * s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
        return [float(x), float(y), float(z), float(w)]

    def _save_debug_visualization(
        self,
        object_id: str,
        rgb: np.ndarray,
        depth: np.ndarray,
        mask: np.ndarray,
        mesh,
        pose: np.ndarray,
        K: np.ndarray,
        worker_logger,
    ) -> None:
        """Generate and save a debug visualization composite image.

        Creates a 4-panel composite: [Original RGB | Masked RGB | Depth colormap | Pose overlay]
        Saves to DEBUG_DIR/{object_id}.png
        """
        try:
            import Utils
            import imageio

            # Compute oriented bounding box for visualization
            to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
            bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)
            center_pose = pose @ np.linalg.inv(to_origin)

            # Draw 3D bounding box and axes on the image
            vis = Utils.draw_posed_3d_box(
                K, img=rgb.copy(), ob_in_cam=center_pose, bbox=bbox
            )
            vis = Utils.draw_xyz_axis(
                vis,
                ob_in_cam=center_pose,
                scale=0.1,
                K=K,
                thickness=3,
                transparency=0,
                is_input_rgb=False,
            )

            # Create masked RGB view
            masked_rgb = rgb.copy()
            masked_rgb[~mask] = 0

            # Create depth colormap visualization
            d_valid = depth[depth > 0]
            if d_valid.size > 0:
                d_min, d_max = d_valid.min(), d_valid.max()
                depth_norm = (
                    ((depth - d_min) / (d_max - d_min + 1e-6) * 255)
                    .clip(0, 255)
                    .astype(np.uint8)
                )
                depth_vis = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
            else:
                depth_vis = np.zeros_like(rgb)

            # Create 4-panel composite: Original | Masked | Depth | Pose result
            composite = np.hstack([rgb, masked_rgb, depth_vis, vis])

            # Save to debug folder
            vis_path = DEBUG_DIR / f"{object_id}.png"

            # Convert BGR -> RGB for saving (imageio expects RGB)
            imageio.imwrite(str(vis_path), composite[..., ::-1])
            worker_logger.info(f"Saved debug visualization to {vis_path}")

        except Exception as e:
            worker_logger.warning(f"Debug visualization failed: {e}")


# --- Global State ---


class ServiceState:
    def __init__(self):
        self.worker: Optional[FoundationPoseWorker] = None
        ctx = mp.get_context("spawn")
        self.task_queue = ctx.Queue()
        self.result_queue = ctx.Queue()

    def ensure_worker(self):
        if self.worker is not None and self.worker.is_alive():
            return

        ctx = mp.get_context("spawn")
        self.task_queue = ctx.Queue()
        self.result_queue = ctx.Queue()

        logger.info("Spawning FoundationPose worker...")
        self.worker = FoundationPoseWorker(
            self.task_queue,
            self.result_queue,
        )
        self.worker.start()

    def unload(self):
        if self.worker is not None:
            logger.info("Terminating worker...")
            self.worker.terminate()
            self.worker.join()
            self.worker = None
            return True
        return False


_state = ServiceState()


# --- API Models ---


class RegisterRequest(BaseModel):
    object_id: str
    mesh_path: str
    rgb_b64: str
    depth_b64: str
    mask_b64: str
    K: List[List[float]]
    iteration: int = 5


class TrackRequest(BaseModel):
    object_id: str
    rgb_b64: str
    depth_b64: str
    K: List[List[float]]
    iteration: int = 2


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


class Size(BaseModel):
    sx: float
    sy: float
    sz: float


class PoseResponse(BaseModel):
    object_id: str
    pose: Pose
    size: Optional[Size] = None
    confidence: float = 1.0


# --- Helper Functions ---


def decode_image(b64_str: str, mode: str = "RGB") -> np.ndarray:
    """Decode base64 image to numpy array."""
    b64_data = b64_str.split(",", 1)[1] if "," in b64_str else b64_str
    img_bytes = base64.b64decode(b64_data)
    pil_image = Image.open(io.BytesIO(img_bytes))
    if mode == "RGB":
        pil_image = pil_image.convert("RGB")
    return np.array(pil_image)


def decode_depth(b64_str: str, shape: tuple) -> np.ndarray:
    """Decode base64 depth to numpy array."""
    b64_data = b64_str.split(",", 1)[1] if "," in b64_str else b64_str
    depth_bytes = base64.b64decode(b64_data)

    try:
        # Try as raw float32
        depth = np.frombuffer(depth_bytes, dtype=np.float32).reshape(shape)
    except ValueError:
        # Try as PNG (uint16 in mm)
        depth_img = Image.open(io.BytesIO(depth_bytes))
        depth = np.array(depth_img).astype(np.float32) / 1000.0

    return depth


def decode_mask(b64_str: str) -> np.ndarray:
    """Decode base64 mask to boolean numpy array."""
    b64_data = b64_str.split(",", 1)[1] if "," in b64_str else b64_str
    mask_bytes = base64.b64decode(b64_data)
    mask_img = Image.open(io.BytesIO(mask_bytes)).convert("L")
    return np.array(mask_img) > 0


# --- FastAPI App ---

app = FastAPI(title="FoundationPose 6D Pose Service", version="2.0.0")


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "worker_alive": _state.worker is not None and _state.worker.is_alive(),
        "weights_path": FP_WEIGHTS_PATH,
    }


@app.post("/register", response_model=PoseResponse)
def register(req: RegisterRequest) -> PoseResponse:
    """Register a new object and estimate its initial 6D pose."""
    try:
        _state.ensure_worker()

        # Decode inputs
        rgb = decode_image(req.rgb_b64, mode="RGB")
        depth = decode_depth(req.depth_b64, rgb.shape[:2])
        mask = decode_mask(req.mask_b64)

        # Convert RGB to BGR for OpenCV compatibility
        rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        task = {
            "type": "register",
            "request_id": str(uuid4()),
            "object_id": req.object_id,
            "mesh_path": req.mesh_path,
            "rgb": rgb_bgr,
            "depth": depth,
            "mask": mask,
            "K": req.K,
            "iteration": req.iteration,
        }

        _state.task_queue.put(task)
        result = _state.result_queue.get(timeout=120)

        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])

        return PoseResponse(
            object_id=result["object_id"],
            pose=Pose(
                position=Position(**result["pose"]["position"]),
                orientation=Orientation(**result["pose"]["orientation"]),
            ),
            size=Size(**result["size"]) if result.get("size") else None,
            confidence=result.get("confidence", 1.0),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Registration failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/track", response_model=PoseResponse)
def track(req: TrackRequest) -> PoseResponse:
    """Track an already-registered object (for future use from DIMOS)."""
    try:
        _state.ensure_worker()

        rgb = decode_image(req.rgb_b64, mode="RGB")
        depth = decode_depth(req.depth_b64, rgb.shape[:2])
        rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        task = {
            "type": "track",
            "request_id": str(uuid4()),
            "object_id": req.object_id,
            "rgb": rgb_bgr,
            "depth": depth,
            "K": req.K,
            "iteration": req.iteration,
        }

        _state.task_queue.put(task)
        result = _state.result_queue.get(timeout=60)

        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])

        return PoseResponse(
            object_id=result["object_id"],
            pose=Pose(
                position=Position(**result["pose"]["position"]),
                orientation=Orientation(**result["pose"]["orientation"]),
            ),
            confidence=result.get("confidence", 1.0),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Tracking failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/unload")
def unload() -> Dict[str, Any]:
    if _state.unload():
        return {"ok": True, "message": "Worker terminated, GPU freed"}
    return {"ok": True, "message": "Worker was not running"}


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    port = int(os.getenv("FOUNDATIONPOSE_PORT", "8093"))
    uvicorn.run(app, host="0.0.0.0", port=port)

