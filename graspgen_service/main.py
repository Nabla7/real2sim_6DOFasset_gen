#!/usr/bin/env python3
"""
GraspGen Service (Worker Process Architecture)

Runs GraspGen inference in a separate process to allow complete GPU memory
reclamation on unload. Supports collision-aware grasp generation.
"""

import base64
import io
import logging
import multiprocessing as mp
import os
import sys
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import trimesh
import uvicorn
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel, Field

# Configure logging
LOG_DIR = Path(os.getenv("LOG_DIR", "/workspace/logs"))
LOG_DIR.mkdir(parents=True, exist_ok=True)


def _configure_logging(name: str = "graspgen_service") -> logging.Logger:
    logger = logging.getLogger(f"spatial_memory.{name}")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        logfile = LOG_DIR / f"{name}.log"
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
GRASPGEN_PATH = os.getenv("GRASPGEN_PATH", "/workspace/third_party/GraspGen")
DEFAULT_GRIPPER = os.getenv("DEFAULT_GRIPPER", "robotiq_2f_140")


# --- Worker Process for Inference ---


class GraspGenWorker(mp.Process):
    """Worker process that loads GraspGen models and processes requests."""

    def __init__(self, task_queue: mp.Queue, result_queue: mp.Queue, gripper_name: str):
        super().__init__()
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.gripper_name = gripper_name
        self._sampler = None
        self._gripper_info = None

    def run(self):
        worker_logger = _configure_logging("graspgen_worker")
        worker_logger.info(f"Worker process started for gripper: {self.gripper_name}")

        # Add GraspGen to path
        graspgen_path = os.getenv("GRASPGEN_PATH", "/workspace/third_party/GraspGen")
        if os.path.exists(graspgen_path) and graspgen_path not in sys.path:
            sys.path.insert(0, graspgen_path)

        # Set EGL for offscreen rendering
        os.environ["PYOPENGL_PLATFORM"] = "egl"

        try:
            from grasp_gen.grasp_server import GraspGenSampler, load_grasp_cfg
            from grasp_gen.robot import get_gripper_info
            from grasp_gen.utils.point_cloud_utils import (
                filter_colliding_grasps,
                depth_and_segmentation_to_point_clouds,
                point_cloud_outlier_removal,
            )
            import trimesh.transformations as tra

            # Load gripper configuration
            gripper_config_path = self._get_gripper_config_path(self.gripper_name)
            worker_logger.info(f"Loading gripper config from {gripper_config_path}")
            
            grasp_cfg = load_grasp_cfg(gripper_config_path)
            self._sampler = GraspGenSampler(grasp_cfg)
            self._gripper_info = get_gripper_info(self.gripper_name)
            
            worker_logger.info("GraspGen models loaded successfully")

            # Process tasks
            while True:
                task = self.task_queue.get()
                if task is None:  # Shutdown signal
                    break

                try:
                    result = self._process_task(
                        task, worker_logger, GraspGenSampler, tra,
                        filter_colliding_grasps, depth_and_segmentation_to_point_clouds,
                        point_cloud_outlier_removal
                    )
                    self.result_queue.put({"success": True, "result": result})
                except Exception as e:
                    worker_logger.error(f"Task failed: {e}", exc_info=True)
                    self.result_queue.put({"success": False, "error": str(e)})

        except Exception as e:
            worker_logger.error(f"Failed to initialize GraspGen: {e}", exc_info=True)
            # Signal failure
            while True:
                task = self.task_queue.get()
                if task is None:
                    break
                self.result_queue.put({"success": False, "error": f"Worker init failed: {e}"})

    def _get_gripper_config_path(self, gripper_name: str) -> str:
        """Get path to gripper configuration YAML file."""
        graspgen_path = os.getenv("GRASPGEN_PATH", "/workspace/third_party/GraspGen")
        
        # Try to find checkpoints in GraspGenModels directory
        models_dir = os.path.join(graspgen_path, "GraspGenModels", "checkpoints")
        if os.path.exists(models_dir):
            config_path = os.path.join(models_dir, f"graspgen_{gripper_name}.yml")
            if os.path.exists(config_path):
                return config_path
        
        # Fallback to relative path
        config_path = os.path.join(graspgen_path, "checkpoints", f"graspgen_{gripper_name}.yml")
        return config_path

    def _process_task(
        self, task: Dict, logger, GraspGenSampler, tra,
        filter_colliding_grasps, depth_and_segmentation_to_point_clouds,
        point_cloud_outlier_removal
    ) -> Dict:
        """Process a single grasp generation task."""
        start_time = time.time()
        
        task_type = task.get("type")
        
        if task_type == "point_cloud":
            return self._process_pointcloud_task(
                task, logger, GraspGenSampler, tra, filter_colliding_grasps, point_cloud_outlier_removal
            )
        elif task_type == "depth_mask":
            return self._process_depth_mask_task(
                task, logger, GraspGenSampler, tra, filter_colliding_grasps,
                depth_and_segmentation_to_point_clouds, point_cloud_outlier_removal
            )
        else:
            raise ValueError(f"Unknown task type: {task_type}")

    def _process_pointcloud_task(
        self, task: Dict, logger, GraspGenSampler, tra,
        filter_colliding_grasps, point_cloud_outlier_removal
    ) -> Dict:
        """Process grasp generation from point cloud."""
        start_time = time.time()
        
        # Decode point cloud
        pc_b64 = task["point_cloud_b64"]
        pc_bytes = base64.b64decode(pc_b64)
        pc = np.frombuffer(pc_bytes, dtype=np.float32).reshape(-1, 3)
        
        # Optional scene point cloud for collision filtering
        scene_pc = None
        if task.get("scene_pc_b64"):
            scene_bytes = base64.b64decode(task["scene_pc_b64"])
            scene_pc = np.frombuffer(scene_bytes, dtype=np.float32).reshape(-1, 3)
        
        # Filter outliers
        pc_torch = torch.from_numpy(pc)
        pc_filtered, _ = point_cloud_outlier_removal(pc_torch)
        pc_filtered = pc_filtered.numpy()
        
        logger.info(f"Processing point cloud with {len(pc_filtered)} points")
        
        # Run inference
        grasps, scores = GraspGenSampler.run_inference(
            pc_filtered,
            self._sampler,
            grasp_threshold=task.get("grasp_threshold", -1.0),
            num_grasps=task.get("num_grasps", 400),
            topk_num_grasps=task.get("topk_num_grasps", 100),
            remove_outliers=False,  # Already filtered
        )
        
        if len(grasps) == 0:
            return {
                "grasps": [],
                "scores": [],
                "collision_free_mask": [],
                "object_centroid": [0.0, 0.0, 0.0],
                "inference_time_ms": (time.time() - start_time) * 1000,
            }
        
        grasps = grasps.cpu().numpy()
        scores = scores.cpu().numpy()
        
        # Center transform
        pc_mean = pc_filtered.mean(axis=0)
        T_center = tra.translation_matrix(-pc_mean)
        pc_centered = tra.transform_points(pc_filtered, T_center)
        grasps_centered = np.array([T_center @ g for g in grasps])
        
        # Collision filtering if requested
        collision_free_mask = None
        if task.get("filter_collisions", False) and scene_pc is not None:
            scene_pc_centered = tra.transform_points(scene_pc, T_center)
            
            # Downsample scene for speed
            max_scene_points = task.get("max_scene_points", 8192)
            if len(scene_pc_centered) > max_scene_points:
                indices = np.random.choice(len(scene_pc_centered), max_scene_points, replace=False)
                scene_pc_downsampled = scene_pc_centered[indices]
            else:
                scene_pc_downsampled = scene_pc_centered
            
            collision_free_mask = filter_colliding_grasps(
                scene_pc=scene_pc_downsampled,
                grasp_poses=grasps_centered,
                gripper_collision_mesh=self._gripper_info.collision_mesh,
                collision_threshold=task.get("collision_threshold", 0.02),
            )
            
            logger.info(f"Collision filtering: {collision_free_mask.sum()}/{len(grasps_centered)} collision-free")
        
        # Transform grasps back to original frame
        T_inv = tra.inverse_matrix(T_center)
        grasps_final = np.array([T_inv @ g for g in grasps_centered])
        
        return {
            "grasps": grasps_final.tolist(),
            "scores": scores.tolist(),
            "collision_free_mask": collision_free_mask.tolist() if collision_free_mask is not None else None,
            "object_centroid": pc_mean.tolist(),
            "inference_time_ms": (time.time() - start_time) * 1000,
        }

    def _process_depth_mask_task(
        self, task: Dict, logger, GraspGenSampler, tra,
        filter_colliding_grasps, depth_and_segmentation_to_point_clouds,
        point_cloud_outlier_removal
    ) -> Dict:
        """Process grasp generation from depth image and segmentation mask."""
        start_time = time.time()
        
        # Decode depth
        depth_b64 = task["depth_b64"]
        depth_bytes = base64.b64decode(depth_b64)
        depth = np.frombuffer(depth_bytes, dtype=np.float32).reshape(task["depth_shape"])
        
        # Decode mask
        mask_b64 = task["mask_b64"]
        mask_img = Image.open(io.BytesIO(base64.b64decode(mask_b64)))
        mask = np.array(mask_img)
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        mask = (mask > 128).astype(np.uint8)
        
        # Decode RGB if provided
        rgb = None
        if task.get("rgb_b64"):
            rgb_img = Image.open(io.BytesIO(base64.b64decode(task["rgb_b64"])))
            rgb = np.array(rgb_img)
        
        # Camera intrinsics
        K = np.array(task["K"])
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        
        # Convert to point clouds
        scene_pc, object_pc, scene_colors, object_colors = depth_and_segmentation_to_point_clouds(
            depth_image=depth,
            segmentation_mask=mask,
            fx=fx, fy=fy, cx=cx, cy=cy,
            rgb_image=rgb,
            target_object_id=1,
            remove_object_from_scene=True,
        )
        
        logger.info(f"Generated point clouds: scene={len(scene_pc)}, object={len(object_pc)}")
        
        # Filter outliers from object point cloud
        object_pc_torch = torch.from_numpy(object_pc)
        object_pc_filtered, _ = point_cloud_outlier_removal(object_pc_torch)
        object_pc_filtered = object_pc_filtered.numpy()
        
        # Run inference
        grasps, scores = GraspGenSampler.run_inference(
            object_pc_filtered,
            self._sampler,
            grasp_threshold=task.get("grasp_threshold", -1.0),
            num_grasps=task.get("num_grasps", 400),
            topk_num_grasps=task.get("topk_num_grasps", 100),
            remove_outliers=False,
        )
        
        if len(grasps) == 0:
            return {
                "grasps": [],
                "scores": [],
                "collision_free_mask": [],
                "object_centroid": [0.0, 0.0, 0.0],
                "inference_time_ms": (time.time() - start_time) * 1000,
            }
        
        grasps = grasps.cpu().numpy()
        scores = scores.cpu().numpy()
        
        # Center transform
        pc_mean = object_pc_filtered.mean(axis=0)
        T_center = tra.translation_matrix(-pc_mean)
        grasps_centered = np.array([T_center @ g for g in grasps])
        
        # Collision filtering if requested
        collision_free_mask = None
        if task.get("filter_collisions", False):
            scene_pc_centered = tra.transform_points(scene_pc, T_center)
            
            # Downsample scene for speed
            max_scene_points = task.get("max_scene_points", 8192)
            if len(scene_pc_centered) > max_scene_points:
                indices = np.random.choice(len(scene_pc_centered), max_scene_points, replace=False)
                scene_pc_downsampled = scene_pc_centered[indices]
            else:
                scene_pc_downsampled = scene_pc_centered
            
            collision_free_mask = filter_colliding_grasps(
                scene_pc=scene_pc_downsampled,
                grasp_poses=grasps_centered,
                gripper_collision_mesh=self._gripper_info.collision_mesh,
                collision_threshold=task.get("collision_threshold", 0.02),
            )
            
            logger.info(f"Collision filtering: {collision_free_mask.sum()}/{len(grasps_centered)} collision-free")
        
        # Transform grasps back to original frame
        T_inv = tra.inverse_matrix(T_center)
        grasps_final = np.array([T_inv @ g for g in grasps_centered])
        
        return {
            "grasps": grasps_final.tolist(),
            "scores": scores.tolist(),
            "collision_free_mask": collision_free_mask.tolist() if collision_free_mask is not None else None,
            "object_centroid": pc_mean.tolist(),
            "inference_time_ms": (time.time() - start_time) * 1000,
        }


# --- FastAPI Service ---


# Pydantic Models
class GraspRequest(BaseModel):
    # Input modes (use one of these)
    point_cloud_b64: Optional[str] = Field(None, description="Base64 float32 Nx3 point cloud")
    depth_b64: Optional[str] = Field(None, description="Base64 float32 depth image")
    mask_b64: Optional[str] = Field(None, description="Base64 PNG segmentation mask")
    rgb_b64: Optional[str] = Field(None, description="Base64 RGB image (optional)")
    K: Optional[List[List[float]]] = Field(None, description="Camera intrinsics 3x3")
    depth_shape: Optional[List[int]] = Field(None, description="Depth image shape [H, W]")
    
    # Optional collision filtering
    scene_pc_b64: Optional[str] = Field(None, description="Base64 scene point cloud for collision filtering")
    filter_collisions: bool = Field(False, description="Enable collision filtering")
    collision_threshold: float = Field(0.02, description="Collision distance threshold (meters)")
    max_scene_points: int = Field(8192, description="Max scene points for collision check")
    
    # Grasp generation parameters
    gripper_type: str = Field(DEFAULT_GRIPPER, description="Gripper type")
    num_grasps: int = Field(400, description="Number of grasps to generate")
    topk_num_grasps: int = Field(100, description="Return top K grasps")
    grasp_threshold: float = Field(-1.0, description="Grasp quality threshold (-1 for auto)")


class GraspPose(BaseModel):
    transform: List[float] = Field(..., description="4x4 transform matrix (16 floats, row-major)")
    score: float = Field(..., description="Grasp quality score")
    collision_free: Optional[bool] = Field(None, description="Whether grasp is collision-free")


class GraspResponse(BaseModel):
    grasps: List[GraspPose]
    gripper_type: str
    object_centroid: List[float] = Field(..., description="Object center in camera frame")
    inference_time_ms: float


# Global worker
_worker: Optional[GraspGenWorker] = None
_task_queue: Optional[mp.Queue] = None
_result_queue: Optional[mp.Queue] = None


def _ensure_worker():
    """Ensure the worker process is running."""
    global _worker, _task_queue, _result_queue
    
    if _worker is not None and _worker.is_alive():
        return
    
    logger.info("Starting GraspGen worker...")
    _task_queue = mp.Queue()
    _result_queue = mp.Queue()
    _worker = GraspGenWorker(_task_queue, _result_queue, DEFAULT_GRIPPER)
    _worker.start()
    logger.info("GraspGen worker started")


app = FastAPI(
    title="GraspGen Service",
    description="Grasp generation service with collision filtering",
    version="1.0.0",
)


@app.on_event("startup")
async def startup():
    """Start worker on service startup."""
    _ensure_worker()


@app.on_event("shutdown")
async def shutdown():
    """Shutdown worker on service stop."""
    if _task_queue:
        _task_queue.put(None)
    if _worker:
        _worker.join(timeout=5)


@app.get("/health")
def health() -> Dict[str, Any]:
    """Health check endpoint."""
    _ensure_worker()
    is_alive = _worker is not None and _worker.is_alive()
    return {
        "status": "ok" if is_alive else "degraded",
        "service": "graspgen",
        "worker_alive": is_alive,
    }


@app.post("/generate", response_model=GraspResponse)
def generate_grasps(req: GraspRequest) -> GraspResponse:
    """
    Generate grasps from point cloud or depth+mask.
    
    Supports two input modes:
    1. Point cloud: provide point_cloud_b64
    2. Depth+mask: provide depth_b64, mask_b64, K, depth_shape
    """
    _ensure_worker()
    
    # Validate inputs
    has_pc = req.point_cloud_b64 is not None
    has_depth = req.depth_b64 is not None and req.mask_b64 is not None
    
    if not has_pc and not has_depth:
        raise HTTPException(
            status_code=400,
            detail="Must provide either point_cloud_b64 or (depth_b64 + mask_b64 + K)"
        )
    
    if has_depth and (req.K is None or req.depth_shape is None):
        raise HTTPException(
            status_code=400,
            detail="depth_b64 mode requires K and depth_shape"
        )
    
    # Prepare task
    if has_pc:
        task = {
            "type": "point_cloud",
            "point_cloud_b64": req.point_cloud_b64,
            "scene_pc_b64": req.scene_pc_b64,
            "filter_collisions": req.filter_collisions,
            "collision_threshold": req.collision_threshold,
            "max_scene_points": req.max_scene_points,
            "num_grasps": req.num_grasps,
            "topk_num_grasps": req.topk_num_grasps,
            "grasp_threshold": req.grasp_threshold,
        }
    else:
        task = {
            "type": "depth_mask",
            "depth_b64": req.depth_b64,
            "depth_shape": req.depth_shape,
            "mask_b64": req.mask_b64,
            "rgb_b64": req.rgb_b64,
            "K": req.K,
            "filter_collisions": req.filter_collisions,
            "collision_threshold": req.collision_threshold,
            "max_scene_points": req.max_scene_points,
            "num_grasps": req.num_grasps,
            "topk_num_grasps": req.topk_num_grasps,
            "grasp_threshold": req.grasp_threshold,
        }
    
    # Submit task
    _task_queue.put(task)
    result = _result_queue.get(timeout=120)
    
    if not result["success"]:
        raise HTTPException(status_code=500, detail=result["error"])
    
    data = result["result"]
    
    # Format response
    grasps = []
    for i, (grasp_transform, score) in enumerate(zip(data["grasps"], data["scores"])):
        collision_free = None
        if data["collision_free_mask"] is not None:
            collision_free = data["collision_free_mask"][i]
        
        grasps.append(GraspPose(
            transform=[float(x) for x in np.array(grasp_transform).flatten()],
            score=float(score),
            collision_free=collision_free,
        ))
    
    return GraspResponse(
        grasps=grasps,
        gripper_type=req.gripper_type,
        object_centroid=data["object_centroid"],
        inference_time_ms=data["inference_time_ms"],
    )


if __name__ == "__main__":
    port = int(os.getenv("GRASPGEN_PORT", "8094"))
    uvicorn.run(app, host="0.0.0.0", port=port)
