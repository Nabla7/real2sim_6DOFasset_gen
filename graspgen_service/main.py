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

import cv2
import numpy as np
import torch
import trimesh
import uvicorn
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel, Field


# --- Debug Visualization Utilities ---

# --- Paths (repo-local by default; Docker still works) ---
REPO_DIR = Path(__file__).resolve().parents[1]
WORKSPACE_DIR = Path(os.getenv("WORKSPACE_DIR", str(REPO_DIR)))
THIRD_PARTY_DIR = Path(os.getenv("THIRD_PARTY_DIR", str(WORKSPACE_DIR / "third_party")))
DEFAULT_LOG_DIR = Path("/tmp/spatial_memory/logs")
DEFAULT_DEBUG_DIR = Path("/tmp/spatial_memory/debug")

DEBUG_DIR = Path(
    os.getenv(
        "DEBUG_DIR",
        str(DEFAULT_DEBUG_DIR if DEFAULT_DEBUG_DIR.exists() else WORKSPACE_DIR / "debug"),
    )
)
DEBUG_DIR.mkdir(parents=True, exist_ok=True)


def get_color_from_score(score: float) -> tuple:
    """Convert score (0-1) to BGR color (red=bad, green=good)."""
    # score 0 -> red (0, 0, 255), score 1 -> green (0, 255, 0)
    r = int(255 * (1 - score))
    g = int(255 * score)
    return (0, g, r)  # BGR for OpenCV


def project_points_to_image(points_3d: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Project 3D points to 2D image coordinates using camera intrinsics.
    
    Args:
        points_3d: (N, 3) array of 3D points in camera frame
        K: 3x3 camera intrinsics matrix
    
    Returns:
        (N, 2) array of 2D pixel coordinates
    """
    # Project: p_2d = K @ p_3d / z
    points_2d_h = (K @ points_3d.T).T  # (N, 3)
    z = points_2d_h[:, 2:3]
    z = np.where(z < 0.01, 0.01, z)  # Avoid division by zero
    points_2d = points_2d_h[:, :2] / z
    return points_2d.astype(np.int32)


def draw_grasp_on_image(
    img: np.ndarray,
    grasp_pose: np.ndarray,
    K: np.ndarray,
    score: float,
    grasp_index: int = 0,
    collision_free: bool = True,
    gripper_depth: float = 0.10,
    gripper_width: float = 0.08,
) -> np.ndarray:
    """Draw a single grasp pose on an image.
    
    GraspGen convention:
    - Origin at gripper wrist/base
    - Z-axis points TOWARD object (approach direction)
    - X-axis is finger opening direction
    - Fingers are at +Z from origin
    
    Args:
        img: BGR image to draw on
        grasp_pose: 4x4 grasp pose in camera frame
        K: 3x3 camera intrinsics
        score: Grasp quality score (0-1)
        grasp_index: Grasp rank for labeling (0 = best)
        collision_free: Whether grasp is collision-free
        gripper_depth: Gripper depth in meters
        gripper_width: Gripper opening width in meters
    """
    color = get_color_from_score(score)
    thickness = 2 if collision_free else 1
    line_type = cv2.LINE_AA
    
    # Define gripper control points in gripper frame
    # Origin at wrist, Z points toward object, fingers at +Z
    wrist_stub = -0.05  # Draw arm stub behind wrist
    finger_length = gripper_depth * 0.4  # Finger prong length
    
    control_points = np.array([
        [0, 0, wrist_stub],                              # 0: Arm stub (behind wrist)
        [0, 0, 0],                                        # 1: Wrist/base
        [gripper_width/2, 0, 0],                          # 2: Right side of palm
        [-gripper_width/2, 0, 0],                         # 3: Left side of palm
        [gripper_width/2, 0, finger_length],              # 4: Right finger tip
        [-gripper_width/2, 0, finger_length],             # 5: Left finger tip
    ])
    
    # Transform to camera frame
    R = grasp_pose[:3, :3]
    t = grasp_pose[:3, 3]
    points_cam = (R @ control_points.T).T + t
    
    # Check if points are in front of camera
    if np.any(points_cam[:, 2] < 0.01):
        return img
    
    # Project to 2D
    pts_2d = project_points_to_image(points_cam, K)
    
    # Draw arm stub (approach axis behind gripper)
    cv2.line(img, tuple(pts_2d[0]), tuple(pts_2d[1]), color, thickness, line_type)
    
    # Draw palm (horizontal bar at wrist connecting the two finger bases)
    cv2.line(img, tuple(pts_2d[2]), tuple(pts_2d[3]), color, thickness + 1, line_type)
    
    # Draw fingers (from palm to fingertips) - open prongs, no connection at tips
    cv2.line(img, tuple(pts_2d[2]), tuple(pts_2d[4]), color, thickness, line_type)  # Right finger
    cv2.line(img, tuple(pts_2d[3]), tuple(pts_2d[5]), color, thickness, line_type)  # Left finger
    
    # Draw small circles at fingertips to make them visible
    cv2.circle(img, tuple(pts_2d[4]), 3, color, -1)
    cv2.circle(img, tuple(pts_2d[5]), 3, color, -1)
    
    # Draw grasp number label at wrist
    label_pos = (int(pts_2d[1][0]) + 5, int(pts_2d[1][1]) - 5)
    cv2.putText(img, str(grasp_index + 1), label_pos, 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
    
    return img


def save_grasp_debug_visualization(
    rgb: np.ndarray,
    depth: np.ndarray,
    mask: np.ndarray,
    grasps: List[np.ndarray],
    scores: List[float],
    collision_free_mask: Optional[List[bool]],
    K: np.ndarray,
    request_id: str,
    gripper_name: str,
    logger,
    max_grasps_to_draw: int = 20,
) -> Optional[str]:
    """Generate and save a debug visualization for grasp predictions.
    
    Creates a 3-panel composite: [RGB with grasps | Masked RGB | Depth colormap]
    
    Returns:
        Path to saved image, or None if failed
    """
    try:
        # Ensure RGB is in BGR format for OpenCV
        if rgb.shape[2] == 3:
            vis_img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        else:
            vis_img = rgb.copy()
        
        # Sort grasps by score (best first)
        if len(grasps) > 0 and len(scores) > 0:
            indices = np.argsort(scores)[::-1]  # Descending
            
            # Draw top-N grasps (worst first so best are on top)
            n_draw = min(max_grasps_to_draw, len(grasps))
            for i in reversed(range(n_draw)):
                idx = indices[i]
                grasp = np.array(grasps[idx])
                score = scores[idx]
                collision_free = collision_free_mask[idx] if collision_free_mask else True
                
                # Normalize score to 0-1 range
                score_norm = (score - min(scores)) / (max(scores) - min(scores) + 1e-6)
                
                draw_grasp_on_image(
                    vis_img, grasp, K, score_norm,
                    grasp_index=i,  # Pass the rank (0 = best)
                    collision_free=collision_free,
                    gripper_depth=0.10 if "robotiq" in gripper_name else 0.05,
                    gripper_width=0.085 if "robotiq" in gripper_name else 0.03,
                )
        
        # Add text overlay
        cv2.putText(
            vis_img, f"Grasps: {len(grasps)} ({gripper_name})",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
        )
        if len(scores) > 0:
            cv2.putText(
                vis_img, f"Best score: {max(scores):.3f}",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )
        if collision_free_mask:
            n_free = sum(collision_free_mask)
            cv2.putText(
                vis_img, f"Collision-free: {n_free}/{len(grasps)}",
                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
            )
        
        # Create masked RGB view
        masked_rgb = rgb.copy()
        if mask is not None and mask.shape[:2] == rgb.shape[:2]:
            mask_3ch = np.stack([mask > 0] * 3, axis=-1)
            masked_rgb = np.where(mask_3ch, rgb, (rgb * 0.3).astype(np.uint8))
        masked_bgr = cv2.cvtColor(masked_rgb, cv2.COLOR_RGB2BGR)
        
        # Create depth colormap
        d_valid = depth[depth > 0]
        if d_valid.size > 0:
            d_min, d_max = d_valid.min(), d_valid.max()
            depth_norm = ((depth - d_min) / (d_max - d_min + 1e-6) * 255).clip(0, 255).astype(np.uint8)
            depth_vis = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
        else:
            depth_vis = np.zeros_like(vis_img)
        
        # Resize panels to same height
        h = vis_img.shape[0]
        w = vis_img.shape[1]
        
        # Create 3-panel composite
        composite = np.hstack([vis_img, masked_bgr, depth_vis])
        
        # Add panel labels
        cv2.putText(composite, "Grasps", (w//2 - 40, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(composite, "Mask", (w + w//2 - 30, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(composite, "Depth", (2*w + w//2 - 35, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Save
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        vis_path = DEBUG_DIR / f"grasp_{request_id}_{timestamp}.png"
        cv2.imwrite(str(vis_path), composite)
        logger.info(f"Saved grasp debug visualization to {vis_path}")
        
        return str(vis_path)
        
    except Exception as e:
        logger.warning(f"Grasp debug visualization failed: {e}")
        return None

# Configure logging
LOG_DIR = Path(
    os.getenv(
        "LOG_DIR",
        str(DEFAULT_LOG_DIR if DEFAULT_LOG_DIR.exists() else WORKSPACE_DIR / "logs"),
    )
)
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
GRASPGEN_PATH = os.getenv("GRASPGEN_PATH", str(THIRD_PARTY_DIR / "GraspGen"))
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
        graspgen_path = os.getenv("GRASPGEN_PATH", str(THIRD_PARTY_DIR / "GraspGen"))
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
        graspgen_path = os.getenv("GRASPGEN_PATH", str(THIRD_PARTY_DIR / "GraspGen"))
        
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
        
        # Generate debug visualization if RGB is available
        debug_path = None
        if rgb is not None:
            import uuid
            request_id = str(uuid.uuid4())[:8]
            debug_path = save_grasp_debug_visualization(
                rgb=rgb,
                depth=depth,
                mask=mask,
                grasps=grasps_final.tolist(),
                scores=scores.tolist(),
                collision_free_mask=collision_free_mask.tolist() if collision_free_mask is not None else None,
                K=K,
                request_id=request_id,
                gripper_name=self.gripper_name,
                logger=logger,
                max_grasps_to_draw=20,
            )
        
        return {
            "grasps": grasps_final.tolist(),
            "scores": scores.tolist(),
            "collision_free_mask": collision_free_mask.tolist() if collision_free_mask is not None else None,
            "object_centroid": pc_mean.tolist(),
            "inference_time_ms": (time.time() - start_time) * 1000,
            "debug_image_path": debug_path,
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
    debug_image_path: Optional[str] = Field(None, description="Path to debug visualization image")


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
        debug_image_path=data.get("debug_image_path"),
    )


if __name__ == "__main__":
    port = int(os.getenv("GRASPGEN_PORT", "8094"))
    uvicorn.run(app, host="0.0.0.0", port=port)
