"""
Pipeline with per-stage queues for parallel processing.

Each stage (SAM3 -> SAM3D -> FoundationPose) runs its own worker,
allowing different requests to be processed concurrently at different stages.
"""

import asyncio
import base64
import io
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from uuid import uuid4

import aiohttp
import numpy as np
from PIL import Image

from .config import (
    SAM3_URL,
    SAM3D_URL,
    FOUNDATIONPOSE_URL,
    SAM3_TIMEOUT,
    SAM3D_TIMEOUT,
    FOUNDATIONPOSE_TIMEOUT,
)

logger = logging.getLogger("spatial_memory.pipeline")


@dataclass
class PipelineJob:
    """Represents a single request flowing through the pipeline."""

    request_id: str
    request: Dict[str, Any]
    future: asyncio.Future
    # Intermediate results passed between stages
    mask_b64: Optional[str] = None
    mesh_path: Optional[str] = None
    mesh_b64: Optional[str] = None
    error: Optional[str] = None


def compute_iou(box1: List[float], box2: List[float]) -> float:
    """Compute IoU between two bounding boxes [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    if x2 <= x1 or y2 <= y1:
        return 0.0

    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


class Pipeline:
    """
    Async pipeline with per-stage queues.

    Stages:
    1. SAM3: Text-prompted segmentation
    2. SAM3D: Mesh reconstruction with depth-based scaling
    3. FoundationPose: 6D pose estimation
    """

    def __init__(self):
        self.sam3_queue: asyncio.Queue[PipelineJob] = asyncio.Queue()
        self.sam3d_queue: asyncio.Queue[PipelineJob] = asyncio.Queue()
        self.fp_queue: asyncio.Queue[PipelineJob] = asyncio.Queue()
        self._running = False
        self._session: Optional[aiohttp.ClientSession] = None
        self._workers: List[asyncio.Task] = []

    async def start(self):
        """Start all pipeline workers."""
        if self._running:
            return
        self._running = True
        self._session = aiohttp.ClientSession()

        self._workers = [
            asyncio.create_task(self._sam3_worker(), name="sam3_worker"),
            asyncio.create_task(self._sam3d_worker(), name="sam3d_worker"),
            asyncio.create_task(self._fp_worker(), name="fp_worker"),
        ]
        logger.info("Pipeline workers started")

    async def stop(self):
        """Stop all pipeline workers."""
        self._running = False
        for worker in self._workers:
            worker.cancel()
        if self._session:
            await self._session.close()
        logger.info("Pipeline workers stopped")

    async def submit(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Submit a request to the pipeline.

        Returns when the full pipeline completes (or fails).
        """
        job = PipelineJob(
            request_id=uuid4().hex,
            request=request,
            future=asyncio.get_event_loop().create_future(),
        )
        logger.info(f"[{job.request_id}] Submitting job for label={request.get('label')}")
        await self.sam3_queue.put(job)
        return await job.future

    async def _sam3_worker(self):
        """Worker for SAM3 segmentation stage."""
        while self._running:
            try:
                job = await asyncio.wait_for(self.sam3_queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            logger.info(f"[{job.request_id}] SAM3: Processing")
            try:
                # Call SAM3 service
                payload = {
                    "image_b64": job.request["image_rgb_b64"],
                    "text_prompts": [job.request["label"]],
                }

                async with self._session.post(
                    f"{SAM3_URL}/segment",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=SAM3_TIMEOUT),
                ) as resp:
                    if resp.status != 200:
                        error = await resp.text()
                        raise RuntimeError(f"SAM3 error: {error}")
                    result = await resp.json()

                detections = result.get("detections", [])
                if not detections:
                    raise RuntimeError(f"SAM3 found no objects for label '{job.request['label']}'")

                # Match detection to input bbox using IoU
                input_bbox = job.request.get("bbox")
                if input_bbox:
                    best_idx = 0
                    best_iou = 0.0
                    for i, det in enumerate(detections):
                        iou = compute_iou(input_bbox, det["bbox"])
                        if iou > best_iou:
                            best_iou = iou
                            best_idx = i
                    matched_det = detections[best_idx]
                    logger.info(f"[{job.request_id}] SAM3: Matched bbox with IoU={best_iou:.3f}")
                else:
                    # Take highest confidence detection
                    matched_det = max(detections, key=lambda d: d["confidence"])

                job.mask_b64 = matched_det["mask"]
                logger.info(f"[{job.request_id}] SAM3: Done, passing to SAM3D")
                await self.sam3d_queue.put(job)

            except Exception as e:
                logger.error(f"[{job.request_id}] SAM3 failed: {e}")
                job.future.set_exception(e)

    async def _sam3d_worker(self):
        """Worker for SAM3D mesh reconstruction stage."""
        while self._running:
            try:
                job = await asyncio.wait_for(self.sam3d_queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            logger.info(f"[{job.request_id}] SAM3D: Processing")
            try:
                # Build payload with depth and intrinsics for pointmap scaling
                payload = {
                    "image_b64": job.request["image_rgb_b64"],
                    "mask_b64": job.mask_b64,
                }

                # Add depth and K if available for real-world scaling
                if "depth_b64" in job.request:
                    payload["depth_b64"] = job.request["depth_b64"]
                if "K" in job.request:
                    payload["K"] = job.request["K"]

                async with self._session.post(
                    f"{SAM3D_URL}/reconstruct",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=SAM3D_TIMEOUT),
                ) as resp:
                    if resp.status != 200:
                        error = await resp.text()
                        raise RuntimeError(f"SAM3D error: {error}")
                    result = await resp.json()

                mesh_path = result.get("mesh_path")
                if not mesh_path:
                    raise RuntimeError("SAM3D returned no mesh path")

                job.mesh_path = mesh_path

                # Also get mesh as base64 if provided
                if "mesh_b64" in result:
                    job.mesh_b64 = result["mesh_b64"]

                logger.info(f"[{job.request_id}] SAM3D: Done, passing to FoundationPose")
                await self.fp_queue.put(job)

            except Exception as e:
                logger.error(f"[{job.request_id}] SAM3D failed: {e}")
                job.future.set_exception(e)

    async def _fp_worker(self):
        """Worker for FoundationPose 6D pose estimation stage."""
        while self._running:
            try:
                job = await asyncio.wait_for(self.fp_queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            logger.info(f"[{job.request_id}] FoundationPose: Processing")
            try:
                # Build payload for FoundationPose
                payload = {
                    "object_id": job.request_id,
                    "mesh_path": job.mesh_path,
                    "rgb_b64": job.request["image_rgb_b64"],
                    "depth_b64": job.request["depth_b64"],
                    "mask_b64": job.mask_b64,
                    "K": job.request["K"],
                }

                async with self._session.post(
                    f"{FOUNDATIONPOSE_URL}/register",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=FOUNDATIONPOSE_TIMEOUT),
                ) as resp:
                    if resp.status != 200:
                        error = await resp.text()
                        raise RuntimeError(f"FoundationPose error: {error}")
                    result = await resp.json()

                # Build final response
                response = {
                    "label": job.request["label"],
                    "mesh_b64": job.mesh_b64,
                    "pose": result.get("pose"),
                    "bbox_3d": result.get("size"),
                    "confidence": result.get("confidence", 1.0),
                }

                logger.info(f"[{job.request_id}] FoundationPose: Done, pipeline complete")
                job.future.set_result(response)

            except Exception as e:
                logger.error(f"[{job.request_id}] FoundationPose failed: {e}")
                job.future.set_exception(e)


# Global pipeline instance
_pipeline: Optional[Pipeline] = None


async def get_pipeline() -> Pipeline:
    """Get or create the global pipeline instance."""
    global _pipeline
    if _pipeline is None:
        _pipeline = Pipeline()
        await _pipeline.start()
    return _pipeline


