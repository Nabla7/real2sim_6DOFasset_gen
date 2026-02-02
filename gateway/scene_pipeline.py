"""Async scene reconstruction pipeline orchestrator."""

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional
from uuid import uuid4

import aiohttp

from .config import (
    COLMAP_TIMEOUT,
    COLMAP_URL,
    NEURAL_RECON_TIMEOUT,
    NEURAL_RECON_URL,
    SCENE_CAPTURE_TIMEOUT,
    SCENE_CAPTURE_URL,
)

logger = logging.getLogger("gateway.scene_pipeline")


@dataclass
class SceneJob:
    run_id: str
    request: Dict[str, Any]
    future: asyncio.Future


class ScenePipeline:
    def __init__(self) -> None:
        self.queue: asyncio.Queue[SceneJob] = asyncio.Queue()
        self._running = False
        self._session: Optional[aiohttp.ClientSession] = None
        self._workers: list[asyncio.Task] = []

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._session = aiohttp.ClientSession()
        self._workers = [asyncio.create_task(self._worker(), name="scene_recon_worker")]
        logger.info("Scene pipeline workers started")

    async def stop(self) -> None:
        self._running = False
        for worker in self._workers:
            worker.cancel()
        if self._session:
            await self._session.close()
        logger.info("Scene pipeline workers stopped")

    async def submit(self, request: Dict[str, Any]) -> str:
        run_id = request.get("run_id") or uuid4().hex
        job = SceneJob(
            run_id=run_id,
            request=request,
            future=asyncio.get_event_loop().create_future(),
        )
        await self.queue.put(job)
        return run_id

    async def _post(self, url: str, payload: Dict[str, Any], timeout_s: int) -> Dict[str, Any]:
        assert self._session is not None
        async with self._session.post(
            url, json=payload, timeout=aiohttp.ClientTimeout(total=timeout_s)
        ) as resp:
            if resp.status != 200:
                error = await resp.text()
                raise RuntimeError(f"{url} failed: {error}")
            return await resp.json()

    async def _worker(self) -> None:
        while self._running:
            try:
                job = await asyncio.wait_for(self.queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            try:
                logger.info("Scene job %s: starting", job.run_id)
                await self._run_job(job)
                job.future.set_result({"run_id": job.run_id})
            except Exception as exc:
                logger.error("Scene job %s failed: %s", job.run_id, exc)
                job.future.set_exception(exc)

    async def _run_job(self, job: SceneJob) -> None:
        req = job.request
        run_id = job.run_id

        if not req.get("run_created", False):
            create_resp = await self._post(
                f"{SCENE_CAPTURE_URL}/runs/create",
                {"run_id": run_id},
                SCENE_CAPTURE_TIMEOUT,
            )
            run_id = create_resp.get("run_id", run_id)
            job.run_id = run_id

        await self._post(
            f"{SCENE_CAPTURE_URL}/runs/{run_id}/extract_frames",
            {
                "video_path": req.get("video_path"),
                "fps": req.get("fps", 3.0),
                "width": req.get("width", 1920),
                "format": req.get("format", "jpg"),
                "jpeg_quality": req.get("jpeg_quality", 2),
                "dedupe": req.get("dedupe", False),
                "scene": req.get("scene"),
            },
            SCENE_CAPTURE_TIMEOUT,
        )

        await self._post(
            f"{COLMAP_URL}/reconstruct_sparse",
            {
                "run_id": run_id,
                "images_dir": req.get("images_dir"),
                "use_gpu": req.get("use_gpu", True),
                "max_image_size": req.get("max_image_size", 2000),
                "overlap": req.get("overlap", 10),
                "single_camera": req.get("single_camera", True),
            },
            COLMAP_TIMEOUT,
        )

        await self._post(
            f"{COLMAP_URL}/runs/{run_id}/select_model",
            {"model_id": req.get("model_id", 0)},
            COLMAP_TIMEOUT,
        )

        await self._post(
            f"{NEURAL_RECON_URL}/prepare_dataset",
            {
                "run_id": run_id,
                "images_dir": req.get("images_dir"),
                "sparse_model_dir": req.get("sparse_model_dir"),
            },
            NEURAL_RECON_TIMEOUT,
        )

        await self._post(
            f"{NEURAL_RECON_URL}/train_3dgrut",
            {
                "run_id": run_id,
                "config_name": req.get("config_name", "apps/colmap_3dgut_mcmc.yaml"),
                "experiment_name": req.get("experiment_name"),
                "overrides": req.get("overrides", []),
            },
            NEURAL_RECON_TIMEOUT,
        )

        if req.get("export", False):
            await self._post(
                f"{NEURAL_RECON_URL}/export",
                {
                    "run_id": run_id,
                    "export_usdz": req.get("export_usdz", True),
                    "render": req.get("render", False),
                    "checkpoint_path": req.get("checkpoint_path"),
                },
                NEURAL_RECON_TIMEOUT,
            )


_scene_pipeline: Optional[ScenePipeline] = None


async def get_scene_pipeline() -> ScenePipeline:
    global _scene_pipeline
    if _scene_pipeline is None:
        _scene_pipeline = ScenePipeline()
        await _scene_pipeline.start()
    return _scene_pipeline
