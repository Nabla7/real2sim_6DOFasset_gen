#!/usr/bin/env python3
"""
SAM 3 Segmentation Service

Runs SAM 3 in its own process/environment and exposes a simple HTTP API:

- GET /health
- POST /segment

Text prompts are used to segment specific objects in the image.
"""

import base64
import io
import logging
import os
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


def _configure_logging() -> logging.Logger:
    logger = logging.getLogger("spatial_memory.sam3_service")
    logger.setLevel(logging.INFO)

    logfile = LOG_DIR / "sam3_service.log"
    if not logger.handlers:
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

# Configuration from environment
WEIGHTS_DIR = os.getenv("WEIGHTS_DIR", "/weights")
SAM3_WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, "sam3")
SAM3_CHECKPOINT = os.path.join(SAM3_WEIGHTS_PATH, "sam3.pt")
SAM3_CONFIG = os.path.join(SAM3_WEIGHTS_PATH, "config.json")

DEFAULT_CONFIDENCE = float(os.getenv("SAM3_DEFAULT_CONF", "0.30"))
MAX_DETECTIONS = int(os.getenv("SAM3_MAX_DETECTIONS", "30"))


# --- Pydantic Models ---


class SegmentRequest(BaseModel):
    image_b64: str
    text_prompts: List[str] = ["object"]
    confidence_threshold: Optional[float] = None


class DetectionResponse(BaseModel):
    bbox: List[float]
    class_name: str
    confidence: float
    mask: str  # base64-encoded PNG mask


class SegmentResponse(BaseModel):
    detections: List[DetectionResponse]


# --- FastAPI App ---

app = FastAPI(
    title="SAM 3 Segmentation Service",
    version="1.0.0",
)

# Global model state
_model = None
_processor = None
_device = "cpu"


def _ensure_model():
    """Lazy-load SAM3 model on first request."""
    global _model, _processor, _device

    if _model is not None and _processor is not None:
        return

    # Import SAM3 modules (must be in PYTHONPATH)
    try:
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
    except ImportError as e:
        logger.error(f"Failed to import SAM3: {e}")
        logger.error("Make sure sam3 is in PYTHONPATH")
        raise RuntimeError(f"SAM3 import failed: {e}")

    candidate_devices = []
    if torch.cuda.is_available():
        candidate_devices.append("cuda")
    candidate_devices.append("cpu")

    last_error: Optional[Exception] = None

    for device in candidate_devices:
        logger.info(f"Loading SAM 3 Image Model on {device}...")
        try:
            # Set environment for weights path
            os.environ["SAM3_CHECKPOINT"] = SAM3_CHECKPOINT
            os.environ["SAM3_CONFIG"] = SAM3_CONFIG

            model = build_sam3_image_model()
            model.to(device if device == "cpu" else "cuda")
            processor = Sam3Processor(model)

            _model = model
            _processor = processor
            _device = device

            if device == "cpu":
                logger.warning("SAM 3 initialized on CPU. Inference will be slower.")
            else:
                logger.info("SAM 3 model initialized successfully on CUDA.")
            return

        except Exception as exc:
            logger.error(f"Failed to initialize SAM 3 model on {device}: {exc}")
            last_error = exc
            _model = None
            _processor = None

    raise RuntimeError("SAM 3 model initialization failed on all devices.") from last_error


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "model_loaded": _model is not None,
        "device": _device,
        "weights_path": SAM3_WEIGHTS_PATH,
    }


@app.post("/segment", response_model=SegmentResponse)
def segment(req: SegmentRequest) -> SegmentResponse:
    """
    Segment objects in an image using SAM 3 with text prompts.
    """
    _ensure_model()

    # Decode image
    try:
        b64_data = req.image_b64.split(",", 1)[1] if "," in req.image_b64 else req.image_b64
        img_bytes = base64.b64decode(b64_data)
        pil_image = Image.open(io.BytesIO(img_bytes))
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")
        pil_image = pil_image.copy()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to decode image: {e}")

    processor = _processor
    if processor is None:
        raise HTTPException(status_code=500, detail="SAM 3 processor not initialized")

    threshold = req.confidence_threshold if req.confidence_threshold is not None else DEFAULT_CONFIDENCE
    threshold = max(0.0, min(1.0, threshold))
    logger.info(f"Using confidence threshold {threshold:.3f} (max detections {MAX_DETECTIONS})")
    processor.set_confidence_threshold(threshold)

    try:
        # Set image once (computes image embeddings)
        inference_state = processor.set_image(pil_image)

        # Process each text prompt separately (as per SAM3 paper)
        prompts = req.text_prompts if req.text_prompts else ["object"]
        detections: List[DetectionResponse] = []

        for prompt_text in prompts:
            prompt_text = (prompt_text or "").strip()
            if not prompt_text:
                continue

            # Reset prompts for this concept
            processor.reset_all_prompts(inference_state)
            output = processor.set_text_prompt(
                state=inference_state,
                prompt=prompt_text,
            )

            masks = output["masks"]  # [N, H, W] tensor
            boxes = output["boxes"]  # [N, 4] tensor
            scores = output["scores"]  # [N] tensor

            logger.info(
                f"SAM3 concept '{prompt_text}' -> masks={tuple(masks.shape)} "
                f"boxes={tuple(boxes.shape)} scores={tuple(scores.shape)}"
            )

            scores_np = scores.detach().cpu().numpy()
            sorted_indices = scores_np.argsort()[::-1]

            for idx in sorted_indices[:MAX_DETECTIONS]:
                score = float(scores_np[idx])
                if score < threshold:
                    break

                box = boxes[idx].tolist()
                mask_tensor = masks[idx]

                # Convert mask to PNG
                mask_np = mask_tensor.squeeze().cpu().numpy().astype(bool)
                mask_img = Image.fromarray((mask_np * 255).astype("uint8"))
                buf = io.BytesIO()
                mask_img.save(buf, format="PNG")
                mask_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

                detections.append(
                    DetectionResponse(
                        bbox=[float(v) for v in box],
                        class_name=prompt_text,
                        confidence=score,
                        mask=mask_b64,
                    )
                )

    except Exception as e:
        logger.error(f"SAM 3 inference error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    logger.info(
        f"Segmentation complete: {len(detections)} detections "
        f"(device={_device}, prompts={req.text_prompts}, threshold={threshold:.3f})"
    )
    return SegmentResponse(detections=detections)


if __name__ == "__main__":
    port = int(os.getenv("SAM3_PORT", "8091"))
    uvicorn.run(app, host="0.0.0.0", port=port)


