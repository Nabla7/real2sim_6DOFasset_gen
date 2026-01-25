"""
Scene Capture Service - port 8095

Manages capture sessions for robot cameras (RealSense, ZED).
Captures images for COLMAP reconstruction pipeline.
"""

import os
from utils import get_token
import uvicorn
from fastapi import FastAPI
from drivers.realsense import RealSenseDriver

driver = RealSenseDriver()
driver.connect()
print(driver.get_intrinsics())

app = FastAPI(
    title="Scene Capture Service",
    description="Capture images from robot cameras for COLMAP reconstruction",
    version="1.0.0",
)


@app.get("/health")
def health():
    return {"status": "ok"}


# TODO: POST /sessions/create
# TODO: POST /sessions/{id}/capture
# TODO: POST /sessions/{id}/auto
# TODO: GET /sessions/{id}/status
# TODO: POST /sessions/{id}/finalize

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(get_token("PORT")))
