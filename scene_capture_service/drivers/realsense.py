"""A thread-safe RealSense camera driver."""

from .base import CameraDriver
import pyrealsense2 as rs
import numpy as np
import threading


class RealSenseDriver(CameraDriver):
    """RealSense D455 driver - uses pyrealsense2."""

    _lock = threading.Lock()

    def __init__(self):
        self.pipe = rs.pipeline()
        self.config = rs.config()
        self.profile = None

    def connect(self) -> None:
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        # self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        # self.config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30)
        self.profile = self.pipe.start(self.config)

    def capture_frame(self) -> bytes:
        with self._lock:
            frames = self.pipe.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                raise RuntimeError("Failed to capture color frame")

            # Convert frame to numpy array then to bytes
            frame_data = np.asanyarray(color_frame.get_data())
            return frame_data.tobytes()

    def get_intrinsics(self) -> dict:
        return (
            self.profile.get_stream(rs.stream.color)
            .as_video_stream_profile()
            .get_intrinsics()
        )

    def disconnect(self) -> None:
        self.pipe.stop()
