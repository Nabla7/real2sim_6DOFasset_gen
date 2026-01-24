"""RealSense camera driver."""
from .base import CameraDriver


class RealSenseDriver(CameraDriver):
    """RealSense D455 driver - uses pyrealsense2."""

    def connect(self) -> None:
        raise NotImplementedError

    def capture_frame(self) -> bytes:
        raise NotImplementedError

    def get_intrinsics(self) -> dict:
        raise NotImplementedError

    def disconnect(self) -> None:
        raise NotImplementedError
