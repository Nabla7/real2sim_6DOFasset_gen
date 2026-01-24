"""Abstract camera driver interface."""
from abc import ABC, abstractmethod


class CameraDriver(ABC):
    """Base class for camera drivers."""

    @abstractmethod
    def connect(self) -> None:
        pass

    @abstractmethod
    def capture_frame(self) -> bytes:
        pass

    @abstractmethod
    def get_intrinsics(self) -> dict:
        pass

    @abstractmethod
    def disconnect(self) -> None:
        pass
