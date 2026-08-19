"""RealSense RGB camera wrapper for X-trainer deployment."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np


TOP_CAMERA_SERIAL = "409122273405"
LEFT_WRIST_CAMERA_SERIAL = "412622272997"
RIGHT_WRIST_CAMERA_SERIAL = "412622271417"


@dataclass(frozen=True)
class XTrainerRealSenseCameraConfig:
    name: str
    serial: str
    observation_key: str
    width: int = 640
    height: int = 480
    fps: int = 30
    warmup_frames: int = 30


DEFAULT_XTRAINER_CAMERA_CONFIGS: dict[str, XTrainerRealSenseCameraConfig] = {
    "top": XTrainerRealSenseCameraConfig(
        name="top",
        serial=TOP_CAMERA_SERIAL,
        observation_key="observation.images.top",
    ),
    "left_wrist": XTrainerRealSenseCameraConfig(
        name="left_wrist",
        serial=LEFT_WRIST_CAMERA_SERIAL,
        observation_key="observation.images.left_wrist",
    ),
    "right_wrist": XTrainerRealSenseCameraConfig(
        name="right_wrist",
        serial=RIGHT_WRIST_CAMERA_SERIAL,
        observation_key="observation.images.right_wrist",
    ),
}


class XTrainerRealSenseCamera:
    """Thin RealSense adapter with delayed SDK import and warmup reads."""

    def __init__(
        self,
        config: XTrainerRealSenseCameraConfig,
        *,
        camera_factory: Any | None = None,
        camera_config_factory: Any | None = None,
    ) -> None:
        self.config = config
        self._camera_factory = camera_factory
        self._camera_config_factory = camera_config_factory
        self._camera = None

    @property
    def is_connected(self) -> bool:
        return self._camera is not None

    def connect(self) -> None:
        if self._camera is not None:
            return
        camera_factory, config_factory = self._resolve_factories()
        camera_config = config_factory(
            serial_number_or_name=self.config.serial,
            fps=self.config.fps,
            width=self.config.width,
            height=self.config.height,
        )
        camera = camera_factory(camera_config)
        try:
            camera.connect()
            for _ in range(max(0, self.config.warmup_frames)):
                camera.read()
        except BaseException:
            disconnect = getattr(camera, "disconnect", None)
            if callable(disconnect):
                disconnect()
            raise
        self._camera = camera

    def read_rgb(self) -> np.ndarray:
        if self._camera is None:
            raise ConnectionError("RealSense camera is not connected")
        image = np.asarray(self._camera.read())
        if image.ndim != 3 or image.shape[2] != 3:
            raise RuntimeError(f"RealSense RGB frame must be HxWx3, got {image.shape}")
        if image.dtype != np.uint8:
            image = image.astype(np.uint8, copy=False)
        return image

    def close(self) -> None:
        camera, self._camera = self._camera, None
        if camera is not None:
            disconnect = getattr(camera, "disconnect", None)
            if callable(disconnect):
                disconnect()

    def _resolve_factories(self):
        if self._camera_factory is not None and self._camera_config_factory is not None:
            return self._camera_factory, self._camera_config_factory

        from lerobot.cameras.realsense import RealSenseCamera, RealSenseCameraConfig

        return self._camera_factory or RealSenseCamera, self._camera_config_factory or RealSenseCameraConfig


def build_xtrainer_cameras(
    *,
    top_serial: str = TOP_CAMERA_SERIAL,
    left_wrist_serial: str = LEFT_WRIST_CAMERA_SERIAL,
    right_wrist_serial: str = RIGHT_WRIST_CAMERA_SERIAL,
    width: int = 640,
    height: int = 480,
    fps: int = 30,
    warmup_frames: int = 30,
) -> dict[str, XTrainerRealSenseCamera]:
    configs = {
        "top": XTrainerRealSenseCameraConfig(
            name="top",
            serial=top_serial,
            observation_key="observation.images.top",
            width=width,
            height=height,
            fps=fps,
            warmup_frames=warmup_frames,
        ),
        "left_wrist": XTrainerRealSenseCameraConfig(
            name="left_wrist",
            serial=left_wrist_serial,
            observation_key="observation.images.left_wrist",
            width=width,
            height=height,
            fps=fps,
            warmup_frames=warmup_frames,
        ),
        "right_wrist": XTrainerRealSenseCameraConfig(
            name="right_wrist",
            serial=right_wrist_serial,
            observation_key="observation.images.right_wrist",
            width=width,
            height=height,
            fps=fps,
            warmup_frames=warmup_frames,
        ),
    }
    return {name: XTrainerRealSenseCamera(config) for name, config in configs.items()}


def read_camera_observations(cameras: dict[str, XTrainerRealSenseCamera]) -> dict[str, np.ndarray]:
    """Read all configured cameras keyed by LeRobot observation field."""

    observations = {}
    for camera in cameras.values():
        observations[camera.config.observation_key] = camera.read_rgb()
    return observations


def warmup_cameras(cameras: dict[str, XTrainerRealSenseCamera], *, seconds: float = 1.0) -> None:
    deadline = time.monotonic() + max(0.0, seconds)
    while time.monotonic() < deadline:
        for camera in cameras.values():
            camera.read_rgb()
