"""Feetech gripper wrapper for X-trainer deployment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class XTrainerFeetechGripperConfig:
    port: str
    motor_id: int
    name: str = "gripper"
    model: str = "sts3215"
    min_position: float = 0.0
    max_position: float = 100.0
    normalized_min: float = 0.0
    normalized_max: float = 1.0


class XTrainerFeetechGripper:
    """Single-motor Feetech gripper with delayed serial SDK import."""

    def __init__(self, config: XTrainerFeetechGripperConfig, *, bus_factory: Any | None = None) -> None:
        self.config = config
        self._bus_factory = bus_factory
        self._bus = None

    @property
    def is_connected(self) -> bool:
        return self._bus is not None

    def connect(self) -> None:
        if self._bus is not None:
            return
        bus_factory, motor_cls, norm_mode = self._resolve_factory()
        bus = bus_factory(
            port=self.config.port,
            motors={self.config.name: motor_cls(self.config.motor_id, self.config.model, norm_mode)},
        )
        try:
            bus.connect()
        except BaseException:
            disconnect = getattr(bus, "disconnect", None)
            if callable(disconnect):
                disconnect()
            raise
        self._bus = bus

    def close(self) -> None:
        bus, self._bus = self._bus, None
        if bus is not None:
            disconnect = getattr(bus, "disconnect", None)
            if callable(disconnect):
                disconnect()

    def read(self) -> float:
        if self._bus is None:
            raise ConnectionError("Feetech gripper is not connected")
        value = self._bus.read("Present_Position", self.config.name)
        return self._to_normalized(float(value))

    def write(self, normalized_position: float) -> None:
        if self._bus is None:
            raise ConnectionError("Feetech gripper is not connected")
        if not np.isfinite(normalized_position):
            raise ValueError("gripper command must be finite")
        normalized = float(np.clip(normalized_position, self.config.normalized_min, self.config.normalized_max))
        self._bus.write("Goal_Position", self.config.name, self._from_normalized(normalized))

    def _to_normalized(self, raw_position: float) -> float:
        span = self.config.max_position - self.config.min_position
        if span <= 0:
            raise ValueError("invalid gripper position range")
        value = (raw_position - self.config.min_position) / span
        return float(np.clip(value, self.config.normalized_min, self.config.normalized_max))

    def _from_normalized(self, normalized_position: float) -> float:
        span = self.config.max_position - self.config.min_position
        return self.config.min_position + normalized_position * span

    def _resolve_factory(self):
        if self._bus_factory is not None:
            return self._bus_factory, _MockMotor, None

        from lerobot.motors import Motor, MotorNormMode
        from lerobot.motors.feetech import FeetechMotorsBus

        return FeetechMotorsBus, Motor, MotorNormMode.RANGE_0_100


@dataclass(frozen=True)
class _MockMotor:
    id: int
    model: str
    norm_mode: Any = None
