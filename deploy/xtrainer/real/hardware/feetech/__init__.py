"""Feetech SMS/STS gripper wrapper for X-trainer deployment."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from .sms_sts import SmsStsGripperBus


_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class XTrainerFeetechGripperConfig:
    port: str
    motor_id: int
    name: str = "gripper"
    min_position: int = 2048
    max_position: int = 3052
    baudrate: int = 1_000_000
    timeout_s: float = 0.1
    normalized_min: float = 0.0
    normalized_max: float = 1.0


class XTrainerFeetechGripper:
    """Single X-trainer SMS/STS gripper with a normalized ``[0, 1]`` API.

    X-trainer's factory grippers respond with model number 10760. The SMS/STS
    protocol and register layout are compatible with the reference deployment,
    but that model is not part of LeRobot's ``FeetechMotorsBus`` model table.
    """

    def __init__(self, config: XTrainerFeetechGripperConfig, *, bus_factory: Any | None = None) -> None:
        if config.max_position <= config.min_position:
            raise ValueError("invalid gripper position range")
        self.config = config
        self._bus_factory = bus_factory
        self._bus: Any | None = None
        self.model_number: int | None = None

    @property
    def is_connected(self) -> bool:
        return self._bus is not None and bool(getattr(self._bus, "is_connected", True))

    def connect(self) -> None:
        if self.is_connected:
            return
        bus = self._make_bus()
        try:
            self.model_number = int(bus.connect())
        except BaseException:
            close = getattr(bus, "close", None)
            if callable(close):
                close()
            raise
        self._bus = bus
        _LOGGER.info(
            "Connected X-trainer Feetech gripper %s (id=%s, model_number=%s)",
            self.config.port,
            self.config.motor_id,
            self.model_number,
        )

    def close(self) -> None:
        bus, self._bus = self._bus, None
        self.model_number = None
        if bus is not None:
            close = getattr(bus, "close", None)
            if callable(close):
                close()

    def read(self) -> float:
        bus = self._require_bus()
        return self._to_normalized(float(bus.read_position()))

    def write(self, normalized_position: float) -> None:
        if not np.isfinite(normalized_position):
            raise ValueError("gripper command must be finite")
        normalized = float(np.clip(normalized_position, self.config.normalized_min, self.config.normalized_max))
        self._require_bus().write_position(int(round(self._from_normalized(normalized))))

    def _make_bus(self) -> Any:
        factory = self._bus_factory or SmsStsGripperBus
        return factory(
            port=self.config.port,
            motor_id=self.config.motor_id,
            baudrate=self.config.baudrate,
            timeout_s=self.config.timeout_s,
        )

    def _require_bus(self) -> Any:
        if not self.is_connected:
            raise ConnectionError("Feetech gripper is not connected")
        assert self._bus is not None
        return self._bus

    def _to_normalized(self, raw_position: float) -> float:
        span = self.config.max_position - self.config.min_position
        value = (raw_position - self.config.min_position) / span
        return float(np.clip(value, self.config.normalized_min, self.config.normalized_max))

    def _from_normalized(self, normalized_position: float) -> float:
        span = self.config.max_position - self.config.min_position
        return self.config.min_position + normalized_position * span


__all__ = ["XTrainerFeetechGripper", "XTrainerFeetechGripperConfig"]
