"""X-trainer real-robot environment adapter.

This layer owns real-hardware safety checks and resource cleanup. It deliberately
stays outside LeRobot's global Robot factory for the first deployment version.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from deploy.xtrainer.image_tools import validate_camera_observations
from deploy.xtrainer.real.hardware.realsense_camera import DEFAULT_XTRAINER_CAMERA_CONFIGS


STATE_KEY = "observation.state"
TASK_KEY = "task"
TOP_IMAGE_KEY = "observation.images.top"
LEFT_WRIST_IMAGE_KEY = "observation.images.left_wrist"
RIGHT_WRIST_IMAGE_KEY = "observation.images.right_wrist"
IMAGE_KEYS = (TOP_IMAGE_KEY, LEFT_WRIST_IMAGE_KEY, RIGHT_WRIST_IMAGE_KEY)
_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class XTrainerSafetyConfig:
    # Policy deployment defaults intentionally do not alter the policy action.
    # Hardware/controller limits remain the final protection; callers that need
    # conservative motion can explicitly supply finite limits.
    max_joint_delta_rad: float = float("inf")
    max_gripper_delta: float = float("inf")
    ramp_step_rad: float = 0.01
    ramp_max_steps: int = 100
    gripper_update_threshold: float = 0.0
    joint_position_limit_rad: tuple[float, float] = (-float("inf"), float("inf"))
    gripper_limit: tuple[float, float] = (0.0, 1.0)
    require_finite_actions: bool = True


@dataclass
class XTrainerRealEnvironment:
    """Adapter exposing reset/get_observation/apply_action/close for X-trainer."""

    left_arm: Any
    right_arm: Any
    left_gripper: Any
    right_gripper: Any
    cameras: dict[str, Any]
    task: str
    safety: XTrainerSafetyConfig = field(default_factory=XTrainerSafetyConfig)
    control_hz: float = 30.0
    sleep_fn: Any = time.sleep
    _last_state: np.ndarray | None = field(default=None, init=False, repr=False)
    _closed: bool = field(default=False, init=False, repr=False)
    _enabled_arms: list[Any] = field(default_factory=list, init=False, repr=False)
    last_action_timing_ms: dict[str, float] = field(default_factory=dict, init=False)

    def reset(self) -> dict[str, Any]:
        self._closed = False
        self._connect_all()
        self._last_state = self._read_state()
        return self.get_observation()

    def get_observation(self) -> dict[str, Any]:
        observation = self.get_observation_snapshot()
        self._last_state = observation[STATE_KEY].copy()
        return observation

    def get_observation_snapshot(self) -> dict[str, Any]:
        """后台只读取反馈；不能用稍早的采集结果覆盖主循环的下发基准。"""
        timing: dict[str, float] = {}
        state = self._read_state(timing)
        camera_observations = self._read_images(timing)
        return {STATE_KEY: state, **camera_observations, TASK_KEY: self.task, "observation.timing_ms": timing}

    def apply_action(self, action: Any, *, pace: bool = True) -> np.ndarray:
        """Validate, limit, and send one action.

        ``pace=False`` 允许外部控制循环或平滑复位连续发送动作，
        避免额外叠加一个控制周期的等待。
        """

        action = self._validate_action(action)
        if self._last_state is None:
            self._last_state = self._read_state()
        limited = self._limit_action(action, self._last_state)

        timing: dict[str, float] = {}
        for name, arm, joints in (
            ("left_arm", self.left_arm, limited[:6]), ("right_arm", self.right_arm, limited[7:13])
        ):
            started = time.monotonic()
            arm.move_joints(joints)
            timing[f"{name}_write_ms"] = (time.monotonic() - started) * 1000
        for name, gripper, index in (
            ("left_gripper", self.left_gripper, 6), ("right_gripper", self.right_gripper, 13)
        ):
            started = time.monotonic()
            wrote = self._write_gripper_if_needed(gripper, limited[index], self._last_state[index])
            timing[f"{name}_write_ms"] = (time.monotonic() - started) * 1000
            timing[f"{name}_lock_wait_ms"] = getattr(gripper, "last_write_lock_wait_ms", 0.0) if wrote else 0.0
        self.last_action_timing_ms = timing
        if pace:
            self._sleep_control_period()
        self._last_state = limited.copy()
        return limited

    def enable_arms(self) -> None:
        """Enable both arms as one fail-closed operation.

        Connecting the environment never enables motion by itself. The real
        runner must call this method only after an explicit execute decision.
        """

        if self._enabled_arms:
            return
        enabled: list[Any] = []
        try:
            for arm in (self.left_arm, self.right_arm):
                enable = getattr(arm, "enable", None)
                if not callable(enable):
                    raise TypeError("X-trainer arm does not provide enable()")
                enable()
                enabled.append(arm)
        except BaseException:
            for arm in reversed(enabled):
                self._disable_arm(arm)
            raise
        self._enabled_arms = enabled

    def disable_arms(self) -> None:
        """Best-effort arm disable used by normal and exceptional cleanup."""

        enabled, self._enabled_arms = self._enabled_arms, []
        for arm in reversed(enabled):
            self._disable_arm(arm)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.disable_arms()
        for resource in [
            *self.cameras.values(),
            self.left_gripper,
            self.right_gripper,
            self.left_arm,
            self.right_arm,
        ]:
            self._close_resource(resource)

    def _connect_all(self) -> None:
        opened: list[Any] = []
        resources = [
            self.left_arm,
            self.right_arm,
            self.left_gripper,
            self.right_gripper,
            *self.cameras.values(),
        ]
        try:
            for resource in resources:
                connect = getattr(resource, "connect", None)
                if callable(connect):
                    connect()
                    opened.append(resource)
        except BaseException:
            for resource in reversed(opened):
                self._close_resource(resource)
            raise

    @staticmethod
    def _close_resource(resource: Any) -> None:
        close = getattr(resource, "close", None)
        if not callable(close):
            return
        try:
            close()
        except Exception:
            _LOGGER.warning("Failed to close X-trainer resource %r", resource, exc_info=True)

    @staticmethod
    def _disable_arm(arm: Any) -> None:
        disable = getattr(arm, "disable", None)
        if not callable(disable):
            return
        try:
            disable()
        except Exception:
            _LOGGER.warning("Failed to disable X-trainer arm %r", arm, exc_info=True)

    def _read_state(self, timing: dict[str, float] | None = None) -> np.ndarray:
        timing = {} if timing is None else timing
        values = []
        for name, reader in (
            ("left_arm", self.left_arm.read_joints), ("right_arm", self.right_arm.read_joints),
            ("left_gripper", self.left_gripper.read), ("right_gripper", self.right_gripper.read),
        ):
            started = time.monotonic()
            values.append(reader())
            timing[f"{name}_read_ms"] = (time.monotonic() - started) * 1000
        left, right = (np.asarray(value, dtype=np.float64) for value in values[:2])
        if left.shape != (6,):
            raise ValueError(f"left arm state must have shape (6,), got {left.shape}")
        if right.shape != (6,):
            raise ValueError(f"right arm state must have shape (6,), got {right.shape}")
        left_gripper, right_gripper = map(float, values[2:])
        state = np.concatenate([left, [left_gripper], right, [right_gripper]]).astype(np.float32)
        if not np.all(np.isfinite(state)):
            raise ValueError("X-trainer state contains NaN or Inf")
        return state

    def _read_images(self, timing: dict[str, float] | None = None) -> dict[str, np.ndarray]:
        timing = {} if timing is None else timing
        observations = {}
        for name, camera in self.cameras.items():
            config = getattr(camera, "config", None)
            observation_key = getattr(config, "observation_key", DEFAULT_XTRAINER_CAMERA_CONFIGS[name].observation_key)
            started = time.monotonic()
            observations[observation_key] = camera.read_rgb()
            timing[f"{name}_camera_ms"] = (time.monotonic() - started) * 1000
        started = time.monotonic()
        validated = validate_camera_observations(observations, IMAGE_KEYS)
        timing["image_validation_ms"] = (time.monotonic() - started) * 1000
        return validated

    def _validate_action(self, action: Any) -> np.ndarray:
        array = np.asarray(action, dtype=np.float64)
        if array.shape != (14,):
            raise ValueError(f"X-trainer action must have shape (14,), got {array.shape}")
        if self.safety.require_finite_actions and not np.all(np.isfinite(array)):
            raise ValueError("X-trainer action contains NaN or Inf")
        return array

    def _limit_action(self, action: np.ndarray, current_state: np.ndarray) -> np.ndarray:
        limited = action.copy()
        joint_min, joint_max = self.safety.joint_position_limit_rad
        gripper_min, gripper_max = self.safety.gripper_limit

        joint_indices = np.r_[0:6, 7:13]
        deltas = limited[joint_indices] - current_state[joint_indices]
        deltas = np.clip(deltas, -self.safety.max_joint_delta_rad, self.safety.max_joint_delta_rad)
        limited[joint_indices] = current_state[joint_indices] + deltas
        limited[joint_indices] = np.clip(limited[joint_indices], joint_min, joint_max)

        for idx in (6, 13):
            delta = float(np.clip(limited[idx] - current_state[idx], -self.safety.max_gripper_delta, self.safety.max_gripper_delta))
            limited[idx] = np.clip(current_state[idx] + delta, gripper_min, gripper_max)
        return limited.astype(np.float32)

    def _write_gripper_if_needed(self, gripper: Any, target: float, current: float) -> bool:
        if abs(float(target) - float(current)) >= self.safety.gripper_update_threshold:
            gripper.write(float(target))
            return True
        return False

    def _sleep_control_period(self) -> None:
        if self.control_hz > 0:
            self.sleep_fn(1.0 / self.control_hz)

    def smooth_reset(self, target_state: Any) -> np.ndarray:
        """按照 LingBot 的复位方式插值到 ``target_state``。"""

        target = self._validate_action(target_state)
        if self._last_state is None:
            self._last_state = self._read_state()
        start = self._last_state.copy()
        max_delta = float(np.max(np.abs(target - start)))
        steps = min(
            int(np.ceil(max_delta / max(self.safety.ramp_step_rad, 1e-6))),
            max(1, self.safety.ramp_max_steps),
        )
        if steps <= 1:
            return self.apply_action(target, pace=False)

        current = start
        for action in np.linspace(start, target, steps):
            current = self.apply_action(action, pace=False).astype(np.float64)
        return current.astype(np.float32)
