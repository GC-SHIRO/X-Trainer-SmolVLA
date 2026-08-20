"""Exercise X-trainer hardware without connecting to a policy server.

This command is intentionally outside ``tests/`` so automatic test discovery
cannot move real hardware. It verifies all three cameras, then moves one arm
joint at a time by +5 and -5 degrees from the observed baseline and exercises
both grippers. It always attempts to return to the observed baseline.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from deploy.xtrainer.real.environment import STATE_KEY, XTrainerRealEnvironment, XTrainerSafetyConfig
from deploy.xtrainer.real.hardware.dobot_xtrainer import XTrainerDobotArm
from deploy.xtrainer.real.hardware.feetech import (
    XTrainerFeetechGripper,
    XTrainerFeetechGripperConfig,
)
from deploy.xtrainer.real.hardware.realsense_camera import (
    LEFT_WRIST_CAMERA_SERIAL,
    RIGHT_WRIST_CAMERA_SERIAL,
    TOP_CAMERA_SERIAL,
    build_xtrainer_cameras,
)


_LOGGER = logging.getLogger(__name__)
_ACTION_DIM = 14
_JOINT_DELTA_RAD = float(np.deg2rad(5.0))
_JOINTS = (
    *(("left", joint + 1, joint) for joint in range(6)),
    *(("right", joint + 1, 7 + joint) for joint in range(6)),
)
_GRIPPERS = (("left", 6), ("right", 13))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify X-trainer cameras, then move each joint +/-5 degrees and cycle both grippers"
    )
    parser.add_argument("--left-robot-ip", default="192.168.5.1")
    parser.add_argument("--right-robot-ip", default="192.168.5.2")
    parser.add_argument("--left-gripper-port", default="/dev/ttyUSB1")
    parser.add_argument("--right-gripper-port", default="/dev/ttyUSB0")
    parser.add_argument("--left-gripper-id", type=int, default=21)
    parser.add_argument("--right-gripper-id", type=int, default=22)
    parser.add_argument("--camera-top-serial", default=TOP_CAMERA_SERIAL)
    parser.add_argument("--camera-left-wrist-serial", default=LEFT_WRIST_CAMERA_SERIAL)
    parser.add_argument("--camera-right-wrist-serial", default=RIGHT_WRIST_CAMERA_SERIAL)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--camera-warmup-frames", type=int, default=30)
    parser.add_argument("--control-hz", type=float, default=20.0)
    parser.add_argument("--hold-seconds", type=float, default=0.5)
    parser.add_argument("--max-gripper-delta", type=float, default=0.02)
    parser.add_argument("--gripper-open", type=float, default=1.0)
    parser.add_argument("--gripper-close", type=float, default=0.0)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Confirm the work area is clear, the emergency stop is reachable, and +/-5 degree motion is safe",
    )
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    if not args.execute:
        raise RuntimeError(
            "Real-hardware test is disabled. Clear the workspace, stand by the emergency stop, then pass --execute."
        )
    positive_values = {
        "camera_fps": args.camera_fps,
        "camera_width": args.camera_width,
        "camera_height": args.camera_height,
        "control_hz": args.control_hz,
        "hold_seconds": args.hold_seconds,
    }
    invalid = [name for name, value in positive_values.items() if value <= 0]
    if invalid:
        raise ValueError(f"Expected positive values for: {', '.join(invalid)}")
    if args.camera_warmup_frames < 0 or args.max_gripper_delta <= 0:
        raise ValueError("camera_warmup_frames must be non-negative and max_gripper_delta must be positive")
    if not 0.0 <= args.gripper_open <= 1.0 or not 0.0 <= args.gripper_close <= 1.0:
        raise ValueError("gripper-open and gripper-close must be in [0, 1]")
    if args.gripper_open == args.gripper_close:
        raise ValueError("gripper-open and gripper-close must differ")


def build_environment(args: argparse.Namespace) -> XTrainerRealEnvironment:
    cameras = build_xtrainer_cameras(
        top_serial=args.camera_top_serial,
        left_wrist_serial=args.camera_left_wrist_serial,
        right_wrist_serial=args.camera_right_wrist_serial,
        width=args.camera_width,
        height=args.camera_height,
        fps=args.camera_fps,
        warmup_frames=args.camera_warmup_frames,
    )
    return XTrainerRealEnvironment(
        left_arm=XTrainerDobotArm.from_parts(ip=args.left_robot_ip),
        right_arm=XTrainerDobotArm.from_parts(ip=args.right_robot_ip),
        left_gripper=XTrainerFeetechGripper(
            XTrainerFeetechGripperConfig(port=args.left_gripper_port, motor_id=args.left_gripper_id)
        ),
        right_gripper=XTrainerFeetechGripper(
            XTrainerFeetechGripperConfig(port=args.right_gripper_port, motor_id=args.right_gripper_id)
        ),
        cameras=cameras,
        task="X-trainer sequential hardware check",
        safety=XTrainerSafetyConfig(
            max_joint_delta_rad=_JOINT_DELTA_RAD,
            max_gripper_delta=args.max_gripper_delta,
            ramp_step_rad=float(np.deg2rad(1.0)),
            gripper_update_threshold=0.0,
        ),
        control_hz=args.control_hz,
    )


def _hold(seconds: float) -> None:
    if seconds > 0:
        time.sleep(seconds)


def _move_joint(environment: XTrainerRealEnvironment, target: np.ndarray, hold_seconds: float) -> np.ndarray:
    applied = environment.smooth_reset(target)
    _hold(hold_seconds)
    return np.asarray(applied, dtype=np.float64)


def _move_gripper(
    environment: XTrainerRealEnvironment,
    current: np.ndarray,
    index: int,
    target_value: float,
    hold_seconds: float,
) -> np.ndarray:
    """Move a gripper through the environment's per-step safety limit."""

    target_value = float(np.clip(target_value, 0.0, 1.0))
    current = np.asarray(current, dtype=np.float64).copy()
    while not np.isclose(current[index], target_value, atol=1e-6):
        action = current.copy()
        action[index] = target_value
        next_state = np.asarray(environment.apply_action(action), dtype=np.float64)
        if np.isclose(next_state[index], current[index], atol=1e-8):
            raise RuntimeError(f"gripper at action index {index} did not advance toward {target_value}")
        current = next_state
    _hold(hold_seconds)
    return current


def run(args: argparse.Namespace) -> None:
    _validate_args(args)
    environment = build_environment(args)
    baseline: np.ndarray | None = None
    current: np.ndarray | None = None
    try:
        observation = environment.reset()
        baseline = np.asarray(observation[STATE_KEY], dtype=np.float64)
        if baseline.shape != (_ACTION_DIM,) or not np.all(np.isfinite(baseline)):
            raise ValueError(f"Expected finite {STATE_KEY} shape ({_ACTION_DIM},), got {baseline.shape}")
        image_shapes = {
            key: value.shape for key, value in observation.items() if key != STATE_KEY and hasattr(value, "shape")
        }
        _LOGGER.info("Camera check passed: %s", image_shapes)
        _LOGGER.info("Baseline state: %s", np.array2string(baseline, precision=4))

        # Match the reference basic-control sequence: reset() samples the
        # current state as baseline; no separate global pose is commanded.
        # Cameras and every serial endpoint are connected before this point,
        # but neither arm is enabled until their observation is verified.
        environment.enable_arms()
        current = baseline.copy()

        for side, joint_number, action_index in _JOINTS:
            for direction, delta in (("+5", _JOINT_DELTA_RAD), ("-5", -_JOINT_DELTA_RAD)):
                target = baseline.copy()
                target[action_index] += delta
                _LOGGER.info("Testing %s joint %s: %s degrees", side, joint_number, direction)
                current = _move_joint(environment, target, args.hold_seconds)
                current = _move_joint(environment, baseline, args.hold_seconds)

        for side, action_index in _GRIPPERS:
            for state_name, target_value in (("open", args.gripper_open), ("close", args.gripper_close)):
                _LOGGER.info("Testing %s gripper: %s (%.3f)", side, state_name, target_value)
                current = _move_gripper(environment, current, action_index, target_value, args.hold_seconds)
            current = _move_gripper(
                environment,
                current,
                action_index,
                float(baseline[action_index]),
                args.hold_seconds,
            )

        _LOGGER.info("Sequential X-trainer hardware check passed")
    finally:
        if baseline is not None:
            _LOGGER.info("Restoring startup baseline state")
            try:
                recovery = current if current is not None else baseline
                recovery = _move_gripper(environment, recovery, 6, float(baseline[6]), 0.0)
                recovery = _move_gripper(environment, recovery, 13, float(baseline[13]), 0.0)
                _move_joint(environment, baseline, 0.0)
            except Exception:
                _LOGGER.exception("Could not return to the startup state; use the emergency stop if needed")
        environment.close()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
