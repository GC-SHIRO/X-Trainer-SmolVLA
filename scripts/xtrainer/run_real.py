#!/usr/bin/env python
"""Run a SmolVLA policy on the X-trainer dual-arm robot."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from deploy.xtrainer.real import XTrainerRealEnvironment, XTrainerSafetyConfig
from deploy.xtrainer.real.environment import (
    LEFT_WRIST_IMAGE_KEY,
    RIGHT_WRIST_IMAGE_KEY,
    STATE_KEY,
    TASK_KEY,
    TOP_IMAGE_KEY,
)
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
from deploy.xtrainer.websocket_client_policy import XTrainerWebSocketPolicyClient

ACTION_DIM = 14
OLD_ACTION_WEIGHT = 0.3
NEW_ACTION_WEIGHT = 0.7
_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class TimedAction:
    action: np.ndarray
    observation_timestep: int
    timestep: int


@dataclass(frozen=True)
class InferenceResult:
    actions: np.ndarray
    observation_timestep: int


def _extract_action_chunk(response: dict[str, Any], action_horizon: int) -> np.ndarray:
    if "action" not in response:
        raise KeyError(f"Missing 'action' in policy response: {tuple(response.keys())}")
    actions = np.asarray(response["action"], dtype=np.float64)
    if actions.ndim == 1:
        actions = actions[None, :]
    if actions.ndim != 2 or actions.shape[1] != ACTION_DIM:
        raise ValueError(f"Expected action shape (H, {ACTION_DIM}), got {actions.shape}")
    if actions.shape[0] == 0:
        raise ValueError("Policy returned an empty action chunk")
    if not np.all(np.isfinite(actions)):
        raise ValueError("Policy returned non-finite actions")
    return actions[:action_horizon].copy()


def _merge_action_queue(
    current_queue: dict[int, TimedAction],
    result: InferenceResult,
    *,
    current_timestep: int,
) -> dict[int, TimedAction]:
    """Replace future actions with a new chunk, blending matching timesteps."""

    merged: dict[int, TimedAction] = {}
    for index, new_action in enumerate(result.actions):
        timestep = result.observation_timestep + index
        if timestep < current_timestep:
            continue
        old_action = current_queue.get(timestep)
        action = np.asarray(new_action, dtype=np.float64).copy()
        if old_action is not None:
            action = OLD_ACTION_WEIGHT * old_action.action + NEW_ACTION_WEIGHT * action
        merged[timestep] = TimedAction(
            action=action,
            observation_timestep=result.observation_timestep,
            timestep=timestep,
        )
    return merged


def _rate_limit_action(
    action: np.ndarray,
    last_action: np.ndarray | None,
    max_delta_per_step: float,
) -> np.ndarray:
    target = np.asarray(action, dtype=np.float64)
    if target.shape != (ACTION_DIM,):
        raise ValueError(f"Expected action shape ({ACTION_DIM},), got {target.shape}")
    if last_action is None or max_delta_per_step <= 0:
        return target.copy()
    previous = np.asarray(last_action, dtype=np.float64)
    if previous.shape != (ACTION_DIM,):
        raise ValueError(f"Expected last action shape ({ACTION_DIM},), got {previous.shape}")
    return previous + np.clip(target - previous, -max_delta_per_step, max_delta_per_step)


def _should_prefetch(queue_size: int, action_horizon: int, threshold: float) -> bool:
    return threshold > 0 and queue_size / action_horizon <= threshold


def _policy_payload(observation: dict[str, Any]) -> dict[str, Any]:
    return {
        "state": observation[STATE_KEY],
        "images": {
            "top": observation[TOP_IMAGE_KEY],
            "left_wrist": observation[LEFT_WRIST_IMAGE_KEY],
            "right_wrist": observation[RIGHT_WRIST_IMAGE_KEY],
        },
        "task": observation[TASK_KEY],
    }


async def _request_action_chunk(
    policy: Any,
    observation: dict[str, Any],
    *,
    action_horizon: int,
    observation_timestep: int,
    request_timeout_s: float,
) -> InferenceResult:
    response = await asyncio.wait_for(policy.infer(_policy_payload(observation)), timeout=request_timeout_s)
    return InferenceResult(
        actions=_extract_action_chunk(response, action_horizon),
        observation_timestep=observation_timestep,
    )


async def run_control_loop(
    policy: Any,
    environment: Any,
    *,
    action_horizon: int,
    control_hz: float,
    max_steps: int,
    prefetch_threshold: float,
    request_timeout_s: float,
    max_delta_per_step: float,
    monotonic_fn: Any = time.monotonic,
    sleep_fn: Any = asyncio.sleep,
) -> None:
    initial_result = await _request_action_chunk(
        policy,
        environment.get_observation(),
        action_horizon=action_horizon,
        observation_timestep=0,
        request_timeout_s=request_timeout_s,
    )
    action_queue = _merge_action_queue({}, initial_result, current_timestep=0)
    last_sent_action: np.ndarray | None = None
    pending_request: asyncio.Task[InferenceResult] | None = None
    period = 1.0 / control_hz
    deadline = monotonic_fn()

    try:
        for step in range(max_steps):
            if pending_request is not None and pending_request.done():
                completed_request, pending_request = pending_request, None
                result = completed_request.result()
                action_queue = _merge_action_queue(action_queue, result, current_timestep=step)

            timed_action = action_queue.pop(step, None)
            if timed_action is None:
                if last_sent_action is None:
                    raise RuntimeError(f"No action available for timestep {step}")
                action = last_sent_action.copy()
                if pending_request is None:
                    pending_request = asyncio.create_task(
                        _request_action_chunk(
                            policy,
                            environment.get_observation(),
                            action_horizon=action_horizon,
                            observation_timestep=step,
                            request_timeout_s=request_timeout_s,
                        )
                    )
            else:
                action = timed_action.action

            action = _rate_limit_action(action, last_sent_action, max_delta_per_step)
            applied_action = environment.apply_action(action, pace=False)
            last_sent_action = np.asarray(applied_action, dtype=np.float64).copy()

            if pending_request is None and _should_prefetch(
                len(action_queue), action_horizon, prefetch_threshold
            ):
                observation_timestep = step + 1
                pending_request = asyncio.create_task(
                    _request_action_chunk(
                        policy,
                        environment.get_observation(),
                        action_horizon=action_horizon,
                        observation_timestep=observation_timestep,
                        request_timeout_s=request_timeout_s,
                    )
                )

            deadline += period
            remaining = deadline - monotonic_fn()
            if remaining > 0:
                await sleep_fn(remaining)
            else:
                deadline = monotonic_fn()
    finally:
        if pending_request is not None:
            pending_request.cancel()
            await asyncio.gather(pending_request, return_exceptions=True)


def _validate_server_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    policy_metadata = metadata.get("policy")
    if not isinstance(policy_metadata, dict):
        raise RuntimeError("Server metadata is missing the policy contract")
    expected = {"model_type": "smolvla", "schema_version": 1, "action_dim": 14, "state_dim": 14}
    for key, value in expected.items():
        if policy_metadata.get(key) != value:
            raise RuntimeError(
                f"Unexpected policy metadata {key}: {policy_metadata.get(key)!r}, expected {value!r}"
            )
    return policy_metadata


def _metadata_reset_pose(policy_metadata: dict[str, Any]) -> np.ndarray | None:
    if "reset_pose" not in policy_metadata:
        return None
    reset_pose = np.asarray(policy_metadata["reset_pose"], dtype=np.float64)
    if reset_pose.shape != (ACTION_DIM,):
        raise RuntimeError(f"Expected reset_pose shape ({ACTION_DIM},), got {reset_pose.shape}")
    if not np.all(np.isfinite(reset_pose)):
        raise RuntimeError("reset_pose contains non-finite values")
    return reset_pose


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
        task=args.task,
        safety=XTrainerSafetyConfig(
            max_joint_delta_rad=args.max_joint_delta,
            max_gripper_delta=args.max_gripper_delta,
            ramp_step_rad=args.ramp_step,
            ramp_max_steps=args.ramp_max_steps,
            gripper_update_threshold=args.gripper_update_threshold,
        ),
        control_hz=args.control_hz,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SmolVLA on an X-trainer robot")
    parser.add_argument("--host", required=True, help="SmolVLA policy server address")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--task", default="pick up the object")
    parser.add_argument("--action-horizon", type=int, default=50)
    parser.add_argument("--control-hz", type=float, default=20.0)
    parser.add_argument("--max-steps", type=int, default=1000)
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
    parser.add_argument("--max-joint-delta", type=float, default=0.17)
    parser.add_argument("--max-gripper-delta", type=float, default=0.05)
    parser.add_argument("--ramp-step", type=float, default=0.01)
    parser.add_argument("--ramp-max-steps", type=int, default=100)
    parser.add_argument("--gripper-update-threshold", type=float, default=0.02)
    parser.add_argument("--prefetch-threshold", type=float, default=0.7)
    parser.add_argument("--request-timeout", type=float, default=10.0)
    parser.add_argument(
        "--max-delta-per-step",
        type=float,
        default=0.0,
        help="Optional final client-side action delta limit; <=0 disables it",
    )
    parser.add_argument(
        "--observation-similarity-epsilon",
        type=float,
        default=None,
        help="Reserved for a later 12-joint observation similarity filter; currently disabled",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Explicitly allow enabling and moving the real robot",
    )
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    positive_values = {
        "port": args.port,
        "action_horizon": args.action_horizon,
        "control_hz": args.control_hz,
        "max_steps": args.max_steps,
        "camera_fps": args.camera_fps,
        "camera_width": args.camera_width,
        "camera_height": args.camera_height,
        "request_timeout": args.request_timeout,
    }
    invalid = [name for name, value in positive_values.items() if value <= 0]
    if invalid:
        raise ValueError(f"Expected positive values for: {', '.join(invalid)}")
    if not 0 <= args.prefetch_threshold <= 1:
        raise ValueError("prefetch_threshold must be in [0, 1]")
    non_negative_values = {
        "camera_warmup_frames": args.camera_warmup_frames,
        "max_joint_delta": args.max_joint_delta,
        "max_gripper_delta": args.max_gripper_delta,
        "ramp_step": args.ramp_step,
        "ramp_max_steps": args.ramp_max_steps,
        "gripper_update_threshold": args.gripper_update_threshold,
        "max_delta_per_step": args.max_delta_per_step,
    }
    invalid = [name for name, value in non_negative_values.items() if value < 0]
    if invalid:
        raise ValueError(f"Expected non-negative values for: {', '.join(invalid)}")


async def run(
    args: argparse.Namespace,
    *,
    policy: Any | None = None,
    environment: Any | None = None,
) -> None:
    _validate_args(args)
    if not args.execute:
        raise RuntimeError("Real-robot motion is disabled; pass --execute only after completing safety checks")
    if args.observation_similarity_epsilon is not None:
        _LOGGER.warning("--observation-similarity-epsilon is reserved and has no effect in this version")

    policy = policy or XTrainerWebSocketPolicyClient(f"http://{args.host}:{args.port}")
    active_environment = environment
    try:
        metadata = await policy.connect()
        policy_metadata = _validate_server_metadata(metadata)
        reset_pose = _metadata_reset_pose(policy_metadata)
        active_environment = active_environment or build_environment(args)

        active_environment.reset()
        active_environment.enable_arms()
        if reset_pose is not None:
            active_environment.smooth_reset(reset_pose)
        await policy.reset()

        await run_control_loop(
            policy,
            active_environment,
            action_horizon=args.action_horizon,
            control_hz=args.control_hz,
            max_steps=args.max_steps,
            prefetch_threshold=args.prefetch_threshold,
            request_timeout_s=args.request_timeout,
            max_delta_per_step=args.max_delta_per_step,
        )
    finally:
        if active_environment is not None:
            active_environment.close()
        await policy.close()


def main() -> None:
    args = parse_args()
    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        _LOGGER.info("Interrupted by user")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
