#!/usr/bin/env python
"""Run a SmolVLA policy on the X-trainer dual-arm robot."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from deploy.xtrainer.msgpack_numpy import PROTOCOL_VERSION, SCHEMA_VERSION
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
DEFAULT_CONTROL_LOG_DIR = REPO_ROOT / "outputs" / "xtrainer" / "control_logs"
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


class ControlActionLog:
    """Optional JSONL trace of the client-side policy/control boundary.

    The server log deliberately contains only raw action chunks. This trace
    makes it possible to compare those chunks with the state sent by the
    client, the action selected from the queue, and the action ultimately
    accepted by the real-environment safety layer. Images are intentionally
    omitted so the trace remains small and easy to share.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("w", encoding="utf-8", buffering=1)
        self._sequence = 0
        self._closed = False

    def write(self, event: str, **fields: Any) -> None:
        """Append one trace record without risking the live control loop."""

        if self._closed:
            return
        record = {
            "event": event,
            "sequence": self._sequence,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            **fields,
        }
        self._sequence += 1
        try:
            self._file.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        except (OSError, TypeError, ValueError):
            # A full disk or malformed diagnostic record must not stop a
            # moving robot. Disable only the optional trace and retain the
            # exception in the normal application log.
            _LOGGER.exception("Disabling client control log after a write failure: %s", self.path)
            self.close()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._file.close()
        except OSError:
            _LOGGER.exception("Could not close client control log: %s", self.path)


def _default_control_log_path() -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return DEFAULT_CONTROL_LOG_DIR / f"control_{timestamp}.jsonl"


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


def _should_prefetch(
    queue_size: int,
    action_horizon: int,
    threshold: float,
    remaining: int | None = None,
) -> bool:
    """Return whether to request the next chunk.

    ``remaining`` is the reference repository's more explicit prefetch
    setting.  When omitted, retain the existing fractional-threshold
    behaviour so current launch commands do not change their timing.
    """

    if remaining is not None:
        return queue_size <= remaining
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
    control_log: ControlActionLog | None = None,
) -> InferenceResult:
    payload = _policy_payload(observation)
    if control_log is not None:
        control_log.write(
            "inference_request",
            observation_timestep=observation_timestep,
            state=np.asarray(payload["state"], dtype=np.float64).tolist(),
            task=str(payload["task"]),
        )
    response = await asyncio.wait_for(policy.infer(payload), timeout=request_timeout_s)
    selected_actions = _extract_action_chunk(response, action_horizon)
    raw_actions = np.asarray(response["action"])
    returned_count = 1 if raw_actions.ndim == 1 else raw_actions.shape[0]
    if control_log is not None:
        control_log.write(
            "inference_response",
            observation_timestep=observation_timestep,
            returned_action_count=int(returned_count),
            retained_actions=selected_actions.tolist(),
        )
    _LOGGER.info(
        "Policy inference for observation step %d returned %d actions; client retained %d (action horizon=%d)",
        observation_timestep,
        returned_count,
        len(selected_actions),
        action_horizon,
    )
    return InferenceResult(
        actions=selected_actions,
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
    prefetch_remaining: int | None = None,
    request_timeout_s: float,
    max_delta_per_step: float,
    control_log: ControlActionLog | None = None,
    monotonic_fn: Any = time.monotonic,
    sleep_fn: Any = asyncio.sleep,
) -> None:
    initial_result = await _request_action_chunk(
        policy,
        environment.get_observation(),
        action_horizon=action_horizon,
        observation_timestep=0,
        request_timeout_s=request_timeout_s,
        control_log=control_log,
    )
    action_queue = _merge_action_queue({}, initial_result, current_timestep=0)
    _LOGGER.info(
        "Initial inference returned %d actions; client action horizon is %d",
        len(initial_result.actions),
        action_horizon,
    )
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
                _LOGGER.info(
                    "Received prefetched chunk of %d actions at control step %d (queue=%d)",
                    len(result.actions),
                    step,
                    len(action_queue),
                )

            timed_action = action_queue.pop(step, None)
            used_fallback = timed_action is None
            if timed_action is None:
                if last_sent_action is None:
                    raise RuntimeError(f"No action available for timestep {step}")
                action = last_sent_action.copy()
                if pending_request is None:
                    _LOGGER.info("Action queue exhausted at step %d; requesting a replacement chunk", step)
                    pending_request = asyncio.create_task(
                        _request_action_chunk(
                            policy,
                            environment.get_observation(),
                            action_horizon=action_horizon,
                            observation_timestep=step,
                            request_timeout_s=request_timeout_s,
                            control_log=control_log,
                        )
                    )
            else:
                action = timed_action.action

            queued_action = np.asarray(action, dtype=np.float64).copy()
            rate_limited_action = _rate_limit_action(
                queued_action, last_sent_action, max_delta_per_step
            )
            applied_action = environment.apply_action(rate_limited_action, pace=False)
            last_sent_action = np.asarray(applied_action, dtype=np.float64).copy()
            if control_log is not None:
                control_log.write(
                    "control_step",
                    control_timestep=step,
                    source_observation_timestep=(
                        None if timed_action is None else timed_action.observation_timestep
                    ),
                    used_fallback=used_fallback,
                    queued_action=queued_action.tolist(),
                    rate_limited_action=rate_limited_action.tolist(),
                    applied_action=last_sent_action.tolist(),
                )

            if pending_request is None and _should_prefetch(
                len(action_queue),
                action_horizon,
                prefetch_threshold,
                prefetch_remaining,
            ):
                observation_timestep = step + 1
                _LOGGER.info(
                    "Requesting prefetch at control step %d (remaining actions=%d)",
                    step,
                    len(action_queue),
                )
                pending_request = asyncio.create_task(
                    _request_action_chunk(
                        policy,
                        environment.get_observation(),
                            action_horizon=action_horizon,
                            observation_timestep=observation_timestep,
                            request_timeout_s=request_timeout_s,
                            control_log=control_log,
                    )
                )

            deadline += period
            remaining = deadline - monotonic_fn()
            if remaining > 0:
                await sleep_fn(remaining)
            else:
                deadline = monotonic_fn()
        _LOGGER.info("Reached --max-steps=%d; ending the control loop", max_steps)
    finally:
        if pending_request is not None:
            pending_request.cancel()
            await asyncio.gather(pending_request, return_exceptions=True)


def _validate_server_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    expected_transport = {
        "protocol": "xtrainer.websocket.msgpack",
        "protocol_version": PROTOCOL_VERSION,
        "schema_version": SCHEMA_VERSION,
    }
    for key, value in expected_transport.items():
        if metadata.get(key) != value:
            raise RuntimeError(
                f"Unexpected transport metadata {key}: {metadata.get(key)!r}, expected {value!r}"
            )

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
    parser.add_argument(
        "--left-robot-ip", "--left-arm-ip", dest="left_robot_ip", default="192.168.5.1"
    )
    parser.add_argument(
        "--right-robot-ip", "--right-arm-ip", dest="right_robot_ip", default="192.168.5.2"
    )
    parser.add_argument("--left-gripper-port", default="/dev/ttyUSB1")
    parser.add_argument("--right-gripper-port", default="/dev/ttyUSB0")
    parser.add_argument("--left-gripper-id", type=int, default=21)
    parser.add_argument("--right-gripper-id", type=int, default=22)
    parser.add_argument(
        "--camera-top-serial", "--top-camera-serial", dest="camera_top_serial", default=TOP_CAMERA_SERIAL
    )
    parser.add_argument(
        "--camera-left-wrist-serial",
        "--left-wrist-camera-serial",
        dest="camera_left_wrist_serial",
        default=LEFT_WRIST_CAMERA_SERIAL,
    )
    parser.add_argument(
        "--camera-right-wrist-serial",
        "--right-wrist-camera-serial",
        dest="camera_right_wrist_serial",
        default=RIGHT_WRIST_CAMERA_SERIAL,
    )
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--camera-warmup-frames", type=int, default=30)
    parser.add_argument(
        "--max-joint-delta",
        type=float,
        default=float("inf"),
        help="Optional environment joint delta limit in radians; default disables it",
    )
    parser.add_argument(
        "--max-gripper-delta",
        type=float,
        default=float("inf"),
        help="Optional environment gripper delta limit; default disables it",
    )
    parser.add_argument(
        "--ramp-step",
        type=float,
        default=0.01,
        help="Joint delta used only while moving to reset pose",
    )
    parser.add_argument(
        "--ramp-max-steps",
        type=int,
        default=100,
        help="Maximum reset interpolation steps",
    )
    parser.add_argument(
        "--gripper-update-threshold",
        type=float,
        default=0.0,
        help="Minimum gripper target change to transmit; default sends every change",
    )
    parser.add_argument("--prefetch-threshold", type=float, default=0.7)
    parser.add_argument(
        "--prefetch-remaining",
        type=int,
        default=None,
        help=(
            "Request the next chunk when this many actions remain (compatible with the reference "
            "launcher; overrides --prefetch-threshold when supplied)"
        ),
    )
    parser.add_argument("--request-timeout", type=float, default=10.0)
    parser.add_argument(
        "--max-delta-per-step",
        type=float,
        default=0.0,
        help="Optional final client-side action delta limit; <=0 disables it",
    )
    parser.add_argument(
        "--log-control",
        action="store_true",
        help=(
            "Write an optional JSONL client trace (state, selected actions, and safety-applied "
            "actions); disabled by default"
        ),
    )
    parser.add_argument(
        "--control-log-path",
        type=Path,
        default=None,
        help="Path for --log-control (default: outputs/xtrainer/control_logs/control_<UTC>.jsonl)",
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
    if args.prefetch_remaining is not None and args.prefetch_remaining < 0:
        raise ValueError("prefetch_remaining must be non-negative")
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
    control_log = None
    if args.log_control:
        control_log = ControlActionLog(args.control_log_path or _default_control_log_path())
        _LOGGER.info("Writing client control trace to %s", control_log.path)
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
            prefetch_remaining=args.prefetch_remaining,
            request_timeout_s=args.request_timeout,
            max_delta_per_step=args.max_delta_per_step,
            control_log=control_log,
        )
    finally:
        if active_environment is not None:
            active_environment.close()
        await policy.close()
        if control_log is not None:
            control_log.close()


def main() -> None:
    args = parse_args()
    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        _LOGGER.info("Interrupted by user")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
