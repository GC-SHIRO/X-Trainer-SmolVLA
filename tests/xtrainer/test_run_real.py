import asyncio
import json

import numpy as np
import pytest

from deploy.xtrainer.msgpack_numpy import protocol_metadata
from deploy.xtrainer.real.environment import (
    LEFT_WRIST_IMAGE_KEY,
    RIGHT_WRIST_IMAGE_KEY,
    STATE_KEY,
    TASK_KEY,
    TOP_IMAGE_KEY,
)
from scripts.xtrainer.run_real import (
    ControlActionLog,
    InferenceResult,
    TimedAction,
    _extract_action_chunk,
    _merge_action_queue,
    _rate_limit_action,
    _should_prefetch,
    parse_args,
    run,
    run_control_loop,
)


def _observation():
    image = np.zeros((4, 5, 3), dtype=np.uint8)
    return {
        STATE_KEY: np.zeros(14, dtype=np.float32),
        TOP_IMAGE_KEY: image.copy(),
        LEFT_WRIST_IMAGE_KEY: image.copy(),
        RIGHT_WRIST_IMAGE_KEY: image.copy(),
        TASK_KEY: "pick",
    }


class MockEnvironment:
    def __init__(self):
        self.actions = []
        self.reset_called = False
        self.enabled = False
        self.closed = False
        self.reset_pose = None

    def reset(self):
        self.reset_called = True
        return _observation()

    def enable_arms(self):
        self.enabled = True

    def smooth_reset(self, reset_pose):
        self.reset_pose = np.asarray(reset_pose, dtype=np.float64).copy()

    def get_observation(self):
        return _observation()

    def apply_action(self, action, *, pace=True):
        assert pace is False
        applied = np.asarray(action, dtype=np.float64).copy()
        self.actions.append(applied)
        return applied

    def close(self):
        self.closed = True


class MockPolicy:
    def __init__(self, chunks, *, metadata=None, hang_after=None):
        self.chunks = [np.asarray(chunk, dtype=np.float64) for chunk in chunks]
        self.server_metadata = metadata or protocol_metadata(
            {
                "policy": {
                    "model_type": "smolvla",
                    "schema_version": 1,
                    "action_dim": 14,
                    "state_dim": 14,
                }
            }
        )
        self.hang_after = hang_after
        self.infer_calls = 0
        self.active_infers = 0
        self.max_active_infers = 0
        self.payloads = []
        self.connected = False
        self.reset_called = False
        self.closed = False

    async def connect(self):
        self.connected = True
        return self.server_metadata

    async def reset(self):
        self.reset_called = True
        return {"ok": True}

    async def infer(self, payload):
        call_index = self.infer_calls
        self.infer_calls += 1
        self.payloads.append(payload)
        self.active_infers += 1
        self.max_active_infers = max(self.max_active_infers, self.active_infers)
        try:
            if self.hang_after is not None and call_index >= self.hang_after:
                await asyncio.Event().wait()
            await asyncio.sleep(0)
            return {"action": self.chunks[min(call_index, len(self.chunks) - 1)].copy()}
        finally:
            self.active_infers -= 1

    async def close(self):
        self.closed = True


def test_extract_action_chunk_validates_shape_and_finite_values():
    chunk = np.zeros((5, 14), dtype=np.float32)

    assert _extract_action_chunk({"action": chunk}, 3).shape == (3, 14)
    with pytest.raises(ValueError, match="shape"):
        _extract_action_chunk({"action": np.zeros((5, 13))}, 5)
    chunk[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        _extract_action_chunk({"action": chunk}, 5)


def test_merge_drops_stale_actions_and_blends_matching_timesteps():
    old = {
        timestep: TimedAction(np.zeros(14), observation_timestep=0, timestep=timestep)
        for timestep in (5, 6, 7)
    }
    incoming = InferenceResult(actions=np.full((4, 14), 10.0), observation_timestep=4)

    merged = _merge_action_queue(old, incoming, current_timestep=5)

    assert tuple(merged) == (5, 6, 7)
    np.testing.assert_allclose(merged[5].action, 7.0)
    np.testing.assert_allclose(merged[6].action, 7.0)
    np.testing.assert_allclose(merged[7].action, 7.0)
    assert all(action.observation_timestep == 4 for action in merged.values())


def test_merge_uses_new_action_directly_after_overlap():
    old = {3: TimedAction(np.zeros(14), observation_timestep=0, timestep=3)}
    incoming = InferenceResult(actions=np.full((3, 14), 10.0), observation_timestep=3)

    merged = _merge_action_queue(old, incoming, current_timestep=3)

    np.testing.assert_allclose(merged[3].action, 7.0)
    np.testing.assert_allclose(merged[4].action, 10.0)
    np.testing.assert_allclose(merged[5].action, 10.0)


def test_final_rate_limit_runs_after_blending():
    blended = np.full(14, 7.0)

    limited = _rate_limit_action(blended, np.zeros(14), max_delta_per_step=0.2)

    np.testing.assert_allclose(limited, 0.2)


def test_cli_uses_planned_camera_defaults_and_reserved_switch():
    args = parse_args(["--host", "127.0.0.1"])

    assert args.camera_top_serial == "409122273405"
    assert args.camera_left_wrist_serial == "412622272997"
    assert args.camera_right_wrist_serial == "412622271417"
    assert args.prefetch_threshold == pytest.approx(0.7)
    assert args.observation_similarity_epsilon is None
    assert args.execute is False
    assert args.log_control is False
    assert args.control_log_path is None


def test_cli_accepts_optional_client_control_log_path(tmp_path):
    log_path = tmp_path / "client.jsonl"

    args = parse_args(["--host", "127.0.0.1", "--log-control", "--control-log-path", str(log_path)])

    assert args.log_control is True
    assert args.control_log_path == log_path


def test_cli_accepts_reference_hardware_option_names_and_prefetch_remaining():
    args = parse_args(
        [
            "--host",
            "127.0.0.1",
            "--left-arm-ip",
            "192.168.5.11",
            "--right-arm-ip",
            "192.168.5.12",
            "--top-camera-serial",
            "top",
            "--left-wrist-camera-serial",
            "left",
            "--right-wrist-camera-serial",
            "right",
            "--prefetch-remaining",
            "2",
        ]
    )

    assert args.left_robot_ip == "192.168.5.11"
    assert args.right_robot_ip == "192.168.5.12"
    assert (args.camera_top_serial, args.camera_left_wrist_serial, args.camera_right_wrist_serial) == (
        "top",
        "left",
        "right",
    )
    assert args.prefetch_remaining == 2
    assert _should_prefetch(2, 5, 0.7, args.prefetch_remaining)
    assert not _should_prefetch(3, 5, 0.7, args.prefetch_remaining)


def test_control_loop_blends_returned_prefetch_and_keeps_one_request_in_flight():
    async def exercise():
        policy = MockPolicy([np.zeros((4, 14)), np.full((4, 14), 10.0)])
        environment = MockEnvironment()

        async def yield_control(_seconds):
            for _ in range(3):
                await asyncio.sleep(0)

        await run_control_loop(
            policy,
            environment,
            action_horizon=4,
            control_hz=20.0,
            max_steps=3,
            prefetch_threshold=0.5,
            request_timeout_s=1.0,
            max_delta_per_step=0.0,
            monotonic_fn=lambda: 0.0,
            sleep_fn=yield_control,
        )
        return policy, environment

    policy, environment = asyncio.run(exercise())

    assert len(environment.actions) == 3
    np.testing.assert_allclose(environment.actions[0], 0.0)
    np.testing.assert_allclose(environment.actions[1], 0.0)
    np.testing.assert_allclose(environment.actions[2], 7.0)
    assert policy.max_active_infers == 1
    assert set(policy.payloads[0]) == {"state", "images", "task"}
    assert set(policy.payloads[0]["images"]) == {"top", "left_wrist", "right_wrist"}


def test_client_control_log_records_state_response_and_applied_action(tmp_path):
    async def exercise(log_path):
        policy = MockPolicy([np.full((2, 14), 0.25)])
        environment = MockEnvironment()
        control_log = ControlActionLog(log_path)
        try:
            await run_control_loop(
                policy,
                environment,
                action_horizon=2,
                control_hz=100.0,
                max_steps=1,
                prefetch_threshold=0.0,
                request_timeout_s=1.0,
                max_delta_per_step=0.0,
                control_log=control_log,
                monotonic_fn=lambda: 0.0,
                sleep_fn=asyncio.sleep,
            )
        finally:
            control_log.close()

    log_path = tmp_path / "control.jsonl"
    asyncio.run(exercise(log_path))
    records = [json.loads(line) for line in log_path.read_text().splitlines()]

    assert [record["event"] for record in records] == [
        "inference_request",
        "inference_response",
        "control_step",
    ]
    assert records[0]["state"] == [0.0] * 14
    assert records[0]["task"] == "pick"
    assert records[1]["retained_actions"] == [[0.25] * 14, [0.25] * 14]
    assert records[2]["source_observation_timestep"] == 0
    assert records[2]["used_fallback"] is False
    assert records[2]["queued_action"] == [0.25] * 14
    assert records[2]["rate_limited_action"] == [0.25] * 14
    assert records[2]["applied_action"] == [0.25] * 14


def test_prefetch_timeout_closes_policy_and_hardware():
    args = parse_args(
        [
            "--host",
            "127.0.0.1",
            "--execute",
            "--action-horizon",
            "2",
            "--max-steps",
            "5",
            "--control-hz",
            "100",
            "--prefetch-threshold",
            "0.5",
            "--request-timeout",
            "0.001",
        ]
    )
    policy = MockPolicy([np.zeros((2, 14))], hang_after=1)
    environment = MockEnvironment()

    with pytest.raises(TimeoutError):
        asyncio.run(run(args, policy=policy, environment=environment))

    assert environment.reset_called
    assert environment.enabled
    assert environment.closed
    assert policy.connected
    assert policy.reset_called
    assert policy.closed
    assert policy.max_active_infers == 1


def test_run_applies_reset_pose_from_policy_metadata_and_cleans_up():
    reset_pose = np.arange(14, dtype=np.float64) / 10
    metadata = protocol_metadata(
        {
            "policy": {
                "model_type": "smolvla",
                "schema_version": 1,
                "action_dim": 14,
                "state_dim": 14,
                "reset_pose": reset_pose.tolist(),
            }
        }
    )
    args = parse_args(
        [
            "--host",
            "127.0.0.1",
            "--execute",
            "--action-horizon",
            "1",
            "--max-steps",
            "1",
            "--control-hz",
            "1000",
            "--prefetch-threshold",
            "0",
        ]
    )
    policy = MockPolicy([np.zeros((1, 14))], metadata=metadata)
    environment = MockEnvironment()

    asyncio.run(run(args, policy=policy, environment=environment))

    np.testing.assert_allclose(environment.reset_pose, reset_pose)
    assert environment.closed
    assert policy.closed


def test_run_requires_explicit_execute_before_connecting_resources():
    args = parse_args(["--host", "127.0.0.1"])
    policy = MockPolicy([np.zeros((1, 14))])
    environment = MockEnvironment()

    with pytest.raises(RuntimeError, match="--execute"):
        asyncio.run(run(args, policy=policy, environment=environment))

    assert not policy.connected
    assert not environment.reset_called
