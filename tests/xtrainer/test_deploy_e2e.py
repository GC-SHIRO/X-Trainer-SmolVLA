import asyncio

import numpy as np
import pytest

from deploy.xtrainer.real.environment import (
    LEFT_WRIST_IMAGE_KEY,
    RIGHT_WRIST_IMAGE_KEY,
    STATE_KEY,
    TASK_KEY,
    TOP_IMAGE_KEY,
)
from deploy.xtrainer.websocket_client_policy import XTrainerWebSocketPolicyClient
from deploy.xtrainer.websocket_policy_server import XTrainerWebSocketPolicyServer
from scripts.xtrainer import run_real as run_real_module
from scripts.xtrainer.run_real import parse_args as parse_run_args, run
from scripts.xtrainer.serve_mock_policy import HoldCurrentMockPolicy

pytest.importorskip("aiohttp")


def _state() -> np.ndarray:
    state = np.array(
        [-0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.25, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.75],
        dtype=np.float32,
    )
    return state


def _observation(state: np.ndarray) -> dict:
    return {
        STATE_KEY: state.copy(),
        TOP_IMAGE_KEY: np.full((4, 5, 3), 11, dtype=np.uint8),
        LEFT_WRIST_IMAGE_KEY: np.full((4, 5, 3), 22, dtype=np.uint8),
        RIGHT_WRIST_IMAGE_KEY: np.full((4, 5, 3), 33, dtype=np.uint8),
        TASK_KEY: "hold the current pose",
    }


class MockHardwareEnvironment:
    def __init__(self):
        self.state = _state()
        self.actions = []
        self.events = []

    def reset(self):
        self.events.append("reset")
        return self.get_observation()

    def enable_arms(self):
        self.events.append("enable_arms")

    def smooth_reset(self, reset_pose):
        self.events.append("smooth_reset")
        self.state = np.asarray(reset_pose, dtype=np.float32).copy()

    def get_observation(self):
        return _observation(self.state)

    def apply_action(self, action, *, pace=True):
        assert pace is False
        applied = np.asarray(action, dtype=np.float32).copy()
        self.events.append("apply_action")
        self.actions.append(applied)
        self.state = applied
        return applied

    def close(self):
        self.events.append("close")


class RecordingHoldCurrentPolicy(HoldCurrentMockPolicy):
    def __init__(self, *, chunk_size: int):
        super().__init__(chunk_size=chunk_size)
        self.payloads = []
        self.action_shapes = []
        self.reset_count = 0

    def reset(self):
        self.reset_count += 1

    def infer(self, payload):
        self.payloads.append(
            {
                "state": payload["state"].copy(),
                "images": {name: image.copy() for name, image in payload["images"].items()},
                "task": payload["task"],
            }
        )
        result = super().infer(payload)
        self.action_shapes.append(result["action"].shape)
        return result


class MetadataOverrideServer(XTrainerWebSocketPolicyServer):
    def __init__(self, *args, transport_overrides=None, policy_overrides=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.transport_overrides = transport_overrides or {}
        self.policy_overrides = policy_overrides or {}

    def metadata(self):
        metadata = super().metadata()
        metadata.update(self.transport_overrides)
        metadata["policy"].update(self.policy_overrides)
        return metadata


def _run_args(port: int, *, max_steps: int = 4):
    return parse_run_args(
        [
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--execute",
            "--action-horizon",
            "4",
            "--max-steps",
            str(max_steps),
            "--control-hz",
            "1000",
            "--prefetch-threshold",
            "0",
        ]
    )


def test_hold_current_policy_repeats_all_14_state_dimensions():
    policy = HoldCurrentMockPolicy(chunk_size=3)
    observation = _observation(_state())
    payload = {
        "state": observation[STATE_KEY],
        "images": {
            "top": observation[TOP_IMAGE_KEY],
            "left_wrist": observation[LEFT_WRIST_IMAGE_KEY],
            "right_wrist": observation[RIGHT_WRIST_IMAGE_KEY],
        },
        "task": observation[TASK_KEY],
    }

    result = policy.infer(payload)

    assert result["action"].shape == (3, 14)
    np.testing.assert_array_equal(result["action"], np.repeat(_state()[None, :], 3, axis=0))
    assert policy.metadata()["mock_policy"] is True


def test_real_websocket_roundtrip_drives_bounded_mock_hardware_loop(monkeypatch):
    async def exercise():
        policy = RecordingHoldCurrentPolicy(chunk_size=4)
        server = XTrainerWebSocketPolicyServer(policy, port=0)
        environment = MockHardwareEnvironment()
        await server.start()
        client = XTrainerWebSocketPolicyClient(f"http://127.0.0.1:{server.port}")
        try:
            await run(_run_args(server.port), policy=client, environment=environment)
        finally:
            await server.stop()
        return policy, client, server, environment

    def fail_build_environment(_args):
        raise AssertionError("E2E test must not construct real hardware")

    monkeypatch.setattr(run_real_module, "build_environment", fail_build_environment)
    policy, client, server, environment = asyncio.run(exercise())

    assert policy.reset_count == 1
    assert policy.action_shapes == [(4, 14)]
    assert len(policy.payloads) == 1
    np.testing.assert_array_equal(policy.payloads[0]["state"], _state())
    assert policy.payloads[0]["task"] == "hold the current pose"
    assert np.all(policy.payloads[0]["images"]["top"] == 11)
    assert np.all(policy.payloads[0]["images"]["left_wrist"] == 22)
    assert np.all(policy.payloads[0]["images"]["right_wrist"] == 33)
    assert all(image.dtype == np.uint8 for image in policy.payloads[0]["images"].values())

    assert len(environment.actions) == 4
    for action in environment.actions:
        np.testing.assert_array_equal(action, _state())
    assert environment.events == [
        "reset",
        "enable_arms",
        "apply_action",
        "apply_action",
        "apply_action",
        "apply_action",
        "close",
    ]
    assert client._session is None
    assert client._ws is None
    assert server._runner is None


@pytest.mark.parametrize(
    ("transport_overrides", "policy_overrides", "error_match"),
    [
        ({"protocol": "wrong.protocol"}, {}, "transport metadata protocol"),
        ({"protocol_version": 99}, {}, "transport metadata protocol_version"),
        ({"schema_version": 99}, {}, "transport metadata schema_version"),
        ({}, {"schema_version": 99}, "policy metadata schema_version"),
        ({}, {"action_dim": 13}, "policy metadata action_dim"),
    ],
)
def test_incompatible_metadata_stops_before_hardware_action(
    transport_overrides,
    policy_overrides,
    error_match,
):
    async def exercise():
        policy = HoldCurrentMockPolicy(chunk_size=4)
        server = MetadataOverrideServer(
            policy,
            port=0,
            transport_overrides=transport_overrides,
            policy_overrides=policy_overrides,
        )
        environment = MockHardwareEnvironment()
        await server.start()
        client = XTrainerWebSocketPolicyClient(f"http://127.0.0.1:{server.port}")
        try:
            with pytest.raises(RuntimeError, match=error_match):
                await run(_run_args(server.port, max_steps=1), policy=client, environment=environment)
        finally:
            await server.stop()
        return client, environment

    client, environment = asyncio.run(exercise())

    assert environment.events == ["close"]
    assert environment.actions == []
    assert client._session is None
    assert client._ws is None
