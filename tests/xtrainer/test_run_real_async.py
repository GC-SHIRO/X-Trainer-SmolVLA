import asyncio
import json

import numpy as np

from deploy.xtrainer.real.environment import (
    LEFT_WRIST_IMAGE_KEY,
    RIGHT_WRIST_IMAGE_KEY,
    STATE_KEY,
    TASK_KEY,
    TOP_IMAGE_KEY,
)
from scripts.xtrainer.run_real import ControlActionLog, run_async_control_loop


class AsyncPolicy:
    def __init__(self):
        self.events = asyncio.Queue()
        self.next_id = 0

    def submit_observation(self, _payload, *, observation_timestep, must_go=False):
        observation_id = self.next_id
        self.next_id += 1
        self.events.put_nowait(
            {
                "status": "actions",
                "observation_id": observation_id,
                "observation_timestep": observation_timestep,
                "payload": {"action": np.ones((3, 14), dtype=np.float32)},
            }
        )
        return observation_id

    async def next_observation_event(self, timeout_s=None):
        return await asyncio.wait_for(self.events.get(), timeout_s)

    def get_observation_event_nowait(self):
        try:
            return self.events.get_nowait()
        except asyncio.QueueEmpty:
            return None


class Environment:
    def __init__(self):
        self.actions = []

    def get_observation(self):
        image = np.zeros((4, 5, 3), dtype=np.uint8)
        return {
            STATE_KEY: np.zeros(14, dtype=np.float32),
            TOP_IMAGE_KEY: image.copy(),
            LEFT_WRIST_IMAGE_KEY: image.copy(),
            RIGHT_WRIST_IMAGE_KEY: image.copy(),
            TASK_KEY: "pick",
        }

    def apply_action(self, action, *, pace=True):
        assert pace is False
        result = np.asarray(action).copy()
        self.actions.append(result)
        return result


def test_async_control_loop_executes_initial_timestep_aligned_chunk():
    policy = AsyncPolicy()
    environment = Environment()

    asyncio.run(
        run_async_control_loop(
            policy,
            environment,
            action_horizon=3,
            control_hz=1000,
            max_steps=3,
            prefetch_threshold=0,
            request_timeout_s=1,
            max_delta_per_step=0,
        )
    )

    assert len(environment.actions) == 3
    np.testing.assert_allclose(environment.actions, 1)


def test_async_control_loop_smooths_source_change_and_logs_diagnostics(tmp_path):
    class ChangingPolicy(AsyncPolicy):
        def submit_observation(self, _payload, *, observation_timestep, must_go=False):
            observation_id = self.next_id
            self.next_id += 1
            value = 0.0 if observation_timestep == 0 else 10.0
            self.events.put_nowait(
                {
                    "status": "actions",
                    "observation_id": observation_id,
                    "observation_timestep": observation_timestep,
                    "payload": {"action": np.full((3, 14), value, dtype=np.float32)},
                    "server_timing": {"policy_call_ms": 1.0},
                }
            )
            return observation_id

    async def exercise(log_path):
        policy = ChangingPolicy()
        environment = Environment()
        control_log = ControlActionLog(log_path)
        try:
            await run_async_control_loop(
                policy,
                environment,
                action_horizon=3,
                control_hz=1000,
                max_steps=2,
                prefetch_threshold=2 / 3,
                request_timeout_s=1,
                max_delta_per_step=0,
                chunk_blend_steps=6,
                control_log=control_log,
                monotonic_fn=lambda: 0.0,
                sleep_fn=asyncio.sleep,
            )
        finally:
            control_log.close()
        return environment

    log_path = tmp_path / "async-control.jsonl"
    environment = asyncio.run(exercise(log_path))
    records = [json.loads(line) for line in log_path.read_text().splitlines()]
    queued = [record for record in records if record["event"] == "async_observation_queued"]
    results = [record for record in records if record["event"] == "async_observation_result"]
    controls = [record for record in records if record["event"] == "control_step"]

    assert queued[0]["observation_capture_ms"] == 0.0
    assert results[-1]["overlap_action_count"] == 2
    assert controls[1]["source_changed"] is True
    assert controls[1]["blend_step"] == 1
    assert controls[1]["raw_action"] == [10.0] * 14
    assert controls[1]["queued_action"] == [7.0] * 14
    joint_indices = np.r_[0:6, 7:13]
    expected_first_joint = 7.0 * (1.0 / 6.0) ** 2 * (3.0 - 2.0 / 6.0)
    np.testing.assert_allclose(environment.actions[1][joint_indices], expected_first_joint)
    np.testing.assert_allclose(environment.actions[1][[6, 13]], 7.0)
