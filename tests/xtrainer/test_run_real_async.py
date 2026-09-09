import asyncio
import json
import time

import numpy as np
import pytest

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
            observation_hz=10,
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
                observation_hz=1000,
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
    assert controls[1]["queued_action"] == [10.0] * 14
    assert queued[0]["state"] == [0.0] * 14
    assert queued[1]["last_applied_action"] == controls[0]["applied_action"]
    assert queued[1]["capture_started_at_utc"] <= queued[1]["observation_ready_at_utc"]
    assert results[0]["returned_actions"] == [[0.0] * 14] * 3
    assert results[1]["returned_actions"] == [[10.0] * 14] * 3
    assert controls[1]["action_index"] == 0
    joint_indices = np.r_[0:6, 7:13]
    expected_first_joint = 10.0 * (1.0 / 6.0) ** 2 * (3.0 - 2.0 / 6.0)
    np.testing.assert_allclose(environment.actions[1][joint_indices], expected_first_joint)
    np.testing.assert_allclose(environment.actions[1][[6, 13]], 10.0)


def test_async_control_loop_yields_transport_after_observation_overrun():
    class DeferredPolicy(AsyncPolicy):
        def submit_observation(self, _payload, *, observation_timestep, must_go=False):
            if self.next_id == 0:
                return super().submit_observation(
                    _payload,
                    observation_timestep=observation_timestep,
                    must_go=must_go,
                )

            observation_id = self.next_id
            self.next_id += 1

            async def deliver_after_control_yields():
                await asyncio.sleep(0)
                self.events.put_nowait(
                    {
                        "status": "actions",
                        "observation_id": observation_id,
                        "observation_timestep": observation_timestep,
                        "payload": {"action": np.full((2, 14), 2.0, dtype=np.float32)},
                    }
                )

            asyncio.create_task(deliver_after_control_yields())
            return observation_id

    class SlowObservationEnvironment(Environment):
        def __init__(self):
            super().__init__()
            self.observation_count = 0

        def get_observation(self):
            self.observation_count += 1
            if self.observation_count > 1:
                time.sleep(0.005)
            return super().get_observation()

    policy = DeferredPolicy()
    environment = SlowObservationEnvironment()

    asyncio.run(
        run_async_control_loop(
            policy,
            environment,
            action_horizon=2,
            control_hz=1000,
            max_steps=3,
            prefetch_threshold=0.5,
            observation_hz=1000,
            request_timeout_s=1,
            max_delta_per_step=0,
            chunk_blend_steps=0,
        )
    )

    np.testing.assert_allclose(environment.actions[0], 1)
    np.testing.assert_allclose(environment.actions[2], 2)


@pytest.mark.parametrize("cadence", [2, 3, 4, 5])
def test_repeated_replans_track_moving_targets_and_reverse_without_gripper_delay(cadence):
    environment = Environment()
    targets = np.zeros((100, 14))
    ramp = np.r_[np.arange(40), 40 - np.arange(60)] * 0.01
    targets[:, 2] = -ramp
    targets[:, 12] = ramp
    targets[:, 13] = (np.arange(100) // 8) % 2

    class ReplanningPolicy(AsyncPolicy):
        def __init__(self):
            super().__init__()
            self.delivered = set()

        def submit_observation(self, _payload, **kwargs):
            self.events.put_nowait({
                "status": "actions", "observation_timestep": 0,
                "payload": {"action": targets.copy()},
            })
            return 0

        def get_observation_event_nowait(self):
            step = len(environment.actions)
            if step and step % cadence == 0 and step not in self.delivered:
                self.delivered.add(step)
                return {
                    "status": "actions", "observation_timestep": step,
                    "payload": {"action": targets[step:].copy()},
                }
            return None

    asyncio.run(run_async_control_loop(
        ReplanningPolicy(), environment, action_horizon=100, control_hz=30,
        max_steps=80, prefetch_threshold=0, observation_hz=30,
        request_timeout_s=1, max_delta_per_step=0, chunk_blend_steps=6,
        monotonic_fn=lambda: 0.0, sleep_fn=lambda _: asyncio.sleep(0),
    ))
    actual = np.array(environment.actions)
    # 对照已撤掉的固定起点方法；同一轨迹频繁换块不能不断压低运动量。
    old = []
    anchor = None
    for step in range(80):
        if step and step % cadence == 0:
            anchor = old[-1]
        progress = min((step % cadence + 1) / 6, 1)
        weight = progress ** 2 * (3 - 2 * progress)
        old.append(targets[step, 12] if anchor is None else anchor + weight * (targets[step, 12] - anchor))
    new_error = np.abs(actual[10:40, 12] - targets[10:40, 12])
    old_error = np.abs(np.array(old)[10:40] - targets[10:40, 12])
    assert new_error.mean() < old_error.mean() * 0.8
    assert new_error.max() < 0.04
    assert np.all(np.diff(actual[:40, 12]) >= 0)
    assert np.all(np.diff(actual[46:, 12]) <= 0)
    np.testing.assert_allclose(actual[:, 2], -actual[:, 12])
    np.testing.assert_array_equal(actual[:, 13], targets[:80, 13])


def test_late_chunk_trace_preserves_full_return_and_execution_index(tmp_path):
    environment = Environment()

    class LatePolicy(AsyncPolicy):
        delivered = False

        def get_observation_event_nowait(self):
            if len(environment.actions) == 4 and not self.delivered:
                self.delivered = True
                return {
                    "status": "actions", "observation_timestep": 1,
                    "observation_id": 99,
                    "client_received_at_utc": "2026-09-09T13:00:00+00:00",
                    "payload": {"action": np.full((8, 14), 2.0)},
                }
            return None

    path = tmp_path / "late.jsonl"
    log = ControlActionLog(path)
    try:
        asyncio.run(run_async_control_loop(
            LatePolicy(), environment, action_horizon=5, control_hz=30,
            max_steps=9, prefetch_threshold=0, observation_hz=30,
            request_timeout_s=1, max_delta_per_step=0, control_log=log,
            monotonic_fn=lambda: 0.0, sleep_fn=lambda _: asyncio.sleep(0),
        ))
    finally:
        log.close()
    records = [json.loads(line) for line in path.read_text().splitlines()]
    result = next(r for r in records if r.get("observation_id") == 99)
    assert result["returned_action_count"] == 8
    assert len(result["returned_actions"]) == 8
    assert result["retained_action_count"] == 5
    assert result["client_received_at_utc"] == "2026-09-09T13:00:00+00:00"
    controls = [r for r in records if r["event"] == "control_step"]
    assert controls[4]["action_index"] == 3
    assert controls[5]["action_index"] == 4
    # 队列耗尽时保持上一条下发值，不继续衰减偏差而意外移动。
    assert all(r["action_index"] is None for r in controls[6:])
    np.testing.assert_allclose(environment.actions[6:], [environment.actions[5]] * 3)
