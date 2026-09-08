import asyncio
import time

import numpy as np

from deploy.xtrainer.real.environment import (
    LEFT_WRIST_IMAGE_KEY,
    RIGHT_WRIST_IMAGE_KEY,
    STATE_KEY,
    TASK_KEY,
    TOP_IMAGE_KEY,
)
from scripts.xtrainer.run_real import run_async_control_loop


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
        )
    )

    np.testing.assert_allclose(environment.actions[0], 1)
    np.testing.assert_allclose(environment.actions[2], 2)
