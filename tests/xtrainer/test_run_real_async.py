import asyncio

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
            request_timeout_s=1,
            max_delta_per_step=0,
        )
    )

    assert len(environment.actions) == 3
    np.testing.assert_allclose(environment.actions, 1)
