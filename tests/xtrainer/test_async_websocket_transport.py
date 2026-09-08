import asyncio
import threading
import time

import numpy as np
import pytest

from deploy.xtrainer.websocket_client_policy import XTrainerWebSocketPolicyClient
from deploy.xtrainer.websocket_policy_server import XTrainerWebSocketPolicyServer

pytest.importorskip("aiohttp")


class BlockingPolicy:
    def __init__(self):
        self.entered = threading.Event()
        self.release = threading.Event()
        self.calls = []
        self.active = 0
        self.max_active = 0

    def metadata(self):
        return {"model_type": "mock", "schema_version": 1, "action_dim": 14}

    def reset(self):
        pass

    def infer(self, payload):
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        self.calls.append(int(payload["state"][0]))
        try:
            if len(self.calls) == 1:
                self.entered.set()
                assert self.release.wait(timeout=5)
            return {"action": np.full((2, 14), self.calls[-1], dtype=np.float32)}
        finally:
            self.active -= 1


def _payload(value):
    return {
        "state": np.full(14, value, dtype=np.float32),
        "images": {},
        "task": "pick",
    }


async def _wait_until(predicate, timeout=2):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise TimeoutError("condition was not reached")
        await asyncio.sleep(0.005)


def test_latest_mode_replaces_only_pending_observation_and_keeps_health_responsive():
    async def scenario():
        policy = BlockingPolicy()
        server = XTrainerWebSocketPolicyServer(policy, port=0)
        await server.start()
        client = XTrainerWebSocketPolicyClient(f"http://127.0.0.1:{server.port}")
        try:
            metadata = await client.connect()
            assert metadata["capabilities"]["async_observation_v1"] is True
            await client.start_async(0)
            await client.reset()

            client.submit_observation(_payload(1), observation_timestep=0, must_go=True)
            assert await asyncio.to_thread(policy.entered.wait, 2)
            assert (await client.get_healthz())["ok"] is True

            client.submit_observation(_payload(2), observation_timestep=1)
            await _wait_until(lambda: client._pending_observation is None and not client._ack_waiters)
            client.submit_observation(_payload(3), observation_timestep=2)
            await _wait_until(lambda: client._pending_observation is None and not client._ack_waiters)
            policy.release.set()

            events = [await client.next_observation_event(2) for _ in range(3)]
            assert {event["status"] for event in events} == {"actions", "superseded"}
            assert sum(event["status"] == "actions" for event in events) == 2
            assert policy.calls == [1, 3]
            assert policy.max_active == 1
        finally:
            policy.release.set()
            await client.close()
            await server.stop()

    asyncio.run(scenario())


def test_similarity_filter_skips_close_joint_state_but_must_go_bypasses_it():
    async def scenario():
        policy = BlockingPolicy()
        policy.release.set()
        server = XTrainerWebSocketPolicyServer(policy, port=0)
        await server.start()
        client = XTrainerWebSocketPolicyClient(f"http://127.0.0.1:{server.port}")
        try:
            await client.connect()
            await client.start_async(0.01)
            await client.reset()

            base = _payload(0)
            close = _payload(0.001)
            close["state"][[6, 13]] = 0
            client.submit_observation(base, observation_timestep=0, must_go=True)
            assert (await client.next_observation_event(2))["status"] == "actions"
            client.submit_observation(close, observation_timestep=1)
            assert (await client.next_observation_event(2))["status"] == "similar"
            client.submit_observation(close, observation_timestep=2, must_go=True)
            assert (await client.next_observation_event(2))["status"] == "actions"
            assert policy.calls == [0, 0]
        finally:
            await client.close()
            await server.stop()

    asyncio.run(scenario())
