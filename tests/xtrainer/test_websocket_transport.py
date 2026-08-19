import asyncio

import numpy as np
import pytest

from deploy.xtrainer import (
    ProtocolError as PublicProtocolError,
    XTrainerWebSocketPolicyClient as PublicXTrainerWebSocketPolicyClient,
    XTrainerWebSocketPolicyServer as PublicXTrainerWebSocketPolicyServer,
    dumps as public_dumps,
    loads as public_loads,
)
from deploy.xtrainer.msgpack_numpy import ProtocolError, dumps, loads
from deploy.xtrainer.websocket_client_policy import XTrainerWebSocketPolicyClient
from deploy.xtrainer.websocket_policy_server import XTrainerWebSocketPolicyServer

pytest.importorskip("aiohttp")


class EchoPolicy:
    def __init__(self):
        self.reset_count = 0

    def metadata(self):
        return {
            "model_type": "mock",
            "schema_version": 1,
            "action_dim": 14,
            "chunk_size": 2,
        }

    def reset(self):
        self.reset_count += 1

    def infer(self, payload):
        return {"action": payload["state"].astype(np.float32) + 1.0}


def test_package_exports_transport_contract():
    assert PublicProtocolError is ProtocolError
    assert PublicXTrainerWebSocketPolicyClient is XTrainerWebSocketPolicyClient
    assert PublicXTrainerWebSocketPolicyServer is XTrainerWebSocketPolicyServer
    assert public_dumps is dumps
    assert public_loads is loads


def test_numpy_msgpack_roundtrip_preserves_dtype_and_shape():
    payload = {
        "state": np.arange(14, dtype=np.float32),
        "image": np.zeros((2, 3, 4), dtype=np.uint8),
    }

    decoded = loads(dumps(payload))

    assert decoded["state"].dtype == np.float32
    assert decoded["state"].shape == (14,)
    assert decoded["image"].dtype == np.uint8
    assert decoded["image"].shape == (2, 3, 4)


@pytest.mark.parametrize(
    "array",
    [
        np.array([object()], dtype=object),
        np.array([1 + 1j], dtype=np.complex64),
        np.zeros((1, 1, 1, 1, 1, 1), dtype=np.float32),
    ],
)
def test_numpy_msgpack_rejects_unsupported_arrays(array):
    with pytest.raises(ProtocolError):
        dumps({"array": array})


def test_msgpack_rejects_oversized_payload():
    with pytest.raises(ProtocolError, match="payload is too large"):
        dumps({"array": np.zeros(128, dtype=np.float32)}, max_payload_bytes=16)


def test_msgpack_rejects_plain_python_objects():
    with pytest.raises(ProtocolError, match="unsupported objects"):
        dumps({"object": object()})


def test_websocket_health_metadata_and_infer_roundtrip():
    asyncio.run(_websocket_health_metadata_and_infer_roundtrip())


async def _websocket_health_metadata_and_infer_roundtrip():
    server = XTrainerWebSocketPolicyServer(EchoPolicy(), port=0)
    await server.start()
    try:
        client = XTrainerWebSocketPolicyClient(f"http://127.0.0.1:{server.port}")
        metadata = await client.connect()
        try:
            health = await client.get_healthz()
            remote_metadata = await client.get_metadata()
            result = await client.infer({"state": np.arange(14, dtype=np.float32)})
            await client.reset()
        finally:
            await client.close()
    finally:
        await server.stop()

    assert health["ok"] is True
    assert metadata["protocol"] == "xtrainer.websocket.msgpack"
    assert metadata["trusted_lan_only"] is True
    assert remote_metadata["policy"]["action_dim"] == 14
    np.testing.assert_array_equal(result["action"], np.arange(14, dtype=np.float32) + 1.0)
    assert server.policy.reset_count == 1


def test_invalid_payload_returns_controlled_error_and_server_survives():
    asyncio.run(_invalid_payload_returns_controlled_error_and_server_survives())


async def _invalid_payload_returns_controlled_error_and_server_survives():
    server = XTrainerWebSocketPolicyServer(EchoPolicy(), port=0)
    await server.start()
    try:
        client = XTrainerWebSocketPolicyClient(f"http://127.0.0.1:{server.port}")
        await client.connect()
        try:
            with pytest.raises(ProtocolError, match="invalid_payload"):
                await client.request({"type": "infer"})
            result = await client.infer({"state": np.zeros(14, dtype=np.float32)})
        finally:
            await client.close()
    finally:
        await server.stop()

    np.testing.assert_array_equal(result["action"], np.ones(14, dtype=np.float32))
