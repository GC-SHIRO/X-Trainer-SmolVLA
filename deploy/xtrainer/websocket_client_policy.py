"""Client for the X-trainer WebSocket policy transport."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

from .msgpack_numpy import PROTOCOL_VERSION, ProtocolError, dumps, loads


class XTrainerWebSocketPolicyClient:
    """Support legacy RPC calls and latest-observation streaming."""

    def __init__(self, base_url: str, *, max_payload_bytes: int | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.max_payload_bytes = max_payload_bytes
        self._session = None
        self._ws = None
        self.metadata: dict[str, Any] | None = None
        self._async_mode = False
        self._session_id: str | None = None
        self._request_id = 0
        self._next_observation_id = 0
        self._receiver_task: asyncio.Task[None] | None = None
        self._sender_task: asyncio.Task[None] | None = None
        self._send_lock = asyncio.Lock()
        self._rpc_waiters: dict[int, asyncio.Future[dict[str, Any]]] = {}
        self._ack_waiters: dict[int, asyncio.Future[dict[str, Any]]] = {}
        # Observation results must never stop the sole WebSocket receiver.  A
        # bounded queue made QueueFull terminate this task and, with it, ACK
        # handling.  The sender already serializes acknowledged observations,
        # and the control loop throttles production, so this queue stays small.
        self._events: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._pending_observation: dict[str, Any] | None = None
        self._pending_event = asyncio.Event()
        self._closed = False

    async def __aenter__(self) -> "XTrainerWebSocketPolicyClient":
        await self.connect()
        return self

    async def __aexit__(self, _exc_type, _exc, _tb) -> None:
        await self.close()

    async def connect(self) -> dict[str, Any]:
        from aiohttp import ClientSession, WSMsgType

        self._closed = False
        self._session = ClientSession()
        self._ws = await self._session.ws_connect(
            f"{self.base_url}/ws", max_msg_size=self.max_payload_bytes or 0
        )
        handshake = await self._ws.receive()
        if handshake.type != WSMsgType.BINARY:
            raise ProtocolError("metadata handshake must be a binary frame")
        response = loads(
            handshake.data,
            max_payload_bytes=self.max_payload_bytes or 64 * 1024 * 1024,
        )
        self._raise_for_error(response)
        self.metadata = response["metadata"]
        return self.metadata

    async def start_async(self, observation_similarity_epsilon: float | None = None) -> dict[str, Any]:
        if self._ws is None:
            raise RuntimeError("client is not connected")
        capabilities = (self.metadata or {}).get("capabilities") or {}
        if capabilities.get("async_observation_v1") is not True:
            raise ProtocolError("server does not support async_observation_v1")
        request_id = self._allocate_request_id()
        await self._send(
            {
                "protocol_version": PROTOCOL_VERSION,
                "type": "start_async",
                "request_id": request_id,
                "options": {"observation_similarity_epsilon": observation_similarity_epsilon},
            }
        )
        response = await self._receive_one()
        if response.get("type") != "async_ready" or response.get("request_id") != request_id:
            raise ProtocolError("unexpected start_async response")
        self._session_id = response.get("session_id")
        if not isinstance(self._session_id, str):
            raise ProtocolError("async_ready is missing session_id")
        self._async_mode = True
        self._receiver_task = asyncio.create_task(self._receiver_loop())
        self._sender_task = asyncio.create_task(self._sender_loop())
        return response

    async def close(self) -> None:
        self._closed = True
        self._pending_event.set()
        if self._ws is not None:
            await self._ws.close()
        tasks = [task for task in (self._sender_task, self._receiver_task) if task is not None]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._fail_waiters(RuntimeError("policy client closed"))
        if self._session is not None:
            await self._session.close()
        self._ws = None
        self._session = None
        self._sender_task = None
        self._receiver_task = None
        self._async_mode = False
        self._session_id = None

    async def get_healthz(self) -> dict[str, Any]:
        from aiohttp import ClientSession

        async with ClientSession() as session:
            async with session.get(f"{self.base_url}/healthz") as response:
                response.raise_for_status()
                return await response.json()

    async def get_metadata(self) -> dict[str, Any]:
        from aiohttp import ClientSession

        async with ClientSession() as session:
            async with session.get(f"{self.base_url}/metadata") as response:
                response.raise_for_status()
                return await response.json()

    async def request(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self._async_mode:
            return await self._async_request(payload)
        await self._send({"protocol_version": PROTOCOL_VERSION, **payload})
        return await self._receive_one()

    async def reset(self) -> dict[str, Any]:
        response = await self.request(
            {"type": "reset", **({"session_id": self._session_id} if self._async_mode else {})}
        )
        if self._async_mode:
            session_id = response.get("session_id")
            if not isinstance(session_id, str):
                raise ProtocolError("reset response is missing session_id")
            self._session_id = session_id
            self._next_observation_id = 0
            self._pending_observation = None
        return response

    async def infer(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self._async_mode:
            raise RuntimeError("infer() is unavailable in latest-observation mode")
        response = await self.request({"type": "infer", "payload": payload})
        result = response.get("payload")
        if not isinstance(result, dict):
            raise ProtocolError("infer response payload must be a map")
        return result

    def submit_observation(
        self,
        payload: dict[str, Any],
        *,
        observation_timestep: int,
        must_go: bool = False,
    ) -> int:
        if not self._async_mode or self._session_id is None:
            raise RuntimeError("latest-observation mode is not active")
        observation_id = self._next_observation_id
        self._next_observation_id += 1
        previous = self._pending_observation
        self._pending_observation = {
            "protocol_version": PROTOCOL_VERSION,
            "type": "observation",
            "session_id": self._session_id,
            "observation_id": observation_id,
            "observation_timestep": observation_timestep,
            "must_go": must_go or bool(previous and previous.get("must_go")),
            "payload": payload,
        }
        self._pending_event.set()
        return observation_id

    async def next_observation_event(self, timeout_s: float | None = None) -> dict[str, Any]:
        if timeout_s is None:
            return await self._events.get()
        return await asyncio.wait_for(self._events.get(), timeout=timeout_s)

    def get_observation_event_nowait(self) -> dict[str, Any] | None:
        try:
            return self._events.get_nowait()
        except asyncio.QueueEmpty:
            return None

    async def _sender_loop(self) -> None:
        while not self._closed:
            await self._pending_event.wait()
            if self._closed:
                return
            message, self._pending_observation = self._pending_observation, None
            self._pending_event.clear()
            if message is None:
                continue
            observation_id = int(message["observation_id"])
            waiter = asyncio.get_running_loop().create_future()
            self._ack_waiters[observation_id] = waiter
            await self._send(message)
            try:
                await waiter
            finally:
                self._ack_waiters.pop(observation_id, None)
            if self._pending_observation is not None:
                self._pending_event.set()

    async def _receiver_loop(self) -> None:
        try:
            while not self._closed:
                message = await self._receive_one()
                message_type = message.get("type")
                if message_type == "observation_ack":
                    waiter = self._ack_waiters.get(message.get("observation_id"))
                    if waiter is not None and not waiter.done():
                        waiter.set_result(message)
                elif message_type == "observation_result":
                    # 接收时间独立于控制循环消费时间，避免把队列等待误算成推理耗时。
                    message["client_received_at_utc"] = datetime.now(timezone.utc).isoformat()
                    self._events.put_nowait(message)
                else:
                    waiter = self._rpc_waiters.get(message.get("request_id"))
                    if waiter is not None and not waiter.done():
                        waiter.set_result(message)
                    else:
                        raise ProtocolError(f"unexpected async message type: {message_type}")
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._fail_waiters(exc)
            if not self._closed:
                try:
                    self._events.put_nowait(
                        {"type": "observation_result", "status": "error", "reason": str(exc)}
                    )
                except asyncio.QueueFull:
                    pass

    async def _async_request(self, payload: dict[str, Any]) -> dict[str, Any]:
        request_id = self._allocate_request_id()
        waiter = asyncio.get_running_loop().create_future()
        self._rpc_waiters[request_id] = waiter
        await self._send({"protocol_version": PROTOCOL_VERSION, **payload, "request_id": request_id})
        try:
            return await waiter
        finally:
            self._rpc_waiters.pop(request_id, None)

    async def _send(self, payload: dict[str, Any]) -> None:
        if self._ws is None:
            raise RuntimeError("client is not connected")
        async with self._send_lock:
            await self._ws.send_bytes(
                dumps(payload, max_payload_bytes=self.max_payload_bytes or 64 * 1024 * 1024)
            )

    async def _receive_one(self) -> dict[str, Any]:
        from aiohttp import WSMsgType

        if self._ws is None:
            raise RuntimeError("client is not connected")
        response = await self._ws.receive()
        if response.type != WSMsgType.BINARY:
            raise ProtocolError("server response must be a binary frame")
        result = loads(
            response.data,
            max_payload_bytes=self.max_payload_bytes or 64 * 1024 * 1024,
        )
        self._raise_for_error(result)
        return result

    def _allocate_request_id(self) -> int:
        request_id = self._request_id
        self._request_id += 1
        return request_id

    def _fail_waiters(self, exc: BaseException) -> None:
        for waiter in (*self._rpc_waiters.values(), *self._ack_waiters.values()):
            if not waiter.done():
                waiter.set_exception(exc)

    @staticmethod
    def _raise_for_error(response: dict[str, Any]) -> None:
        if response.get("ok") is False:
            error = response.get("error") or {}
            raise ProtocolError(f"{error.get('code', 'error')}: {error.get('message', 'unknown error')}")
