"""WebSocket policy transport for X-trainer deployment."""

from __future__ import annotations

import asyncio
import functools
import inspect
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from .async_observation import PendingObservation, observations_similar, validate_similarity_epsilon
from .msgpack_numpy import ProtocolError, dumps, loads, protocol_metadata


class XTrainerWebSocketPolicyServer:
    """Serve legacy RPC inference and latest-observation streaming over WebSocket."""

    def __init__(
        self,
        policy: Any,
        *,
        host: str = "127.0.0.1",
        port: int = 8765,
        max_payload_bytes: int | None = None,
    ) -> None:
        self.policy = policy
        self.host = host
        self.port = port
        self.max_payload_bytes = max_payload_bytes
        self._runner = None
        self._site = None
        self._policy_lock = asyncio.Lock()
        self._policy_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="xtrainer-policy")
        self._async_owner: Any | None = None

    def metadata(self) -> dict[str, Any]:
        policy_metadata = self.policy.metadata() if hasattr(self.policy, "metadata") else {}
        return protocol_metadata(
            {
                "policy": policy_metadata,
                "capabilities": {"async_observation_v1": True},
            }
        )

    async def start(self) -> None:
        from aiohttp import web

        app = web.Application()
        app.router.add_get("/healthz", self._handle_healthz)
        app.router.add_get("/metadata", self._handle_metadata)
        app.router.add_get("/ws", self._handle_websocket)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        self._site = web.TCPSite(self._runner, self.host, self.port)
        await self._site.start()
        sockets = getattr(self._site, "_server", None)
        if self.port == 0 and sockets is not None and sockets.sockets:
            self.port = sockets.sockets[0].getsockname()[1]

    async def stop(self) -> None:
        if self._runner is not None:
            await self._runner.cleanup()
        await self._call_policy("close", required=False)
        self._policy_executor.shutdown(wait=True)
        self._runner = None
        self._site = None

    async def _call_policy(self, name: str, *args: Any, required: bool = True) -> Any:
        method = getattr(self.policy, name, None)
        if not callable(method):
            if required:
                raise ProtocolError(f"policy does not implement {name}()")
            return None
        async with self._policy_lock:
            if inspect.iscoroutinefunction(method):
                return await method(*args)
            result = await asyncio.get_running_loop().run_in_executor(
                self._policy_executor, functools.partial(method, *args)
            )
            if inspect.isawaitable(result):
                return await result
            return result

    async def _handle_healthz(self, _request):
        from aiohttp import web

        return web.json_response({"ok": True, "metadata": self.metadata()})

    async def _handle_metadata(self, _request):
        from aiohttp import web

        return web.json_response(self.metadata())

    async def _handle_websocket(self, request):
        from aiohttp import WSMsgType, web

        ws = web.WebSocketResponse(max_msg_size=self.max_payload_bytes or 0)
        await ws.prepare(request)
        outgoing: asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=32)

        async def sender() -> None:
            while True:
                frame = await outgoing.get()
                if frame is None:
                    return
                await ws.send_bytes(frame)

        async def send(payload: dict[str, Any]) -> None:
            await outgoing.put(self._ok(payload))

        sender_task = asyncio.create_task(sender())
        await send({"type": "metadata", "metadata": self.metadata()})
        session: _LatestSession | None = None
        try:
            async for message in ws:
                if message.type != WSMsgType.BINARY:
                    if message.type == WSMsgType.ERROR:
                        break
                    await outgoing.put(self._error("invalid_frame", "expected a binary MessagePack frame"))
                    continue
                try:
                    decoded = loads(
                        message.data,
                        max_payload_bytes=self.max_payload_bytes or 64 * 1024 * 1024,
                    )
                    if decoded.get("protocol_version") not in (
                        None,
                        self.metadata()["protocol_version"],
                    ):
                        raise ProtocolError("unsupported protocol_version")
                    request_type = decoded.get("type")
                    if request_type == "start_async":
                        if session is not None:
                            raise ProtocolError("async mode is already active")
                        if self._async_owner not in (None, ws):
                            raise ProtocolError("another async control session is active")
                        epsilon = validate_similarity_epsilon(
                            (decoded.get("options") or {}).get("observation_similarity_epsilon")
                        )
                        self._async_owner = ws
                        session = _LatestSession(self, send, epsilon)
                        await session.start()
                        await send(
                            {
                                "type": "async_ready",
                                "request_id": decoded.get("request_id"),
                                "session_id": session.session_id,
                                "options": {"observation_similarity_epsilon": epsilon},
                            }
                        )
                    elif session is not None and request_type == "observation":
                        await session.submit(decoded)
                    elif session is not None and request_type == "reset":
                        await session.reset(decoded.get("request_id"), decoded.get("session_id"))
                    elif session is not None:
                        raise ProtocolError("async session accepts only observation and reset")
                    else:
                        await send(await self._dispatch(decoded, websocket=ws))
                except ProtocolError as exc:
                    await outgoing.put(self._error("invalid_payload", str(exc)))
                except Exception as exc:
                    await outgoing.put(self._error("server_error", str(exc)))
        finally:
            if session is not None:
                await session.close()
            if self._async_owner is ws:
                self._async_owner = None
            await outgoing.put(None)
            await sender_task
        return ws

    async def _dispatch(self, request: dict[str, Any], *, websocket: Any | None = None) -> dict[str, Any]:
        request_type = request.get("type")
        if request_type in {"reset", "infer"} and self._async_owner not in (None, websocket):
            raise ProtocolError("another async control session owns the policy")
        if request_type == "metadata":
            return {"type": "metadata", "metadata": self.metadata()}
        if request_type == "reset":
            await self._call_policy("reset", required=False)
            return {
                "type": "reset",
                "ok": True,
                "request_id": request.get("request_id"),
            }
        if request_type == "infer":
            payload = request.get("payload")
            if not isinstance(payload, dict):
                raise ProtocolError("infer request requires a payload map")
            result = await self._call_policy("infer", payload)
            if not isinstance(result, dict):
                raise ProtocolError("policy infer(payload) must return a map")
            return {
                "type": "infer",
                "payload": result,
                "request_id": request.get("request_id"),
            }
        raise ProtocolError("request type must be one of metadata, reset, infer, start_async")

    def _ok(self, payload: dict[str, Any]) -> bytes:
        return dumps(
            {"ok": True, **payload},
            max_payload_bytes=self.max_payload_bytes or 64 * 1024 * 1024,
        )

    def _error(self, code: str, message: str) -> bytes:
        return dumps(
            {"ok": False, "error": {"code": code, "message": message}},
            max_payload_bytes=self.max_payload_bytes or 64 * 1024 * 1024,
        )


class _LatestSession:
    def __init__(
        self,
        server: XTrainerWebSocketPolicyServer,
        send: Any,
        epsilon: float | None,
    ) -> None:
        self.server = server
        self.send = send
        self.epsilon = epsilon
        self.session_id = uuid.uuid4().hex
        self.pending: PendingObservation | None = None
        self.pending_event = asyncio.Event()
        self.idle = asyncio.Event()
        self.idle.set()
        self.closed = False
        self.generation = 0
        self.highest_id = -1
        self.highest_timestep = -1
        self.last_successful_payload: dict[str, Any] | None = None
        self.worker_task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        self.worker_task = asyncio.create_task(self._worker())

    async def submit(self, request: dict[str, Any]) -> None:
        if request.get("session_id") != self.session_id:
            raise ProtocolError("observation belongs to an inactive session")
        try:
            observation_id = int(request["observation_id"])
            timestep = int(request["observation_timestep"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ProtocolError("observation requires integer id and timestep") from exc
        if observation_id <= self.highest_id or timestep <= self.highest_timestep:
            await self._terminal(request, "duplicate", "id or timestep is not increasing")
            return
        payload = request.get("payload")
        if not isinstance(payload, dict):
            raise ProtocolError("observation requires a payload map")
        validator = getattr(self.server.policy, "validate_payload", None)
        if callable(validator):
            validator(payload)
        item = PendingObservation(
            observation_id,
            timestep,
            bool(request.get("must_go")),
            payload,
            self.session_id,
        )
        self.highest_id = observation_id
        self.highest_timestep = timestep
        await self.send(
            {
                "type": "observation_ack",
                "session_id": self.session_id,
                "observation_id": observation_id,
                "observation_timestep": timestep,
                "status": "accepted",
            }
        )
        replaced = self.pending
        self.pending = item
        self.pending_event.set()
        if replaced is not None:
            await self._result(replaced, "superseded", reason="replaced by a newer observation")

    async def reset(self, request_id: Any, session_id: Any) -> None:
        if session_id != self.session_id:
            raise ProtocolError("reset belongs to an inactive session")
        self.generation += 1
        pending, self.pending = self.pending, None
        self.pending_event.clear()
        if pending is not None:
            await self._result(pending, "superseded", reason="session reset")
        await self.idle.wait()
        await self.server._call_policy("reset", required=False)
        self.session_id = uuid.uuid4().hex
        self.highest_id = -1
        self.highest_timestep = -1
        self.last_successful_payload = None
        await self.send(
            {
                "type": "reset",
                "request_id": request_id,
                "session_id": self.session_id,
                "ok": True,
            }
        )

    async def close(self) -> None:
        self.closed = True
        self.generation += 1
        self.pending = None
        self.pending_event.set()
        if self.worker_task is not None:
            await self.worker_task

    async def _worker(self) -> None:
        while True:
            await self.pending_event.wait()
            if self.closed:
                return
            item, self.pending = self.pending, None
            self.pending_event.clear()
            if item is None:
                continue
            generation = self.generation
            if (
                not item.must_go
                and self.last_successful_payload is not None
                and observations_similar(item.payload, self.last_successful_payload, self.epsilon)
            ):
                await self._result(item, "similar", reason="joint state is below epsilon")
                continue
            self.idle.clear()
            started = time.perf_counter()
            try:
                result = await self.server._call_policy("infer", item.payload)
                elapsed_ms = (time.perf_counter() - started) * 1000
                if generation != self.generation or self.closed:
                    continue
                if not isinstance(result, dict):
                    raise ProtocolError("policy infer(payload) must return a map")
                self.last_successful_payload = item.payload
                await self._result(
                    item,
                    "actions",
                    payload=result,
                    server_timing={"policy_call_ms": elapsed_ms},
                )
            except Exception as exc:
                if generation == self.generation and not self.closed:
                    await self._result(item, "error", reason=str(exc))
            finally:
                self.idle.set()
            if self.pending is not None:
                self.pending_event.set()

    async def _terminal(self, request: dict[str, Any], status: str, reason: str) -> None:
        await self.send(
            {
                "type": "observation_result",
                "session_id": self.session_id,
                "observation_id": request.get("observation_id"),
                "observation_timestep": request.get("observation_timestep"),
                "status": status,
                "reason": reason,
            }
        )

    async def _result(self, item: PendingObservation, status: str, **fields: Any) -> None:
        await self.send(
            {
                "type": "observation_result",
                "session_id": item.session_id,
                "observation_id": item.observation_id,
                "observation_timestep": item.observation_timestep,
                "status": status,
                **fields,
            }
        )


async def serve_forever(policy: Any, *, host: str = "127.0.0.1", port: int = 8765) -> None:
    server = XTrainerWebSocketPolicyServer(policy, host=host, port=port)
    await server.start()
    try:
        await asyncio.Event().wait()
    finally:
        await server.stop()
