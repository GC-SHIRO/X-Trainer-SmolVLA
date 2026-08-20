"""WebSocket policy transport for X-trainer deployment.

The first deployment version is intentionally LAN-only: no authentication and
no TLS are implemented here. Put it behind a trusted network boundary.
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable
from typing import Any

from .msgpack_numpy import ProtocolError, dumps, loads, protocol_metadata

PolicyInfer = Callable[[dict[str, Any]], dict[str, Any] | Awaitable[dict[str, Any]]]


class XTrainerWebSocketPolicyServer:
    """Serve a policy-like object over HTTP metadata and binary WebSocket frames."""

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

    def metadata(self) -> dict[str, Any]:
        policy_metadata = {}
        if hasattr(self.policy, "metadata"):
            policy_metadata = self.policy.metadata()
        return protocol_metadata({"policy": policy_metadata})

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
        close = getattr(self.policy, "close", None)
        if callable(close):
            result = close()
            if inspect.isawaitable(result):
                await result
        if self._runner is not None:
            await self._runner.cleanup()
        self._runner = None
        self._site = None

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
        await ws.send_bytes(self._ok({"type": "metadata", "metadata": self.metadata()}))

        async for message in ws:
            if message.type == WSMsgType.BINARY:
                response = await self._handle_binary_request(message.data)
                await ws.send_bytes(response)
            elif message.type == WSMsgType.ERROR:
                break
            else:
                await ws.send_bytes(self._error("invalid_frame", "expected a binary MessagePack frame"))
        return ws

    async def _handle_binary_request(self, data: bytes) -> bytes:
        try:
            request = loads(data, max_payload_bytes=self.max_payload_bytes or 64 * 1024 * 1024)
            response = await self._dispatch(request)
            return self._ok(response)
        except ProtocolError as exc:
            return self._error("invalid_payload", str(exc))
        except Exception as exc:  # Keep malformed requests from taking down the server.
            return self._error("server_error", str(exc))

    async def _dispatch(self, request: dict[str, Any]) -> dict[str, Any]:
        request_type = request.get("type")
        if request.get("protocol_version") not in (None, self.metadata()["protocol_version"]):
            raise ProtocolError("unsupported protocol_version")
        if request_type == "metadata":
            return {"type": "metadata", "metadata": self.metadata()}
        if request_type == "reset":
            reset = getattr(self.policy, "reset", None)
            if callable(reset):
                result = reset()
                if inspect.isawaitable(result):
                    await result
            return {"type": "reset", "ok": True}
        if request_type == "infer":
            payload = request.get("payload")
            if not isinstance(payload, dict):
                raise ProtocolError("infer request requires a payload map")
            infer = getattr(self.policy, "infer", None)
            if not callable(infer):
                raise ProtocolError("policy does not implement infer(payload)")
            result = infer(payload)
            if inspect.isawaitable(result):
                result = await result
            if not isinstance(result, dict):
                raise ProtocolError("policy infer(payload) must return a map")
            return {"type": "infer", "payload": result}
        raise ProtocolError("request type must be one of metadata, reset, infer")

    def _ok(self, payload: dict[str, Any]) -> bytes:
        return dumps({"ok": True, **payload}, max_payload_bytes=self.max_payload_bytes or 64 * 1024 * 1024)

    def _error(self, code: str, message: str) -> bytes:
        return dumps(
            {"ok": False, "error": {"code": code, "message": message}},
            max_payload_bytes=self.max_payload_bytes or 64 * 1024 * 1024,
        )


async def serve_forever(policy: Any, *, host: str = "127.0.0.1", port: int = 8765) -> None:
    server = XTrainerWebSocketPolicyServer(policy, host=host, port=port)
    await server.start()
    try:
        await asyncio.Event().wait()
    finally:
        await server.stop()
