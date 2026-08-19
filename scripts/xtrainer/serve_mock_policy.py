#!/usr/bin/env python
"""Serve a lightweight hold-current policy over the X-trainer WebSocket transport."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from deploy.xtrainer.websocket_policy_server import XTrainerWebSocketPolicyServer

ACTION_DIM = 14
CAMERA_NAMES = ("top", "left_wrist", "right_wrist")


class HoldCurrentMockPolicy:
    """Repeat the observed state as an action chunk without loading a model."""

    def __init__(self, *, chunk_size: int = 50) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        self.chunk_size = chunk_size

    def metadata(self) -> dict[str, Any]:
        return {
            "model_type": "smolvla",
            "schema_version": 1,
            "action_dim": ACTION_DIM,
            "state_dim": ACTION_DIM,
            "chunk_size": self.chunk_size,
            "mock_policy": True,
        }

    def reset(self) -> None:
        return None

    def infer(self, payload: dict[str, Any]) -> dict[str, np.ndarray]:
        state = np.asarray(payload.get("state"), dtype=np.float32)
        if state.shape != (ACTION_DIM,):
            raise ValueError(f"'state' must have shape ({ACTION_DIM},), got {state.shape}")
        if not np.all(np.isfinite(state)):
            raise ValueError("'state' contains non-finite values")

        images = payload.get("images")
        if not isinstance(images, dict):
            raise ValueError("payload is missing 'images' map")
        for name in CAMERA_NAMES:
            image = np.asarray(images.get(name))
            if image.dtype != np.uint8:
                raise ValueError(f"image '{name}' must be uint8, got {image.dtype}")
            if image.ndim != 3 or image.shape[2] != 3:
                raise ValueError(f"image '{name}' must have shape (H, W, 3), got {image.shape}")

        task = payload.get("task")
        if not isinstance(task, str) or not task:
            raise ValueError("payload is missing a non-empty string 'task'")

        actions = np.repeat(state[None, :], self.chunk_size, axis=0)
        return {"action": actions}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve a hold-current policy for X-trainer tests")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--max-payload-mb", type=int, default=64)
    return parser.parse_args(argv)


async def _serve(args: argparse.Namespace) -> None:
    if args.port < 0 or args.port > 65535:
        raise ValueError("port must be in [0, 65535]")
    if args.max_payload_mb <= 0:
        raise ValueError("max_payload_mb must be positive")

    policy = HoldCurrentMockPolicy(chunk_size=args.chunk_size)
    server = XTrainerWebSocketPolicyServer(
        policy,
        host=args.host,
        port=args.port,
        max_payload_bytes=args.max_payload_mb * 1024 * 1024,
    )
    await server.start()
    logging.info("Serving hold-current mock policy on %s:%d", args.host, server.port)
    try:
        await asyncio.Event().wait()
    finally:
        await server.stop()


def main() -> None:
    try:
        asyncio.run(_serve(parse_args()))
    except KeyboardInterrupt:
        logging.info("Interrupted by user")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
