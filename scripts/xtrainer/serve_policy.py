#!/usr/bin/env python
"""Serve a SmolVLA checkpoint over the X-trainer WebSocket transport.

Reads the deployment contract from configs/xtrainer/deploy.yaml (or an
override) and starts XTrainerWebSocketPolicyServer with a SmolVLAXTrainerPolicy.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from deploy.xtrainer.smolvla_policy import SmolVLAXTrainerPolicy
from deploy.xtrainer.websocket_policy_server import XTrainerWebSocketPolicyServer

DEFAULT_CONFIG = REPO_ROOT / "configs" / "xtrainer" / "deploy.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve a SmolVLA checkpoint for X-trainer")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to deploy.yaml")
    parser.add_argument("--checkpoint", default=None, help="Override policy.checkpoint")
    parser.add_argument("--lora-adapter", default=None, help="Override policy.lora_adapter")
    parser.add_argument("--device", default=None, help="Override policy.device")
    parser.add_argument("--host", default=None, help="Override network.host")
    parser.add_argument("--port", type=int, default=None, help="Override network.port")
    parser.add_argument("--no-warmup", action="store_true", help="Skip the startup warmup inference")
    return parser.parse_args()


def load_deploy_config(path: str) -> dict:
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def main() -> None:
    args = parse_args()
    config = load_deploy_config(args.config)

    policy_cfg = config["policy"]
    network_cfg = config["network"]
    xtrainer_cfg = config["xtrainer"]

    checkpoint = args.checkpoint or policy_cfg["checkpoint"]
    lora_adapter = args.lora_adapter or policy_cfg.get("lora_adapter")
    device = args.device or policy_cfg.get("device", "cuda")
    host = args.host or network_cfg.get("host", "0.0.0.0")
    port = args.port if args.port is not None else network_cfg.get("port", 8000)

    observation_keys = xtrainer_cfg["observation_keys"]
    camera_keys = observation_keys["images"]

    policy = SmolVLAXTrainerPolicy(
        checkpoint=checkpoint,
        lora_adapter=lora_adapter,
        device=device,
        actions_per_chunk=policy_cfg.get("actions_per_chunk", xtrainer_cfg.get("chunk_size", 50)),
        camera_keys=camera_keys,
        state_key=observation_keys.get("state", "observation.state"),
        action_key=xtrainer_cfg.get("action_key", "action"),
        warmup=not args.no_warmup,
    )

    server = XTrainerWebSocketPolicyServer(
        policy,
        host=host,
        port=port,
        max_payload_bytes=int(network_cfg.get("max_payload_mb", 64)) * 1024 * 1024,
    )

    logging.info("Serving SmolVLA policy on %s:%d (checkpoint=%s)", host, port, checkpoint)
    asyncio.run(_serve_forever(server))


async def _serve_forever(server: XTrainerWebSocketPolicyServer) -> None:
    await server.start()
    try:
        await asyncio.Event().wait()
    finally:
        await server.stop()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
