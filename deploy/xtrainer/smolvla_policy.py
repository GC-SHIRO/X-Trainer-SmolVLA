"""SmolVLA policy adapter for the X-trainer WebSocket deployment server.

Loads a full SmolVLA checkpoint or a base model + LoRA adapter pair using the
standard LeRobot policy/processor factories, validates X-trainer observations
before they reach the model, and exposes the ``metadata()``/``reset()``/
``infer()`` contract expected by :class:`XTrainerWebSocketPolicyServer`.

The service is single-client, single-batch: only one ``infer()`` call is
handled at a time, matching the transport layer's one-request/one-response
semantics.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors
from lerobot.policies import make_pre_post_processors
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE
from lerobot.utils.import_utils import require_package

STATE_DIM = 14
ACTION_DIM = 14
logger = logging.getLogger(__name__)


class SmolVLAXTrainerPolicy:
    """Wraps a SmolVLA policy for X-trainer's 14-dim dual-arm state/action layout."""

    def __init__(
        self,
        *,
        checkpoint: str,
        lora_adapter: str | None = None,
        device: str = "cuda",
        actions_per_chunk: int = 50,
        camera_keys: dict[str, str] | None = None,
        state_key: str = OBS_STATE,
        action_key: str = ACTION,
        reset_pose: list[float] | None = None,
        warmup: bool = True,
        action_log_path: str | Path | None = None,
    ) -> None:
        self.device = device
        self.actions_per_chunk = actions_per_chunk
        self.camera_keys = camera_keys or {
            "top": f"{OBS_IMAGES}.top",
            "left_wrist": f"{OBS_IMAGES}.left_wrist",
            "right_wrist": f"{OBS_IMAGES}.right_wrist",
        }
        self.state_key = state_key
        self.action_key = action_key
        self.reset_pose = reset_pose
        self._action_log_path = Path(action_log_path) if action_log_path is not None else None
        self._action_log_file = None

        self.policy = self._load_policy(checkpoint, lora_adapter, device)
        self.preprocessor, self.postprocessor = self._load_processors(checkpoint, lora_adapter)

        if warmup:
            self._warmup()
        if self._action_log_path is not None:
            self._open_action_log()

    def _load_policy(self, checkpoint: str, lora_adapter: str | None, device: str) -> SmolVLAPolicy:
        if lora_adapter is None:
            policy = SmolVLAPolicy.from_pretrained(checkpoint)
        else:
            require_package("peft", extra="peft")
            from peft import PeftConfig, PeftModel

            # The adapter config records the base model it was trained on, so the base
            # model is loaded first and the adapter is applied on top of it.
            peft_config = PeftConfig.from_pretrained(lora_adapter)
            base_path = peft_config.base_model_name_or_path or checkpoint
            policy = SmolVLAPolicy.from_pretrained(base_path)
            policy = PeftModel.from_pretrained(policy, lora_adapter, config=peft_config, is_trainable=False)
        policy.to(device)
        policy.eval()
        return policy

    def _load_processors(self, checkpoint: str, lora_adapter: str | None):
        processor_source = lora_adapter or checkpoint
        try:
            return make_pre_post_processors(
                policy_cfg=self.policy.config,
                pretrained_path=processor_source,
            )
        except (FileNotFoundError, OSError):
            # No saved processor pipeline next to the checkpoint: build defaults from config.
            return make_smolvla_pre_post_processors(config=self.policy.config)

    def _warmup(self) -> None:
        state = np.zeros(STATE_DIM, dtype=np.float32)
        images = {name: np.zeros((480, 640, 3), dtype=np.uint8) for name in self.camera_keys}
        self.infer({"state": state, "images": images, "task": "warmup"})
        self.reset()

    def metadata(self) -> dict[str, Any]:
        info = {
            "model_type": "smolvla",
            "schema_version": 1,
            "action_dim": ACTION_DIM,
            "state_dim": STATE_DIM,
            "chunk_size": self.actions_per_chunk,
        }
        if self.reset_pose is not None:
            info["reset_pose"] = list(self.reset_pose)
        return info

    def reset(self) -> None:
        self.policy.reset()

    def close(self) -> None:
        """Close the optional action log file when the policy server stops."""
        if self._action_log_file is not None:
            self._action_log_file.close()
            self._action_log_file = None

    def infer(self, payload: dict[str, Any]) -> dict[str, Any]:
        self._validate_payload(payload)
        batch = self._build_batch(payload)

        with torch.inference_mode():
            transition = self.preprocessor(batch)
            actions = self.policy.predict_action_chunk(transition)
            actions = self.postprocessor(actions)

        actions_np = actions.squeeze(0).to(dtype=torch.float32).cpu().numpy()
        actions_np = actions_np[: self.actions_per_chunk]

        if not np.isfinite(actions_np).all():
            raise ValueError("policy produced non-finite actions")
        if actions_np.shape[-1] != ACTION_DIM:
            raise ValueError(f"policy produced action dim {actions_np.shape[-1]}, expected {ACTION_DIM}")

        self._record_actions(actions_np)
        return {self.action_key: actions_np.astype(np.float32)}

    def _open_action_log(self) -> None:
        assert self._action_log_path is not None
        self._action_log_path.parent.mkdir(parents=True, exist_ok=True)
        self._action_log_file = self._action_log_path.open("a", encoding="utf-8", buffering=1)
        logger.info("Action logging enabled: %s", self._action_log_path)

    def _record_actions(self, actions: np.ndarray) -> None:
        """Append one action chunk per line for every non-warmup policy response."""
        if self._action_log_file is None:
            return
        try:
            self._action_log_file.write(json.dumps(actions.tolist(), allow_nan=False) + "\n")
        except OSError:
            logger.exception("Could not write action log; disabling action logging")
            self.close()

    def _validate_payload(self, payload: dict[str, Any]) -> None:
        state = payload.get("state")
        if state is None:
            raise ValueError("payload is missing 'state'")
        state_arr = np.asarray(state)
        if state_arr.shape != (STATE_DIM,):
            raise ValueError(f"'state' must have shape ({STATE_DIM},), got {state_arr.shape}")
        if not np.isfinite(state_arr).all():
            raise ValueError("'state' contains non-finite values")

        images = payload.get("images")
        if not isinstance(images, dict):
            raise ValueError("payload is missing 'images' map")
        for name in self.camera_keys:
            if name not in images:
                raise ValueError(f"payload is missing image '{name}'")
            image = np.asarray(images[name])
            if image.dtype != np.uint8:
                raise ValueError(f"image '{name}' must be uint8, got {image.dtype}")
            if image.ndim != 3 or image.shape[2] != 3:
                raise ValueError(f"image '{name}' must have shape (H, W, 3), got {image.shape}")

        task = payload.get("task")
        if not isinstance(task, str) or not task:
            raise ValueError("payload is missing a non-empty string 'task'")

    def _build_batch(self, payload: dict[str, Any]) -> dict[str, Any]:
        state = np.asarray(payload["state"], dtype=np.float32)
        images = payload["images"]

        batch: dict[str, Any] = {self.state_key: torch.from_numpy(state)}
        for name, obs_key in self.camera_keys.items():
            # Preprocessor steps expect float32 CHW in [0, 1], matching what the
            # dataset loader yields at training time (raw frames arrive HWC uint8).
            image = torch.from_numpy(np.asarray(images[name], dtype=np.uint8))
            image = image.to(dtype=torch.float32) / 255.0
            batch[obs_key] = image.permute(2, 0, 1).contiguous()
        batch["task"] = payload["task"]
        return batch
