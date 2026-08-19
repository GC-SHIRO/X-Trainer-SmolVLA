"""CPU-only tests for the SmolVLA X-trainer policy adapter.

These tests avoid loading a real SmolVLA checkpoint: SmolVLAPolicy.from_pretrained
and the processor factory are monkeypatched with lightweight fakes so the tests
run on CPU without network access or GPU.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from deploy.xtrainer import smolvla_policy as smolvla_policy_module
from deploy.xtrainer.smolvla_policy import ACTION_DIM, STATE_DIM, SmolVLAXTrainerPolicy

CAMERA_KEYS = {
    "top": "observation.images.top",
    "left_wrist": "observation.images.left_wrist",
    "right_wrist": "observation.images.right_wrist",
}


class FakeConfig:
    pass


class FakePolicy:
    """Stands in for SmolVLAPolicy: records calls and echoes a deterministic chunk."""

    def __init__(self):
        self.config = FakeConfig()
        self.reset_calls = 0
        self.chunk_len = 5
        self.last_batch = None

    def to(self, _device):
        return self

    def eval(self):
        return self

    def reset(self):
        self.reset_calls += 1

    def predict_action_chunk(self, batch):
        self.last_batch = batch
        return torch.zeros(1, self.chunk_len, ACTION_DIM)


class IdentityPipeline:
    """Stands in for a PolicyProcessorPipeline: passes data through unchanged."""

    def __call__(self, data):
        return data


def _make_policy(monkeypatch, *, fake_policy=None, load_calls=None):
    fake_policy = fake_policy or FakePolicy()
    load_calls = load_calls if load_calls is not None else []

    def fake_from_pretrained(path):
        load_calls.append(path)
        return fake_policy

    monkeypatch.setattr(smolvla_policy_module.SmolVLAPolicy, "from_pretrained", fake_from_pretrained)
    monkeypatch.setattr(
        smolvla_policy_module,
        "make_smolvla_pre_post_processors",
        lambda config: (IdentityPipeline(), IdentityPipeline()),
    )

    def fail_make_pre_post_processors(**_kwargs):
        raise FileNotFoundError("no saved processors for this fake checkpoint")

    monkeypatch.setattr(smolvla_policy_module, "make_pre_post_processors", fail_make_pre_post_processors)

    policy = SmolVLAXTrainerPolicy(
        checkpoint="fake/checkpoint",
        device="cpu",
        camera_keys=CAMERA_KEYS,
        warmup=False,
    )
    return policy, fake_policy


def _valid_payload():
    return {
        "state": np.zeros(STATE_DIM, dtype=np.float32),
        "images": {name: np.zeros((480, 640, 3), dtype=np.uint8) for name in CAMERA_KEYS},
        "task": "pick up the object",
    }


def test_infer_returns_expected_action_shape(monkeypatch):
    policy, fake_policy = _make_policy(monkeypatch)

    result = policy.infer(_valid_payload())

    assert set(result.keys()) == {"action"}
    assert result["action"].shape == (fake_policy.chunk_len, ACTION_DIM)
    assert result["action"].dtype == np.float32
    assert np.isfinite(result["action"]).all()


def test_infer_truncates_to_actions_per_chunk(monkeypatch):
    policy, _ = _make_policy(monkeypatch)
    policy.actions_per_chunk = 3

    result = policy.infer(_valid_payload())

    assert result["action"].shape == (3, ACTION_DIM)


def test_infer_rejects_wrong_state_shape_before_calling_model(monkeypatch):
    policy, fake_policy = _make_policy(monkeypatch)
    payload = _valid_payload()
    payload["state"] = np.zeros(10, dtype=np.float32)

    with pytest.raises(ValueError, match="state"):
        policy.infer(payload)
    assert fake_policy.last_batch is None


def test_infer_rejects_non_finite_state_before_calling_model(monkeypatch):
    policy, fake_policy = _make_policy(monkeypatch)
    payload = _valid_payload()
    payload["state"][0] = np.nan

    with pytest.raises(ValueError, match="non-finite"):
        policy.infer(payload)
    assert fake_policy.last_batch is None


def test_infer_rejects_missing_camera_before_calling_model(monkeypatch):
    policy, fake_policy = _make_policy(monkeypatch)
    payload = _valid_payload()
    del payload["images"]["left_wrist"]

    with pytest.raises(ValueError, match="left_wrist"):
        policy.infer(payload)
    assert fake_policy.last_batch is None


def test_infer_rejects_wrong_image_dtype_before_calling_model(monkeypatch):
    policy, fake_policy = _make_policy(monkeypatch)
    payload = _valid_payload()
    payload["images"]["top"] = np.zeros((480, 640, 3), dtype=np.float32)

    with pytest.raises(ValueError, match="uint8"):
        policy.infer(payload)
    assert fake_policy.last_batch is None


def test_infer_rejects_wrong_image_shape_before_calling_model(monkeypatch):
    policy, fake_policy = _make_policy(monkeypatch)
    payload = _valid_payload()
    payload["images"]["top"] = np.zeros((480, 640, 4), dtype=np.uint8)

    with pytest.raises(ValueError, match="shape"):
        policy.infer(payload)
    assert fake_policy.last_batch is None


def test_infer_rejects_missing_task_before_calling_model(monkeypatch):
    policy, fake_policy = _make_policy(monkeypatch)
    payload = _valid_payload()
    payload["task"] = ""

    with pytest.raises(ValueError, match="task"):
        policy.infer(payload)
    assert fake_policy.last_batch is None


def test_infer_rejects_non_finite_model_output(monkeypatch):
    fake_policy = FakePolicy()
    fake_policy.predict_action_chunk = lambda batch: torch.full((1, 5, ACTION_DIM), float("nan"))
    policy, _ = _make_policy(monkeypatch, fake_policy=fake_policy)

    with pytest.raises(ValueError, match="non-finite"):
        policy.infer(_valid_payload())


def test_reset_delegates_to_underlying_policy(monkeypatch):
    policy, fake_policy = _make_policy(monkeypatch)

    policy.reset()

    assert fake_policy.reset_calls == 1


def test_metadata_reports_schema_and_dims(monkeypatch):
    policy, _ = _make_policy(monkeypatch)

    metadata = policy.metadata()

    assert metadata["model_type"] == "smolvla"
    assert metadata["action_dim"] == ACTION_DIM
    assert metadata["state_dim"] == STATE_DIM


def test_lora_adapter_loads_base_model_before_adapter(monkeypatch):
    load_calls: list[str] = []
    fake_policy = FakePolicy()

    def fake_from_pretrained(path):
        load_calls.append(f"base:{path}")
        return fake_policy

    monkeypatch.setattr(smolvla_policy_module.SmolVLAPolicy, "from_pretrained", fake_from_pretrained)
    monkeypatch.setattr(
        smolvla_policy_module,
        "make_smolvla_pre_post_processors",
        lambda config: (IdentityPipeline(), IdentityPipeline()),
    )
    monkeypatch.setattr(
        smolvla_policy_module,
        "make_pre_post_processors",
        lambda **_kwargs: (_ for _ in ()).throw(FileNotFoundError()),
    )

    class FakePeftConfig:
        base_model_name_or_path = "fake/base-model"

    class FakePeftModule:
        PeftConfig = type("PeftConfig", (), {"from_pretrained": staticmethod(lambda _path: FakePeftConfig())})

        class PeftModel:
            @staticmethod
            def from_pretrained(policy, adapter_path, *, config, is_trainable):
                load_calls.append(f"adapter:{adapter_path}")
                return policy

    monkeypatch.setattr(smolvla_policy_module, "require_package", lambda *_a, **_k: None)
    monkeypatch.setitem(__import__("sys").modules, "peft", FakePeftModule)

    SmolVLAXTrainerPolicy(
        checkpoint="fake/checkpoint",
        lora_adapter="fake/adapter",
        device="cpu",
        camera_keys=CAMERA_KEYS,
        warmup=False,
    )

    assert load_calls == ["base:fake/base-model", "adapter:fake/adapter"]
