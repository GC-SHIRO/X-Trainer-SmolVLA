"""Integration coverage for routing X-trainer v2.1 datasets through the factory."""

from pathlib import Path

import pytest
import yaml

pytest_plugins = ["tests.datasets.v21.conftest"]

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")
pytest.importorskip("pyarrow", reason="pyarrow is required (install lerobot[dataset])")
pytest.importorskip("av", reason="av is required (install lerobot[dataset])")
torch = pytest.importorskip("torch", reason="torch is required (install lerobot[dataset])")

from lerobot.configs.default import DatasetConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.datasets.v21 import LeRobotDatasetV21
from lerobot.policies.smolvla import SmolVLAConfig


def _make_config(root):
    return TrainPipelineConfig(
        dataset=DatasetConfig(repo_id="test/xtrainer", root=root, format_version="v2.1"),
        policy=SmolVLAConfig(chunk_size=3, n_action_steps=3),
    )


def test_factory_routes_v21_to_read_only_adapter(v21_xtrainer_dataset, monkeypatch):
    root, _ = v21_xtrainer_dataset

    def unexpected_v3_loader(*args, **kwargs):
        raise AssertionError("v2.1 data must not be passed to the v3 metadata loader")

    def fake_decode(path, timestamps, tolerance_s, backend, return_uint8):
        return torch.zeros((len(timestamps), 3, 2, 2), dtype=torch.uint8)

    monkeypatch.setattr("lerobot.datasets.factory.LeRobotDatasetMetadata", unexpected_v3_loader)
    monkeypatch.setattr("lerobot.datasets.v21.dataset.decode_video_frames", fake_decode)

    dataset = make_dataset(_make_config(root))

    assert isinstance(dataset, LeRobotDatasetV21)
    assert dataset.delta_timestamps["action"] == [0.0, 0.1, 0.2]
    assert dataset[2]["action_is_pad"].tolist() == [False, True, True]


@pytest.mark.parametrize("filename", ["train_smolvla.yaml", "train_smolvla_lora.yaml"])
def test_xtrainer_training_config_selects_v21(filename):
    config_path = Path(__file__).parents[2] / "configs" / "xtrainer" / filename
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    dataset_config = DatasetConfig(**config["dataset"])

    assert dataset_config.format_version == "v2.1"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"format_version": "v2.1", "streaming": True}, "does not support streaming"),
        ({"format_version": "v2.1", "repo_type": "bucket"}, "only supports repo_type='dataset'"),
        ({"format_version": "v2.0"}, "format_version"),
    ],
)
def test_v21_unsupported_dataset_configurations_fail_early(kwargs, message):
    with pytest.raises(ValueError, match=message):
        DatasetConfig(repo_id="test/xtrainer", **kwargs)


def test_factory_rejects_v21_multi_dataset_configuration():
    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id="test/xtrainer", format_version="v2.1"),
        policy=SmolVLAConfig(chunk_size=3, n_action_steps=3),
    )
    cfg.dataset.repo_id = ["test/first", "test/second"]

    with pytest.raises(NotImplementedError, match="MultiLeRobotDataset"):
        make_dataset(cfg)
