"""Contract tests for the X-trainer SmolVLA training launchers."""

import os
import subprocess
from pathlib import Path

import draccus
import pytest

from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig

REPO_ROOT = Path(__file__).resolve().parents[2]
BASH_LAUNCHER = REPO_ROOT / "scripts" / "xtrainer" / "train_smolvla.sh"
TRAIN_CONFIG = REPO_ROOT / "configs" / "xtrainer" / "train_smolvla.yaml"


def test_bash_launcher_help():
    result = subprocess.run(["bash", str(BASH_LAUNCHER), "--help"], capture_output=True, text=True)

    assert result.returncode == 0
    assert "--dataset-root PATH" in result.stdout
    assert "--resume-checkpoint PATH" in result.stdout


def test_xtrainer_training_config_parses_with_lerobot_train():
    parser._config_path_args.clear()
    parser._config_yaml_overrides.clear()
    cleaned_config = parser.extract_path_fields_from_config(
        str(TRAIN_CONFIG), TrainPipelineConfig.__get_path_fields__()
    )
    try:
        config = draccus.parse(config_class=TrainPipelineConfig, config_path=cleaned_config, args=[])
    finally:
        parser._config_path_args.clear()
        parser._config_yaml_overrides.clear()
        if Path(cleaned_config) != TRAIN_CONFIG:
            Path(cleaned_config).unlink(missing_ok=True)

    assert config.dataset.format_version == "v2.1"
    assert config.xtrainer["schema_version"] == 1


def test_bash_launcher_rejects_nonexistent_dataset_root(tmp_path):
    missing_root = tmp_path / "missing"
    result = subprocess.run(
        ["bash", str(BASH_LAUNCHER), "--dataset-root", str(missing_root)],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "dataset root does not exist" in result.stderr


def test_bash_launcher_forwards_overrides_to_lerobot_train(tmp_path):
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    captured_args = tmp_path / "train_args.txt"
    fake_train = bin_dir / "lerobot-train"
    fake_train.write_text(
        '#!/usr/bin/env bash\nprintf "%s\\n" "$@" > "$CAPTURED_TRAIN_ARGS"\n', encoding="utf-8"
    )
    fake_train.chmod(0o755)
    environment = {
        **os.environ,
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "CAPTURED_TRAIN_ARGS": str(captured_args),
    }

    result = subprocess.run(
        [
            "bash",
            str(BASH_LAUNCHER),
            "--dataset-root",
            str(dataset_root),
            "--output-dir",
            "outputs/smoke",
            "--device",
            "cuda",
            "--batch-size",
            "1",
            "--steps",
            "2",
            "--skip-validation",
        ],
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.returncode == 0, result.stderr
    assert captured_args.read_text(encoding="utf-8").splitlines() == [
        f"--config_path={TRAIN_CONFIG}",
        f"--dataset.root={dataset_root}",
        "--output_dir=outputs/smoke",
        "--policy.device=cuda",
        "--batch_size=1",
        "--steps=2",
    ]


def test_bash_launcher_uses_checkpoint_config_when_resuming(tmp_path):
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    captured_args = tmp_path / "train_args.txt"
    fake_train = bin_dir / "lerobot-train"
    fake_train.write_text(
        '#!/usr/bin/env bash\nprintf "%s\\n" "$@" > "$CAPTURED_TRAIN_ARGS"\n', encoding="utf-8"
    )
    fake_train.chmod(0o755)
    environment = {
        **os.environ,
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "CAPTURED_TRAIN_ARGS": str(captured_args),
    }

    result = subprocess.run(
        [
            "bash",
            str(BASH_LAUNCHER),
            "--dataset-root",
            str(dataset_root),
            "--resume-checkpoint",
            "outputs/run/checkpoints/last/pretrained_model",
            "--skip-validation",
        ],
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.returncode == 0, result.stderr
    assert captured_args.read_text(encoding="utf-8").splitlines() == [
        "--resume=true",
        "--config_path=outputs/run/checkpoints/last/pretrained_model",
        f"--dataset.root={dataset_root}",
    ]
