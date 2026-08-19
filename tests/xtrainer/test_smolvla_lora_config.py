"""Configuration and launcher contracts for X-trainer SmolVLA LoRA training."""

import os
import subprocess
from pathlib import Path

import draccus
import pytest
import yaml

from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig

REPO_ROOT = Path(__file__).resolve().parents[2]
FULL_CONFIG = REPO_ROOT / "configs" / "xtrainer" / "train_smolvla.yaml"
LORA_CONFIG = REPO_ROOT / "configs" / "xtrainer" / "train_smolvla_lora.yaml"
LORA_LAUNCHER = REPO_ROOT / "scripts" / "xtrainer" / "train_smolvla_lora.sh"


def _parse_train_config(config_path: Path) -> tuple[TrainPipelineConfig, str | None]:
    parser._config_path_args.clear()
    parser._config_yaml_overrides.clear()
    cleaned_config = parser.extract_path_fields_from_config(
        str(config_path), TrainPipelineConfig.__get_path_fields__()
    )
    try:
        config = draccus.parse(config_class=TrainPipelineConfig, config_path=cleaned_config, args=[])
        policy_path = parser.get_path_arg("policy")
    finally:
        parser._config_path_args.clear()
        parser._config_yaml_overrides.clear()
        if Path(cleaned_config) != config_path:
            Path(cleaned_config).unlink(missing_ok=True)
    return config, policy_path


def test_lora_config_uses_smolvla_base_and_expected_peft_settings():
    config, policy_path = _parse_train_config(LORA_CONFIG)
    full_config = yaml.safe_load(FULL_CONFIG.read_text(encoding="utf-8"))

    assert policy_path == "lerobot/smolvla_base"
    assert config.dataset.format_version == "v2.1"
    assert config.peft is not None
    assert config.peft.method_type == "LORA"
    assert config.peft.r == 64
    assert config.peft.lora_alpha == 64
    assert config.peft.target_modules is None
    assert config.ema.enable is False
    assert config.parallelism.dp_replicate == 1
    assert config.parallelism.dp_shard == 1
    assert str(config.output_dir) != full_config["output_dir"]


def test_lora_config_rejects_sharded_parallelism():
    config, _ = _parse_train_config(LORA_CONFIG)
    config.parallelism.dp_shard = 2

    with pytest.raises(ValueError, match="PEFT is not supported under sharded training"):
        config._validate_distributed()


def test_lora_launcher_forwards_to_standard_training_loop(tmp_path):
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
            str(LORA_LAUNCHER),
            "--dataset-root",
            str(dataset_root),
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
        f"--config_path={LORA_CONFIG}",
        f"--dataset.root={dataset_root}",
        "--batch_size=1",
        "--steps=2",
    ]


def test_lora_launcher_help_identifies_the_lora_config():
    result = subprocess.run(["bash", str(LORA_LAUNCHER), "--help"], capture_output=True, text=True)

    assert result.returncode == 0
    assert "SmolVLA LoRA" in result.stdout
    assert "configs/xtrainer/train_smolvla_lora.yaml" in result.stdout
