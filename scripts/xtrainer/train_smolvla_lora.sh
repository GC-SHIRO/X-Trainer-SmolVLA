#!/usr/bin/env bash
# Train SmolVLA LoRA adapters on a local X-trainer LeRobot Dataset v2.1 recording.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

export XTRAINER_TRAIN_CONFIG="${REPO_ROOT}/configs/xtrainer/train_smolvla_lora.yaml"
export XTRAINER_TRAINING_DESCRIPTION="SmolVLA LoRA 微调"

exec "${SCRIPT_DIR}/train_smolvla.sh" "$@"
