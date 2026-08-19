#!/usr/bin/env bash
# Train SmolVLA on a local X-trainer LeRobot Dataset v2.1 recording.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
TRAIN_CONFIG="${XTRAINER_TRAIN_CONFIG:-${REPO_ROOT}/configs/xtrainer/train_smolvla.yaml}"
TRAINING_DESCRIPTION="${XTRAINER_TRAINING_DESCRIPTION:-SmolVLA 全量微调}"
VALIDATOR="${SCRIPT_DIR}/validate_dataset_v21.py"

DATASET_ROOT=""
OUTPUT_DIR=""
DEVICE=""
BATCH_SIZE=""
STEPS=""
RESUME_CHECKPOINT=""
SKIP_VALIDATION=false

usage() {
    cat <<EOF
Usage: scripts/xtrainer/train_smolvla.sh --dataset-root PATH [options]

Run ${TRAINING_DESCRIPTION} with ${TRAIN_CONFIG#"${REPO_ROOT}/"} and the standard
lerobot-train loop. The v2.1 dataset is validated before training by default.

Required:
  --dataset-root PATH        Local LeRobot Dataset v2.1 directory.

Options:
  --output-dir PATH          Override the training output directory.
  --device DEVICE            Override policy.device (for example: cuda).
  --batch-size N             Override batch_size.
  --steps N                  Override steps.
  --resume-checkpoint PATH   Resume from a checkpoint train_config.json or
                             pretrained_model directory. Its saved config is used.
  --skip-validation          Do not run validate_dataset_v21.py before training.
  -h, --help                 Show this help message.
EOF
}

require_value() {
    if [[ $# -lt 2 || -z "$2" || "$2" == --* ]]; then
        echo "error: $1 requires a value" >&2
        usage >&2
        exit 2
    fi
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset-root)
            require_value "$1" "${2:-}"
            DATASET_ROOT="$2"
            shift 2
            ;;
        --output-dir)
            require_value "$1" "${2:-}"
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --device)
            require_value "$1" "${2:-}"
            DEVICE="$2"
            shift 2
            ;;
        --batch-size)
            require_value "$1" "${2:-}"
            BATCH_SIZE="$2"
            shift 2
            ;;
        --steps)
            require_value "$1" "${2:-}"
            STEPS="$2"
            shift 2
            ;;
        --resume-checkpoint)
            require_value "$1" "${2:-}"
            RESUME_CHECKPOINT="$2"
            shift 2
            ;;
        --skip-validation)
            SKIP_VALIDATION=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "error: unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ -z "$DATASET_ROOT" ]]; then
    echo "error: --dataset-root is required" >&2
    usage >&2
    exit 2
fi
if [[ ! -d "$DATASET_ROOT" ]]; then
    echo "error: dataset root does not exist or is not a directory: $DATASET_ROOT" >&2
    exit 2
fi
if [[ ! -f "$TRAIN_CONFIG" ]]; then
    echo "error: training config does not exist: $TRAIN_CONFIG" >&2
    exit 2
fi

if ! command -v python >/dev/null 2>&1; then
    echo "error: python was not found; activate the LeRobot environment first" >&2
    exit 127
fi
if ! command -v lerobot-train >/dev/null 2>&1; then
    echo "error: lerobot-train was not found; install/activate LeRobot first" >&2
    exit 127
fi

if [[ "$SKIP_VALIDATION" == false ]]; then
    python "$VALIDATOR" --root "$DATASET_ROOT"
fi

train_args=()
if [[ -n "$RESUME_CHECKPOINT" ]]; then
    train_args+=("--resume=true" "--config_path=$RESUME_CHECKPOINT")
else
    train_args+=("--config_path=$TRAIN_CONFIG")
fi
train_args+=("--dataset.root=$DATASET_ROOT")

if [[ -n "$OUTPUT_DIR" ]]; then
    train_args+=("--output_dir=$OUTPUT_DIR")
fi
if [[ -n "$DEVICE" ]]; then
    train_args+=("--policy.device=$DEVICE")
fi
if [[ -n "$BATCH_SIZE" ]]; then
    train_args+=("--batch_size=$BATCH_SIZE")
fi
if [[ -n "$STEPS" ]]; then
    train_args+=("--steps=$STEPS")
fi

exec lerobot-train "${train_args[@]}"
