#!/usr/bin/env bash
set -Eeuo pipefail

# Complete X-trainer SmolVLA environment for Ubuntu 24.04 x86_64.
# The default path installs CUDA-enabled training, deployment, and hardware
# dependencies into an isolated Conda environment.

export PYTHONNOUSERSITE=1
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_NO_INPUT=1

ENV_NAME="xtrainer-smolvla"
PYTHON_VERSION="3.12"
TORCH_VERSION="2.8.0"
TORCHVISION_VERSION="0.23.0"
TORCHCODEC_VERSION="0.6.0"
MIN_DRIVER_VERSION="570.26"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
RECREATE=0
CPU_ONLY=0
INSTALL_SYSTEM_PACKAGES=1
CURRENT_STAGE="initialization"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

log() {
  printf '[xtrainer-env] %s\n' "$*"
}

warn() {
  printf '[xtrainer-env] WARN: %s\n' "$*" >&2
}

die() {
  printf '[xtrainer-env] ERROR: %s\n' "$*" >&2
  exit 1
}

on_error() {
  local exit_code=$?
  printf '[xtrainer-env] ERROR: stage=%s line=%s (exit %s): %s\n' \
    "${CURRENT_STAGE:-unknown}" "${BASH_LINENO[0]:-unknown}" "${exit_code}" \
    "${BASH_COMMAND:-unknown}" >&2
  exit "${exit_code}"
}
trap on_error ERR

stage() {
  CURRENT_STAGE="$1"
  log "stage: ${CURRENT_STAGE}"
}

usage() {
  cat <<'USAGE'
Usage: bash tools/install_xtrainer_env.sh [OPTIONS]

Create the complete X-trainer SmolVLA environment for Ubuntu 24.04 x86_64.
By default the script:
  - installs required Ubuntu runtime packages with apt
  - creates or reuses Conda environment "xtrainer-smolvla"
  - installs Python 3.12 and PyTorch 2.8.0 CUDA 12.8 wheels
  - installs training, SmolVLA, LoRA, WebSocket, Feetech, and RealSense dependencies
  - installs this repository in editable mode and validates key imports

Options:
  --env-name NAME           Conda environment name (default: xtrainer-smolvla)
  --recreate                Remove and rebuild an existing environment
  --cpu-only                Install CPU-only PyTorch; intended for Mock/server checks
  --skip-system-packages    Do not run apt-get; use when OS packages are already installed
  -h, --help                Show this help

Environment overrides:
  TORCH_INDEX_URL=<url>     Override the PyTorch wheel index

Examples:
  bash tools/install_xtrainer_env.sh
  bash tools/install_xtrainer_env.sh --recreate
  bash tools/install_xtrainer_env.sh --cpu-only --skip-system-packages
  bash tools/install_xtrainer_env.sh --env-name xtrainer-dev

The script does not install NVIDIA drivers, download models or datasets, or
change serial/USB permissions. Activate the finished environment with:
  conda activate <environment-name>
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-name)
      [[ $# -ge 2 ]] || die "--env-name requires a value"
      ENV_NAME="$2"
      shift 2
      ;;
    --recreate)
      RECREATE=1
      shift
      ;;
    --cpu-only)
      CPU_ONLY=1
      TORCH_INDEX_URL="https://download.pytorch.org/whl/cpu"
      shift
      ;;
    --skip-system-packages)
      INSTALL_SYSTEM_PACKAGES=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown argument: $1"
      ;;
  esac
done

[[ "${ENV_NAME}" =~ ^[A-Za-z0-9._-]+$ ]] || die "invalid Conda environment name: ${ENV_NAME}"

require_command() {
  command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"
}

stage "system preflight"
[[ "$(uname -s)" == "Linux" ]] || die "this installer requires Linux"
[[ "$(uname -m)" == "x86_64" ]] || die "this installer currently requires x86_64"
[[ -r /etc/os-release ]] || die "cannot read /etc/os-release"
# shellcheck disable=SC1091
source /etc/os-release
[[ "${ID:-}" == "ubuntu" && "${VERSION_ID:-}" == "24.04" ]] || \
  die "expected Ubuntu 24.04, detected ${PRETTY_NAME:-unknown}"
require_command conda

if [[ "${CPU_ONLY}" == "0" ]]; then
  require_command nvidia-smi
  require_command dpkg
  DRIVER_VERSIONS="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader)"
  DRIVER_VERSION="${DRIVER_VERSIONS%%$'\n'*}"
  DRIVER_VERSION="${DRIVER_VERSION//[[:space:]]/}"
  dpkg --compare-versions "${DRIVER_VERSION}" ge "${MIN_DRIVER_VERSION}" || \
    die "NVIDIA driver >=${MIN_DRIVER_VERSION} is required; detected ${DRIVER_VERSION}"
  log "detected NVIDIA driver: ${DRIVER_VERSION}"
fi

if [[ "${INSTALL_SYSTEM_PACKAGES}" == "1" ]]; then
  stage "Ubuntu system packages"
  if [[ "${EUID}" -eq 0 ]]; then
    SUDO_CMD=()
  else
    require_command sudo
    SUDO_CMD=(sudo)
  fi
  "${SUDO_CMD[@]}" apt-get update
  "${SUDO_CMD[@]}" apt-get install -y \
    build-essential \
    ffmpeg \
    git \
    libusb-1.0-0 \
    udev
else
  warn "system package installation skipped"
fi

stage "Conda environment"
if conda env list | awk 'NF && $1 !~ /^#/ {print $1}' | grep -Fxq "${ENV_NAME}"; then
  if [[ "${RECREATE}" == "1" ]]; then
    log "removing existing environment: ${ENV_NAME}"
    conda env remove -n "${ENV_NAME}" -y
    conda create -n "${ENV_NAME}" "python=${PYTHON_VERSION}" pip -y
  else
    log "reusing existing environment: ${ENV_NAME}"
  fi
else
  conda create -n "${ENV_NAME}" "python=${PYTHON_VERSION}" pip -y
fi

CONDA_PYTHON=(conda run --no-capture-output -n "${ENV_NAME}" python)
"${CONDA_PYTHON[@]}" -c \
  'import sys; assert sys.version_info[:2] == (3, 12), "existing environment must use Python 3.12; rerun with --recreate"'

stage "Python packaging tools"
"${CONDA_PYTHON[@]}" -m pip install --upgrade pip setuptools wheel

stage "PyTorch"
"${CONDA_PYTHON[@]}" -m pip install --index-url "${TORCH_INDEX_URL}" \
  "torch==${TORCH_VERSION}" \
  "torchvision==${TORCHVISION_VERSION}"
"${CONDA_PYTHON[@]}" -m pip install "torchcodec==${TORCHCODEC_VERSION}"

stage "X-trainer training and deployment dependencies"
"${CONDA_PYTHON[@]}" -m pip install -e \
  "${REPO_ROOT}[training,smolvla,peft,feetech,intelrealsense]"

stage "environment validation"
"${CONDA_PYTHON[@]}" - "${CPU_ONLY}" <<'PY'
import sys

import accelerate
import aiohttp
import datasets
import msgpack
import peft
import pyrealsense2
import torch
import transformers
import wandb
from lerobot.motors.feetech import FeetechMotorsBus

assert torch.__version__.split("+", 1)[0] == "2.8.0", torch.__version__
if sys.argv[1] == "0":
    assert torch.cuda.is_available(), "CUDA PyTorch was installed but no GPU is available"

print("environment validation passed")
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("transformers", transformers.__version__)
print("datasets", datasets.__version__)
PY

log "environment ready: ${ENV_NAME}"
log "activate with: conda activate ${ENV_NAME}"
log "models, checkpoints, and datasets were not downloaded"
log "serial/USB permissions must be configured separately before real-hardware deployment"
