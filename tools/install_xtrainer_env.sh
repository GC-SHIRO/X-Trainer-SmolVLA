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
CUSTOM_PIP_INDEX_URL="${PIP_INDEX_URL:-}"
CUSTOM_TORCH_INDEX_URL="${TORCH_INDEX_URL:-}"
SOURCE="china"
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
  --source NAME             Package source: china or official (default: china)
  --skip-system-packages    Do not run apt-get; use when OS packages are already installed
  -h, --help                Show this help

Environment overrides:
  PIP_INDEX_URL=<url>       Override the Python package index
  TORCH_INDEX_URL=<url>     Override the PyTorch wheel index

Examples:
  bash tools/install_xtrainer_env.sh
  bash tools/install_xtrainer_env.sh --recreate
  bash tools/install_xtrainer_env.sh --source official
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
      shift
      ;;
    --source)
      [[ $# -ge 2 ]] || die "--source requires a value"
      SOURCE="$2"
      shift 2
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

CONDA_SOURCE_ARGS=()
APT_SOURCE_URL=""
case "${SOURCE}" in
  official)
    DEFAULT_PIP_INDEX_URL="https://pypi.org/simple"
    if [[ "${CPU_ONLY}" == "1" ]]; then
      DEFAULT_TORCH_INDEX_URL="https://download.pytorch.org/whl/cpu"
    else
      DEFAULT_TORCH_INDEX_URL="https://download.pytorch.org/whl/cu128"
    fi
    ;;
  china)
    DEFAULT_PIP_INDEX_URL="https://pypi.tuna.tsinghua.edu.cn/simple"
    if [[ "${CPU_ONLY}" == "1" ]]; then
      DEFAULT_TORCH_INDEX_URL="https://mirrors.aliyun.com/pytorch-wheels/cpu"
    else
      DEFAULT_TORCH_INDEX_URL="https://mirrors.aliyun.com/pytorch-wheels/cu128"
    fi
    APT_SOURCE_URL="https://mirrors.tuna.tsinghua.edu.cn/ubuntu"
    CONDA_SOURCE_ARGS=(
      --override-channels
      -c "https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main"
      -c "https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r"
    )
    ;;
  *)
    die "unsupported source: ${SOURCE}; expected official or china"
    ;;
esac

PIP_INDEX_URL="${CUSTOM_PIP_INDEX_URL:-${DEFAULT_PIP_INDEX_URL}}"
TORCH_INDEX_URL="${CUSTOM_TORCH_INDEX_URL:-${DEFAULT_TORCH_INDEX_URL}}"
export PIP_INDEX_URL

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
log "package source: ${SOURCE}"
log "Python index: ${PIP_INDEX_URL}"
log "PyTorch index: ${TORCH_INDEX_URL}"

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
  APT_SOURCE_ARGS=()
  if [[ -n "${APT_SOURCE_URL}" ]]; then
    APT_SOURCE_FILE="$(mktemp)"
    trap 'rm -f "${APT_SOURCE_FILE:-}"' EXIT
    cat >"${APT_SOURCE_FILE}" <<EOF
deb [arch=amd64] ${APT_SOURCE_URL} noble main restricted universe multiverse
deb [arch=amd64] ${APT_SOURCE_URL} noble-updates main restricted universe multiverse
deb [arch=amd64] ${APT_SOURCE_URL} noble-backports main restricted universe multiverse
deb [arch=amd64] ${APT_SOURCE_URL} noble-security main restricted universe multiverse
EOF
    APT_SOURCE_ARGS=(
      -o "Dir::Etc::sourcelist=${APT_SOURCE_FILE}"
      -o "Dir::Etc::sourceparts=-"
      -o "APT::Get::List-Cleanup=0"
    )
  fi
  "${SUDO_CMD[@]}" apt-get "${APT_SOURCE_ARGS[@]}" update
  "${SUDO_CMD[@]}" apt-get "${APT_SOURCE_ARGS[@]}" install -y \
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
    conda create "${CONDA_SOURCE_ARGS[@]}" -n "${ENV_NAME}" "python=${PYTHON_VERSION}" pip -y
  else
    log "reusing existing environment: ${ENV_NAME}"
  fi
else
  conda create "${CONDA_SOURCE_ARGS[@]}" -n "${ENV_NAME}" "python=${PYTHON_VERSION}" pip -y
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
"${CONDA_PYTHON[@]}" -m pip install modelscope

stage "environment validation"
"${CONDA_PYTHON[@]}" - "${CPU_ONLY}" <<'PY'
import sys

import accelerate
import aiohttp
import datasets
import msgpack
import modelscope
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
