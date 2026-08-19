#!/usr/bin/env bash
set -Eeuo pipefail

ENV_NAME="xtrainer-smolvla"
REPO_ID="lerobot/smolvla_base"
REVISION=""
ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUTPUT_DIR="${REPO_ROOT}/models/smolvla_base"

usage() {
  cat <<'USAGE'
Usage: bash tools/download_smolvla_weights_hf.sh [OPTIONS]

Download a SmolVLA model snapshot from Hugging Face.

Options:
  --repo-id ID          Model ID (default: lerobot/smolvla_base)
  --output-dir PATH     Local model directory (default: models/smolvla_base)
  --revision REVISION   Optional branch, tag, or commit SHA
  --endpoint URL        Hugging Face endpoint (default: https://huggingface.co)
  --env-name NAME       Conda environment name (default: xtrainer-smolvla)
  -h, --help            Show this help

For private or gated repositories, authenticate with `hf auth login` or provide
HF_TOKEN. Existing downloaded files are reused.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo-id) REPO_ID="${2:?--repo-id requires a value}"; shift 2 ;;
    --output-dir) OUTPUT_DIR="${2:?--output-dir requires a value}"; shift 2 ;;
    --revision) REVISION="${2:?--revision requires a value}"; shift 2 ;;
    --endpoint) ENDPOINT="${2:?--endpoint requires a value}"; shift 2 ;;
    --env-name) ENV_NAME="${2:?--env-name requires a value}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'ERROR: unknown argument: %s\n' "$1" >&2; exit 1 ;;
  esac
done

[[ "${ENV_NAME}" =~ ^[A-Za-z0-9._-]+$ ]] || {
  printf 'ERROR: invalid Conda environment name: %s\n' "${ENV_NAME}" >&2
  exit 1
}
command -v conda >/dev/null 2>&1 || { printf 'ERROR: conda was not found\n' >&2; exit 1; }

mkdir -p "${OUTPUT_DIR}"
printf '[xtrainer-download-hf] model: %s\n' "${REPO_ID}"
printf '[xtrainer-download-hf] endpoint: %s\n' "${ENDPOINT}"
printf '[xtrainer-download-hf] destination: %s\n' "${OUTPUT_DIR}"

conda run --no-capture-output -n "${ENV_NAME}" \
  python - "${REPO_ID}" "${OUTPUT_DIR}" "${REVISION}" "${ENDPOINT}" <<'PY'
import os
import sys

from huggingface_hub import snapshot_download

repo_id, output_dir, revision, endpoint = sys.argv[1:]
downloaded_path = snapshot_download(
    repo_id=repo_id,
    repo_type="model",
    revision=revision or None,
    local_dir=output_dir,
    endpoint=endpoint,
    token=os.environ.get("HF_TOKEN"),
)
print(f"model download complete: {downloaded_path}")
PY

printf '[xtrainer-download-hf] deployment checkpoint: %s\n' "${OUTPUT_DIR}"
