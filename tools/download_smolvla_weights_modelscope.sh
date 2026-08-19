#!/usr/bin/env bash
set -Eeuo pipefail

ENV_NAME="xtrainer-smolvla"
MODEL_ID="lerobot/smolvla_base"
REVISION=""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUTPUT_DIR="${REPO_ROOT}/models/smolvla_base"

usage() {
  cat <<'USAGE'
Usage: bash tools/download_smolvla_weights_modelscope.sh [OPTIONS]

Download a SmolVLA model snapshot from ModelScope.

Options:
  --model-id ID         ModelScope model ID (default: lerobot/smolvla_base)
  --output-dir PATH     Local model directory (default: models/smolvla_base)
  --revision REVISION   Optional branch, tag, or commit ID
  --env-name NAME       Conda environment name (default: xtrainer-smolvla)
  -h, --help            Show this help

For private repositories, provide MODELSCOPE_API_TOKEN. Existing downloaded
files are reused.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-id) MODEL_ID="${2:?--model-id requires a value}"; shift 2 ;;
    --output-dir) OUTPUT_DIR="${2:?--output-dir requires a value}"; shift 2 ;;
    --revision) REVISION="${2:?--revision requires a value}"; shift 2 ;;
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
printf '[xtrainer-download-modelscope] model: %s\n' "${MODEL_ID}"
printf '[xtrainer-download-modelscope] destination: %s\n' "${OUTPUT_DIR}"

conda run --no-capture-output -n "${ENV_NAME}" \
  python - "${MODEL_ID}" "${OUTPUT_DIR}" "${REVISION}" <<'PY'
import os
import sys

from modelscope import snapshot_download

model_id, output_dir, revision = sys.argv[1:]
token = os.environ.get("MODELSCOPE_API_TOKEN")
if token:
    from modelscope.hub.api import HubApi

    HubApi().login(token)

downloaded_path = snapshot_download(
    model_id=model_id,
    revision=revision or None,
    local_dir=output_dir,
)
print(f"model download complete: {downloaded_path}")
PY

printf '[xtrainer-download-modelscope] deployment checkpoint: %s\n' "${OUTPUT_DIR}"
