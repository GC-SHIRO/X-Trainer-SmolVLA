#!/usr/bin/env bash
set -Eeuo pipefail

ENV_NAME="xtrainer-smolvla"
REPO_ID="lerobot/smolvla_base"
REVISION=""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUTPUT_DIR="${REPO_ROOT}/models/smolvla_base"

usage() {
  cat <<'USAGE'
Usage: bash tools/download_smolvla_weights.sh [OPTIONS]

Download a SmolVLA model snapshot into a local directory.

Options:
  --repo-id ID          Hugging Face model ID (default: lerobot/smolvla_base)
  --output-dir PATH     Local model directory (default: models/smolvla_base)
  --revision REVISION   Optional branch, tag, or commit SHA
  --env-name NAME       Conda environment name (default: xtrainer-smolvla)
  -h, --help            Show this help

For private or gated repositories, authenticate first with `hf auth login` or
provide the HF_TOKEN environment variable. Existing downloaded files are reused.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo-id)
      [[ $# -ge 2 ]] || { printf 'ERROR: --repo-id requires a value\n' >&2; exit 1; }
      REPO_ID="$2"
      shift 2
      ;;
    --output-dir)
      [[ $# -ge 2 ]] || { printf 'ERROR: --output-dir requires a value\n' >&2; exit 1; }
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --revision)
      [[ $# -ge 2 ]] || { printf 'ERROR: --revision requires a value\n' >&2; exit 1; }
      REVISION="$2"
      shift 2
      ;;
    --env-name)
      [[ $# -ge 2 ]] || { printf 'ERROR: --env-name requires a value\n' >&2; exit 1; }
      ENV_NAME="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      printf 'ERROR: unknown argument: %s\n' "$1" >&2
      exit 1
      ;;
  esac
done

[[ "${ENV_NAME}" =~ ^[A-Za-z0-9._-]+$ ]] || {
  printf 'ERROR: invalid Conda environment name: %s\n' "${ENV_NAME}" >&2
  exit 1
}
command -v conda >/dev/null 2>&1 || {
  printf 'ERROR: conda was not found\n' >&2
  exit 1
}

mkdir -p "${OUTPUT_DIR}"
printf '[xtrainer-download] model: %s\n' "${REPO_ID}"
printf '[xtrainer-download] destination: %s\n' "${OUTPUT_DIR}"

conda run --no-capture-output -n "${ENV_NAME}" \
  python - "${REPO_ID}" "${OUTPUT_DIR}" "${REVISION}" <<'PY'
import os
import sys

from huggingface_hub import snapshot_download

repo_id, output_dir, revision = sys.argv[1:]
downloaded_path = snapshot_download(
    repo_id=repo_id,
    repo_type="model",
    revision=revision or None,
    local_dir=output_dir,
    token=os.environ.get("HF_TOKEN"),
)
print(f"model download complete: {downloaded_path}")
PY

printf '[xtrainer-download] use for training: --policy.repo_id=%s\n' "${OUTPUT_DIR}"
printf '[xtrainer-download] use for deployment: --checkpoint %s\n' "${OUTPUT_DIR}"
