#!/usr/bin/env bash
# Download local missing model weights for this machine.
#
# Defaults (can be overridden by env vars):
#   PARTCRAFT_CKPT_ROOT=/mnt/cfs/vffey4/3dedit/ckpts
#   VLM_REPO_ID=Qwen/Qwen3.5-27B
#   EDIT_REPO_ID=black-forest-labs/FLUX.2-klein-9B
#
# Usage:
#   bash scripts/tools/download_local_missing_weights.sh
#   MODE=vlm  bash scripts/tools/download_local_missing_weights.sh
#   MODE=edit bash scripts/tools/download_local_missing_weights.sh
#   VLM_REPO_ID=<repo> EDIT_REPO_ID=<repo> bash scripts/tools/download_local_missing_weights.sh

set -euo pipefail

MODE="${MODE:-all}"  # all | vlm | edit
ROOT="${PARTCRAFT_CKPT_ROOT:-/mnt/cfs/vffey4/3dedit/ckpts}"
FORCE="${FORCE:-0}"  # 1: force re-download even if marker exists

VLM_DIR_NAME="${VLM_DIR_NAME:-Qwen3.5-27B}"
VLM_REPO_ID="${VLM_REPO_ID:-Qwen/Qwen3.5-27B}"

EDIT_DIR_NAME="${EDIT_DIR_NAME:-FLUX.2-klein-9B}"
EDIT_REPO_ID="${EDIT_REPO_ID:-black-forest-labs/FLUX.2-klein-9B}"

mkdir -p "${ROOT}"

if ! command -v hf >/dev/null 2>&1; then
  echo "[ERROR] huggingface cli (hf) not found."
  echo "Install with: pip install \"huggingface_hub[cli]\""
  exit 1
fi

is_download_complete() {
  local local_dir="$1"
  local marker="$2"
  [[ -f "${local_dir}/${marker}" ]]
}

download_repo() {
  local repo_id="$1"
  local local_dir="$2"
  local tag="$3"
  local marker="$4"

  if [[ "${FORCE}" != "1" ]]; then
    if is_download_complete "${local_dir}" "${marker}"; then
      echo "[SKIP] ${tag} already present: ${local_dir} (${marker})"
      return 0
    fi
    if [[ -d "${local_dir}" ]] && [[ -n "$(ls -A "${local_dir}" 2>/dev/null || true)" ]]; then
      echo "[WARN] ${tag} directory exists but marker missing, will resume/fix:"
      echo "       missing ${local_dir}/${marker}"
    fi
  else
    echo "[INFO] FORCE=1, re-downloading ${tag}"
  fi

  echo "[INFO] Downloading ${tag}"
  echo "       repo_id=${repo_id}"
  echo "       local_dir=${local_dir}"
  hf download "${repo_id}" --local-dir "${local_dir}"

  if ! is_download_complete "${local_dir}" "${marker}"; then
    echo "[ERROR] ${tag} download finished but marker still missing:"
    echo "        ${local_dir}/${marker}"
    exit 1
  fi
}

case "${MODE}" in
  all)
    download_repo "${VLM_REPO_ID}" "${ROOT}/${VLM_DIR_NAME}" "VLM" "config.json"
    download_repo "${EDIT_REPO_ID}" "${ROOT}/${EDIT_DIR_NAME}" "ImageEdit" "model_index.json"
    ;;
  vlm)
    download_repo "${VLM_REPO_ID}" "${ROOT}/${VLM_DIR_NAME}" "VLM" "config.json"
    ;;
  edit)
    download_repo "${EDIT_REPO_ID}" "${ROOT}/${EDIT_DIR_NAME}" "ImageEdit" "model_index.json"
    ;;
  *)
    echo "[ERROR] Unsupported MODE=${MODE} (use: all | vlm | edit)"
    exit 1
    ;;
esac

echo "[DONE] Checkpoints ready under: ${ROOT}"
