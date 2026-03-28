#!/usr/bin/env bash
set -euo pipefail

# One-click launcher for Step3 (2D edit) with arbitrary GPU combination.
#
# Behavior:
# 1) Start local image_edit_server on selected GPUs (default 3/4/5/6/7)
# 2) Wait for health checks
# 3) Run pipeline Step3 once, passing all image-edit URLs
#
# Usage:
#   bash scripts/tools/run_step3_parallel_s00_0328.sh
#   bash scripts/tools/run_step3_parallel_s00_0328.sh 3 4 5 6 7
#
# Optional overrides:
#   CONFIG=/path/to/config.yaml TAG=0328 STEP3_WORKERS=32 bash ...
#   GPU_LIST=3,4,5,6,7 BASE_PORT=8004 bash ...
#   KEEP_EDIT_SERVERS=1 bash ...   # keep servers alive after Step3

PROJECT_ROOT="${PROJECT_ROOT:-/root/workspace/PartCraft3D}"
CONFIG="${CONFIG:-/root/workspace/PartCraft3D/configs/partverse_local_parallel_0328_shard00.yaml}"
TAG="${TAG:-0328}"
STEP3_WORKERS="${STEP3_WORKERS:-32}"
KEEP_EDIT_SERVERS="${KEEP_EDIT_SERVERS:-0}"
GPU_LIST="${GPU_LIST:-}"
BASE_PORT="${BASE_PORT:-8004}"
LOG_DIR="${PROJECT_ROOT}/logs/step3_edit_${TAG}"
mkdir -p "${LOG_DIR}"

cd "${PROJECT_ROOT}"

source /root/miniconda3/etc/profile.d/conda.sh
conda activate pipeline_server

PIDS=()
URLS=()

# Parse GPU list from args > GPU_LIST > default
if [[ "$#" -gt 0 ]]; then
  GPUS=("$@")
elif [[ -n "${GPU_LIST}" ]]; then
  IFS=',' read -r -a GPUS <<< "${GPU_LIST}"
else
  GPUS=(3 4 5 6 7)
fi

if [[ "${#GPUS[@]}" -eq 0 ]]; then
  echo "[ERROR] No GPUs specified"
  exit 1
fi

_healthcheck() {
  local url="$1"
  python - "$url" <<'PY'
import json, sys, urllib.request
url = sys.argv[1].rstrip("/") + "/health"
try:
    with urllib.request.urlopen(url, timeout=3) as r:
        body = r.read().decode("utf-8", "replace").strip()
        if r.status == 200 and not body:
            ok = True
        else:
            try:
                d = json.loads(body)
                ok = (d.get("status") == "ok")
            except Exception:
                ok = (r.status == 200)
except Exception:
    ok = False
sys.exit(0 if ok else 1)
PY
}

cleanup() {
  if [[ "${KEEP_EDIT_SERVERS}" == "1" ]]; then
    echo "[INFO] KEEP_EDIT_SERVERS=1, skip stopping edit servers."
    return
  fi
  if [[ "${#PIDS[@]}" -gt 0 ]]; then
    echo "[INFO] Stopping edit servers: ${PIDS[*]}"
    kill "${PIDS[@]}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

echo "[INFO] Starting image edit servers on GPUs ${GPUS[*]} ..."
for idx in "${!GPUS[@]}"; do
  gpu="${GPUS[$idx]}"
  port=$((BASE_PORT + idx))
  url="http://127.0.0.1:${port}"
  log_file="${LOG_DIR}/edit_gpu${gpu}_port${port}.log"

  python scripts/tools/image_edit_server.py --gpu "${gpu}" --port "${port}" \
    >"${log_file}" 2>&1 &
  pid=$!
  PIDS+=("${pid}")
  URLS+=("${url}")
  echo "[INFO] gpu=${gpu} port=${port} pid=${pid} log=${log_file}"
done

echo "[INFO] Waiting for image edit health checks ..."
for idx in "${!GPUS[@]}"; do
  port=$((BASE_PORT + idx))
  base="http://127.0.0.1:${port}"
  ok=0
  for _ in $(seq 1 180); do
    if _healthcheck "${base}"; then
      ok=1
      break
    fi
    sleep 2
  done
  if [[ "${ok}" -ne 1 ]]; then
    echo "[ERROR] Edit server not healthy: ${base}. Check ${LOG_DIR}/edit_gpu${GPUS[$idx]}_port${port}.log"
    exit 1
  fi
  echo "[INFO] Healthy: ${base}"
done

echo "[INFO] Running Step3 with URLs: ${URLS[*]}"
python scripts/run_pipeline.py \
  --config "${CONFIG}" \
  --steps 3 \
  --tag "${TAG}" \
  --workers "${STEP3_WORKERS}" \
  --image-edit-urls "${URLS[@]}" \
  --max-parallel

echo "[INFO] Step3 done."
