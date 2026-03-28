#!/usr/bin/env bash
set -euo pipefail

# One-click launcher:
# 1) Start local VLM servers on chosen GPUs (default 3/4/5/6)
#    Ports are assigned from BASE_PORT sequentially.
# 2) Run pipeline Step1+Step2 for shard00 with object sharding across VLM URLs
#
# Usage:
#   bash scripts/tools/run_step12_parallel_s00_0328.sh
#   bash scripts/tools/run_step12_parallel_s00_0328.sh 3 4 5 6 7
#
# Optional overrides:
#   CONFIG=/path/to/config.yaml TAG=0328 STEP1_WORKERS=16 bash ...
#   GPU_LIST=3,4,5,6,7 BASE_PORT=8003 bash ...
#   KEEP_VLM_SERVERS=1 bash ...   # keep VLM servers alive after Step1+2

PROJECT_ROOT="${PROJECT_ROOT:-/root/workspace/PartCraft3D}"
CONFIG="${CONFIG:-/root/workspace/PartCraft3D/configs/partverse_local_parallel_0328_shard00.yaml}"
TAG="${TAG:-0328}"
STEP1_WORKERS="${STEP1_WORKERS:-16}"
KEEP_VLM_SERVERS="${KEEP_VLM_SERVERS:-0}"
GPU_LIST="${GPU_LIST:-}"

BASE_PORT=8003
BASE_PORT="${BASE_PORT:-8003}"
LOG_DIR="${PROJECT_ROOT}/logs/step1_vlm_${TAG}"
mkdir -p "${LOG_DIR}"

cd "${PROJECT_ROOT}"

# Activate conda env
source /root/miniconda3/etc/profile.d/conda.sh
conda activate pipeline_server

PIDS=()
URLS=()

# Parse GPUs from args > GPU_LIST > default
if [[ "$#" -gt 0 ]]; then
  GPUS=("$@")
elif [[ -n "${GPU_LIST}" ]]; then
  IFS=',' read -r -a GPUS <<< "${GPU_LIST}"
else
  GPUS=(3 4 5 6)
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
        # SGLang health endpoint may return empty body with HTTP 200.
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
  if [[ "${KEEP_VLM_SERVERS}" == "1" ]]; then
    echo "[INFO] KEEP_VLM_SERVERS=1, skip stopping VLM servers."
    return
  fi
  if [[ "${#PIDS[@]}" -gt 0 ]]; then
    echo "[INFO] Stopping VLM servers: ${PIDS[*]}"
    kill "${PIDS[@]}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

echo "[INFO] Starting VLM servers on GPUs ${GPUS[*]} ..."
for idx in "${!GPUS[@]}"; do
  gpu="${GPUS[$idx]}"
  port=$((BASE_PORT + idx))
  url="http://127.0.0.1:${port}/v1"
  log_file="${LOG_DIR}/vlm_gpu${gpu}_port${port}.log"

  CUDA_VISIBLE_DEVICES="${gpu}" VLM_PORT="${port}" \
    bash scripts/tools/launch_local_vlm.sh >"${log_file}" 2>&1 &
  pid=$!
  PIDS+=("${pid}")
  URLS+=("${url}")
  echo "[INFO] gpu=${gpu} port=${port} pid=${pid} log=${log_file}"
done

echo "[INFO] Waiting for VLM health checks ..."
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
    echo "[ERROR] VLM not healthy: ${base}. Check ${LOG_DIR}/vlm_gpu${GPUS[$idx]}_port${port}.log"
    exit 1
  fi
  echo "[INFO] Healthy: ${base}"
done

echo "[INFO] Running Step1+Step2 (tag=${TAG}) ..."
python scripts/run_pipeline.py \
  --config "${CONFIG}" \
  --steps 1 2 \
  --tag "${TAG}" \
  --step1-vlm-urls "${URLS[@]}" \
  --step1-gpus "${GPUS[@]}" \
  --step1-workers "${STEP1_WORKERS}" \
  --max-parallel

echo "[INFO] Step1+Step2 done."
