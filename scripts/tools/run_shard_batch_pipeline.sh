#!/usr/bin/env bash
set -euo pipefail

# Shard-first staged batch runner with per-step checks.
# Runs phases separately (12 -> 3 -> 4 -> 5 -> 6) with health checks,
# diagnostics checks, and canonical output-path validation.

PROJECT_ROOT="${PROJECT_ROOT:-/root/workspace/PartCraft3D}"
BASE_CONFIG="${BASE_CONFIG:-${PROJECT_ROOT}/configs/partverse_local_parallel_shard00.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/zsn/data/partverse/outputs/partverse}"
SHARD="${SHARD:-00}"
STEPS="${STEPS:-12,3,4,5,6}"      # e.g. STEPS=12,3,4 or STEPS=all
MAX_PARALLEL="${MAX_PARALLEL:-1}"

# Step1+2 (VLM)
VLM_GPUS="${VLM_GPUS:-3,4,5,6,7}"
VLM_BASE_PORT="${VLM_BASE_PORT:-8003}"
STEP1_WORKERS="${STEP1_WORKERS:-16}"

# Step3 (2D edit servers)
EDIT_GPUS="${EDIT_GPUS:-3,4,5,6,7}"
EDIT_BASE_PORT="${EDIT_BASE_PORT:-8004}"
STEP3_WORKERS="${STEP3_WORKERS:-32}"

# Step4 (TRELLIS)
STEP4_GPUS="${STEP4_GPUS:-3,4,5,6,7}"
USE_2D="${USE_2D:-1}"
CACHE_ONLY_2D="${CACHE_ONLY_2D:-0}"

KEEP_SERVERS="${KEEP_SERVERS:-0}"
LOG_DIR="${PROJECT_ROOT}/logs/batch_shard${SHARD}"
mkdir -p "${LOG_DIR}"

cd "${PROJECT_ROOT}"
source /root/miniconda3/etc/profile.d/conda.sh
conda activate pipeline_server

RUNTIME_CONFIG="${LOG_DIR}/runtime_config_shard${SHARD}.yaml"
RUN_TOKEN="shard${SHARD}"
SHARD_ROOT="${OUTPUT_ROOT}/shard_${SHARD}"

VLM_PIDS=()
EDIT_PIDS=()

to_args_from_csv() {
  local csv="$1"
  local -n out_ref="$2"
  out_ref=()
  IFS=',' read -r -a _raw <<< "${csv}"
  for x in "${_raw[@]}"; do
    x="$(echo "${x}" | xargs)"
    [[ -n "${x}" ]] && out_ref+=("${x}")
  done
}

healthcheck() {
  local url="$1"
  python - "$url" <<'PY'
import json, sys, urllib.request
u = sys.argv[1].rstrip("/") + "/health"
try:
    with urllib.request.urlopen(u, timeout=3) as r:
        body = r.read().decode("utf-8", "replace").strip()
        if r.status == 200 and not body:
            ok = True
        else:
            try:
                d = json.loads(body)
                ok = d.get("status") == "ok"
            except Exception:
                ok = r.status == 200
except Exception:
    ok = False
sys.exit(0 if ok else 1)
PY
}

create_runtime_config() {
  python - "${BASE_CONFIG}" "${RUNTIME_CONFIG}" "${SHARD}" "${OUTPUT_ROOT}" <<'PY'
import sys, yaml
base, out, shard, out_root = sys.argv[1:]
cfg = yaml.safe_load(open(base, "r", encoding="utf-8"))
cfg.setdefault("data", {})
cfg["data"]["shards"] = [str(shard).zfill(2)]
cfg["data"]["output_by_shard"] = False
cfg["data"]["output_dir"] = f"{out_root}/shard_{str(shard).zfill(2)}"
for k, v in {
    "phase0": "cache/phase0",
    "phase1": "cache/phase1",
    "phase2": "cache/phase2",
    "phase2_5": "cache/phase2_5",
    "phase3": "cache/phase3",
    "phase4": "cache/phase4",
}.items():
    if k in cfg and isinstance(cfg[k], dict):
        cfg[k]["cache_dir"] = v
with open(out, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
PY
}

stop_vlm_servers() {
  if [[ "${#VLM_PIDS[@]}" -gt 0 ]]; then
    kill "${VLM_PIDS[@]}" 2>/dev/null || true
    VLM_PIDS=()
  fi
}

stop_edit_servers() {
  if [[ "${#EDIT_PIDS[@]}" -gt 0 ]]; then
    kill "${EDIT_PIDS[@]}" 2>/dev/null || true
    EDIT_PIDS=()
  fi
}

cleanup() {
  if [[ "${KEEP_SERVERS}" == "1" ]]; then
    return
  fi
  stop_vlm_servers
  stop_edit_servers
}
trap cleanup EXIT

start_vlm_servers() {
  local -a gpus=("$@")
  local idx=0
  VLM_URLS=()
  for gpu in "${gpus[@]}"; do
    local port=$((VLM_BASE_PORT + idx))
    local base="http://127.0.0.1:${port}"
    local url="${base}/v1"
    local log="${LOG_DIR}/vlm_gpu${gpu}_port${port}.log"
    CUDA_VISIBLE_DEVICES="${gpu}" VLM_PORT="${port}" \
      bash scripts/tools/launch_local_vlm.sh >"${log}" 2>&1 &
    VLM_PIDS+=("$!")
    VLM_URLS+=("${url}")
    idx=$((idx + 1))
  done
  for i in "${!gpus[@]}"; do
    local port=$((VLM_BASE_PORT + i))
    local base="http://127.0.0.1:${port}"
    local ok=0
    for _ in $(seq 1 180); do
      if healthcheck "${base}"; then ok=1; break; fi
      sleep 2
    done
    [[ "${ok}" -eq 1 ]] || { echo "[ERROR] VLM not healthy: ${base}"; exit 1; }
  done
}

start_edit_servers() {
  local -a gpus=("$@")
  local idx=0
  EDIT_URLS=()
  for gpu in "${gpus[@]}"; do
    local port=$((EDIT_BASE_PORT + idx))
    local base="http://127.0.0.1:${port}"
    local log="${LOG_DIR}/edit_gpu${gpu}_port${port}.log"
    python scripts/tools/image_edit_server.py --gpu "${gpu}" --port "${port}" \
      >"${log}" 2>&1 &
    EDIT_PIDS+=("$!")
    EDIT_URLS+=("${base}")
    idx=$((idx + 1))
  done
  for i in "${!gpus[@]}"; do
    local port=$((EDIT_BASE_PORT + i))
    local base="http://127.0.0.1:${port}"
    local ok=0
    for _ in $(seq 1 180); do
      if healthcheck "${base}"; then ok=1; break; fi
      sleep 2
    done
    [[ "${ok}" -eq 1 ]] || { echo "[ERROR] Edit server not healthy: ${base}"; exit 1; }
  done
}

check_step_file() {
  local path="$1"
  local label="$2"
  [[ -s "${path}" ]] || { echo "[ERROR] Missing ${label}: ${path}"; exit 1; }
}

check_jsonl_min_rows() {
  local path="$1"
  local min_rows="$2"
  local label="$3"
  python - "${path}" "${min_rows}" "${label}" <<'PY'
import sys
p, n, label = sys.argv[1], int(sys.argv[2]), sys.argv[3]
rows = 0
with open(p, "r", encoding="utf-8", errors="replace") as f:
    for line in f:
        if line.strip():
            rows += 1
if rows < n:
    raise SystemExit(f"[ERROR] {label} has only {rows} rows (< {n}): {p}")
print(f"[CHECK] {label}: {rows} rows")
PY
}

run_step12() {
  local -a gpus=()
  to_args_from_csv "${VLM_GPUS}" gpus
  [[ "${#gpus[@]}" -gt 0 ]] || { echo "[ERROR] VLM_GPUS is empty"; exit 1; }
  start_vlm_servers "${gpus[@]}"
  local -a cmd=(
    python scripts/run_pipeline.py
    --config "${RUNTIME_CONFIG}"
    --shard "${SHARD}"
    --steps 1 2
    --step1-vlm-urls "${VLM_URLS[@]}"
    --step1-gpus "${gpus[@]}"
    --step1-workers "${STEP1_WORKERS}"
  )
  [[ "${MAX_PARALLEL}" == "1" ]] && cmd+=(--max-parallel)
  "${cmd[@]}"
  [[ "${KEEP_SERVERS}" == "1" ]] || stop_vlm_servers

  check_step_file "${SHARD_ROOT}/cache/phase0/semantic_labels_${RUN_TOKEN}.jsonl" "step1 labels"
  check_step_file "${SHARD_ROOT}/cache/phase1/edit_specs_${RUN_TOKEN}.jsonl" "step2 specs"
  check_step_file "${SHARD_ROOT}/pipeline/reports/step1_semantic.json" "step1 diagnostic"
  check_step_file "${SHARD_ROOT}/pipeline/reports/step2_planning.json" "step2 diagnostic"
  check_jsonl_min_rows "${SHARD_ROOT}/cache/phase1/edit_specs_${RUN_TOKEN}.jsonl" 100 "step2 specs"
}

run_step3() {
  local -a gpus=()
  to_args_from_csv "${EDIT_GPUS}" gpus
  [[ "${#gpus[@]}" -gt 0 ]] || { echo "[ERROR] EDIT_GPUS is empty"; exit 1; }
  start_edit_servers "${gpus[@]}"
  local -a cmd=(
    python scripts/run_pipeline.py
    --config "${RUNTIME_CONFIG}"
    --shard "${SHARD}"
    --steps 3
    --workers "${STEP3_WORKERS}"
    --image-edit-urls "${EDIT_URLS[@]}"
  )
  [[ "${MAX_PARALLEL}" == "1" ]] && cmd+=(--max-parallel)
  "${cmd[@]}"
  [[ "${KEEP_SERVERS}" == "1" ]] || stop_edit_servers

  check_step_file "${SHARD_ROOT}/cache/phase2_5/2d_edits_${RUN_TOKEN}/manifest.jsonl" "step3 manifest"
  check_step_file "${SHARD_ROOT}/pipeline/reports/step3_2d_edit.json" "step3 diagnostic"
  check_jsonl_min_rows "${SHARD_ROOT}/cache/phase2_5/2d_edits_${RUN_TOKEN}/manifest.jsonl" 100 "step3 manifest"
}

run_step4() {
  local -a gpus=()
  to_args_from_csv "${STEP4_GPUS}" gpus
  [[ "${#gpus[@]}" -gt 0 ]] || { echo "[ERROR] STEP4_GPUS is empty"; exit 1; }

  local -a cmd=(
    python scripts/run_pipeline.py
    --config "${RUNTIME_CONFIG}"
    --shard "${SHARD}"
    --steps 4
    --gpus "${gpus[@]}"
  )
  [[ "${USE_2D}" == "1" ]] || cmd+=(--no-2d-edit)
  [[ "${CACHE_ONLY_2D}" == "1" ]] && cmd+=(--2d-cache-only)
  "${cmd[@]}"

  check_step_file "${SHARD_ROOT}/cache/phase2_5/edit_results_${RUN_TOKEN}.jsonl" "step4 results"
  check_step_file "${SHARD_ROOT}/pipeline/reports/step4_3d_edit.json" "step4 diagnostic"
  check_jsonl_min_rows "${SHARD_ROOT}/cache/phase2_5/edit_results_${RUN_TOKEN}.jsonl" 100 "step4 results"
}

run_step5() {
  python scripts/run_pipeline.py \
    --config "${RUNTIME_CONFIG}" \
    --shard "${SHARD}" \
    --steps 5
  check_step_file "${SHARD_ROOT}/pipeline/reports/step5_quality.json" "step5 diagnostic"
}

run_step6() {
  python scripts/run_pipeline.py \
    --config "${RUNTIME_CONFIG}" \
    --shard "${SHARD}" \
    --steps 6
  check_step_file "${SHARD_ROOT}/pipeline/reports/step6_export.json" "step6 diagnostic"
}

create_runtime_config

if [[ "${STEPS}" == "all" ]]; then
  STEPS="12,3,4,5,6"
fi

IFS=',' read -r -a STEP_LIST <<< "${STEPS}"
for step in "${STEP_LIST[@]}"; do
  step="$(echo "${step}" | xargs)"
  case "${step}" in
    12) run_step12 ;;
    3) run_step3 ;;
    4) run_step4 ;;
    5) run_step5 ;;
    6) run_step6 ;;
    *) echo "[ERROR] Unsupported step token: ${step}"; exit 1 ;;
  esac
done

check_step_file "${SHARD_ROOT}/pipeline/reports/pipeline_summary.json" "pipeline summary"
echo "[DONE] Shard ${SHARD} batch run complete. Reports: ${SHARD_ROOT}/pipeline/reports"
