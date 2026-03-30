#!/usr/bin/env bash
set -euo pipefail

# Shard-first staged batch runner with per-step checks.
# Runs phases separately (12 -> 3 -> 4 -> 5 -> 6) with health checks,
# diagnostics checks, and canonical output-path validation.
#
# Machine-specific paths are loaded from configs/machine/<hostname>.env.
# Override with: MACHINE_ENV=configs/machine/mybox.env SHARD=01 bash $0

# ── Resolve project root from script location ─────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"

# ── Load machine config ───────────────────────────────────────────
MACHINE_ENV="${MACHINE_ENV:-${PROJECT_ROOT}/configs/machine/$(hostname).env}"
if [[ ! -f "${MACHINE_ENV}" ]]; then
  echo "[ERROR] Machine config not found: ${MACHINE_ENV}"
  echo "  Create it from the template: configs/machine/node39.env"
  exit 1
fi
# shellcheck disable=SC1090
source "${MACHINE_ENV}"

# ── Conda setup ───────────────────────────────────────────────────
CONDA_INIT="${CONDA_INIT:?CONDA_INIT not set in ${MACHINE_ENV}}"
CONDA_ENV_SERVER="${CONDA_ENV_SERVER:?CONDA_ENV_SERVER not set in ${MACHINE_ENV}}"
CONDA_ENV_PIPELINE="${CONDA_ENV_PIPELINE:?CONDA_ENV_PIPELINE not set in ${MACHINE_ENV}}"
# shellcheck disable=SC1090
set +u; source "${CONDA_INIT}"; set -u

# ── Checkpoint paths ──────────────────────────────────────────────
VLM_CKPT="${VLM_CKPT:?VLM_CKPT not set in ${MACHINE_ENV}}"
EDIT_CKPT="${EDIT_CKPT:?EDIT_CKPT not set in ${MACHINE_ENV}}"
TRELLIS_CKPT_ROOT="${TRELLIS_CKPT_ROOT:-${PROJECT_ROOT}/checkpoints}"

# ── Data / output paths ──────────────────────────────────────────
DATA_DIR="${DATA_DIR:?DATA_DIR not set in ${MACHINE_ENV}}"
OUTPUT_ROOT="${OUTPUT_ROOT:?OUTPUT_ROOT not set in ${MACHINE_ENV}}"

# ── Pipeline YAML config ─────────────────────────────────────────
BASE_CONFIG="${BASE_CONFIG:-${PROJECT_ROOT}/configs/partverse_local_parallel_shard00.yaml}"
SHARD="${SHARD:-00}"
STEPS="${STEPS:-12,3,4,5,6}"      # e.g. STEPS=12,3,4 or STEPS=all
MAX_PARALLEL="${MAX_PARALLEL:-1}"

# Step1+2 (VLM)
VLM_GPUS="${VLM_GPUS:-3,4,5,6,7}"
VLM_TP="${VLM_TP:-1}"              # tensor parallel per VLM instance
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

# Misc
LIMIT="${LIMIT:-}"                 # e.g. LIMIT=3 for quick validation
KEEP_SERVERS="${KEEP_SERVERS:-0}"
LOG_DIR="${PROJECT_ROOT}/logs/batch_shard${SHARD}"
mkdir -p "${LOG_DIR}"

cd "${PROJECT_ROOT}"

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
  python - "${BASE_CONFIG}" "${RUNTIME_CONFIG}" "${SHARD}" "${OUTPUT_ROOT}" \
           "${DATA_DIR}" "${TRELLIS_CKPT_ROOT}" <<'PY'
import sys, yaml
base, out, shard, out_root, data_dir, ckpt_root = sys.argv[1:]
cfg = yaml.safe_load(open(base, "r", encoding="utf-8"))
cfg.setdefault("data", {})
cfg["data"]["shards"] = [str(shard).zfill(2)]
cfg["data"]["output_by_shard"] = False
cfg["data"]["output_dir"] = f"{out_root}/shard_{str(shard).zfill(2)}"
cfg["data"]["data_dir"] = data_dir
if ckpt_root:
    cfg["ckpt_root"] = ckpt_root
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

_kill_tree() {
  # Kill process and all descendants
  local pid="$1"
  # Kill children first
  pkill -P "${pid}" 2>/dev/null || true
  kill "${pid}" 2>/dev/null || true
}

stop_vlm_servers() {
  if [[ "${#VLM_PIDS[@]}" -gt 0 ]]; then
    for pid in "${VLM_PIDS[@]}"; do _kill_tree "${pid}"; done
    # Also kill any remaining sglang processes spawned by our servers
    pkill -f "sglang::scheduler" 2>/dev/null || true
    pkill -f "sglang::detokenizer" 2>/dev/null || true
    sleep 2
    VLM_PIDS=()
    echo "[INFO] VLM servers stopped"
  fi
}

stop_edit_servers() {
  if [[ "${#EDIT_PIDS[@]}" -gt 0 ]]; then
    for pid in "${EDIT_PIDS[@]}"; do _kill_tree "${pid}"; done
    sleep 1
    EDIT_PIDS=()
    echo "[INFO] Edit servers stopped"
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
  local tp="${VLM_TP}"
  local idx=0
  local num_instances=$(( ${#gpus[@]} / tp ))
  VLM_URLS=()
  set +u; source "${CONDA_INIT}" && conda activate "${CONDA_ENV_SERVER}"; set -u
  if [[ "${num_instances}" -lt 1 ]]; then
    echo "[ERROR] VLM_GPUS has ${#gpus[@]} GPUs but VLM_TP=${tp} requires at least ${tp}"
    exit 1
  fi
  echo "[INFO] Launching ${num_instances} VLM instance(s) with TP=${tp}"
  for (( i=0; i<num_instances; i++ )); do
    # Gather tp consecutive GPUs for this instance
    local gpu_group=""
    for (( j=0; j<tp; j++ )); do
      local gi=$(( i * tp + j ))
      [[ -n "${gpu_group}" ]] && gpu_group+=","
      gpu_group+="${gpus[$gi]}"
    done
    local port=$((VLM_BASE_PORT + i))
    local base="http://127.0.0.1:${port}"
    local url="${base}/v1"
    local log="${LOG_DIR}/vlm_tp${tp}_gpu${gpu_group}_port${port}.log"
    echo "[INFO]   Instance $i: GPUs=${gpu_group} port=${port}"
    CUDA_VISIBLE_DEVICES="${gpu_group}" VLM_MODEL="${VLM_CKPT}" VLM_PORT="${port}" \
      VLM_TP="${tp}" VLM_MAX_LEN="${VLM_MAX_LEN:-16384}" \
      VLM_MEM_FRAC="${VLM_MEM_FRAC:-0.85}" \
      SGLANG_DISABLE_CUDNN_CHECK=1 \
      bash scripts/tools/launch_local_vlm.sh >"${log}" 2>&1 &
    VLM_PIDS+=("$!")
    VLM_URLS+=("${url}")
  done
  for (( i=0; i<num_instances; i++ )); do
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
  set +u; source "${CONDA_INIT}" && conda activate "${CONDA_ENV_SERVER}"; set -u
  for gpu in "${gpus[@]}"; do
    local port=$((EDIT_BASE_PORT + idx))
    local base="http://127.0.0.1:${port}"
    local log="${LOG_DIR}/edit_gpu${gpu}_port${port}.log"
    CUDA_VISIBLE_DEVICES="${gpu}" \
      python scripts/tools/image_edit_server.py \
        --model "${EDIT_CKPT}" --port "${port}" \
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

run_pipeline_cmd() {
  # Ensure conda function is available, then switch to pipeline env
  set +u; source "${CONDA_INIT}" && conda activate "${CONDA_ENV_PIPELINE}"; set -u
  ATTN_BACKEND=flash_attn PARTCRAFT_CKPT_ROOT="${TRELLIS_CKPT_ROOT}" "$@"
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
  [[ -n "${LIMIT}" ]] && cmd+=(--limit "${LIMIT}")
  run_pipeline_cmd "${cmd[@]}"
  [[ "${KEEP_SERVERS}" == "1" ]] || stop_vlm_servers

  check_step_file "${SHARD_ROOT}/cache/phase0/semantic_labels_${RUN_TOKEN}.jsonl" "step1 labels"
  check_step_file "${SHARD_ROOT}/cache/phase1/edit_specs_${RUN_TOKEN}.jsonl" "step2 specs"
  check_step_file "${SHARD_ROOT}/pipeline/reports/step1_semantic.json" "step1 diagnostic"
  check_step_file "${SHARD_ROOT}/pipeline/reports/step2_planning.json" "step2 diagnostic"
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
  [[ -n "${LIMIT}" ]] && cmd+=(--limit "${LIMIT}")
  run_pipeline_cmd "${cmd[@]}"
  [[ "${KEEP_SERVERS}" == "1" ]] || stop_edit_servers

  check_step_file "${SHARD_ROOT}/cache/phase2_5/2d_edits_${RUN_TOKEN}/manifest.jsonl" "step3 manifest"
  check_step_file "${SHARD_ROOT}/pipeline/reports/step3_2d_edit.json" "step3 diagnostic"
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
  [[ -n "${LIMIT}" ]] && cmd+=(--limit "${LIMIT}")
  run_pipeline_cmd "${cmd[@]}"

  check_step_file "${SHARD_ROOT}/cache/phase2_5/edit_results_${RUN_TOKEN}.jsonl" "step4 results"
  check_step_file "${SHARD_ROOT}/pipeline/reports/step4_3d_edit.json" "step4 diagnostic"
}

run_step5() {
  local -a cmd=(
    python scripts/run_pipeline.py
    --config "${RUNTIME_CONFIG}"
    --shard "${SHARD}"
    --steps 5
  )
  [[ -n "${LIMIT}" ]] && cmd+=(--limit "${LIMIT}")
  run_pipeline_cmd "${cmd[@]}"
  check_step_file "${SHARD_ROOT}/pipeline/reports/step5_quality.json" "step5 diagnostic"
}

run_step6() {
  local -a cmd=(
    python scripts/run_pipeline.py
    --config "${RUNTIME_CONFIG}"
    --shard "${SHARD}"
    --steps 6
  )
  [[ -n "${LIMIT}" ]] && cmd+=(--limit "${LIMIT}")
  run_pipeline_cmd "${cmd[@]}"
  check_step_file "${SHARD_ROOT}/pipeline/reports/step6_export.json" "step6 diagnostic"
}

# ── Main ──────────────────────────────────────────────────────────
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
