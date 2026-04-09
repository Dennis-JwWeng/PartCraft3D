#!/usr/bin/env bash
# Smoke test: run full pipeline (all phases) for a fixed list of obj_ids.
# Mirrors run_pipeline_v2_shard.sh but uses --obj-ids instead of --all,
# so you can test 2 objects without waiting for the whole shard.
#
# Usage:
#   bash scripts/tools/run_pipeline_v2_smoke.sh <tag> <config> <obj_id> [obj_id ...]
# Example:
#   bash scripts/tools/run_pipeline_v2_smoke.sh smoke00 configs/pipeline_v2_gpu01.yaml \
#       0008dc75fb3648f2af4ca8c4d711e53e 000ec112ae7f4a8a93f847ccfd4031be
#
# Env overrides (same as the main shard script):
#   STAGES=A,C,D     override phase list
#   MACHINE_ENV=...  point to a specific machine env file

set -euo pipefail

TAG="${1:?usage: $0 <tag> <config> <obj_id> [obj_id ...]}"
CFG="${2:?usage: $0 <tag> <config> <obj_id> [obj_id ...]}"
shift 2
OBJ_IDS=("$@")
[ "${#OBJ_IDS[@]}" -gt 0 ] || { echo "ERROR: provide at least one obj_id"; exit 1; }

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
[ -f "$CFG" ] || { echo "missing $CFG"; exit 1; }

# Derive shard from tag (strip leading non-digits, zero-pad to 2)
SHARD=$(echo "$TAG" | grep -o '[0-9]*' | tail -1)
SHARD=$(printf "%02d" "${SHARD:-0}")

# ── machine env ──────────────────────────────────────────────────────
ENV_FILE="${MACHINE_ENV:-configs/machine/$(hostname).env}"
[ -f "$ENV_FILE" ] || { echo "[ERROR] Machine config not found: ${ENV_FILE}"; exit 1; }
source "$ENV_FILE"

CONDA_INIT="${CONDA_INIT:?}"
CONDA_ENV_SERVER="${CONDA_ENV_SERVER:?}"
CONDA_ENV_PIPELINE="${CONDA_ENV_PIPELINE:?}"
VLM_CKPT="${VLM_CKPT:?}"
EDIT_CKPT="${EDIT_CKPT:?}"

set +u; source "${CONDA_INIT}"; set -u
PY_PIPE="$(conda run -n "${CONDA_ENV_PIPELINE}" which python 2>/dev/null)" \
    || PY_PIPE="${CONDA_PREFIX:-/root/miniconda3}/envs/${CONDA_ENV_PIPELINE}/bin/python"
PY_SRV="$(conda run -n "${CONDA_ENV_SERVER}" which python 2>/dev/null)" \
    || PY_SRV="${CONDA_PREFIX:-/root/miniconda3}/envs/${CONDA_ENV_SERVER}/bin/python"
[ -x "$PY_PIPE" ] || { echo "[ERROR] Pipeline python not found: $PY_PIPE"; exit 1; }
[ -x "$PY_SRV"  ] || { echo "[ERROR] Server python not found: $PY_SRV";   exit 1; }

LOG_DIR="logs/smoke_${TAG}"
mkdir -p "$LOG_DIR"
mkdir -p "$LOG_DIR"

# ── scheduling plan ──────────────────────────────────────────────────
plan=$("$PY_PIPE" -c "
import yaml
from partcraft.pipeline_v2.scheduler import dump_shell_env
cfg = yaml.safe_load(open('$CFG'))
print(dump_shell_env(cfg))
")
eval "$plan"
N_GPUS=${#GPUS[@]}
N_VLM_SERVERS="${N_VLM_SERVERS:-$N_GPUS}"

if [ -n "${STAGES:-}" ]; then
    IFS=',' read -r -a SELECTED_STAGES <<< "$STAGES"
else
    SELECTED_STAGES=("${DEFAULT_STAGES[@]}")
fi

OBJ_IDS_STR="${OBJ_IDS[*]}"

echo "============================================================"
echo "  pipeline_v2 SMOKE TEST"
echo "============================================================"
echo "  tag         : $TAG"
echo "  config      : $CFG"
echo "  shard       : $SHARD"
echo "  obj_ids     : ${OBJ_IDS[*]}"
echo "  gpus        : ${GPUS[*]}  (N=$N_GPUS)"
echo "  vlm_ports   : ${VLM_PORTS[*]}"
echo "  flux_ports  : ${FLUX_PORTS[*]}"
echo "  stages      : ${SELECTED_STAGES[*]}"
echo "  log dir     : $LOG_DIR"
echo "============================================================"

# ── server helpers (same as shard script) ────────────────────────────
start_vlm() {
    echo "[VLM] starting ${N_VLM_SERVERS} servers"
    : > "$LOG_DIR/vlm.pids"
    for i in $(seq 0 $((N_VLM_SERVERS-1))); do
        local gpu="${GPUS[i]}" port="${VLM_PORTS[i]}"
        echo "  GPU $gpu -> :$port  log=$LOG_DIR/vlm_${port}.log"
        set +u; source "${CONDA_INIT}" && conda activate "${CONDA_ENV_SERVER}"; set -u
        CUDA_VISIBLE_DEVICES="$gpu" VLM_MODEL="$VLM_CKPT" VLM_PORT="$port" VLM_TP=1 \
        SGLANG_DISABLE_CUDNN_CHECK=1 \
            bash scripts/tools/launch_local_vlm.sh > "$LOG_DIR/vlm_${port}.log" 2>&1 &
        echo $! >> "$LOG_DIR/vlm.pids"
    done
    local deadline=$(( $(date +%s) + 900 ))
    for port in "${VLM_PORTS[@]}"; do
        while :; do
            curl -s -m 2 "http://localhost:${port}/v1/models" >/dev/null 2>&1 && { echo "[VLM] :$port ready"; break; }
            [ "$(date +%s)" -gt "$deadline" ] && { echo "[VLM] :$port TIMEOUT"; tail -30 "$LOG_DIR/vlm_${port}.log"; stop_vlm; return 1; }
            sleep 5
        done
    done
    echo "[VLM] all ready"
}
stop_vlm() {
    [ -f "$LOG_DIR/vlm.pids" ] || return 0
    echo "[VLM] killing servers"
    while read -r pid; do kill -9 "$pid" 2>/dev/null || true; done < "$LOG_DIR/vlm.pids"
    pkill -9 -f "sglang.launch_server.*${VLM_CKPT}" 2>/dev/null || true
    rm -f "$LOG_DIR/vlm.pids"; sleep 2
}
start_flux() {
    echo "[FLUX] starting ${N_GPUS} servers"
    : > "$LOG_DIR/flux.pids"
    for i in $(seq 0 $((N_GPUS-1))); do
        local gpu="${GPUS[i]}" port="${FLUX_PORTS[i]}"
        echo "  GPU $gpu -> :$port  log=$LOG_DIR/flux_${port}.log"
        set +u; source "${CONDA_INIT}" && conda activate "${CONDA_ENV_SERVER}"; set -u
        CUDA_VISIBLE_DEVICES="$gpu" "$PY_SRV" scripts/tools/image_edit_server.py \
            --model "$EDIT_CKPT" --port "$port" > "$LOG_DIR/flux_${port}.log" 2>&1 &
        echo $! >> "$LOG_DIR/flux.pids"
    done
    local deadline=$(( $(date +%s) + 900 ))
    for port in "${FLUX_PORTS[@]}"; do
        while :; do
            curl -s -m 2 -o /dev/null -w "%{http_code}" "http://localhost:${port}/health" 2>/dev/null | grep -q "200" && { echo "[FLUX] :$port ready"; break; }
            [ "$(date +%s)" -gt "$deadline" ] && { echo "[FLUX] :$port TIMEOUT"; tail -20 "$LOG_DIR/flux_${port}.log"; stop_flux; return 1; }
            sleep 5
        done
    done
    echo "[FLUX] all ready"
}
stop_flux() {
    [ -f "$LOG_DIR/flux.pids" ] || return 0
    echo "[FLUX] killing servers"
    while read -r pid; do kill -9 "$pid" 2>/dev/null || true; done < "$LOG_DIR/flux.pids"
    pkill -9 -f "image_edit_server.py" 2>/dev/null || true
    rm -f "$LOG_DIR/flux.pids"; sleep 2
}
cleanup_all() { stop_vlm; stop_flux; }
trap cleanup_all EXIT

# ── per-phase runner ─────────────────────────────────────────────────
run_stage() {
    local stage="$1"
    local log="$LOG_DIR/stage_${stage}.log"
    eval "$("$PY_PIPE" -c "
import yaml
from partcraft.pipeline_v2.scheduler import dump_shell_env
cfg = yaml.safe_load(open('$CFG'))
print(dump_shell_env(cfg, stage_name='$stage'))
")"
    echo
    echo "▶ Stage ${STAGE_NAME} — ${STAGE_DESC}  (steps=${STAGE_STEPS[*]} servers=${STAGE_SERVERS} gpus=${STAGE_USE_GPUS})"
    case "$STAGE_SERVERS" in vlm) start_vlm ;; flux) start_flux ;; esac

    local gpus_flag=""
    [ "${STAGE_USE_GPUS}" = "1" ] && gpus_flag="--gpus $(IFS=,; echo "${GPUS[*]}")"

    ATTN_BACKEND=flash_attn \
    "$PY_PIPE" -m partcraft.pipeline_v2.run \
        --config "$CFG" \
        --shard "$SHARD" \
        --obj-ids ${OBJ_IDS_STR} \
        --stage "$stage" \
        $gpus_flag \
        2>&1 | tee "$log"
    local rc=${PIPESTATUS[0]}

    case "$STAGE_SERVERS" in vlm) stop_vlm ;; flux) stop_flux ;; esac
    [ "$rc" = "0" ] || { echo "[smoke] stage $stage exit=$rc — abort"; exit "$rc"; }
}

# ── MAIN ─────────────────────────────────────────────────────────────
for stage in "${SELECTED_STAGES[@]}"; do
    run_stage "$stage"
done
echo
echo "=== SMOKE TEST DONE (${#OBJ_IDS[@]} objects, stages: ${SELECTED_STAGES[*]}) ==="
