#!/usr/bin/env bash
# pipeline_v2 stage scheduler — config-driven, GPU-count-agnostic.
#
# Scheduling (GPUs, ports, which servers each stage needs) comes from the
# YAML `pipeline:` + `services` sections. The shell:
#   1. asks Python for the resolved plan
#   2. starts/stops server pools
#   3. invokes `python -m partcraft.pipeline_v2.run --stage <name>`
#
# Usage:
#   bash scripts/tools/run_pipeline_v2_shard.sh shard01 \
#        configs/pipeline_v2_shard01.yaml
#
#   STAGES="A,C,D,D2,E,F"  bash ... shard01 cfg     # custom subset
#   STAGES=A               bash ... shard01 cfg     # single stage
#   WITH_OPTIONAL=1        bash ... shard01 cfg     # include optional stages
#
# Each stage logs to logs/v2_<tag>/<stage>.log; aborts on first failure.

set -euo pipefail

# ─── args ─────────────────────────────────────────────────────────────
TAG="${1:?usage: $0 <tag> <config> }"
CFG="${2:?usage: $0 <tag> <config> }"
WITH_OPTIONAL="${WITH_OPTIONAL:-0}"

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
[ -f "$CFG" ] || { echo "missing $CFG"; exit 1; }

# ─── env from machine/<hostname>.env ─────────────────────────────────
ENV_FILE="${MACHINE_ENV:-configs/machine/$(hostname).env}"
[ -f "$ENV_FILE" ] || { echo "[ERROR] Machine config not found: ${ENV_FILE}"; echo "  Create it from: configs/machine/node39.env"; exit 1; }
# shellcheck disable=SC1090
source "$ENV_FILE"

CONDA_INIT="${CONDA_INIT:?CONDA_INIT not set in ${ENV_FILE}}"
CONDA_ENV_SERVER="${CONDA_ENV_SERVER:?CONDA_ENV_SERVER not set in ${ENV_FILE}}"
CONDA_ENV_PIPELINE="${CONDA_ENV_PIPELINE:?CONDA_ENV_PIPELINE not set in ${ENV_FILE}}"
VLM_CKPT="${VLM_CKPT:?VLM_CKPT not set in ${ENV_FILE}}"
EDIT_CKPT="${EDIT_CKPT:?EDIT_CKPT not set in ${ENV_FILE}}"

# Resolve Python binaries via conda environments
# shellcheck disable=SC1090
set +u; source "${CONDA_INIT}"; set -u
PY_PIPE="$(conda run -n "${CONDA_ENV_PIPELINE}" which python 2>/dev/null)" \
    || PY_PIPE="${CONDA_PREFIX:-/root/miniconda3}/envs/${CONDA_ENV_PIPELINE}/bin/python"
PY_SRV="$(conda run -n "${CONDA_ENV_SERVER}" which python 2>/dev/null)" \
    || PY_SRV="${CONDA_PREFIX:-/root/miniconda3}/envs/${CONDA_ENV_SERVER}/bin/python"
[ -x "$PY_PIPE" ] || { echo "[ERROR] Pipeline python not found: $PY_PIPE"; exit 1; }
[ -x "$PY_SRV"  ] || { echo "[ERROR] Server python not found: $PY_SRV";   exit 1; }

LOG_DIR="logs/v2_${TAG}"
mkdir -p "$LOG_DIR"

# ─── ask python for the run plan ─────────────────────────────────────
# Sets: GPUS=(...) VLM_PORTS=(...) FLUX_PORTS=(...) DEFAULT_STAGES/ALL_STAGES=(...)
plan=$(
    "$PY_PIPE" -c "
import sys, yaml
from partcraft.pipeline_v2.scheduler import dump_shell_env
cfg = yaml.safe_load(open('$CFG'))
print(dump_shell_env(cfg))
"
)
eval "$plan"
N_GPUS=${#GPUS[@]}
N_VLM_SERVERS="${N_VLM_SERVERS:-$N_GPUS}"

# Stage selection: env STAGES override > default list from config
if [ -n "${STAGES:-}" ]; then
    IFS=',' read -r -a SELECTED_STAGES <<< "$STAGES"
elif [ "$WITH_OPTIONAL" = "1" ]; then
    SELECTED_STAGES=("${ALL_STAGES[@]}")
else
    SELECTED_STAGES=("${DEFAULT_STAGES[@]}")
fi

echo "============================================================"
echo "  pipeline_v2 phased run"
echo "============================================================"
echo "  tag         : $TAG"
echo "  config      : $CFG"
echo "  gpus        : ${GPUS[*]}  (N=$N_GPUS)"
echo "  vlm_ports   : ${VLM_PORTS[*]}"
echo "  flux_ports  : ${FLUX_PORTS[*]}"
echo "  stages      : ${SELECTED_STAGES[*]}"
echo "  log dir     : $LOG_DIR"
echo "============================================================"

# ─── server lifecycle ────────────────────────────────────────────────

start_vlm() {
    echo "[VLM] starting ${N_VLM_SERVERS} servers (of ${N_GPUS} total GPUs)"
    : > "$LOG_DIR/vlm.pids"
    for i in $(seq 0 $((N_VLM_SERVERS-1))); do
        local gpu="${GPUS[i]}" port="${VLM_PORTS[i]}"
        local log="$LOG_DIR/vlm_${port}.log"
        echo "  GPU $gpu -> port $port"
        set +u; source "${CONDA_INIT}" && conda activate "${CONDA_ENV_SERVER}"; set -u
        CUDA_VISIBLE_DEVICES="$gpu" \
        VLM_MODEL="$VLM_CKPT" \
        VLM_PORT="$port" \
        VLM_TP=1 \
        SGLANG_DISABLE_CUDNN_CHECK=1 \
            bash scripts/tools/launch_local_vlm.sh \
                > "$log" 2>&1 &
        echo $! >> "$LOG_DIR/vlm.pids"
    done
    local deadline=$(( $(date +%s) + 900 ))
    for port in "${VLM_PORTS[@]}"; do
        while :; do
            if curl -s -m 2 "http://localhost:${port}/v1/models" \
                    >/dev/null 2>&1; then
                echo "[VLM] :${port} ready"; break
            fi
            if [ "$(date +%s)" -gt "$deadline" ]; then
                echo "[VLM] :${port} TIMEOUT"
                tail -30 "$LOG_DIR/vlm_${port}.log" || true
                stop_vlm; return 1
            fi
            sleep 5
        done
    done
    echo "[VLM] all ${N_GPUS} servers ready"
}

stop_vlm() {
    if [ -f "$LOG_DIR/vlm.pids" ]; then
        echo "[VLM] killing all servers"
        while read -r pid; do kill -9 "$pid" 2>/dev/null || true; done \
            < "$LOG_DIR/vlm.pids"
        pkill -9 -f "sglang.launch_server.*${VLM_CKPT}" 2>/dev/null || true
        rm -f "$LOG_DIR/vlm.pids"
        sleep 2
    fi
}

start_flux() {
    echo "[FLUX] starting ${N_GPUS} servers"
    : > "$LOG_DIR/flux.pids"
    for i in $(seq 0 $((N_GPUS-1))); do
        local gpu="${GPUS[i]}" port="${FLUX_PORTS[i]}"
        local log="$LOG_DIR/flux_${port}.log"
        echo "  GPU $gpu -> port $port"
        set +u; source "${CONDA_INIT}" && conda activate "${CONDA_ENV_SERVER}"; set -u
        CUDA_VISIBLE_DEVICES="$gpu" \
            "$PY_SRV" scripts/tools/image_edit_server.py \
                --model "$EDIT_CKPT" --port "$port" \
                > "$log" 2>&1 &
        echo $! >> "$LOG_DIR/flux.pids"
    done
    local deadline=$(( $(date +%s) + 900 ))
    for port in "${FLUX_PORTS[@]}"; do
        while :; do
            if curl -s -m 2 -o /dev/null -w "%{http_code}" \
                    "http://localhost:${port}/health" 2>/dev/null \
                    | grep -q "200"; then
                echo "[FLUX] :${port} ready"; break
            fi
            if [ "$(date +%s)" -gt "$deadline" ]; then
                echo "[FLUX] :${port} TIMEOUT"
                tail -20 "$LOG_DIR/flux_${port}.log" || true
                stop_flux; return 1
            fi
            sleep 5
        done
    done
    echo "[FLUX] all ${N_GPUS} servers ready"
}

stop_flux() {
    if [ -f "$LOG_DIR/flux.pids" ]; then
        echo "[FLUX] killing all servers"
        while read -r pid; do kill -9 "$pid" 2>/dev/null || true; done \
            < "$LOG_DIR/flux.pids"
        pkill -9 -f "image_edit_server.py" 2>/dev/null || true
        rm -f "$LOG_DIR/flux.pids"
        sleep 2
    fi
}

cleanup_all() { stop_vlm; stop_flux; }
trap cleanup_all EXIT

# ─── per-phase invocation ────────────────────────────────────────────


run_pipeline_stage() {
    local stage="$1"
    local log="$LOG_DIR/stage_${stage}.log"

    eval "$(
        "$PY_PIPE" -c "
import yaml
from partcraft.pipeline_v2.scheduler import dump_shell_env
cfg = yaml.safe_load(open('$CFG'))
print(dump_shell_env(cfg, stage_name='$stage'))
"
    )"

    echo
    echo "▶ Stage ${STAGE_NAME} — ${STAGE_DESC}  (steps=${STAGE_STEPS[*]} servers=${STAGE_SERVERS} use_gpus=${STAGE_USE_GPUS})"

    # Pre-check: count objects with pending work before starting servers
    _SERVERS_STARTED=0
    if [ "$STAGE_SERVERS" != "none" ]; then
        _PENDING=$(LIMIT="${LIMIT:-}" "$PY_PIPE" -m partcraft.pipeline_v2.run             --config "$CFG" --shard "${TAG#shard}" --all             --stage "$stage" --count-pending 2>/dev/null || echo 1)
        if [ "${_PENDING}" = "0" ]; then
            echo "[scheduler] stage $stage: all objects already complete — skipping server startup"
        else
            echo "[scheduler] stage $stage: ${_PENDING} objects pending"
            case "$STAGE_SERVERS" in
                vlm)  start_vlm  && _SERVERS_STARTED=1 ;;
                flux) start_flux && _SERVERS_STARTED=1 ;;
                *) echo "[scheduler] unknown servers=$STAGE_SERVERS"; return 1 ;;
            esac
        fi
    fi

    # OBJ_IDS env var: space-separated object IDs to process instead of --all
    if [ -n "${OBJ_IDS:-}" ]; then
        read -ra _OBJ_FLAG <<< "$OBJ_IDS"
        _OBJ_FLAG=(--obj-ids "${_OBJ_FLAG[@]}")
    else
        _OBJ_FLAG=(--all)
    fi

    ATTN_BACKEND=flash_attn \
    "$PY_PIPE" -m partcraft.pipeline_v2.run \
        --config "$CFG" \
        --shard "${TAG#shard}" \
        "${_OBJ_FLAG[@]}" \
        --stage "$stage" \
        2>&1 | tee "$log"
    local rc=${PIPESTATUS[0]}

    if [ "$_SERVERS_STARTED" = "1" ]; then
        case "$STAGE_SERVERS" in
            vlm)  stop_vlm ;;
            flux) stop_flux ;;
        esac
    fi

    if [ "$rc" != 0 ]; then
        echo "[scheduler] stage $stage exit=$rc — aborting"
        exit "$rc"
    fi
}

# ═══ MAIN LOOP ═══════════════════════════════════════════════════════
_stage_idx=0
while [ "$_stage_idx" -lt "${#SELECTED_STAGES[@]}" ]; do
    stage="${SELECTED_STAGES[$_stage_idx]}"
    run_pipeline_stage "$stage"
    _stage_idx=$(( _stage_idx + 1 ))
done

echo
echo "=== ALL STAGES DONE ==="
