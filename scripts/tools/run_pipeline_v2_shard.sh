#!/usr/bin/env bash
# pipeline_v2 phased scheduler — config-driven, GPU-count-agnostic.
#
# All scheduling decisions (which GPUs, which ports, which phases,
# which step needs which servers) come from the YAML config's
# `pipeline:` section. The shell only knows how to:
#   1. ask python "what does this config look like?"
#   2. start/stop server pools
#   3. invoke `python -m partcraft.pipeline_v2.run --phase <name>`
#
# Usage:
#   bash scripts/tools/run_pipeline_v2_shard.sh shard01 \
#        configs/pipeline_v2_shard01.yaml
#
#   PHASES="A,C,D,D2,E,F"  bash ... shard01 cfg     # custom subset
#   PHASES=A               bash ... shard01 cfg     # single phase
#   WITH_OPTIONAL=1        bash ... shard01 cfg     # include optional phases
#
# Each phase logs to logs/v2_<tag>/<phase>.log; the script aborts on
# the first non-zero exit and tears down any running server.

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
# Sets: GPUS=(...) VLM_PORTS=(...) FLUX_PORTS=(...) DEFAULT_STAGES/DEFAULT_PHASES=(...)
plan=$(
    "$PY_PIPE" -c "
import sys, yaml
from partcraft.pipeline_v2.scheduler import dump_shell_env
cfg = yaml.safe_load(open('$CFG'))
print(dump_shell_env(cfg))
"
)
eval "$plan"
# Backward compat if an older scheduler omitted *_STAGES arrays
if [ ${#ALL_STAGES[@]} -eq 0 ] && [ ${#ALL_PHASES[@]} -gt 0 ]; then
    ALL_STAGES=("${ALL_PHASES[@]}")
fi
if [ ${#DEFAULT_STAGES[@]} -eq 0 ] && [ ${#DEFAULT_PHASES[@]} -gt 0 ]; then
    DEFAULT_STAGES=("${DEFAULT_PHASES[@]}")
fi
N_GPUS=${#GPUS[@]}
N_VLM_SERVERS="${N_VLM_SERVERS:-$N_GPUS}"

# Stage selection: STAGES (preferred) > PHASES (deprecated) > default
if [ -n "${STAGES:-}" ]; then
    IFS=',' read -r -a SELECTED_PHASES <<< "$STAGES"
elif [ -n "${PHASES:-}" ]; then
    echo "[WARN] PHASES is deprecated; use STAGES" >&2
    IFS=',' read -r -a SELECTED_PHASES <<< "$PHASES"
elif [ "$WITH_OPTIONAL" = "1" ]; then
    SELECTED_PHASES=("${ALL_STAGES[@]}")
else
    SELECTED_PHASES=("${DEFAULT_STAGES[@]}")
fi

echo "============================================================"
echo "  pipeline_v2 phased run"
echo "============================================================"
echo "  tag         : $TAG"
echo "  config      : $CFG"
echo "  gpus        : ${GPUS[*]}  (N=$N_GPUS)"
echo "  vlm_ports   : ${VLM_PORTS[*]}"
echo "  flux_ports  : ${FLUX_PORTS[*]}"
echo "  stages      : ${SELECTED_PHASES[*]}"
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

run_pipeline_phase() {
    local phase="$1"
    local log="$LOG_DIR/phase${phase}.log"

    # Ask python what this phase needs
    eval "$(
        "$PY_PIPE" -c "
import yaml
from partcraft.pipeline_v2.scheduler import dump_shell_env
cfg = yaml.safe_load(open('$CFG'))
print(dump_shell_env(cfg, phase_name='$phase'))
"
    )"

    echo
    echo "▶ Phase ${PHASE_NAME} — ${PHASE_DESC}  (steps=${PHASE_STEPS[*]} servers=${PHASE_SERVERS} use_gpus=${PHASE_USE_GPUS})"

    case "$PHASE_SERVERS" in
        vlm)  start_vlm ;;
        flux) start_flux ;;
        none) ;;
        *) echo "[scheduler] unknown servers=$PHASE_SERVERS"; return 1 ;;
    esac

    ATTN_BACKEND=flash_attn \
    "$PY_PIPE" -m partcraft.pipeline_v2.run \
        --config "$CFG" \
        --shard "${TAG#shard}" \
        --all \
        --phase "$phase" \
        2>&1 | tee "$log"
    local rc=${PIPESTATUS[0]}

    case "$PHASE_SERVERS" in
        vlm)  stop_vlm ;;
        flux) stop_flux ;;
    esac

    if [ "$rc" != 0 ]; then
        echo "[scheduler] phase $phase exit=$rc — aborting"
        exit "$rc"
    fi
}

# ═══ MAIN LOOP ═══════════════════════════════════════════════════════
for phase in "${SELECTED_PHASES[@]}"; do
    run_pipeline_phase "$phase"
done

echo
echo "=== ALL PHASES DONE ==="
