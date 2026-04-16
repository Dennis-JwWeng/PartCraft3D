#!/usr/bin/env bash
# pipeline_v3 stage scheduler — config-driven, GPU-count-agnostic.
#
# Mirrors scripts/tools/run_pipeline_v2_shard.sh but targets pipeline_v3.
# Scheduling (GPUs, ports, server pools, parallel batches) is driven entirely
# by the YAML pipeline: + services: sections via partcraft.pipeline_v3.scheduler.
#
# The shell is responsible only for:
#   1. Reading the resolved plan from Python (GPUs, ports, stage order)
#   2. Starting / stopping VLM and FLUX server pools per stage
#   3. Invoking `python -m partcraft.pipeline_v3.run --stage <name>`
#   4. Running parallel stage batches concurrently (& + wait)
#
# GPU-bound steps (trellis_3d, preview_flux, render_3d) are dispatched
# internally by pipeline_v3.run via dispatch_gpus() — the shell does NOT
# need to fork per-GPU subprocesses for those steps.
#
# Usage:
#   bash scripts/tools/run_pipeline_v3_shard.sh <tag> <config.yaml>
#
# Env overrides (all optional):
#   OBJ_IDS_FILE=<path>   Limit run to object IDs listed in this file
#   STAGES="a,b,c"        Comma-separated subset of stage names to run
#   FORCE=1               Re-run already-completed steps
#   LIMIT=N               Cap objects processed (useful for smoke tests)
#   MACHINE_ENV=<path>    Override machine env file
#   VLM_MEM_FRAC=0.57     VLM VRAM fraction passed to SGLang
#
# Each stage logs to logs/v3_<tag>/<stage>.log.
# Pipeline aborts on the first stage failure and shows the relevant log tail.

set -euo pipefail

# ─── args ────────────────────────────────────────────────────────────
TAG="${1:?usage: $0 <tag> <config.yaml>}"
CFG="${2:?usage: $0 <tag> <config.yaml>}"

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
[ -f "$CFG" ] || { echo "[ERROR] config not found: $CFG"; exit 1; }

# ─── machine env ─────────────────────────────────────────────────────
ENV_FILE="${MACHINE_ENV:-configs/machine/$(hostname).env}"
[ -f "$ENV_FILE" ] || {
    echo "[ERROR] Machine config not found: ${ENV_FILE}"
    echo "  Create from: configs/machine/H200.env"
    exit 1
}
# shellcheck disable=SC1090
source "$ENV_FILE"

: "${CONDA_INIT:?CONDA_INIT not set in ${ENV_FILE}}"
: "${CONDA_ENV_SERVER:?CONDA_ENV_SERVER not set in ${ENV_FILE}}"
: "${CONDA_ENV_PIPELINE:?CONDA_ENV_PIPELINE not set in ${ENV_FILE}}"
: "${VLM_CKPT:?VLM_CKPT not set in ${ENV_FILE}}"
: "${EDIT_CKPT:?EDIT_CKPT not set in ${ENV_FILE}}"

# Resolve Python binaries via conda environments.
# shellcheck disable=SC1090
set +u; source "${CONDA_INIT}"; set -u
PY_PIPE="$(conda run -n "${CONDA_ENV_PIPELINE}" which python 2>/dev/null)" \
    || PY_PIPE="${CONDA_PREFIX:-/root/miniconda3}/envs/${CONDA_ENV_PIPELINE}/bin/python"
PY_SRV="$(conda run -n "${CONDA_ENV_SERVER}" which python 2>/dev/null)" \
    || PY_SRV="${CONDA_PREFIX:-/root/miniconda3}/envs/${CONDA_ENV_SERVER}/bin/python"
[ -x "$PY_PIPE" ] || { echo "[ERROR] Pipeline python not found: $PY_PIPE"; exit 1; }
[ -x "$PY_SRV"  ] || { echo "[ERROR] Server python not found: $PY_SRV";   exit 1; }

LOG_DIR="logs/v3_${TAG}"
mkdir -p "$LOG_DIR"

# ─── resolve GPU/port plan from Python ───────────────────────────────
plan=$(
    "$PY_PIPE" -c "
import sys, yaml
from partcraft.pipeline_v3.scheduler import dump_shell_env
cfg = yaml.safe_load(open('$CFG'))
print(dump_shell_env(cfg))
"
)
eval "$plan"
N_GPUS=${#GPUS[@]}

# ─── stage selection ─────────────────────────────────────────────────
if [ -n "${STAGES:-}" ]; then
    IFS=',' read -r -a SELECTED_STAGES <<< "$STAGES"
else
    SELECTED_STAGES=("${DEFAULT_STAGES[@]}")
fi

# ─── object-ids flag: OBJ_IDS_FILE env → --obj-ids-file; else --all ─
if [ -n "${OBJ_IDS_FILE:-}" ]; then
    [ -f "$OBJ_IDS_FILE" ] || { echo "[ERROR] OBJ_IDS_FILE not found: $OBJ_IDS_FILE"; exit 1; }
    _OBJ_FLAG=(--obj-ids-file "$OBJ_IDS_FILE")
else
    _OBJ_FLAG=(--all)
fi

# FORCE=1 → --force
_FORCE_FLAG=()
[ "${FORCE:-0}" = "1" ] && _FORCE_FLAG=(--force)

# Shard: strip leading "shard" prefix (tag="shard08" → shard="08")
SHARD="${TAG#shard}"

echo "============================================================"
echo "  pipeline_v3 shard run"
echo "============================================================"
echo "  tag         : $TAG"
echo "  shard       : $SHARD"
echo "  config      : $CFG"
echo "  gpus        : ${GPUS[*]}  (N=$N_GPUS)"
echo "  vlm_ports   : ${VLM_PORTS[*]}"
echo "  flux_ports  : ${FLUX_PORTS[*]}"
echo "  stages      : ${SELECTED_STAGES[*]}"
echo "  obj scope   : ${OBJ_IDS_FILE:-<all objects>}"
echo "  log dir     : $LOG_DIR"
echo "============================================================"

# ─── ANSI colours (disabled when stdout is not a tty) ────────────────
if [ -t 1 ]; then
    _RED='\033[0;31m'; _YEL='\033[1;33m'; _CYN='\033[0;36m'
    _GRN='\033[0;32m'; _BOLD='\033[1m';   _RST='\033[0m'
else
    _RED=''; _YEL=''; _CYN=''; _GRN=''; _BOLD=''; _RST=''
fi

# ─── server lifecycle ────────────────────────────────────────────────

start_vlm() {
    echo "[VLM] starting ${N_VLM_SERVERS} servers"
    : > "$LOG_DIR/vlm.pids"
    for i in $(seq 0 $((N_VLM_SERVERS - 1))); do
        local gpu="${GPUS[$i]}" port="${VLM_PORTS[$i]}"
        local log="$LOG_DIR/vlm_${port}.log"
        printf "  GPU %-3s -> port %s\n" "$gpu" "$port"
        (
            set +u; source "${CONDA_INIT}" && conda activate "${CONDA_ENV_SERVER}"; set -u
            CUDA_VISIBLE_DEVICES="$gpu" \
            VLM_MODEL="$VLM_CKPT" \
            VLM_PORT="$port" \
            VLM_TP=1 \
            VLM_MEM_FRAC="${VLM_MEM_FRAC:-0.57}" \
            SGLANG_DISABLE_CUDNN_CHECK=1 \
                bash scripts/tools/launch_local_vlm.sh
        ) > "$log" 2>&1 &
        echo $! >> "$LOG_DIR/vlm.pids"
    done
    local deadline=$(( $(date +%s) + 900 ))
    for port in "${VLM_PORTS[@]}"; do
        while :; do
            if curl -s -m 2 "http://localhost:${port}/v1/models" >/dev/null 2>&1; then
                echo "[VLM] :${port} ready"; break
            fi
            if [ "$(date +%s)" -gt "$deadline" ]; then
                echo "[VLM] :${port} TIMEOUT after 15 min"
                tail -30 "$LOG_DIR/vlm_${port}.log" || true
                stop_vlm; return 1
            fi
            sleep 5
        done
    done
    echo "[VLM] all ${N_VLM_SERVERS} servers ready"
}

stop_vlm() {
    if [ -f "$LOG_DIR/vlm.pids" ]; then
        echo "[VLM] stopping servers"
        while read -r pid; do kill -9 "$pid" 2>/dev/null || true; done \
            < "$LOG_DIR/vlm.pids"
        pkill -9 -f "sglang.launch_server" 2>/dev/null || true
        rm -f "$LOG_DIR/vlm.pids"
        sleep 2
    fi
}

start_flux() {
    echo "[FLUX] starting $N_GPUS servers"
    : > "$LOG_DIR/flux.pids"
    for i in $(seq 0 $((N_GPUS - 1))); do
        local gpu="${GPUS[$i]}" port="${FLUX_PORTS[$i]}"
        local log="$LOG_DIR/flux_${port}.log"
        printf "  GPU %-3s -> port %s\n" "$gpu" "$port"
        (
            set +u; source "${CONDA_INIT}" && conda activate "${CONDA_ENV_SERVER}"; set -u
            CUDA_VISIBLE_DEVICES="$gpu" \
                "$PY_SRV" scripts/tools/image_edit_server.py \
                    --model "$EDIT_CKPT" --port "$port"
        ) > "$log" 2>&1 &
        echo $! >> "$LOG_DIR/flux.pids"
    done
    local deadline=$(( $(date +%s) + 600 ))
    for port in "${FLUX_PORTS[@]}"; do
        while :; do
            if curl -s -m 2 -o /dev/null -w "%{http_code}" \
                    "http://localhost:${port}/health" 2>/dev/null | grep -q "200"; then
                echo "[FLUX] :${port} ready"; break
            fi
            if [ "$(date +%s)" -gt "$deadline" ]; then
                echo "[FLUX] :${port} TIMEOUT after 10 min"
                tail -20 "$LOG_DIR/flux_${port}.log" || true
                stop_flux; return 1
            fi
            sleep 5
        done
    done
    echo "[FLUX] all $N_GPUS servers ready"
}

stop_flux() {
    if [ -f "$LOG_DIR/flux.pids" ]; then
        echo "[FLUX] stopping servers"
        while read -r pid; do kill -9 "$pid" 2>/dev/null || true; done \
            < "$LOG_DIR/flux.pids"
        pkill -9 -f "image_edit_server.py" 2>/dev/null || true
        rm -f "$LOG_DIR/flux.pids"
        sleep 2
    fi
}

cleanup_all() { stop_vlm; stop_flux; }
trap cleanup_all EXIT

# ─── error display ───────────────────────────────────────────────────

show_stage_errors() {
    local log_file="$1" stage_name="$2"
    [ -f "$log_file" ] || return
    local width=72
    local bar; bar=$(printf '%*s' "$width" '' | tr ' ' '-')
    printf "${_RED}+%s+${_RST}\n" "$bar"
    printf "${_RED}|${_RST}  ${_BOLD}${_RED}STAGE FAILED: %-$((width - 14))s${_RST}${_RED}|${_RST}\n" "$stage_name"
    printf "${_RED}|${_RST}  log: %-$((width - 7))s${_RED}|${_RST}\n" "$log_file"
    printf "${_RED}+%s+${_RST}\n" "$bar"
    tail -60 "$log_file" | while IFS= read -r line; do
        if echo "$line" | grep -qE '(Traceback|TypeError|Error:|Exception:|FAILED|exit=[^0])'; then
            printf "${_RED}|${_RST}  ${_RED}%s${_RST}\n" "$line"
        elif echo "$line" | grep -qE '(WARNING|warn)'; then
            printf "${_RED}|${_RST}  ${_YEL}%s${_RST}\n" "$line"
        else
            printf "${_RED}|${_RST}  %s\n" "$line"
        fi
    done
    printf "${_RED}+%s+${_RST}\n\n" "$bar"
}

# ─── live heartbeat for parallel batches ─────────────────────────────
# Prints the last log line of each running stage every N seconds so the
# user sees progress without interleaved output.

_live_monitor() {
    local interval="$1"; shift
    local -a log_files names
    while [ "$#" -ge 2 ]; do log_files+=("$1"); names+=("$2"); shift 2; done
    while true; do
        sleep "$interval" || return
        local ts; ts=$(date '+%H:%M:%S')
        for i in "${!log_files[@]}"; do
            local lf="${log_files[$i]}" nm="${names[$i]}"
            if [ -f "$lf" ]; then
                local last; last=$(tail -1 "$lf" 2>/dev/null | sed 's/\[[0-9;]*m//g')
                printf "${_CYN}[%s %-20s]${_RST} %s\n" "$ts" "$nm" "$last"
            else
                printf "${_CYN}[%s %-20s]${_RST} (waiting for log...)\n" "$ts" "$nm"
            fi
        done
    done
}

# ─── helper: fetch stage metadata into shell vars ────────────────────

_load_stage_meta() {
    # Populates STAGE_NAME STAGE_DESC STAGE_SERVERS STAGE_STEPS STAGE_USE_GPUS
    local stage="$1"
    eval "$(
        "$PY_PIPE" -c "
import yaml
from partcraft.pipeline_v3.scheduler import dump_shell_env
cfg = yaml.safe_load(open('$CFG'))
print(dump_shell_env(cfg, stage_name='$stage'))
"
    )"
}

# ─── single-stage invocation ─────────────────────────────────────────

run_stage() {
    # run_stage <stage_name> [servers_already_started=0|1]
    # When servers_already_started=1 (called from a parallel batch that
    # already started servers), skip the server lifecycle entirely.
    local stage="$1"
    local servers_up="${2:-0}"
    local log="$LOG_DIR/stage_${stage}.log"

    _load_stage_meta "$stage"

    printf "\n${_BOLD}> Stage %-24s — %s${_RST}  (steps=[%s] servers=%s gpu=%s)\n" \
        "$STAGE_NAME" "$STAGE_DESC" "${STAGE_STEPS[*]}" "$STAGE_SERVERS" "$STAGE_USE_GPUS"

    # Pre-check: skip server startup if no objects have pending work.
    local _started=0
    if [ "$servers_up" = "0" ] && [ "$STAGE_SERVERS" != "none" ]; then
        local _pending
        _pending=$(
            LIMIT="${LIMIT:-}" \
            "$PY_PIPE" -m partcraft.pipeline_v3.run \
                --config "$CFG" --shard "$SHARD" \
                "${_OBJ_FLAG[@]}" --stage "$stage" \
                --count-pending 2>/dev/null
        ) || _pending=1

        if [ "${_pending}" = "0" ]; then
            echo "[scheduler] stage $stage: all objects done — skipping server startup"
        else
            printf "[scheduler] stage %s: %s objects pending\n" "$stage" "$_pending"
            case "$STAGE_SERVERS" in
                vlm)  start_vlm  && _started=1 ;;
                flux) start_flux && _started=1 ;;
                *)    echo "[scheduler] unknown servers=$STAGE_SERVERS"; return 1 ;;
            esac
        fi
    fi

    LIMIT="${LIMIT:-}" \
    ATTN_BACKEND="${ATTN_BACKEND:-flash_attn}" \
    "$PY_PIPE" -m partcraft.pipeline_v3.run \
        --config "$CFG" \
        --shard "$SHARD" \
        "${_OBJ_FLAG[@]}" \
        "${_FORCE_FLAG[@]}" \
        --stage "$stage" \
        2>&1 | tee "$log"
    local rc=${PIPESTATUS[0]}

    if [ "$_started" = "1" ]; then
        case "$STAGE_SERVERS" in
            vlm)  stop_vlm ;;
            flux) stop_flux ;;
        esac
    fi

    if [ "$rc" != "0" ]; then
        show_stage_errors "$log" "$stage"
        echo "[scheduler] stage $stage exit=$rc — aborting"
        exit "$rc"
    fi
}

# ─── parallel batch executor ─────────────────────────────────────────

run_parallel_batch() {
    # run_parallel_batch <stage1> [<stage2> ...]
    # 1 stage  → serial (run_stage).
    # N stages → start servers for the batch, fork all stages in parallel,
    #             live-monitor progress, collect exit codes, stop servers.
    if [ "$#" -eq 1 ]; then
        run_stage "$1"
        return $?
    fi

    local names=("$@") pids=() _any_fail=0

    printf "\n${_BOLD}> Parallel batch [%s] — launching${_RST}\n" "${names[*]}"

    # Determine what servers the batch needs (union across all stages).
    local batch_servers="none"
    local server_stage=""
    for _s in "${names[@]}"; do
        _load_stage_meta "$_s"
        if [ "$STAGE_SERVERS" != "none" ]; then
            batch_servers="$STAGE_SERVERS"
            server_stage="$_s"
        fi
    done

    # Start servers for the whole batch once (pending precheck on server stage).
    local _batch_servers_up=0
    if [ "$batch_servers" != "none" ] && [ -n "$server_stage" ]; then
        local _pending
        _pending=$(
            LIMIT="${LIMIT:-}" \
            "$PY_PIPE" -m partcraft.pipeline_v3.run \
                --config "$CFG" --shard "$SHARD" \
                "${_OBJ_FLAG[@]}" --stage "$server_stage" \
                --count-pending 2>/dev/null
        ) || _pending=1

        if [ "${_pending}" = "0" ]; then
            echo "[scheduler] batch [${names[*]}]: server stage done — skipping startup"
        else
            case "$batch_servers" in
                vlm)  start_vlm  && _batch_servers_up=1 ;;
                flux) start_flux && _batch_servers_up=1 ;;
            esac
        fi
    fi

    # Fork all stages; each stage call gets servers_already_started=1 so it
    # skips the server lifecycle (servers are owned by this function).
    for _stage in "${names[@]}"; do
        (
            _load_stage_meta "$_stage"
            printf "  ${_BOLD}> %s${_RST}  steps=[%s]\n" "$_stage" "${STAGE_STEPS[*]}"
            LIMIT="${LIMIT:-}" \
            ATTN_BACKEND="${ATTN_BACKEND:-flash_attn}" \
            "$PY_PIPE" -m partcraft.pipeline_v3.run \
                --config "$CFG" --shard "$SHARD" \
                "${_OBJ_FLAG[@]}" "${_FORCE_FLAG[@]}" \
                --stage "$_stage"
        ) > "$LOG_DIR/stage_${_stage}.log" 2>&1 &
        pids+=($!)
        printf "  ${_CYN}%s${_RST} -> PID %s\n" "$_stage" "${pids[-1]}"
    done

    # Background heartbeat — one status line per stage every 15 s.
    local _mon_args=()
    for _s in "${names[@]}"; do
        _mon_args+=("$LOG_DIR/stage_${_s}.log" "$_s")
    done
    _live_monitor 15 "${_mon_args[@]}" &
    local _mon_pid=$!

    for _i in "${!pids[@]}"; do
        local _rc=0
        wait "${pids[$_i]}" || _rc=$?
        if [ "$_rc" -ne 0 ]; then
            printf "${_RED}[scheduler] %s FAILED (exit=%s)${_RST}\n" "${names[$_i]}" "$_rc"
            show_stage_errors "$LOG_DIR/stage_${names[$_i]}.log" "${names[$_i]}"
            _any_fail=$_rc
        else
            printf "${_GRN}[scheduler] %s OK${_RST}\n" "${names[$_i]}"
        fi
    done

    kill "$_mon_pid" 2>/dev/null || true
    wait "$_mon_pid" 2>/dev/null || true

    if [ "$_batch_servers_up" = "1" ]; then
        case "$batch_servers" in
            vlm)  stop_vlm ;;
            flux) stop_flux ;;
        esac
    fi

    if [ "$_any_fail" -ne 0 ]; then
        printf "${_RED}[scheduler] parallel batch [%s] had failures — aborting${_RST}\n" "${names[*]}"
        exit "$_any_fail"
    fi
}

# ═══ MAIN LOOP ═══════════════════════════════════════════════════════
# Ask Python to group SELECTED_STAGES into execution batches respecting
# parallel_group annotations.  Each output line is a space-separated
# list of stage names; single-element lines run serially, multi-element
# lines run concurrently (& + wait).
#
# Example output from dump_stage_batches:
#   text_gen_gate_a
#   deletion_cpu flux_2d     <- parallel (same parallel_group)
#   trellis_preview
#   gate_quality

_stages_str="${SELECTED_STAGES[*]}"

while IFS=' ' read -ra _batch; do
    [ "${#_batch[@]}" -eq 0 ] && continue
    run_parallel_batch "${_batch[@]}"
done < <(
    "$PY_PIPE" -c "
import yaml
from partcraft.pipeline_v3.scheduler import dump_stage_batches, stages_for
cfg = yaml.safe_load(open('$CFG'))
# Preserve config order, filter to the requested stages.
all_names = [s.name for s in stages_for(cfg)]
requested = set('$_stages_str'.split())
ordered = [n for n in all_names if n in requested]
for batch in dump_stage_batches(cfg, ordered):
    print(' '.join(batch))
"
)

echo
printf "${_GRN}${_BOLD}=== ALL STAGES DONE  tag=%s ===${_RST}\n" "$TAG"
