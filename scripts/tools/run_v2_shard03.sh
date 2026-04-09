#!/usr/bin/env bash
# Pipeline v2 — shard03 fully-automated tmux launcher (node37)
#
# Usage:
#   bash scripts/tools/run_v2_shard03.sh [SESSION_NAME]
#
# Creates a tmux session and runs the full pipeline without manual input:
#
#   pipe   Orchestrator — starts servers, health-checks, runs each phase,
#          then tears down servers before moving to the next phase.
#
#   vlm    Tails VLM server logs (read-only monitor).
#   flux   Tails FLUX server logs (read-only monitor).
#
# Phase sequence (fully automatic):
#   [phase A  s1]  VLM annotation       — starts 3 VLM servers, waits ready
#   [phase C  s4]  FLUX 2D edits        — kills VLM, starts 3 FLUX servers
#   [phases D-F]   TRELLIS 3D + cleanup — kills FLUX, runs s5/s5b/s6/s6b/s7
#
# Resume-safe: objects with completed steps in status.json are skipped.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

SESSION="${1:-pv2s03}"
CONFIG="${PROJECT_ROOT}/configs/pipeline_v2_shard03.yaml"
SHARD="03"

CONDA_INIT="/home/artgen/anaconda3/etc/profile.d/conda.sh"
ENV_SERVER="qwen_test"
ENV_PIPE="vinedresser3d"

VLM_CKPT="/Node11_nvme/zsn/checkpoints/Qwen3.5-27B"
EDIT_CKPT="/Node11_nvme/wjw/checkpoints/FLUX.2-klein-9B"

GPUS=(3 4 6)
VLM_PORTS=(8002 8012 8022)
FLUX_PORTS=(8004 8005 8006)

LOG_DIR="/tmp/pv2s03_logs"

# ── guards ───────────────────────────────────────────────────────────
if [[ ! -f "${CONFIG}" ]]; then
    echo "[ERROR] Config not found: ${CONFIG}"
    exit 1
fi
if tmux has-session -t "${SESSION}" 2>/dev/null; then
    echo "[tmux] session '${SESSION}' already exists — attaching."
    exec tmux attach-session -t "${SESSION}"
fi

mkdir -p "${LOG_DIR}"

# ── helpers ──────────────────────────────────────────────────────────
send_line() { tmux send-keys -t "${SESSION}:${1}" "${2}" Enter; }

# ── build the fully-automated pipe script ────────────────────────────
TMPSCRIPT="/tmp/pv2s03_pipe.sh"
cat > "${TMPSCRIPT}" <<SCRIPT
#!/usr/bin/env bash
set -euo pipefail
cd "${PROJECT_ROOT}"
source "${CONDA_INIT}"
conda activate "${ENV_PIPE}"

LOG_DIR="${LOG_DIR}"
VLM_CKPT="${VLM_CKPT}"
EDIT_CKPT="${EDIT_CKPT}"
GPUS=(${GPUS[*]})
VLM_PORTS=(${VLM_PORTS[*]})
FLUX_PORTS=(${FLUX_PORTS[*]})

# ── health check helper ───────────────────────────────────────────────
wait_port() {
    local port="\$1" label="\$2" elapsed=0 max=360
    printf "  waiting %-20s (port %s)" "\${label}" "\${port}"
    while ! curl -sf "http://localhost:\${port}/health" > /dev/null 2>&1; do
        sleep 5; elapsed=\$((elapsed+5))
        printf "."
        if [[ \${elapsed} -ge \${max} ]]; then
            echo " TIMEOUT — aborting"; exit 1
        fi
    done
    echo " OK"
}

# ── kill helpers ─────────────────────────────────────────────────────
kill_ports() {
    local ports=("\$@")
    for p in "\${ports[@]}"; do
        pids=\$(lsof -ti tcp:\${p} 2>/dev/null || true)
        if [[ -n "\${pids}" ]]; then
            echo "  killing port \${p} (pids \${pids})"
            kill \${pids} 2>/dev/null || true
        fi
    done
    sleep 3
}

# ════════════════════════════════════════════════════════════════════
echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  Pipeline v2 — shard03  (node37)  \$(date '+%Y-%m-%d %H:%M')"
echo "══════════════════════════════════════════════════════════════"

# ────────────────────────────────────────────────────────────────────
echo ""
echo "▶ PHASE A — VLM annotation (s1)"
echo "  Starting VLM servers on GPUs \${GPUS[*]} ..."
VLM_PIDS=()
for i in "\${!GPUS[@]}"; do
    gpu="\${GPUS[\$i]}"
    port="\${VLM_PORTS[\$i]}"
    echo "  GPU \${gpu} → port \${port} (log: \${LOG_DIR}/vlm_\${port}.log)"
    SGLANG_DISABLE_CUDNN_CHECK=1 VLM_MEM_FRAC=0.85 \\
        CUDA_VISIBLE_DEVICES="\${gpu}" VLM_MODEL="\${VLM_CKPT}" VLM_PORT="\${port}" \\
        bash scripts/tools/launch_local_vlm.sh \\
        > "\${LOG_DIR}/vlm_\${port}.log" 2>&1 &
    VLM_PIDS+=(\$!)
    # stagger to avoid simultaneous checkpoint I/O
    [[ \$i -lt \$((\${#GPUS[@]}-1)) ]] && sleep 30
done

echo "  Health-checking VLM servers..."
for i in "\${!VLM_PORTS[@]}"; do
    wait_port "\${VLM_PORTS[\$i]}" "vlm-gpu\${GPUS[\$i]}"
done
echo "  All VLM servers ready."

echo ""
echo "  Running Phase A (s1)..."
ATTN_BACKEND=xformers python -m partcraft.pipeline_v2.run \\
    --config configs/pipeline_v2_shard03.yaml \\
    --shard 03 --all --phase A

echo ""
echo "  Phase A done. Killing VLM servers..."
kill_ports "\${VLM_PORTS[@]}"
wait "\${VLM_PIDS[@]}" 2>/dev/null || true
echo "  VLM servers stopped."

# ────────────────────────────────────────────────────────────────────
echo ""
echo "▶ PHASE C — FLUX 2D edits (s4)"
echo "  Starting FLUX servers on GPUs \${GPUS[*]} ..."
FLUX_PIDS=()
for i in "\${!GPUS[@]}"; do
    gpu="\${GPUS[\$i]}"
    port="\${FLUX_PORTS[\$i]}"
    echo "  GPU \${gpu} → port \${port} (log: \${LOG_DIR}/flux_\${port}.log)"
    CUDA_VISIBLE_DEVICES="\${gpu}" python scripts/tools/image_edit_server.py \\
        --ckpt "\${EDIT_CKPT}" --port "\${port}" \\
        > "\${LOG_DIR}/flux_\${port}.log" 2>&1 &
    FLUX_PIDS+=(\$!)
done

echo "  Health-checking FLUX servers..."
for i in "\${!FLUX_PORTS[@]}"; do
    wait_port "\${FLUX_PORTS[\$i]}" "flux-gpu\${GPUS[\$i]}"
done
echo "  All FLUX servers ready."

echo ""
echo "  Running Phase C (s4)..."
ATTN_BACKEND=xformers python -m partcraft.pipeline_v2.run \\
    --config configs/pipeline_v2_shard03.yaml \\
    --shard 03 --all --phase C

echo ""
echo "  Phase C done. Killing FLUX servers..."
kill_ports "\${FLUX_PORTS[@]}"
wait "\${FLUX_PIDS[@]}" 2>/dev/null || true
echo "  FLUX servers stopped."

# ────────────────────────────────────────────────────────────────────
echo ""
echo "▶ PHASES D→F — TRELLIS 3D + cleanup (s5 s5b s6 s6b s7)"
echo "  Running across GPUs \${GPUS[*]}..."
ATTN_BACKEND=xformers python -m partcraft.pipeline_v2.run \\
    --config configs/pipeline_v2_shard03.yaml \\
    --shard 03 --all \\
    --steps s5,s5b,s6,s6b,s7 --gpus "\$(IFS=,; echo "\${GPUS[*]}")"

echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  All phases complete.  \$(date '+%Y-%m-%d %H:%M')"
echo "══════════════════════════════════════════════════════════════"
SCRIPT

chmod +x "${TMPSCRIPT}"

# ── create session ───────────────────────────────────────────────────
tmux new-session -d -s "${SESSION}" -n pipe

# ── window: pipe ─────────────────────────────────────────────────────
send_line pipe "bash ${TMPSCRIPT} 2>&1 | tee ${LOG_DIR}/pipe.log"

# ── window: vlm (log monitor) ────────────────────────────────────────
tmux new-window -t "${SESSION}" -n vlm
send_line vlm "mkdir -p ${LOG_DIR} && echo 'Waiting for VLM logs...' && until ls ${LOG_DIR}/vlm_*.log 2>/dev/null | head -1 | grep -q .; do sleep 2; done && tail -f ${LOG_DIR}/vlm_*.log"

# ── window: flux (log monitor) ───────────────────────────────────────
tmux new-window -t "${SESSION}" -n flux
send_line flux "mkdir -p ${LOG_DIR} && echo 'Waiting for FLUX logs...' && until ls ${LOG_DIR}/flux_*.log 2>/dev/null | head -1 | grep -q .; do sleep 2; done && tail -f ${LOG_DIR}/flux_*.log"

# ── focus pipe and attach ─────────────────────────────────────────────
tmux select-window -t "${SESSION}:pipe"

echo ""
echo "Session '${SESSION}' created — fully automated."
echo ""
echo "  pipe  : orchestrates the full run (VLM→PhaseA→FLUX→PhaseC→3D)"
echo "  vlm   : tails VLM server logs"
echo "  flux  : tails FLUX server logs"
echo ""
echo "  All logs also in: ${LOG_DIR}/"
echo "  Pipeline output:  outputs/partverse/pipeline_v2_shard03/"
echo ""
echo "Attaching..."
exec tmux attach-session -t "${SESSION}"
