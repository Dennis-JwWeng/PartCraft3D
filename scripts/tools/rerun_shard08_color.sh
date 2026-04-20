#!/usr/bin/env bash
# Re-run pipeline steps for shard08 color edits only.
#
# Assumes cleanup_color_rerun_shard08.py --execute has already cleared the
# clr_* entries from edit_status.json. The per-edit resume logic then
# naturally restricts trellis_3d / preview_flux / gate_quality to clr_*.
#
# Intended to co-run alongside shard06 pipeline. Uses GPUs 1-7 (skips GPU 0
# which is loaded by Trellis500k encoder) and reuses the already-running
# VLM servers on ports 8200-8207 for gate_quality (same model spec).
#
# Logs: logs/v3_shard08_color_rerun/{s5,s6p,gate_e,pull,index}_<ts>.log
set -euo pipefail

cd /mnt/zsn/zsn_workspace/PartCraft3D

export CONDA_INIT=${CONDA_INIT:-/mnt/zsn/miniconda3/etc/profile.d/conda.sh}
# shellcheck disable=SC1090
source "$CONDA_INIT"
conda activate vinedresser3d

CFG=configs/pipeline_v3_shard08.yaml
SHARD=08
GPUS="${GPUS:-1,2,3,4,5,6,7}"
TS=$(date +%Y%m%d_%H%M%S)
LOG_DIR=logs/v3_shard08_color_rerun
mkdir -p "$LOG_DIR"

echo "===== [rerun_shard08_color] $(date) ====="
echo "cfg=$CFG  shard=$SHARD  gpus=$GPUS  log_dir=$LOG_DIR"

# ── 1) Trellis 3D (s5) ─────────────────────────────────────────────────
# Per-edit resume is keyed on stages.s5; we cleared clr_* so only those
# become pending. Other types (mod/scl/mat/glb) have s5=done and are
# skipped automatically.
echo "[1/5] trellis_3d"
python -m partcraft.pipeline_v3.run \
    --config "$CFG" --shard "$SHARD" --all \
    --steps trellis_3d --gpus "$GPUS" \
    2>&1 | tee "$LOG_DIR/s5_${TS}.log"

# ── 2) Preview (s6p) ──────────────────────────────────────────────────
echo "[2/5] preview_flux"
python -m partcraft.pipeline_v3.run \
    --config "$CFG" --shard "$SHARD" --all \
    --steps preview_flux --gpus "$GPUS" \
    2>&1 | tee "$LOG_DIR/s6p_${TS}.log"

# ── 3) Gate quality (gate_e), color only ──────────────────────────────
# Reuses the VLM servers already running from shard06 (ports 8200-8207).
# QC_ONLY_TYPES restricts Gate E to color.
echo "[3/5] gate_quality (color only)"
QC_ONLY_TYPES=color python -m partcraft.pipeline_v3.run \
    --config "$CFG" --shard "$SHARD" --all \
    --steps gate_quality --gpus "$GPUS" \
    2>&1 | tee "$LOG_DIR/gate_e_${TS}.log"

# ── 4) Promote passing color edits into H3D_v1 release ────────────────
echo "[4/5] pull_flux --types color"
python -m scripts.cleaning.h3d_v1.pull_flux \
    --pipeline-cfg "$CFG" --shard "$SHARD" \
    --dataset-root data/H3D_v1 --types color \
    2>&1 | tee "$LOG_DIR/pull_${TS}.log"

# ── 5) Rebuild aggregated manifest ────────────────────────────────────
echo "[5/5] build_h3d_v1_index"
python -m scripts.cleaning.h3d_v1.build_h3d_v1_index \
    --dataset-root data/H3D_v1 \
    2>&1 | tee "$LOG_DIR/index_${TS}.log"

echo "===== [rerun_shard08_color] DONE $(date) ====="
