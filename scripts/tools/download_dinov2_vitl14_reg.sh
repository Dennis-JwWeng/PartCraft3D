#!/usr/bin/env bash
# Download DINOv2 ViT-L/14 (4 registers) pretrained weights for PartCraft encode.
# Default: PARTCRAFT_CKPT_ROOT or /mnt/zsn/ckpts, else ~/.cache/torch/partcraft_ckpts
set -euo pipefail
ROOT="${PARTCRAFT_CKPT_ROOT:-}"
if [[ -z "$ROOT" ]]; then
  if [[ -d /mnt/zsn/ckpts ]]; then
    ROOT="/mnt/zsn/ckpts"
  else
    ROOT="${HOME}/.cache/torch/partcraft_ckpts"
  fi
fi
OUT="${ROOT}/dinov2/dinov2_vitl14_reg4_pretrain.pth"
mkdir -p "$(dirname "$OUT")"
if [[ -f "$OUT" ]]; then
  echo "Already present: $OUT"
  exit 0
fi
URL="https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_reg4_pretrain.pth"
echo "Downloading to $OUT (~1.2GB) ..."
if command -v curl >/dev/null 2>&1; then
  curl -fL --retry 5 --continue-at - -o "$OUT" "$URL"
else
  wget -O "$OUT" "$URL"
fi
echo "Done: $OUT"
echo "Optional: export PARTCRAFT_CKPT_ROOT=$ROOT"
