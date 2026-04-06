#!/usr/bin/env bash
# Download TRELLIS + CLIP checkpoints into PartCraft3D/checkpoints/
# Run when Hugging Face is reachable. If you see SSL UNEXPECTED_EOF, fix
# https_proxy / try another network, then re-run.
set -euo pipefail
ROOT="${PARTCRAFT_CKPT_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)/checkpoints}"
mkdir -p "$ROOT"

if ! command -v hf &>/dev/null; then
  echo "Install: pip install huggingface_hub[cli]"
  exit 1
fi

echo "==> CLIP (text conditioning for TRELLIS-text)"
hf download openai/clip-vit-large-patch14 \
  --local-dir "$ROOT/clip-vit-large-patch14"

echo "==> TRELLIS image pipeline"
hf download JeffreyXiang/TRELLIS-image-large \
  --local-dir "$ROOT/TRELLIS-image-large"

echo "==> TRELLIS text pipeline"
hf download JeffreyXiang/TRELLIS-text-xlarge \
  --local-dir "$ROOT/TRELLIS-text-xlarge"

echo "Done. Checkpoints in: $ROOT"
