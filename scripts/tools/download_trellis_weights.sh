#!/usr/bin/env bash
# Download TRELLIS + CLIP checkpoints into PartCraft3D/checkpoints/
# Run when Hugging Face is reachable. If you see SSL UNEXPECTED_EOF, fix
# https_proxy / try another network, then re-run.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

if ! command -v hf &>/dev/null; then
  echo "Install: pip install huggingface_hub[cli]"
  exit 1
fi

echo "==> CLIP (text conditioning for TRELLIS-text)"
hf download openai/clip-vit-large-patch14 \
  --local-dir "$ROOT/checkpoints/clip-vit-large-patch14"

echo "==> TRELLIS image pipeline"
hf download JeffreyXiang/TRELLIS-image-large \
  --local-dir "$ROOT/checkpoints/TRELLIS-image-large"

echo "==> TRELLIS text pipeline"
hf download JeffreyXiang/TRELLIS-text-xlarge \
  --local-dir "$ROOT/checkpoints/TRELLIS-text-xlarge"

echo "Done. Expected: checkpoints/clip-vit-large-patch14/config.json and *.safetensors under TRELLIS-*/ckpts/"
