#!/usr/bin/env bash
# Standardized bench input validator (reference workflow).
#
# Purpose:
#   Validate PartVerse bench inputs (mesh/images/slat) for a target shard/object set
#   using the same PartVerseDataset checks as pipeline_v3.
#
# Default target (this repo):
#   /mnt/zsn/zsn_workspace/PartCraft3D/data/partverse/bench/inputs
#
# Usage:
#   bash scripts/tools/validate_bench_inputs.sh
#   bash scripts/tools/validate_bench_inputs.sh --all
#   bash scripts/tools/validate_bench_inputs.sh --obj-ids-file configs/shard08_test20_obj_ids.txt
#   INPUT_ROOT=/path/to/bench/inputs bash scripts/tools/validate_bench_inputs.sh --shard 08 --all

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

INPUT_ROOT="${INPUT_ROOT:-$ROOT/data/partverse/bench/inputs}"
TEMPLATE_CFG="${TEMPLATE_CFG:-$ROOT/configs/pipeline_v3_shard08_test20.yaml}"
SHARD="${SHARD:-08}"
MODE="ids_file"
OBJ_IDS_FILE="${OBJ_IDS_FILE:-$ROOT/configs/shard08_test20_obj_ids.txt}"
VERBOSE=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input-root)
      INPUT_ROOT="$2"; shift 2 ;;
    --template-cfg)
      TEMPLATE_CFG="$2"; shift 2 ;;
    --shard)
      SHARD="$2"; shift 2 ;;
    --obj-ids-file)
      MODE="ids_file"; OBJ_IDS_FILE="$2"; shift 2 ;;
    --all)
      MODE="all"; shift ;;
    --quiet)
      VERBOSE=0; shift ;;
    -h|--help)
      sed -n '1,40p' "$0"; exit 0 ;;
    *)
      echo "[ERROR] Unknown arg: $1" >&2
      exit 2 ;;
  esac
done

MESH_ROOT="$INPUT_ROOT/mesh"
IMAGES_ROOT="$INPUT_ROOT/images"
SLAT_DIR="$INPUT_ROOT/slat"

for p in "$TEMPLATE_CFG" "$MESH_ROOT" "$IMAGES_ROOT" "$SLAT_DIR"; do
  [[ -e "$p" ]] || { echo "[ERROR] missing required path: $p" >&2; exit 1; }
done

[[ -d "$MESH_ROOT/$SHARD" ]]   || { echo "[ERROR] missing shard dir: $MESH_ROOT/$SHARD" >&2; exit 1; }
[[ -d "$IMAGES_ROOT/$SHARD" ]] || { echo "[ERROR] missing shard dir: $IMAGES_ROOT/$SHARD" >&2; exit 1; }
[[ -d "$SLAT_DIR/$SHARD" ]]    || { echo "[ERROR] missing shard dir: $SLAT_DIR/$SHARD" >&2; exit 1; }

if [[ "$MODE" == "ids_file" ]]; then
  [[ -f "$OBJ_IDS_FILE" ]] || { echo "[ERROR] obj_ids_file not found: $OBJ_IDS_FILE" >&2; exit 1; }
fi

TMP_CFG="$(mktemp /tmp/bench_inputs_validate.XXXXXX.yaml)"
trap 'rm -f "$TMP_CFG"' EXIT

python - <<'PY' "$TEMPLATE_CFG" "$TMP_CFG" "$MESH_ROOT" "$IMAGES_ROOT" "$SLAT_DIR"
from pathlib import Path
import sys, yaml

tpl = Path(sys.argv[1])
out = Path(sys.argv[2])
mesh_root, images_root, slat_dir = sys.argv[3], sys.argv[4], sys.argv[5]

cfg = yaml.safe_load(tpl.read_text())
cfg.setdefault("data", {})
cfg["data"]["mesh_root"] = mesh_root
cfg["data"]["images_root"] = images_root
cfg["data"]["slat_dir"] = slat_dir
out.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
print(out)
PY

COMMON=(python scripts/tools/validate_v3_inputs.py --config "$TMP_CFG" --shard "$SHARD")

if [[ "$VERBOSE" -eq 1 ]]; then
  COMMON+=( -v )
fi

if [[ "$MODE" == "all" ]]; then
  echo "[bench-validate] mode=all"
  echo "[bench-validate] input_root=$INPUT_ROOT shard=$SHARD"
  "${COMMON[@]}" --all
else
  N_IDS="$(python - <<'PY' "$OBJ_IDS_FILE"
from pathlib import Path
import sys
p=Path(sys.argv[1])
ids=[l.strip() for l in p.read_text().splitlines() if l.strip() and not l.strip().startswith('#')]
print(len(ids))
PY
)"
  echo "[bench-validate] mode=obj_ids_file"
  echo "[bench-validate] input_root=$INPUT_ROOT shard=$SHARD obj_ids_file=$OBJ_IDS_FILE (n=$N_IDS)"
  "${COMMON[@]}" --obj-ids-file "$OBJ_IDS_FILE"
fi

echo "[bench-validate] PASS"
