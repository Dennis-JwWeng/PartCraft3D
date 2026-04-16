#!/usr/bin/env bash
# Run shard08 test benchmark (20 objects, stratified sample from shard08).
#
# Usage:
#   bash scripts/tools/run_shard08_test.sh [STAGES]
#
#   STAGES defaults to "gate_a,flux_branch,del_branch,gate_e_qc,gate_e_encode"
#   Pass a comma-separated subset to run only those stages, e.g.:
#     bash scripts/tools/run_shard08_test.sh gate_a
#
# The 20 object IDs are embedded via OBJ_IDS — no --all flag is used, so only
# these objects are processed regardless of what else is in the shard.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

CFG="configs/pipeline_v2_shard08_test.yaml"
TAG="shard08"

OBJ_IDS="ced5d54c132c4ac697d9d37f0e7732db \
d03ee70eac80426c99e2cb6d8b0c27f2 \
d81c17b371424a068f5c6a1c3e806615 \
db215ccb1d04490dae6de8e2c8c9707e \
d1290034c0724e78acb31a354451321c \
c89669068aea40beac2bad8dc2d403a0 \
c40020f3804c4370832cc9444709bbb6 \
d025942952774f50808e0511dc918556 \
c69887e12a164e20bfdc40c5a57c9d77 \
d3940528075e497d9c63661d3a5fe67d \
c3d88711e2f34164b1eb8803a3e2448a \
c5c9b0c4722a4cb5a5752c6459394881 \
c449a455a19647ccbfc39e5cedb471c0 \
d9d0c5c687f14147ae6be6da981e1854 \
db31b226b97b4d73b5284563ea2f17e7 \
c9e81ddcd0534b3186c18998919e4ed2 \
c578c8c12a984e79a6d958a90f86cdc8 \
d26028b85e364a3aa7685a72be5b10b2 \
d13e2260de04423698a90b4ca50516ac \
c3f0d5bcd6ff441aa89bda7886575199"

if [ -n "${1:-}" ]; then
    export STAGES="$1"
fi

OBJ_IDS="$OBJ_IDS" bash scripts/tools/run_pipeline_v2_shard.sh "$TAG" "$CFG"
