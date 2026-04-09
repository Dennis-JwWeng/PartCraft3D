#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/setup_env_common.sh"

SHOW_HELP=0
parse_common_args "$@"
if [[ "${SHOW_HELP}" == "1" ]]; then
  cat <<'EOF'
Usage: bash scripts/tools/check_machine_env_for_pipeline.sh [--machine-env <path>]

Validates required machine env variables and that referenced paths exist
for full pipeline_v2 runs (all phases). Does not install packages.

Options are the same as other setup scripts (--machine-env, --check).
EOF
  exit 0
fi

load_machine_env
validate_pipeline_machine_env_paths
echo "[OK] Machine profile is ready for pipeline_v2 (paths + required vars)."
