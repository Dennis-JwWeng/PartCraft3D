#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/setup_env_common.sh"

SHOW_HELP=0
parse_common_args "$@"

if [[ "${SHOW_HELP}" == "1" ]]; then
  cat <<EOF
Usage: bash scripts/tools/setup_pipeline_env.sh [options]

Setup pipeline runtime environment (run_pipeline.py / run_streaming.py).

$(usage_common)
EOF
  exit 0
fi

load_machine_env
require_vars CONDA_ENV_PIPELINE
init_conda

ENV_NAME="${CONDA_ENV_PIPELINE}"
ensure_env_exists "${ENV_NAME}" "3.10"
activate_env "${ENV_NAME}"
print_runtime_info "${ENV_NAME}"

if [[ "${CHECK_ONLY}" == "1" ]]; then
  echo "[CHECK] Pipeline env activation succeeded."
  exit 0
fi

echo "[INFO] Installing pipeline dependencies..."
pip_install_cmd install --upgrade pip
pip_install_cmd install -r "${PROJECT_ROOT}/requirements.txt"
if [[ "${REINSTALL}" == "1" ]]; then
  python -m pip install --upgrade --force-reinstall -e "${PROJECT_ROOT}"
else
  python -m pip install -e "${PROJECT_ROOT}"
fi

echo "[INFO] Verifying pipeline imports..."
python - <<'PY'
import importlib
mods = ("partcraft", "numpy", "yaml", "trimesh")
missing = []
for m in mods:
    try:
        importlib.import_module(m)
    except Exception:
        missing.append(m)
if missing:
    raise SystemExit(f"[ERROR] Missing pipeline modules: {', '.join(missing)}")
print("[CHECK] Pipeline modules ok:", ", ".join(mods))
PY

echo "[DONE] Pipeline environment is ready: ${ENV_NAME}"
