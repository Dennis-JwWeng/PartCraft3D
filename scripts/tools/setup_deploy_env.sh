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
Usage: bash scripts/tools/setup_deploy_env.sh [options]

Setup server-side deploy environment (VLM + image edit service).

$(usage_common)
EOF
  exit 0
fi

load_machine_env
require_vars CONDA_ENV_SERVER
init_conda

ENV_NAME="${CONDA_ENV_SERVER}"
ensure_env_exists "${ENV_NAME}" "3.10"
activate_env "${ENV_NAME}"
print_runtime_info "${ENV_NAME}"

if [[ "${CHECK_ONLY}" == "1" ]]; then
  echo "[CHECK] Deploy env activation succeeded."
  exit 0
fi

echo "[INFO] Installing deploy dependencies..."
pip_install_cmd install --upgrade pip
pip_install_cmd install "sglang[all]" diffusers transformers accelerate

echo "[INFO] Verifying deploy imports..."
python - <<'PY'
import importlib
mods = ("sglang", "diffusers", "transformers", "accelerate")
missing = []
for m in mods:
    try:
        importlib.import_module(m)
    except Exception:
        missing.append(m)
if missing:
    raise SystemExit(f"[ERROR] Missing deploy modules: {', '.join(missing)}")
print("[CHECK] Deploy modules ok:", ", ".join(mods))
PY

echo "[DONE] Deploy environment is ready: ${ENV_NAME}"
