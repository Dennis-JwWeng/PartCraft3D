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
Ensures diffusers>=0.37.1 (required for FLUX.2-klein-9B / Flux2KleinPipeline).

$(usage_common)
EOF
  exit 0
fi

load_machine_env
if [[ "${CHECK_ONLY}" == "1" ]]; then
  validate_pipeline_machine_env_paths
fi
require_vars CONDA_ENV_SERVER
init_conda

ENV_NAME="${CONDA_ENV_SERVER}"
ensure_env_exists "${ENV_NAME}" "3.10"
activate_env "${ENV_NAME}"
print_runtime_info "${ENV_NAME}"

if [[ "${CHECK_ONLY}" == "1" ]]; then
  echo "[CHECK] Deploy env activation succeeded."
  python - <<'PY'
import importlib
checks = {
    "sglang": "sglang",
    "diffusers": "diffusers",
    "transformers": "transformers",
    "accelerate": "accelerate",
}
missing = []
for label, mod in checks.items():
    try:
        m = importlib.import_module(mod)
        v = getattr(m, "__version__", "?")
        print(f"  {label} {v}")
    except Exception:
        missing.append(label)
if missing:
    print(f"[WARN] Missing: {', '.join(missing)}")
else:
    print("[CHECK] All deploy modules present")

# Verify Flux2KleinPipeline
try:
    from diffusers import Flux2KleinPipeline
    print("[CHECK] Flux2KleinPipeline importable")
except ImportError as e:
    print(f"[WARN] Flux2KleinPipeline NOT importable: {e}")
    print("       Upgrade diffusers: pip install 'diffusers>=0.37.1'")
PY
  exit 0
fi

echo "[INFO] Installing deploy dependencies..."
pip_install_cmd install --upgrade pip

# diffusers>=0.37.1 is required for FLUX.2-klein-9B (Flux2KleinPipeline)
pip_install_cmd install "sglang[all]" "diffusers>=0.37.1" transformers accelerate

echo "[INFO] Verifying deploy imports..."
python - <<'PY'
import importlib, sys

mods = ("sglang", "diffusers", "transformers", "accelerate")
missing = []
for m in mods:
    try:
        importlib.import_module(m)
    except Exception:
        missing.append(m)
if missing:
    raise SystemExit(f"[ERROR] Missing deploy modules: {', '.join(missing)}")
print("[OK] Deploy modules:", ", ".join(mods))

# Verify diffusers version and Flux2KleinPipeline
import diffusers
from packaging.version import Version
v = Version(diffusers.__version__)
if v < Version("0.37.1"):
    print(f"[WARN] diffusers {diffusers.__version__} < 0.37.1, "
          "Flux2KleinPipeline may not be available")
    print("  Fix: pip install 'diffusers>=0.37.1'")
else:
    print(f"[OK] diffusers {diffusers.__version__} (>=0.37.1)")

try:
    from diffusers import Flux2KleinPipeline
    print("[OK] Flux2KleinPipeline importable")
except ImportError:
    print("[ERROR] Flux2KleinPipeline not found — upgrade diffusers to >=0.37.1")
    sys.exit(1)
PY

echo "[DONE] Deploy environment is ready: ${ENV_NAME}"
