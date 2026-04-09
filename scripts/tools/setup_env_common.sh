#!/usr/bin/env bash

# Shared helpers for one-click environment setup scripts.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"

usage_common() {
  cat <<EOF
Options:
  --machine-env <path>  Path to machine env file (default: configs/machine/\$(hostname).env)
  --reinstall           Force reinstall pip packages
  --check               Only validate machine env and conda activation, no installs
  -h, --help            Show help
EOF
}

parse_common_args() {
  REINSTALL=0
  CHECK_ONLY=0
  MACHINE_ENV="${MACHINE_ENV:-${PROJECT_ROOT}/configs/machine/$(hostname).env}"

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --machine-env)
        [[ $# -ge 2 ]] || { echo "[ERROR] --machine-env requires a value"; exit 1; }
        MACHINE_ENV="$2"
        shift 2
        ;;
      --reinstall)
        REINSTALL=1
        shift
        ;;
      --check)
        CHECK_ONLY=1
        shift
        ;;
      -h|--help)
        SHOW_HELP=1
        shift
        ;;
      *)
        echo "[ERROR] Unknown argument: $1"
        exit 1
        ;;
    esac
  done

  if [[ "${SHOW_HELP:-0}" == "1" ]]; then
    return 0
  fi

  if [[ ! -f "${MACHINE_ENV}" ]]; then
    echo "[ERROR] Machine config not found: ${MACHINE_ENV}"
    echo "  Create it from template: ${PROJECT_ROOT}/configs/machine/node39.env"
    exit 1
  fi
}

load_machine_env() {
  # shellcheck disable=SC1090
  source "${MACHINE_ENV}"
}

require_vars() {
  local name
  for name in "$@"; do
    if [[ -z "${!name:-}" ]]; then
      echo "[ERROR] ${name} is required in ${MACHINE_ENV}"
      exit 1
    fi
  done
}

# Required for full pipeline_v2 (deploy + pipeline conda + ckpt roots + data roots).
# CONDA_ENV_* values are conda environment *names*, not filesystem paths.
PIPELINE_REQUIRED_MACHINE_VARS=(
  CONDA_INIT
  CONDA_ENV_SERVER
  CONDA_ENV_PIPELINE
  VLM_CKPT
  EDIT_CKPT
  TRELLIS_CKPT_ROOT
  DATA_DIR
  OUTPUT_ROOT
)

# Paths that must exist on disk (see validate_pipeline_machine_env_paths).
PIPELINE_PATH_VARS=(
  CONDA_INIT
  VLM_CKPT
  EDIT_CKPT
  TRELLIS_CKPT_ROOT
  DATA_DIR
  OUTPUT_ROOT
)

validate_pipeline_machine_env_paths() {
  local name val
  require_vars "${PIPELINE_REQUIRED_MACHINE_VARS[@]}"

  val="${CONDA_INIT}"
  if [[ ! -f "${val}" ]]; then
    echo "[ERROR] CONDA_INIT is not a file: ${val}"
    echo "  Fix in: ${MACHINE_ENV}"
    exit 1
  fi
  # shellcheck disable=SC1090
  set +u; source "${CONDA_INIT}"; set -u
  for name in CONDA_ENV_SERVER CONDA_ENV_PIPELINE; do
    val="${!name}"
    if ! conda env list 2>/dev/null | awk -v env="${val}" '$1 == env { found=1 } END { exit(found ? 0 : 1) }'; then
      echo "[ERROR] ${name} conda env not found: ${val}"
      echo "  Fix in: ${MACHINE_ENV}"
      exit 1
    fi
  done

  for name in "${PIPELINE_PATH_VARS[@]}"; do
    [[ "${name}" == "CONDA_INIT" ]] && continue
    val="${!name}"
    if [[ ! -e "${val}" ]]; then
      echo "[ERROR] ${name} path does not exist: ${val}"
      echo "  Fix in: ${MACHINE_ENV}"
      exit 1
    fi
  done

  if [[ -n "${BLENDER_PATH:-}" ]] && [[ ! -f "${BLENDER_PATH}" ]] && [[ ! -x "${BLENDER_PATH}" ]]; then
    echo "[WARN] BLENDER_PATH set but not a file: ${BLENDER_PATH}"
  fi
  echo "[CHECK] Machine env OK (conda envs + paths): ${PIPELINE_REQUIRED_MACHINE_VARS[*]}"
}

init_conda() {
  require_vars CONDA_INIT
  if [[ ! -f "${CONDA_INIT}" ]]; then
    echo "[ERROR] CONDA_INIT does not exist: ${CONDA_INIT}"
    exit 1
  fi
  # shellcheck disable=SC1090
  set +u; source "${CONDA_INIT}"; set -u
}

ensure_env_exists() {
  local env_name="$1"
  local python_version="${2:-3.10}"
  if conda env list | awk -v env="${env_name}" '$1 == env { found=1 } END { exit(found ? 0 : 1) }'; then
    echo "[INFO] Conda env exists: ${env_name}"
  else
    echo "[INFO] Creating conda env: ${env_name} (python=${python_version})"
    conda create -y -n "${env_name}" "python=${python_version}"
  fi
}

activate_env() {
  local env_name="$1"
  set +u; conda activate "${env_name}"; set -u
}

pip_install_cmd() {
  local mode="${1:-install}"
  shift || true
  local -a args=("$@")
  if [[ "${mode}" == "install" && "${REINSTALL}" == "1" ]]; then
    python -m pip install --upgrade --force-reinstall "${args[@]}"
  else
    python -m pip install "${args[@]}"
  fi
}

print_runtime_info() {
  local env_name="$1"
  echo "[INFO] Machine env: ${MACHINE_ENV}"
  echo "[INFO] Conda env: ${env_name}"
  python --version
  python -m pip --version
}

# ── CUDA helpers ────────────────────────────────────────────────

detect_cuda_suffix() {
  # Detect CUDA runtime version and output the pip suffix (e.g. "cu124").
  # Falls back to "cu121" if detection fails.
  python - <<'PY'
import re, subprocess, sys
for cmd in (["nvcc", "--version"], ["nvidia-smi"]):
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
        m = re.search(r"(?:release |CUDA Version: )(\d+)\.(\d+)", out)
        if m:
            print(f"cu{m.group(1)}{m.group(2)}")
            sys.exit(0)
    except Exception:
        pass
try:
    import torch
    cv = torch.version.cuda
    if cv:
        parts = cv.split(".")
        print(f"cu{parts[0]}{parts[1]}")
        sys.exit(0)
except Exception:
    pass
print("cu121")
PY
}

check_flash_attn() {
  # Returns 0 if flash_attn is importable and functional, 1 otherwise.
  python - <<'PY' 2>/dev/null
import torch, flash_attn
q = torch.randn(1, 1, 2, 64, dtype=torch.float16, device="cuda")
k = torch.randn(1, 1, 2, 64, dtype=torch.float16, device="cuda")
v = torch.randn(1, 1, 2, 64, dtype=torch.float16, device="cuda")
flash_attn.flash_attn_func(q, k, v)
print(f"flash_attn {flash_attn.__version__} ok")
PY
}

check_xformers() {
  # Returns 0 if xformers memory_efficient_attention works, 1 otherwise.
  python - <<'PY' 2>/dev/null
import torch, xformers.ops
q = torch.randn(1, 2, 64, dtype=torch.float16, device="cuda")
k = torch.randn(1, 2, 64, dtype=torch.float16, device="cuda")
v = torch.randn(1, 2, 64, dtype=torch.float16, device="cuda")
xformers.ops.memory_efficient_attention(q, k, v)
import xformers
print(f"xformers {xformers.__version__} ok")
PY
}

resolve_attn_backend() {
  # Determine the best attention backend. Priority: flash_attn > xformers.
  # Prints the chosen backend name and returns 0.
  # If neither works, prints "xformers" (safest default) and returns 1.
  if check_flash_attn; then
    echo "flash_attn"
    return 0
  fi
  if check_xformers; then
    echo "xformers"
    return 0
  fi
  echo "xformers"
  return 1
}
