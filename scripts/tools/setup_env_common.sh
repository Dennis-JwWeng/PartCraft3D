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
  if conda env list | awk '{print $1}' | rg -xq "${env_name}"; then
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
