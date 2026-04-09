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

Setup pipeline runtime environment for partcraft.pipeline_v2.
Installs all required dependencies including CUDA-matched spconv, attention
backend (flash_attn preferred, xformers as fallback), and warp-lang.

$(usage_common)
EOF
  exit 0
fi

load_machine_env
if [[ "${CHECK_ONLY}" == "1" ]]; then
  validate_pipeline_machine_env_paths
fi
require_vars CONDA_ENV_PIPELINE
init_conda

ENV_NAME="${CONDA_ENV_PIPELINE}"
ensure_env_exists "${ENV_NAME}" "3.10"
activate_env "${ENV_NAME}"
print_runtime_info "${ENV_NAME}"

if [[ "${CHECK_ONLY}" == "1" ]]; then
  echo "[CHECK] Pipeline env activation succeeded."
  echo "[CHECK] Detecting attention backend..."
  ATTN="$(resolve_attn_backend)" && echo "[CHECK] Attention backend: ${ATTN}" \
    || echo "[WARN] No working attention backend detected (will install on full run)"
  exit 0
fi

# ── 1. Basic pip + requirements.txt ──────────────────────────────
echo ""
echo "================================================================"
echo "[1/5] Core dependencies (requirements.txt)"
echo "================================================================"
pip_install_cmd install --upgrade pip
pip_install_cmd install -r "${PROJECT_ROOT}/requirements.txt"
if [[ "${REINSTALL}" == "1" ]]; then
  python -m pip install --upgrade --force-reinstall -e "${PROJECT_ROOT}"
else
  python -m pip install -e "${PROJECT_ROOT}"
fi

# ── 2. spconv + cumm (CUDA-version-matched) ─────────────────────
echo ""
echo "================================================================"
echo "[2/5] spconv + cumm (CUDA-matched)"
echo "================================================================"
CUDA_SUFFIX="$(detect_cuda_suffix)"
echo "[INFO] Detected CUDA suffix: ${CUDA_SUFFIX}"

SPCONV_PKG="spconv-${CUDA_SUFFIX}"
CURRENT_SPCONV="$(python -c "import spconv; print(spconv.__version__)" 2>/dev/null || echo "none")"

python - "${SPCONV_PKG}" "${CUDA_SUFFIX}" <<'PY'
import subprocess, sys
pkg, cuda_suffix = sys.argv[1], sys.argv[2]
try:
    import spconv
    ver = spconv.__version__
    try:
        from cumm import tensorview as tv
        tv_ok = True
    except Exception:
        tv_ok = False
    installed_cuda = None
    for dist in __import__("importlib.metadata", fromlist=["metadata"]).packages_distributions().get("spconv", []):
        if "cu" in dist:
            installed_cuda = dist.split("-")[-1] if "-" in dist else None
    # Check if the installed spconv matches the current CUDA
    needs_reinstall = not tv_ok
    if installed_cuda and cuda_suffix not in installed_cuda:
        needs_reinstall = True
        print(f"[INFO] spconv CUDA mismatch: installed={installed_cuda}, need={cuda_suffix}")
    if needs_reinstall:
        print(f"[INFO] Reinstalling {pkg}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", pkg])
    else:
        print(f"[OK] spconv {ver} with cumm already installed and CUDA-matched")
except ImportError:
    print(f"[INFO] spconv not found, installing {pkg}...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
PY

# Verify spconv functional
python -c "
from spconv.core import ConvAlgo
import cumm.tensorview as tv
print('[OK] spconv + cumm import verified')
" || { echo "[ERROR] spconv installation verification failed"; exit 1; }

# ── 3. warp-lang ────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "[3/5] warp-lang (NVIDIA Warp)"
echo "================================================================"
python -c "import warp; print(f'[OK] warp-lang {warp.__version__} already installed')" 2>/dev/null \
  || { echo "[INFO] Installing warp-lang..."; pip_install_cmd install "warp-lang>=1.0"; }

# ── 4. Attention backend: flash_attn (preferred) or xformers ───
echo ""
echo "================================================================"
echo "[4/5] Attention backend (flash_attn / xformers)"
echo "================================================================"

HAS_FLASH=0
HAS_XFORMERS=0

# Check xformers (should be installed via requirements.txt or TRELLIS deps)
if check_xformers; then
  HAS_XFORMERS=1
  echo "[OK] xformers available"
else
  echo "[INFO] xformers not working, installing..."
  pip_install_cmd install xformers 2>/dev/null && {
    check_xformers && HAS_XFORMERS=1 && echo "[OK] xformers installed"
  } || echo "[WARN] xformers installation failed (non-fatal if flash_attn works)"
fi

# Try flash_attn
if check_flash_attn; then
  HAS_FLASH=1
  echo "[OK] flash_attn already available"
else
  echo "[INFO] flash_attn not available, attempting install..."
  # Method 1: pip install (works if pre-built wheel matches torch+CUDA)
  pip_install_cmd install flash-attn --no-build-isolation 2>/dev/null && {
    check_flash_attn && HAS_FLASH=1 && echo "[OK] flash_attn installed via pip"
  } || true

  if [[ "${HAS_FLASH}" == "0" ]]; then
    # Method 2: look for flash_attn in sibling conda environments
    echo "[INFO] pip install failed, scanning sibling conda envs for flash_attn..."
    CONDA_ENVS_DIR="$(conda info --base)/envs"
    FLASH_SRC=""
    for env_dir in "${CONDA_ENVS_DIR}"/*/; do
      candidate="${env_dir}lib/python3.10/site-packages/flash_attn"
      if [[ -d "${candidate}" ]]; then
        FLASH_SRC="${candidate}"
        echo "[INFO] Found flash_attn in: ${env_dir}"
        break
      fi
    done
    if [[ -n "${FLASH_SRC}" ]]; then
      TARGET_DIR="$(python -c 'import site; print(site.getsitepackages()[0])')"
      if [[ -d "${TARGET_DIR}" ]]; then
        echo "[INFO] Copying flash_attn from ${FLASH_SRC} to ${TARGET_DIR}/"
        cp -r "${FLASH_SRC}" "${TARGET_DIR}/"
        # Also copy the dist-info if present
        FLASH_DIST="$(dirname "${FLASH_SRC}")/flash_attn-"*".dist-info"
        for d in ${FLASH_DIST}; do
          [[ -d "${d}" ]] && cp -r "${d}" "${TARGET_DIR}/"
        done
        check_flash_attn && HAS_FLASH=1 && echo "[OK] flash_attn copied from sibling env"
      fi
    fi
  fi

  if [[ "${HAS_FLASH}" == "0" ]]; then
    echo "[WARN] flash_attn could not be installed."
    echo "       Manual install options:"
    echo "         pip install flash-attn --no-build-isolation"
    echo "         (or copy from an env that has it built for matching torch+CUDA)"
  fi
fi

# Decide final backend
if [[ "${HAS_FLASH}" == "1" ]]; then
  CHOSEN_ATTN="flash_attn"
elif [[ "${HAS_XFORMERS}" == "1" ]]; then
  CHOSEN_ATTN="xformers"
else
  echo "[ERROR] Neither flash_attn nor xformers is functional. At least one is required."
  echo "  Install xformers:   pip install xformers"
  echo "  Install flash_attn: pip install flash-attn --no-build-isolation"
  exit 1
fi
echo ""
echo "[INFO] ✓ Selected attention backend: ${CHOSEN_ATTN}"

# Update machine env ATTN_BACKEND if it differs
if [[ -n "${MACHINE_ENV:-}" && -f "${MACHINE_ENV}" ]]; then
  CURRENT_ATTN="$(grep -oP '^\s*ATTN_BACKEND=\K\S+' "${MACHINE_ENV}" 2>/dev/null || echo "")"
  if [[ "${CURRENT_ATTN}" != "${CHOSEN_ATTN}" ]]; then
    if grep -q '^\s*ATTN_BACKEND=' "${MACHINE_ENV}" 2>/dev/null; then
      sed -i "s|^\(\s*\)ATTN_BACKEND=.*|\1ATTN_BACKEND=${CHOSEN_ATTN}|" "${MACHINE_ENV}"
      echo "[INFO] Updated ${MACHINE_ENV}: ATTN_BACKEND=${CHOSEN_ATTN} (was: ${CURRENT_ATTN:-unset})"
    else
      echo "" >> "${MACHINE_ENV}"
      echo "ATTN_BACKEND=${CHOSEN_ATTN}" >> "${MACHINE_ENV}"
      echo "[INFO] Added ATTN_BACKEND=${CHOSEN_ATTN} to ${MACHINE_ENV}"
    fi
  else
    echo "[INFO] ${MACHINE_ENV} already has ATTN_BACKEND=${CHOSEN_ATTN}"
  fi
fi

# Also update YAML configs that reference attn_backend
for yaml_cfg in "${PROJECT_ROOT}"/configs/partverse_*.yaml; do
  [[ -f "${yaml_cfg}" ]] || continue
  if grep -q 'attn_backend:' "${yaml_cfg}"; then
    OLD_YAML_ATTN="$(grep -oP 'attn_backend:\s*\K\S+' "${yaml_cfg}" | tr -d '"' | tr -d "'")"
    if [[ "${OLD_YAML_ATTN}" != "${CHOSEN_ATTN}" ]]; then
      sed -i "s|attn_backend:.*|attn_backend: \"${CHOSEN_ATTN}\"|" "${yaml_cfg}"
      echo "[INFO] Updated $(basename "${yaml_cfg}"): attn_backend=${CHOSEN_ATTN} (was: ${OLD_YAML_ATTN})"
    fi
  fi
done

# ── 5. Final verification ───────────────────────────────────────
echo ""
echo "================================================================"
echo "[5/5] Final verification"
echo "================================================================"
python - <<'PY'
import importlib, sys
checks = {
    "partcraft": "partcraft",
    "numpy": "numpy",
    "yaml": "yaml",
    "trimesh": "trimesh",
    "spconv": "spconv",
    "warp": "warp",
    "torch": "torch",
}
missing = []
for label, mod in checks.items():
    try:
        importlib.import_module(mod)
    except Exception:
        missing.append(label)
if missing:
    raise SystemExit(f"[ERROR] Missing pipeline modules: {', '.join(missing)}")
print("[OK] Core modules:", ", ".join(checks.keys()))

# Report attention
try:
    import flash_attn
    print(f"[OK] flash_attn {flash_attn.__version__}")
except ImportError:
    print("[--] flash_attn not available")
try:
    import xformers
    print(f"[OK] xformers {xformers.__version__}")
except ImportError:
    print("[--] xformers not available")

# Report CUDA
import torch
print(f"[OK] torch {torch.__version__}, CUDA {torch.version.cuda}, "
      f"GPUs: {torch.cuda.device_count()}")
PY

echo ""
echo "[DONE] Pipeline environment is ready: ${ENV_NAME}"
echo "       Attention backend: ${CHOSEN_ATTN}"
