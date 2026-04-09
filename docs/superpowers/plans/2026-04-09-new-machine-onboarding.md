# New machine onboarding — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Operators on a new host can validate conda/paths once, then run **all** `pipeline_v2` phases using the canonical shell entrypoint, with **one** canonical doc and **no duplicated runbook text** elsewhere.

**Architecture:** Add a small **machine-env validator** (bash, shared helpers) invoked from existing `--check` flows; add **one** markdown doc under `docs/` that is the single onboarding narrative; add a **short pointer** in `docs/ARCH.md` (link only). Optional **compatibility note** in that doc for known env/YAML duplicate keys (VLM path), without changing Python merge logic in the first pass unless trivial.

**Tech Stack:** Bash (`set -euo pipefail`), existing `scripts/tools/setup_env_common.sh`, YAML configs unchanged for phase 1.

**Spec:** `docs/superpowers/specs/2026-04-09-new-machine-onboarding-design.md`

---

## File map (what changes)

| Path | Role |
|------|------|
| `scripts/tools/setup_env_common.sh` | Add reusable helpers: `validate_pipeline_machine_env_paths()` that checks required names are set and paths exist. |
| `scripts/tools/setup_deploy_env.sh` | When `CHECK_ONLY=1`, call path validation after `load_machine_env`. |
| `scripts/tools/setup_pipeline_env.sh` | When `CHECK_ONLY=1`, same call. |
| `scripts/tools/check_machine_env_for_pipeline.sh` | **New:** standalone entry so users run one command without full conda install. |
| `docs/new-machine-onboarding.md` | **New:** canonical onboarding (clone → copy env → validate → setup scripts → run all phases). |
| `docs/ARCH.md` | **Small edit:** link to `docs/new-machine-onboarding.md` only (no full copy-paste). |

---

### Task 1: Shared validation helpers in `setup_env_common.sh`

**Files:**
- Modify: `scripts/tools/setup_env_common.sh`

- [ ] **Step 1:** After `require_vars` (after line ~74), add:

```bash
# Required for full pipeline_v2 (deploy + pipeline conda + ckpt roots + data roots).
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

validate_pipeline_machine_env_paths() {
  local name
  require_vars "${PIPELINE_REQUIRED_MACHINE_VARS[@]}"
  for name in "${PIPELINE_REQUIRED_MACHINE_VARS[@]}"; do
    local val="${!name}"
    if [[ ! -e "${val}" ]]; then
      echo "[ERROR] ${name} path does not exist: ${val}"
      echo "  Fix in: ${MACHINE_ENV}"
      exit 1
    fi
  done
  if [[ -n "${BLENDER_PATH:-}" ]] && [[ ! -x "${BLENDER_PATH}" ]] && [[ ! -f "${BLENDER_PATH}" ]]; then
    echo "[WARN] BLENDER_PATH set but not a file: ${BLENDER_PATH}"
  fi
  echo "[CHECK] Machine env paths exist for keys: ${PIPELINE_REQUIRED_MACHINE_VARS[*]}"
}
```

- [ ] **Step 2:** Run `bash -n scripts/tools/setup_env_common.sh` — expect exit 0.

---

### Task 2: Wire validation into `setup_deploy_env.sh` and `setup_pipeline_env.sh` for `--check`

**Files:**
- Modify: `scripts/tools/setup_deploy_env.sh`
- Modify: `scripts/tools/setup_pipeline_env.sh`

- [ ] **Step 1 (`setup_deploy_env.sh`):** After `load_machine_env`, before `require_vars CONDA_ENV_SERVER`, insert:

```bash
if [[ "${CHECK_ONLY}" == "1" ]]; then
  validate_pipeline_machine_env_paths
fi
```

- [ ] **Step 2 (`setup_pipeline_env.sh`):** After `load_machine_env`, before `require_vars CONDA_ENV_PIPELINE`, insert the same block.

**Verify:**

```bash
bash scripts/tools/setup_deploy_env.sh --check
bash scripts/tools/setup_pipeline_env.sh --check
```

Expected: paths validated when `--check`; full non-`--check` runs still only `require_vars` the single env name as today (deploy does not require all paths before pip install).

**Note:** If `require_vars CONDA_ENV_SERVER` becomes redundant when `CHECK_ONLY` already ran full validation, keep both for clarity or drop duplicate — either is fine as long as behavior matches.

---

### Task 3: Standalone `scripts/tools/check_machine_env_for_pipeline.sh`

**Files:**
- Create: `scripts/tools/check_machine_env_for_pipeline.sh`

- [ ] **Step 1:** Create `scripts/tools/check_machine_env_for_pipeline.sh` with:

```bash
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
```

- [ ] **Step 2:** `chmod +x scripts/tools/check_machine_env_for_pipeline.sh`

**Verify:**

```bash
bash scripts/tools/check_machine_env_for_pipeline.sh
echo $?
```

Expected: `0` on a correctly filled machine env.

---

### Task 4: Canonical doc `docs/new-machine-onboarding.md`

**Files:**
- Create: `docs/new-machine-onboarding.md`

Sections: purpose; copy machine env template; run `check_machine_env_for_pipeline.sh`; `setup_deploy_env.sh` / `setup_pipeline_env.sh`; run `run_pipeline_v2_shard.sh` for all phases; **one** subsection on machine vs YAML + VLM path note (launcher uses `VLM_CKPT`).

---

### Task 5: Link from `docs/ARCH.md` only

**Files:**
- Modify: `docs/ARCH.md`

Add a short bullet pointing to `docs/new-machine-onboarding.md` as the single detailed new-machine guide.

---

### Task 6: Edge case — `DATA_DIR` / `OUTPUT_ROOT` empty on disk

If the team does not use those env keys literally, document: create dirs or symlinks so validation passes, or follow-up task to relax checks — note in `docs/new-machine-onboarding.md`.

---

## Spec self-review (plan vs spec)

| Spec requirement | Covered by |
|------------------|------------|
| Pre-flight checks | Tasks 1–3 |
| Full phases runbook | Task 4 |
| Single canonical doc | Tasks 4–5 |
| No duplicate config literals (code) | Out of scope; doc note only |
| Layer 1 Python `[CONFIG_ERROR]` | Unchanged (existing) |

---

## Execution handoff

Plan complete and saved to `docs/superpowers/plans/2026-04-09-new-machine-onboarding.md`.

**1. Subagent-driven (recommended)** — fresh subagent per task, review between tasks.

**2. Inline execution** — execute tasks in this session using executing-plans, checkpoints after Tasks 3 and 5.

Which approach do you want?
