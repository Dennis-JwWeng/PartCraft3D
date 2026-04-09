# New machine onboarding (scope A) — design

**Date:** 2026-04-09  
**Status:** Draft for review  
**Scope:** Make a **new machine** reach a reliable first run: correct dependencies and paths, then **all** pipeline v2 phases (`A`, `C`, `D`, `D2`, `E`, `F`; optional `B` per config), assuming **packed data** (`images` / `mesh` NPZ, etc.) already exists on that machine.

---

## 1. Problem

- Switching hosts currently costs long debugging time.
- Configuration feels split across machine env, YAML, and ad-hoc exports; the **same path** sometimes appears in more than one place, so it is unclear which value wins.
- Documentation is easy to duplicate; readers want **one** place to read.

---

## 2. Success criteria

1. **Pre-flight:** After copying and filling the machine profile, documented setup/check commands (`setup_deploy_env.sh` / `setup_pipeline_env.sh` with `--check` or equivalent) either pass or **fail fast** with a clear missing key/path and fix hint.
2. **Runtime:** Using the **same** `configs/machine/<hostname>.env` (or explicit `MACHINE_ENV=`) plus a chosen pipeline YAML, the operator can run the canonical entrypoint (`bash scripts/tools/run_pipeline_v2_shard.sh …` or `python -m partcraft.pipeline_v2.run …`) through **all phases** the project supports for that run, without guessing env vars.
3. **Auditability:** Resolved paths remain traceable (existing `[CONFIG_PATH]` / `[CONFIG_ERROR]` style is the baseline; extend only where gaps are found for "all phases").

---

## 3. Design principles

### 3.1 Single source of truth (documentation)

- **One canonical narrative** for "clone → machine profile → checks → run full pipeline."
- Other docs (including `docs/ARCH.md`) **link** to that section instead of copying long instructions. Short summaries or pointers are fine; **full duplicated runbooks are not**.

### 3.2 Single source of truth (configuration values)

- **Rule:** A given semantic (e.g. "VLM weights directory", "FLUX weights directory", "default dataset root") must have **at most one authoritative place** where the human edits the path.
- **Today:** machine env and pipeline YAML may both contain **absolute paths** to the same resource (e.g. VLM). That is **accidental duplication** and should be treated as **technical debt**, not a pattern to extend.
- **Direction for implementation:**
  - Prefer **one write location** per concern; the other layer **references** it (e.g. env substitution in YAML, or Python merge defaults from env, or shell exports only what the launcher needs from the same file)—exact mechanism is an implementation decision; the **contract** is no duplicate literals for the same meaning.
  - Until code removes all duplicates, the onboarding doc may include a **minimal compatibility table** listing known overlapping keys and **which value must be authoritative today** (shrinks as debt is paid down).

### 3.3 Two layers, different jobs (not two copies of the same path)


| Layer               | File(s)                          | Answers                                                                                                                                          |
| ------------------- | -------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Machine profile** | `configs/machine/<hostname>.env` | *Where* this computer lives: conda init, env names, host-local roots for weights/data/output if that is the convention, GPU lists for launchers. |
| **Pipeline run**    | `configs/pipeline_v2_*.yaml`     | *Which* job: shard/experiment layout, `data.`* paths for this run, `pipeline.phases`, ports/strides, per-phase options.                          |


**Mental model:** machine profile = host; YAML = experiment. **Neither** should repeat the same editable path unless the codebase explicitly defines precedence and migration is incomplete.

---

## 4. Validation strategy (recommended)

1. **Layer 0 — Shell / deploy:** Extend `setup_deploy_env.sh` and `setup_pipeline_env.sh --check` so required fields match what **full-phase** runs need, aligned with `run_pipeline_v2_shard.sh` and `docs/ARCH.md` "machine env 必填字段".
2. **Layer 1 — Python:** Keep strict load semantics: missing or invalid paths → `[CONFIG_ERROR]` with key, resolved value, and source; log `[CONFIG_PATH]` for resolved paths.
3. **Optional later:** A small smoke (`LIMIT=1`) through all phases is **not** part of this deliverable unless explicitly added; it can be a follow-up for harder proof.

---

## 5. Documentation deliverable (scope A)

- Add **one** onboarding section (exact path to be chosen in implementation; default candidate: new file under `docs/` or a dedicated subsection in `ARCH.md` with a stable anchor).
- Contents: ordered steps, copy-from template for `configs/machine/<hostname>.env`, required variables, how `MACHINE_ENV` overrides hostname-based path, how to run **all** phases, and **where** pipeline YAML is chosen.
- **Do not** duplicate full content in multiple top-level docs; link to the single canonical section.

---

## 6. Out of scope (this spec)

- Changing NPZ layout or step algorithms.
- Large refactors of `pipeline_v2` structure beyond what is needed for checks and clarity.
- Broader "config redesign" (B/C) unless strictly required to remove duplication for onboarding.

---

## 7. Next steps

1. **Review:** Stakeholder approval of this spec (and any tweak to canonical doc location).
2. **Implementation plan:** Use `writing-plans` to produce `docs/superpowers/plans/2026-04-09-new-machine-onboarding.md` with concrete file edits (scripts, `config`, docs, optional deduplication of VLM/FLUX paths).

---

## 8. Spec self-review (short)

- **Placeholders:** None intentional; canonical doc path left to implementation on purpose.
- **Consistency:** Scope A + full phases + "no duplicate config values" as principle is consistent; duplicate literals are explicitly **debt** until removed.
- **Ambiguity:** "Authoritative" path when env and YAML still differ pre-refactor — addressed by **minimal compatibility table** during transition only.

