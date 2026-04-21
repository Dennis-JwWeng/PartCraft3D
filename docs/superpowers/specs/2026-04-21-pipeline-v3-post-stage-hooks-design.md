# Pipeline v3 Post-Stage Hooks — External Command Orchestration

**Status**: Design (approved by user 2026-04-21)
**Date**: 2026-04-21
**Owners**: PartCraft3D pipeline team
**Related**:
- `docs/ARCH.md` — pipeline v3 architecture & scheduler
- `docs/runbooks/h3d-v1-promote.md` — H3D_v1 promote runbook (current manual invocation)
- `docs/superpowers/specs/2026-04-19-h3d-v1-design.md` — H3D_v1 dataset layout & `pull_*` CLIs

---

## 1. Why

Pipeline v3 produces per-edit artifacts that downstream tools (first consumer: `scripts.cleaning.h3d_v1.pull_deletion --phase render`) promote into `data/H3D_v1/`. Today these tools are **run manually after the shard finishes** via `run_pipeline_v3_shard.sh`. Two concrete losses:

1. **Idle CPU during GPU work.** `pull_deletion --phase render` is Blender-only (no GPU, no torch). While the `flux_2d > trellis_preview` chain occupies all GPUs for hours, the CPU side is idle. If render kicks off the moment `del_mesh` finishes, the two runs overlap and render wall-time is absorbed "for free" inside the FLUX chain.
2. **No declarative place to land future post-processing.** As more pulls / packers / index-builders land (pull_flux, pull_addition, pull_deletion encode phase, pack_shard, build_h3d_v1_index, …) each one becomes another hand-typed command in the runbook. We want a single config surface that future post-processors slot into without new ad-hoc wiring.

## 2. Goals & non-goals

**Goals**
- A **single** YAML surface (`pipeline.hooks`) declaring post-stage external commands, trigger stages, and resource class.
- First hook delivered: `pull_deletion_render` — triggers on `del_mesh` completion and runs in parallel with the `flux_2d > trellis_preview` chain in the same `edit_branches` batch.
- Reuse existing stage/chain/parallel_group scheduling; hooks are modelled as appended chain tails so no new top-level batch concept is introduced.
- Additive only: existing `pipeline.stages` YAML parsing and shell driver control flow are unchanged; shards without `pipeline.hooks` run exactly as today.

**Non-goals (v1)**
- GPU-class hooks with compute-serialisation (`uses: gpu`). Needed when `pull_deletion --phase encode` is wired up — deferred to a follow-up spec.
- Cross-hook DAG (`after_hook: ...`). Needed when `pull_addition` is wired up (waits for `pull_deletion` promote) — deferred.
- Retries / timeouts / structured status persistence. Hooks are expected to be idempotent CLIs; failures abort the run and are re-invoked by re-running the shard driver.
- Per-object hook triggering. Hooks are per-shard, invoked **once** after the upstream stage's per-object loop terminates.

## 3. Data model

### 3.1 YAML surface

```yaml
pipeline:
  stages: [ ... unchanged ... ]

  hooks:
    - name: pull_deletion_render
      after_stage: del_mesh
      uses: cpu                 # enum: cpu | none  (v1); gpu deferred
      command:
        - "{py_pipe}"
        - "-m"
        - "scripts.cleaning.h3d_v1.pull_deletion"
        - "--pipeline-cfg"
        - "{cfg}"
        - "--shard"
        - "{shard}"
        - "--dataset-root"
        - "{h3d_dataset_root}"
        - "--phase"
        - "render"
        - "--encode-work-dir"
        - "{h3d_encode_work_dir}"
        - "--blender"
        - "{blender}"
      env_passthrough: [PARTCRAFT_CKPT_ROOT]
```

### 3.2 Hook schema

| Field | Type | Required | Meaning |
|---|---|---|---|
| `name` | str | yes | Hook identifier; must be unique across `pipeline.stages` names and `pipeline.hooks` names. Used in logs (`logs/v3_<tag>/hook_<name>.log`) and in scheduler chain output as `<name>@hook`. |
| `after_stage` | str | yes | Name of a stage in `pipeline.stages`. Hook runs after that stage's chain completes. v1 requires this to be an existing stage name; unknown names raise a config error. |
| `uses` | str | yes | `cpu` or `none`. v1 treats both identically (no mutex); the distinction exists so `uses: gpu` can be added later without schema churn. |
| `command` | list[str] | yes | argv list with placeholders (see §3.3). Each element is a string; `{...}` is expanded before exec. No shell word-splitting — exec is direct. |
| `env_passthrough` | list[str] | no | Names of env vars that must be present and forwarded into the hook process. Missing vars raise an error at hook launch. Other env vars are inherited from the shell driver (standard behaviour). |

All other fields are reserved for future versions and raise `ValueError` if present in v1 to force early failure on typos.

### 3.3 Placeholder resolution

Placeholders are `{name}` literals in any `command` element. Resolution happens once, in the shell driver, via a Python helper `partcraft.pipeline_v3.scheduler.resolve_hook_command`. Recognised names (v1):

| Placeholder | Source | Error if missing |
|---|---|---|
| `{py_pipe}` | The `$PY_PIPE` already resolved by the shell driver (conda env `pipeline` python). | yes |
| `{cfg}` | `--config` (= the YAML path passed to `run_pipeline_v3_shard.sh`). | yes |
| `{shard}` | `$SHARD` resolved by the shell driver. | yes |
| `{blender}` | YAML top-level `blender:` key, else env `$BLENDER_PATH`. | yes |
| `{h3d_dataset_root}` | env `$H3D_DATASET_ROOT`, default `data/H3D_v1` (resolved relative to repo root). | no (defaulted) |
| `{h3d_encode_work_dir}` | env `$H3D_ENCODE_WORK_DIR`, default `outputs/h3d_v1_encode/<shard>` (resolved relative to repo root). | no (defaulted) |

Unknown placeholders raise a hard error at resolution time — v1 does not support free-form env interpolation in YAML. If a new source is needed, it's added to the resolver explicitly (keeps the surface closed and greppable).

## 4. Scheduling semantics

### 4.1 Trigger model

Hooks are flattened into the existing batch/chain structure by appending them as chain tails:

- For each hook, find the chain in `dump_stage_chains` output that **ends with** `hook.after_stage`. Append `<hook.name>@hook` to that chain.
- Multiple hooks on the same `after_stage` append in YAML declaration order; they run **sequentially** within that chain (because chains are serial).
  - *Rationale:* v1 doesn't need parallel sibling hooks. If two hooks on the same stage must run in parallel, the second hook is hoisted into a sibling chain via a follow-up feature; v1 raises a warning if it detects this (≥ 2 hooks with the same `after_stage`) so we notice early.
- Chains in sibling chains of the same `parallel_group` continue to run **in parallel** with the hook-bearing chain. This is exactly what delivers the requested speedup: `pull_deletion_render` runs concurrently with `flux_2d > trellis_preview`.

**Example — shard06 config + v1 hook:**

Before (batches from `dump_stage_chains`):

```
text_gen_gate_a
del_mesh | flux_2d > trellis_preview
gate_quality
```

After:

```
text_gen_gate_a
del_mesh > pull_deletion_render@hook | flux_2d > trellis_preview
gate_quality
```

### 4.2 Selection & skip semantics

- Hooks inherit the selection of their `after_stage`. If `STAGES=...` does not include `del_mesh`, `pull_deletion_render` is not executed and not listed in the batch plan.
- There is **no** separate `HOOKS=...` env in v1.
- If a stage is **skipped** by the per-stage pending-count pre-check (see `_run_stage_bg` in the shell driver), the hook still runs. Rationale: skipping means no new per-object work, but the shard's outputs still exist and the hook's upstream CLI is idempotent — running it is cheap and guarantees consistency when the dataset root is stale.
  - *Escape hatch:* `SKIP_HOOKS=1` env suppresses **all** hook execution for the current run (no partial control in v1).

### 4.3 Failure semantics

- Hook exit code ≠ 0 aborts the entire shard run, identical to stage failure. `show_stage_errors` is reused, tagged as `HOOK FAILED:`.
- No retries in v1; re-running `run_pipeline_v3_shard.sh` with the same args re-triggers the hook (CLIs are idempotent per the existing runbook).
- Hook logs land at `logs/v3_<tag>/hook_<name>.log`.

### 4.4 Interaction with server lifecycle

- v1 hooks declare `uses: cpu|none` and therefore do not need to wait for VLM/FLUX server shutdown.
- The hook's upstream stage (e.g. `del_mesh` with `servers: none`) also doesn't hold a server, so the trigger point has no server teardown cost.
- When `uses: gpu` is added in a follow-up, the contract will be: (a) hook runs only after all GPU-holding stages in the same batch finish, and (b) hook inherits the batch's GPU pool. That contract is explicitly out of scope here but the schema leaves room (`uses` is already an enum).

## 5. Code layout

| File | Change |
|---|---|
| `partcraft/pipeline_v3/scheduler.py` | Add `Hook` dataclass, `hooks_for(cfg) -> list[Hook]`, extend `dump_stage_chains` to splice hooks into chain tails, update `format_stage_chains_text` to emit `<name>@hook`. Add `resolve_hook_command(cfg, hook, *, py_pipe, cfg_path, shard, blender) -> list[str]`. |
| `partcraft/pipeline_v3/scheduler.py` | Add `dump_hook_meta(cfg, name) -> dict` exposing `{name, command, env_passthrough}` for the shell driver to eval (mirrors `dump_shell_env` pattern). |
| `scripts/tools/run_pipeline_v3_shard.sh` | `_run_stage_bg` recognises `<name>@hook` suffix: resolve command via `dump_hook_meta`, exec with env passthrough, log to `logs/v3_<tag>/hook_<name>.log`. Honour `SKIP_HOOKS=1`. |
| `configs/pipeline_v3_shard06.yaml` | Add `pipeline.hooks: [pull_deletion_render]` entry with the command skeleton from §3.1. |
| `docs/ARCH.md` | New subsection after "调度": "Post-stage hooks" — short description + pointer to this spec. |
| `docs/runbooks/h3d-v1-promote.md` | Note at the top: "If you invoked via `run_pipeline_v3_shard.sh` on a config with `pull_deletion_render` hook, the render pass already ran — skip §2 step 1 `--phase render`." |
| `tests/pipeline_v3/test_hooks.py` | New file. Unit tests for `hooks_for` parsing, `dump_stage_chains` hook splicing, and `resolve_hook_command` placeholder expansion (including error paths: unknown placeholder, missing `env_passthrough`, unknown `after_stage`). |

Net estimated change: ≤ 200 LOC production + ~120 LOC tests + ~60 LOC docs.

## 6. Testing strategy

### 6.1 Unit

- `test_hook_parsing`: malformed YAML (missing `name`/`after_stage`/`command`, unknown field, non-existent `after_stage`) all raise with clear messages.
- `test_chain_splice`: fixture config with two stages + one hook; confirm output batches/chains contain `<hook>@hook` at the correct position.
- `test_chain_splice_parallel_group`: hook on a stage inside a `parallel_group` with a sibling chain; confirm the sibling chain is unchanged and still parallel to the hook-bearing chain.
- `test_chain_splice_multi_hook_same_stage`: two hooks both on `del_mesh`; confirm both appended in YAML order and a warning is logged.
- `test_resolve_command`: all placeholders resolve; unknown placeholder raises; `env_passthrough` with a missing env var raises.

### 6.2 Smoke (manual, documented in runbook)

On shard06 (or any existing pipeline_v3 shard):

```bash
LIMIT=3 STAGES=text_gen_gate_a,del_mesh \
    bash scripts/tools/run_pipeline_v3_shard.sh shard06 \
         configs/pipeline_v3_shard06.yaml
```

Expected:

1. `del_mesh` completes per §3.1 of ARCH.
2. `pull_deletion_render` hook auto-executes after `del_mesh` (inherited from its `after_stage: del_mesh`); log visible at `logs/v3_shard06/hook_pull_deletion_render.log`.
3. `outputs/h3d_v1_encode/06/06_*/render.done` markers present for all accepted deletion edits in the 3-object subset.
4. Negative: with `BLENDER_PATH=/tmp/nope` exported into the run, the shard aborts with `HOOK FAILED: pull_deletion_render` and the log tail of the hook.

### 6.3 Parallelism verification (manual)

On a full shard with all stages enabled, inspect `logs/v3_<tag>/chain_del_mesh.log` and `logs/v3_<tag>/chain_flux_2d.log` timestamps to confirm `hook_pull_deletion_render.log`'s render loop overlaps in wallclock with the FLUX chain's stages. Acceptance: ≥ 50 % of render wall-time is within the FLUX chain's lifespan on shard06.

## 7. Rollout

1. Land the scheduler + shell driver + tests (no config change). Hooks-empty configs behave identically to today.
2. Add `pipeline.hooks: [pull_deletion_render]` to `configs/pipeline_v3_shard06.yaml`. Re-run shard06 end-to-end; verify §6.2 smoke.
3. After one successful shard, propagate the hook definition into other shard configs (07/08/09/…).
4. Follow-up spec (not this one) extends the mechanism with `uses: gpu`, `after_hook:`, and wires up `pull_deletion_encode`, `pull_flux`, `pull_addition` in declaration order.

## 8. Open questions (answered by brainstorming)

- **Q:** Put external commands as `kind: external` stages inside `pipeline.stages`, or separate `pipeline.hooks`?
  **A:** Separate `pipeline.hooks`. Rationale: keeps the "core pipeline" stages and "post-processing" hooks visually distinct in YAML; hooks are tied to **their upstream stage name**, which is a cleaner dependency than composing stages via `chain_id`/`chain_order`.
- **Q:** Should hooks be selectable independently via `HOOKS=...`?
  **A:** Not in v1. Selection follows the upstream stage. Re-running with `STAGES=del_mesh` triggers the hook; `SKIP_HOOKS=1` opts out globally.
- **Q:** What about `pull_flux` / `pull_addition` / encode phase?
  **A:** Out of scope for v1 per explicit user decision. Once `uses: gpu` lands, `pull_deletion_encode` can be added with `after_stage: trellis_preview uses: gpu`; `pull_flux` with `after_stage: gate_quality`; `pull_addition` requires cross-hook DAG (`after_hook: pull_deletion_encode`).
