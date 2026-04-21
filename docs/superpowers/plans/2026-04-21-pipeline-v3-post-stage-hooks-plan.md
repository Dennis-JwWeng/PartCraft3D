# Pipeline v3 Post-Stage Hooks Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `pipeline.hooks` YAML surface so external CLIs (first user: `scripts.cleaning.h3d_v1.pull_deletion --phase render`) run as chain-tail followers of pipeline v3 stages — the render hook absorbs Blender wall-time into the concurrent `flux_2d > trellis_preview` chain.

**Architecture:** Add a `Hook` dataclass + parser + placeholder resolver to `partcraft/pipeline_v3/scheduler.py`; extend `dump_stage_chains` to splice hook names (tagged `<name>@hook`) onto the end of the chain containing their `after_stage`; teach `scripts/tools/run_pipeline_v3_shard.sh` to exec external commands when it sees the `@hook` suffix. No change to per-object pipeline steps; hooks run once per shard invocation.

**Tech Stack:** Python 3 (dataclasses, stdlib logging, unittest), bash 4+, PyYAML (already used), existing `partcraft.pipeline_v3.services_cfg.pipeline_stages_raw` for YAML validation.

**Spec:** `docs/superpowers/specs/2026-04-21-pipeline-v3-post-stage-hooks-design.md`

---

## File Structure

| File | Role | Task |
|---|---|---|
| `partcraft/pipeline_v3/scheduler.py` | Add `Hook` dataclass, `hooks_for(cfg)`, chain splicing, placeholder resolver, `dump_hook_meta(cfg, name)` | 1–4 |
| `scripts/tools/run_pipeline_v3_shard.sh` | Recognise `<name>@hook` stages, exec external command, per-hook log, `SKIP_HOOKS=1` opt-out | 5 |
| `tests/test_pipeline_v3_hooks.py` | Unit tests for parsing, splicing, resolver, dump | 1–4 |
| `configs/pipeline_v3_shard06.yaml` | Declare `pull_deletion_render` hook | 6 |
| `docs/ARCH.md` | New subsection under "调度": "Post-stage hooks" | 7 |
| `docs/runbooks/h3d-v1-promote.md` | Note that `run_pipeline_v3_shard.sh` auto-runs render when hook is declared | 7 |

**Test file naming note:** existing pipeline tests live flat at `tests/test_*.py` (no `tests/pipeline_v3/` subdir exists today). The spec's `tests/pipeline_v3/test_hooks.py` reference is normalised to `tests/test_pipeline_v3_hooks.py` to match convention.

---

### Task 0: Baseline sanity

**Files:**
- Read only: `tests/test_pipeline_services.py`, `partcraft/pipeline_v3/scheduler.py`

- [ ] **Step 1: Confirm existing tests are green**

Run: `python -m pytest tests/test_pipeline_services.py tests/test_pipeline_smoke.py -v`
Expected: all green (no hooks code introduced yet).

- [ ] **Step 2: Confirm current shard06 YAML loads with scheduler**

Run:

```bash
python -c "
import yaml
from partcraft.pipeline_v3.scheduler import dump_stage_chains
cfg = yaml.safe_load(open('configs/pipeline_v3_shard06.yaml'))
stages = [s['name'] for s in cfg['pipeline']['stages']]
print(dump_stage_chains(cfg, stages))
"
```

Expected output contains `[['text_gen_gate_a'], ['del_mesh', 'flux_2d', 'trellis_preview'], ...]` (flux chain+del in the edit_branches batch, etc.).

No commit for Task 0; this is a baseline check only.

---

### Task 1: `Hook` dataclass + `hooks_for(cfg)` parser

**Files:**
- Modify: `partcraft/pipeline_v3/scheduler.py` (add after `Phase` dataclass near top)
- Create: `tests/test_pipeline_v3_hooks.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_pipeline_v3_hooks.py` with:

```python
"""Tests for pipeline_v3 post-stage hooks (spec 2026-04-21)."""
from __future__ import annotations

import unittest

from partcraft.pipeline_v3.scheduler import Hook, hooks_for


def _cfg_with_stages(**extra) -> dict:
    base = {
        "services": {"vlm": {"model": "m"}, "image_edit": {}},
        "pipeline": {
            "gpus": [0],
            "stages": [
                {"name": "del_mesh", "servers": "none", "steps": ["del_mesh"]},
                {"name": "flux_2d", "servers": "flux", "steps": ["flux_2d"]},
            ],
        },
    }
    base["pipeline"].update(extra)
    return base


class TestHooksParsing(unittest.TestCase):
    def test_no_hooks_block_returns_empty(self):
        cfg = _cfg_with_stages()
        self.assertEqual(hooks_for(cfg), [])

    def test_minimal_hook_parses(self):
        cfg = _cfg_with_stages(hooks=[{
            "name": "pull_deletion_render",
            "after_stage": "del_mesh",
            "uses": "cpu",
            "command": ["echo", "{shard}"],
        }])
        out = hooks_for(cfg)
        self.assertEqual(len(out), 1)
        h = out[0]
        self.assertIsInstance(h, Hook)
        self.assertEqual(h.name, "pull_deletion_render")
        self.assertEqual(h.after_stage, "del_mesh")
        self.assertEqual(h.uses, "cpu")
        self.assertEqual(h.command, ["echo", "{shard}"])
        self.assertEqual(h.env_passthrough, [])

    def test_env_passthrough_preserved(self):
        cfg = _cfg_with_stages(hooks=[{
            "name": "h1", "after_stage": "del_mesh", "uses": "none",
            "command": ["true"], "env_passthrough": ["PARTCRAFT_CKPT_ROOT", "HOME"],
        }])
        self.assertEqual(hooks_for(cfg)[0].env_passthrough,
                         ["PARTCRAFT_CKPT_ROOT", "HOME"])

    def test_missing_required_fields_raise(self):
        for bad in [
            {"after_stage": "del_mesh", "uses": "cpu", "command": ["x"]},
            {"name": "h", "uses": "cpu", "command": ["x"]},
            {"name": "h", "after_stage": "del_mesh", "command": ["x"]},
            {"name": "h", "after_stage": "del_mesh", "uses": "cpu"},
        ]:
            with self.subTest(bad=bad):
                cfg = _cfg_with_stages(hooks=[bad])
                with self.assertRaises(ValueError):
                    hooks_for(cfg)

    def test_unknown_after_stage_raises(self):
        cfg = _cfg_with_stages(hooks=[{
            "name": "h", "after_stage": "nope",
            "uses": "cpu", "command": ["true"],
        }])
        with self.assertRaises(ValueError) as ctx:
            hooks_for(cfg)
        self.assertIn("after_stage", str(ctx.exception))
        self.assertIn("nope", str(ctx.exception))

    def test_unknown_uses_raises(self):
        cfg = _cfg_with_stages(hooks=[{
            "name": "h", "after_stage": "del_mesh",
            "uses": "gpu", "command": ["true"],
        }])
        with self.assertRaises(ValueError):
            hooks_for(cfg)

    def test_unknown_field_raises(self):
        cfg = _cfg_with_stages(hooks=[{
            "name": "h", "after_stage": "del_mesh",
            "uses": "cpu", "command": ["true"],
            "retry": 3,
        }])
        with self.assertRaises(ValueError) as ctx:
            hooks_for(cfg)
        self.assertIn("retry", str(ctx.exception))

    def test_duplicate_name_collides_with_stage(self):
        cfg = _cfg_with_stages(hooks=[{
            "name": "del_mesh",
            "after_stage": "del_mesh",
            "uses": "cpu", "command": ["true"],
        }])
        with self.assertRaises(ValueError):
            hooks_for(cfg)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_pipeline_v3_hooks.py -v`
Expected: all fail — `ImportError: cannot import name 'Hook'` (or similar).

- [ ] **Step 3: Implement `Hook` + `hooks_for` in `partcraft/pipeline_v3/scheduler.py`**

Insert **after** the `Phase` dataclass definition (i.e. after the existing `chain_order: int = 0` line in `Phase`).

```python
_ALLOWED_HOOK_USES = frozenset({"cpu", "none"})
_ALLOWED_HOOK_FIELDS = frozenset({
    "name", "after_stage", "uses", "command", "env_passthrough",
})


@dataclass
class Hook:
    """One post-stage hook row from ``pipeline.hooks`` (spec 2026-04-21)."""

    name: str
    after_stage: str
    uses: str  # "cpu" | "none" in v1
    command: list[str] = field(default_factory=list)
    env_passthrough: list[str] = field(default_factory=list)


def hooks_for(cfg: dict) -> list[Hook]:
    """Parse ``pipeline.hooks`` into a list of :class:`Hook`.

    Returns ``[]`` when the block is absent. Validates:
      * every required field is present;
      * ``after_stage`` names an existing stage in ``pipeline.stages``;
      * ``uses`` is one of ``cpu`` / ``none`` (v1; ``gpu`` is reserved);
      * no unknown keys (guard against typos like ``timeout``);
      * hook names do not collide with stage names.
    """
    p = _pipeline(cfg)
    raw = p.get("hooks")
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError("[CONFIG] pipeline.hooks: must be a list")

    stage_names = {ph.name for ph in stages_for(cfg)}
    out: list[Hook] = []
    seen_names: set[str] = set()

    for idx, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise ValueError(f"[CONFIG] pipeline.hooks[{idx}] not a mapping: {entry!r}")

        unknown = set(entry) - _ALLOWED_HOOK_FIELDS
        if unknown:
            raise ValueError(
                f"[CONFIG] pipeline.hooks[{idx}] unknown fields: {sorted(unknown)}; "
                f"allowed: {sorted(_ALLOWED_HOOK_FIELDS)}"
            )
        for req in ("name", "after_stage", "uses", "command"):
            if req not in entry:
                raise ValueError(
                    f"[CONFIG] pipeline.hooks[{idx}] missing required field {req!r}"
                )

        name = str(entry["name"])
        after_stage = str(entry["after_stage"])
        uses = str(entry["uses"])
        command = list(entry["command"])
        env_passthrough = list(entry.get("env_passthrough") or [])

        if not command or not all(isinstance(c, str) for c in command):
            raise ValueError(
                f"[CONFIG] pipeline.hooks[{idx}] command must be a non-empty list of strings"
            )
        if uses not in _ALLOWED_HOOK_USES:
            raise ValueError(
                f"[CONFIG] pipeline.hooks[{idx}] uses={uses!r}; allowed v1: "
                f"{sorted(_ALLOWED_HOOK_USES)} (gpu reserved for follow-up spec)"
            )
        if after_stage not in stage_names:
            raise ValueError(
                f"[CONFIG] pipeline.hooks[{idx}] after_stage={after_stage!r} is not a "
                f"declared stage; known stages: {sorted(stage_names)}"
            )
        if name in stage_names:
            raise ValueError(
                f"[CONFIG] pipeline.hooks[{idx}] name={name!r} collides with an "
                "existing stage name"
            )
        if name in seen_names:
            raise ValueError(
                f"[CONFIG] pipeline.hooks[{idx}] duplicate hook name {name!r}"
            )
        seen_names.add(name)

        out.append(Hook(
            name=name,
            after_stage=after_stage,
            uses=uses,
            command=command,
            env_passthrough=env_passthrough,
        ))
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_pipeline_v3_hooks.py -v`
Expected: 8 tests pass.

- [ ] **Step 5: Commit**

```bash
git add tests/test_pipeline_v3_hooks.py partcraft/pipeline_v3/scheduler.py
git commit -m "feat(pipeline_v3): Hook dataclass + hooks_for parser"
```

---

### Task 2: Splice hooks into chain tails in `dump_stage_chains`

**Files:**
- Modify: `partcraft/pipeline_v3/scheduler.py` (extend `dump_stage_chains` + `format_stage_chains_text`)
- Modify: `tests/test_pipeline_v3_hooks.py`

- [ ] **Step 1: Append splicing tests**

Add to `tests/test_pipeline_v3_hooks.py`:

```python
from partcraft.pipeline_v3.scheduler import (  # noqa: E402
    dump_stage_chains, format_stage_chains_text,
)


def _cfg_with_parallel_group() -> dict:
    """Minimal shard06-shaped config: del_mesh || (flux_2d > trellis_preview)."""
    return {
        "services": {"vlm": {"model": "m"}, "image_edit": {}},
        "pipeline": {
            "gpus": [0],
            "stages": [
                {"name": "text_gen_gate_a", "servers": "vlm", "steps": ["gen_edits"]},
                {"name": "del_mesh", "servers": "none", "steps": ["del_mesh"],
                 "parallel_group": "edit_branches"},
                {"name": "flux_2d", "servers": "flux", "steps": ["flux_2d"],
                 "parallel_group": "edit_branches",
                 "chain_id": "flux_chain", "chain_order": 0},
                {"name": "trellis_preview", "servers": "none", "steps": ["trellis_3d"],
                 "parallel_group": "edit_branches",
                 "chain_id": "flux_chain", "chain_order": 1},
                {"name": "gate_quality", "servers": "vlm", "steps": ["gate_quality"]},
            ],
        },
    }


class TestHookChainSplice(unittest.TestCase):
    def test_chain_splice_basic(self):
        cfg = _cfg_with_parallel_group()
        cfg["pipeline"]["hooks"] = [{
            "name": "pull_deletion_render", "after_stage": "del_mesh",
            "uses": "cpu", "command": ["true"],
        }]
        stages = [s["name"] for s in cfg["pipeline"]["stages"]]
        batches = dump_stage_chains(cfg, stages)
        edit_batch = batches[1]
        chains = {c[0]: c for c in edit_batch}
        self.assertIn("del_mesh", chains)
        self.assertEqual(chains["del_mesh"],
                         ["del_mesh", "pull_deletion_render@hook"])
        self.assertIn("flux_2d", chains)
        self.assertEqual(chains["flux_2d"], ["flux_2d", "trellis_preview"])

    def test_chain_splice_format(self):
        cfg = _cfg_with_parallel_group()
        cfg["pipeline"]["hooks"] = [{
            "name": "pull_deletion_render", "after_stage": "del_mesh",
            "uses": "cpu", "command": ["true"],
        }]
        stages = [s["name"] for s in cfg["pipeline"]["stages"]]
        text = format_stage_chains_text(dump_stage_chains(cfg, stages))
        self.assertIn("del_mesh>pull_deletion_render@hook", text)
        self.assertIn("flux_2d>trellis_preview", text)

    def test_hook_dropped_when_after_stage_not_selected(self):
        cfg = _cfg_with_parallel_group()
        cfg["pipeline"]["hooks"] = [{
            "name": "pull_deletion_render", "after_stage": "del_mesh",
            "uses": "cpu", "command": ["true"],
        }]
        batches = dump_stage_chains(cfg, ["flux_2d", "trellis_preview"])
        flat = [s for batch in batches for chain in batch for s in chain]
        self.assertNotIn("pull_deletion_render@hook", flat)

    def test_multi_hook_same_stage_appended_in_order(self):
        cfg = _cfg_with_parallel_group()
        cfg["pipeline"]["hooks"] = [
            {"name": "h_a", "after_stage": "del_mesh",
             "uses": "cpu", "command": ["a"]},
            {"name": "h_b", "after_stage": "del_mesh",
             "uses": "cpu", "command": ["b"]},
        ]
        stages = [s["name"] for s in cfg["pipeline"]["stages"]]
        batches = dump_stage_chains(cfg, stages)
        del_chain = next(c for c in batches[1] if c[0] == "del_mesh")
        self.assertEqual(del_chain, ["del_mesh", "h_a@hook", "h_b@hook"])

    def test_no_hooks_matches_legacy_output(self):
        cfg = _cfg_with_parallel_group()
        stages = [s["name"] for s in cfg["pipeline"]["stages"]]
        batches = dump_stage_chains(cfg, stages)
        flat = [s for batch in batches for chain in batch for s in chain]
        self.assertNotIn("@hook", " ".join(flat))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_pipeline_v3_hooks.py::TestHookChainSplice -v`
Expected: all five splice tests fail (except `test_no_hooks_matches_legacy_output`, which passes trivially even without the change).

- [ ] **Step 3: Extend `dump_stage_chains` to splice hooks**

Edit `partcraft/pipeline_v3/scheduler.py`. At the **end** of `dump_stage_chains`, just before the final `return batches`, insert:

```python
    # Post-stage hooks (spec 2026-04-21): append each hook after the chain
    # ending with its after_stage. Hooks whose after_stage is not in the
    # selected stage_names drop silently (§4.2 of the spec).
    hooks = hooks_for(cfg)
    if hooks:
        by_after: dict[str, list[Hook]] = {}
        for h in hooks:
            by_after.setdefault(h.after_stage, []).append(h)
        for batch in batches:
            for chain in batch:
                last = chain[-1] if chain else None
                if last in by_after:
                    if len(by_after[last]) > 1:
                        log.warning(
                            "[scheduler] %d hooks share after_stage=%s; they "
                            "will run sequentially in declaration order.",
                            len(by_after[last]), last,
                        )
                    for h in by_after[last]:
                        chain.append(f"{h.name}@hook")
```

`log` is already defined at the top of `dump_stage_chains` (`log = logging.getLogger("scheduler")`). Reuse it.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_pipeline_v3_hooks.py -v`
Expected: Task 1 (8) + Task 2 (5) = 13 tests pass.

- [ ] **Step 5: Sanity-check existing scheduler tests still green**

Run: `python -m pytest tests/test_pipeline_services.py tests/test_pipeline_smoke.py -v`
Expected: unchanged pass.

- [ ] **Step 6: Commit**

```bash
git add tests/test_pipeline_v3_hooks.py partcraft/pipeline_v3/scheduler.py
git commit -m "feat(pipeline_v3): splice hooks into stage chain tails"
```

---

### Task 3: `resolve_hook_command` placeholder resolver

**Files:**
- Modify: `partcraft/pipeline_v3/scheduler.py` (append resolver)
- Modify: `tests/test_pipeline_v3_hooks.py` (new test class)

- [ ] **Step 1: Write failing resolver tests**

Append to `tests/test_pipeline_v3_hooks.py`:

```python
from pathlib import Path  # noqa: E402
from partcraft.pipeline_v3.scheduler import resolve_hook_command  # noqa: E402


class TestResolveHookCommand(unittest.TestCase):
    def _hook(self, command):
        return Hook(
            name="h", after_stage="del_mesh", uses="cpu",
            command=command, env_passthrough=[],
        )

    def _ctx(self, **over):
        base = dict(
            py_pipe="/usr/bin/python3",
            cfg_path=Path("configs/pipeline_v3_shard06.yaml"),
            shard="06",
            blender="/opt/blender/blender",
            h3d_dataset_root=Path("data/H3D_v1"),
            h3d_encode_work_dir=Path("outputs/h3d_v1_encode/06"),
        )
        base.update(over)
        return base

    def test_resolves_all_known_placeholders(self):
        h = self._hook([
            "{py_pipe}", "-m", "mod", "--cfg", "{cfg}", "--shard", "{shard}",
            "--dataset-root", "{h3d_dataset_root}",
            "--encode-work-dir", "{h3d_encode_work_dir}",
            "--blender", "{blender}",
        ])
        argv = resolve_hook_command(h, **self._ctx())
        self.assertEqual(argv, [
            "/usr/bin/python3", "-m", "mod",
            "--cfg", "configs/pipeline_v3_shard06.yaml",
            "--shard", "06",
            "--dataset-root", "data/H3D_v1",
            "--encode-work-dir", "outputs/h3d_v1_encode/06",
            "--blender", "/opt/blender/blender",
        ])

    def test_literal_argv_unchanged(self):
        h = self._hook(["echo", "no-placeholders-here"])
        argv = resolve_hook_command(h, **self._ctx())
        self.assertEqual(argv, ["echo", "no-placeholders-here"])

    def test_unknown_placeholder_raises(self):
        h = self._hook(["{unknown_key}"])
        with self.assertRaises(ValueError) as ctx:
            resolve_hook_command(h, **self._ctx())
        self.assertIn("unknown_key", str(ctx.exception))

    def test_partial_placeholder_like_string_is_literal(self):
        h = self._hook(["{", "}", "{not a placeholder}"])
        argv = resolve_hook_command(h, **self._ctx())
        self.assertEqual(argv, ["{", "}", "{not a placeholder}"])

    def test_multiple_placeholders_in_one_arg(self):
        h = self._hook(["{h3d_dataset_root}/shard-{shard}.tar"])
        argv = resolve_hook_command(h, **self._ctx())
        self.assertEqual(argv, ["data/H3D_v1/shard-06.tar"])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_pipeline_v3_hooks.py::TestResolveHookCommand -v`
Expected: fail with `ImportError: cannot import name 'resolve_hook_command'`.

- [ ] **Step 3: Implement `resolve_hook_command`**

Append to `partcraft/pipeline_v3/scheduler.py`:

```python
import re as _re
from pathlib import Path as _Path

_HOOK_PLACEHOLDER_RE = _re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")


def resolve_hook_command(
    hook: Hook,
    *,
    py_pipe: str,
    cfg_path: _Path,
    shard: str,
    blender: str,
    h3d_dataset_root: _Path,
    h3d_encode_work_dir: _Path,
) -> list[str]:
    """Expand ``{placeholder}`` tokens in ``hook.command``.

    Known placeholders (spec 2026-04-21 §3.3):
      py_pipe, cfg, shard, blender, h3d_dataset_root, h3d_encode_work_dir

    Unknown placeholders raise :class:`ValueError` — v1 keeps the surface
    closed; add new sources explicitly here.
    """
    table = {
        "py_pipe": str(py_pipe),
        "cfg": str(cfg_path),
        "shard": str(shard),
        "blender": str(blender),
        "h3d_dataset_root": str(h3d_dataset_root),
        "h3d_encode_work_dir": str(h3d_encode_work_dir),
    }

    def _sub(match: _re.Match[str]) -> str:
        key = match.group(1)
        if key not in table:
            raise ValueError(
                f"[hook:{hook.name}] unknown placeholder {{{key}}}; "
                f"known: {sorted(table)}"
            )
        return table[key]

    return [_HOOK_PLACEHOLDER_RE.sub(_sub, arg) for arg in hook.command]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_pipeline_v3_hooks.py::TestResolveHookCommand -v`
Expected: 5 tests pass.

- [ ] **Step 5: Commit**

```bash
git add tests/test_pipeline_v3_hooks.py partcraft/pipeline_v3/scheduler.py
git commit -m "feat(pipeline_v3): resolve_hook_command placeholder expansion"
```

---

### Task 4: `dump_hook_meta(cfg, name)` — shell-eval metadata

**Files:**
- Modify: `partcraft/pipeline_v3/scheduler.py` (append dumper)
- Modify: `tests/test_pipeline_v3_hooks.py` (new tests)

- [ ] **Step 1: Write failing tests**

Append to `tests/test_pipeline_v3_hooks.py`:

```python
from partcraft.pipeline_v3.scheduler import dump_hook_meta  # noqa: E402


class TestDumpHookMeta(unittest.TestCase):
    def _cfg_with_hook(self):
        return {
            "services": {"vlm": {"model": "m"}, "image_edit": {}},
            "pipeline": {
                "gpus": [0],
                "stages": [
                    {"name": "del_mesh", "servers": "none", "steps": ["del_mesh"]},
                ],
                "hooks": [{
                    "name": "pull_deletion_render",
                    "after_stage": "del_mesh",
                    "uses": "cpu",
                    "command": [
                        "{py_pipe}", "-m", "scripts.cleaning.h3d_v1.pull_deletion",
                        "--shard", "{shard}",
                    ],
                    "env_passthrough": ["PARTCRAFT_CKPT_ROOT"],
                }],
            },
        }

    def test_dump_hook_meta_shape(self):
        meta = dump_hook_meta(self._cfg_with_hook(), "pull_deletion_render")
        self.assertEqual(meta["name"], "pull_deletion_render")
        self.assertEqual(meta["uses"], "cpu")
        self.assertEqual(meta["after_stage"], "del_mesh")
        self.assertEqual(meta["command"], [
            "{py_pipe}", "-m", "scripts.cleaning.h3d_v1.pull_deletion",
            "--shard", "{shard}",
        ])
        self.assertEqual(meta["env_passthrough"], ["PARTCRAFT_CKPT_ROOT"])

    def test_dump_hook_meta_unknown_name_raises(self):
        with self.assertRaises(KeyError):
            dump_hook_meta(self._cfg_with_hook(), "nope")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_pipeline_v3_hooks.py::TestDumpHookMeta -v`
Expected: fail on import.

- [ ] **Step 3: Implement `dump_hook_meta`**

Append to `partcraft/pipeline_v3/scheduler.py`:

```python
def get_hook(cfg: dict, name: str) -> Hook:
    for h in hooks_for(cfg):
        if h.name == name:
            return h
    raise KeyError(f"hook {name!r} not in pipeline.hooks")


def dump_hook_meta(cfg: dict, name: str) -> dict:
    """Return a JSON-serialisable snapshot of one hook for shell drivers."""
    h = get_hook(cfg, name)
    return {
        "name": h.name,
        "after_stage": h.after_stage,
        "uses": h.uses,
        "command": list(h.command),
        "env_passthrough": list(h.env_passthrough),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_pipeline_v3_hooks.py -v`
Expected: all test classes (parsing + splice + resolver + dump) green.

- [ ] **Step 5: Commit**

```bash
git add tests/test_pipeline_v3_hooks.py partcraft/pipeline_v3/scheduler.py
git commit -m "feat(pipeline_v3): dump_hook_meta for shell-driver consumption"
```

---

### Task 5: Shell driver — recognise `@hook` suffix, exec command, log

**Files:**
- Modify: `scripts/tools/run_pipeline_v3_shard.sh` (new `_run_hook` + `_run_stage_bg` dispatch)

- [ ] **Step 1: Read current `_run_stage_bg` structure**

Open `scripts/tools/run_pipeline_v3_shard.sh` and locate the `_run_stage_bg()` function (≈ line 230). It runs `"$PY_PIPE" -m partcraft.pipeline_v3.run --stage <stage>`. We will add a fast-path branch at its top for `@hook` names.

- [ ] **Step 2: Insert `_run_hook` helper**

Insert the following function **above** `_run_stage_bg()` (just before the comment `# ─── single-stage invocation (background-friendly: no tee) ──────────`):

```bash
# ─── hook invocation (post-stage external command) ───────────────────

_run_hook() {
    # _run_hook <hook_name>
    # Looks up pipeline.hooks.<name>, resolves placeholders via the
    # scheduler helper, and exec's the command with env_passthrough.
    local name="$1"
    local log="$LOG_DIR/hook_${name}.log"

    if [ "${SKIP_HOOKS:-0}" = "1" ]; then
        printf "${_YEL}> Hook %-22s SKIPPED (SKIP_HOOKS=1)${_RST}\n" "$name"
        return 0
    fi

    # Resolve hook metadata + argv in one Python invocation. Output
    # layout: line 1 = 'ENV:<space-separated names>', then 'ARGV:',
    # followed by one argv element per line.
    local resolved rc
    resolved=$(
        H3D_DATASET_ROOT_DEFAULT="${H3D_DATASET_ROOT:-data/H3D_v1}" \
        H3D_ENCODE_WORK_DIR_DEFAULT="${H3D_ENCODE_WORK_DIR:-outputs/h3d_v1_encode/${SHARD}}" \
        PY_PIPE_FOR_HOOK="$PY_PIPE" \
        BLENDER_FOR_HOOK="${BLENDER_PATH:-}" \
        "$PY_PIPE" -c "
import os, yaml
from pathlib import Path
from partcraft.pipeline_v3.scheduler import get_hook, resolve_hook_command
cfg = yaml.safe_load(open('$CFG'))
h = get_hook(cfg, '$name')
blender = cfg.get('blender') or os.environ.get('BLENDER_FOR_HOOK') or ''
if not blender:
    raise SystemExit('[hook:$name] no blender path (YAML blender: or \$BLENDER_PATH)')
argv = resolve_hook_command(
    h,
    py_pipe=os.environ['PY_PIPE_FOR_HOOK'],
    cfg_path=Path('$CFG'),
    shard='$SHARD',
    blender=blender,
    h3d_dataset_root=Path(os.environ['H3D_DATASET_ROOT_DEFAULT']),
    h3d_encode_work_dir=Path(os.environ['H3D_ENCODE_WORK_DIR_DEFAULT']),
)
print('ENV:' + ' '.join(h.env_passthrough))
print('ARGV:')
for a in argv:
    print(a)
"
    )
    rc=$?
    if [ "$rc" != "0" ]; then
        echo "[scheduler] hook $name resolve failed: $resolved"
        return "$rc"
    fi

    local env_line
    env_line=$(printf '%s\n' "$resolved" | sed -n '1s/^ENV://p')
    local -a HOOK_ARGV=()
    local in_argv=0
    while IFS= read -r line; do
        if [ "$in_argv" = "1" ]; then
            HOOK_ARGV+=("$line")
        elif [ "$line" = "ARGV:" ]; then
            in_argv=1
        fi
    done <<< "$resolved"

    if [ "${#HOOK_ARGV[@]}" = "0" ]; then
        echo "[scheduler] hook $name produced empty argv"
        return 1
    fi

    local -a env_assigns=()
    if [ -n "$env_line" ]; then
        for v in $env_line; do
            if [ -z "${!v:-}" ]; then
                echo "[scheduler] hook $name env_passthrough missing: $v"
                return 1
            fi
            env_assigns+=("$v=${!v}")
        done
    fi

    printf "\n${_BOLD}> Hook  %-24s${_RST}  (after-stage tail, uses=cpu|none)\n" "$name"
    printf "  argv: %s\n" "${HOOK_ARGV[*]}"
    env "${env_assigns[@]}" "${HOOK_ARGV[@]}" > "$log" 2>&1
    rc=$?
    if [ "$rc" != "0" ]; then
        show_stage_errors "$log" "HOOK FAILED: $name"
        echo "[scheduler] hook $name exit=$rc — aborting"
    fi
    return "$rc"
}
```

- [ ] **Step 3: Dispatch `@hook` entries from `_run_stage_bg`**

At the **top** of `_run_stage_bg()` (immediately after `local stage="$1"`), insert:

```bash
    if [[ "$stage" == *@hook ]]; then
        local hook_name="${stage%@hook}"
        _run_hook "$hook_name"
        return $?
    fi
```

- [ ] **Step 4: Bash syntax check**

Run: `bash -n scripts/tools/run_pipeline_v3_shard.sh && echo OK`
Expected: `OK`.

Optional: `shellcheck scripts/tools/run_pipeline_v3_shard.sh || true` — note any new warnings vs. the pre-change baseline; pre-existing warnings are acceptable.

- [ ] **Step 5: Python-level resolver smoke test**

Write a scratch config:

```bash
cat > /tmp/hooks_smoke.yaml <<'YML'
blender: /bin/true
ckpt_root: /tmp/nonexistent
data:
  output_dir: /tmp/pipeline_v3_smoke
  mesh_root: /tmp/mesh
  images_root: /tmp/images
  slat_dir: /tmp/slat
pipeline:
  gpus: [0]
  stages:
    - name: del_mesh
      servers: none
      steps: [del_mesh]
  hooks:
    - name: echo_probe
      after_stage: del_mesh
      uses: cpu
      command: ["/bin/echo", "shard={shard}", "cfg={cfg}"]
services:
  vlm: {model: m}
  image_edit: {enabled: false}
YML
```

Then:

```bash
python -c "
import yaml
from pathlib import Path
from partcraft.pipeline_v3.scheduler import (
    get_hook, resolve_hook_command, format_stage_chains_text, dump_stage_chains,
)
cfg = yaml.safe_load(open('/tmp/hooks_smoke.yaml'))
print(format_stage_chains_text(dump_stage_chains(cfg, ['del_mesh'])))
h = get_hook(cfg, 'echo_probe')
print(resolve_hook_command(
    h, py_pipe='/usr/bin/python3',
    cfg_path=Path('/tmp/hooks_smoke.yaml'),
    shard='99', blender='/bin/true',
    h3d_dataset_root=Path('/tmp/ds'),
    h3d_encode_work_dir=Path('/tmp/enc'),
))
"
```

Expected output:

```
del_mesh>echo_probe@hook
['/bin/echo', 'shard=99', 'cfg=/tmp/hooks_smoke.yaml']
```

- [ ] **Step 6: Commit**

```bash
git add scripts/tools/run_pipeline_v3_shard.sh
git commit -m "feat(pipeline_v3): shell driver execs @hook entries"
```

---

### Task 6: Wire `pull_deletion_render` into shard06 config

**Files:**
- Modify: `configs/pipeline_v3_shard06.yaml`

- [ ] **Step 1: Append `pipeline.hooks` to shard06 config**

Edit `configs/pipeline_v3_shard06.yaml`. Insert the following block **inside** the `pipeline:` mapping, immediately after the final `- name: gate_quality` stage (as a sibling key to `stages:`):

```yaml
  hooks:
    # After del_mesh produces after_new.glb for every accepted deletion,
    # launch the Blender-only render pass of pull_deletion in parallel
    # with the flux chain. Spec: docs/superpowers/specs/
    # 2026-04-21-pipeline-v3-post-stage-hooks-design.md.
    - name: pull_deletion_render
      after_stage: del_mesh
      uses: cpu
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

- [ ] **Step 2: Lint check — scheduler can still parse**

Run:

```bash
python -c "
import yaml
from partcraft.pipeline_v3.scheduler import dump_stage_chains, format_stage_chains_text
cfg = yaml.safe_load(open('configs/pipeline_v3_shard06.yaml'))
stages = [s['name'] for s in cfg['pipeline']['stages']]
print(format_stage_chains_text(dump_stage_chains(cfg, stages)))
"
```

Expected: plan-text contains `del_mesh>pull_deletion_render@hook|flux_2d>trellis_preview` on one line, and the other batches are unchanged.

- [ ] **Step 3: Commit**

```bash
git add configs/pipeline_v3_shard06.yaml
git commit -m "config(pipeline_v3): enable pull_deletion_render hook on shard06"
```

---

### Task 7: Documentation

**Files:**
- Modify: `docs/ARCH.md` (add "Post-stage hooks" subsection)
- Modify: `docs/runbooks/h3d-v1-promote.md` (note auto-render path)

- [ ] **Step 1: Add ARCH.md subsection**

Open `docs/ARCH.md` and locate the section titled `## Trellis 多 worker`. Insert a new subsection **before** that header:

```markdown
## Post-stage hooks

在 `pipeline.hooks:` 下声明"跑完某个 stage 之后触发的外部命令"，调度器会把它作为 `<hook>@hook` 追加到对应 chain 尾部，与其它 chain 自然并行。典型用法：`del_mesh` 完成后自动启动 `pull_deletion --phase render`，和 `flux_2d > trellis_preview` 同 batch 并行跑 Blender。

- **YAML schema**、占位符清单、触发/跳过语义、失败处理见 `docs/superpowers/specs/2026-04-21-pipeline-v3-post-stage-hooks-design.md`。
- **禁用全部 hooks**：`SKIP_HOOKS=1 bash scripts/tools/run_pipeline_v3_shard.sh ...`
- **v1 限制**：`uses` 只支持 `cpu | none`；`uses: gpu` 以及 `after_hook` 跨 hook 依赖在后续 spec 中扩展。

---
```

- [ ] **Step 2: Add runbook note**

Open `docs/runbooks/h3d-v1-promote.md`. Immediately below the `# H3D_v1 promote runbook` header, insert:

```markdown
> **Auto-render via pipeline_v3 shard driver**: shards whose config declares a
> `pull_deletion_render` hook (see `pipeline.hooks` block in
> `configs/pipeline_v3_shard06.yaml`) already run `--phase render` as part of
> `run_pipeline_v3_shard.sh`. For those shards, skip the `--phase render`
> command under §2 and go straight to `--phase encode`.

```

- [ ] **Step 3: Commit**

```bash
git add docs/ARCH.md docs/runbooks/h3d-v1-promote.md
git commit -m "docs(pipeline_v3): post-stage hooks in ARCH + h3d-v1 runbook"
```

---

### Task 8: Live smoke on shard06 (manual, documented)

**Files:** None (operational verification only).

- [ ] **Step 1: Run a 3-object slice with del_mesh + hook**

Assuming shard06 data is present and machine env is sourced:

```bash
LIMIT=3 STAGES=text_gen_gate_a,del_mesh \
    bash scripts/tools/run_pipeline_v3_shard.sh shard06 \
         configs/pipeline_v3_shard06.yaml
```

- [ ] **Step 2: Verify hook ran**

```bash
ls logs/v3_shard06/hook_pull_deletion_render.log
tail -40 logs/v3_shard06/hook_pull_deletion_render.log
ls outputs/h3d_v1_encode/06/06_*/render.done 2>/dev/null | wc -l
```

Expected:
- Log exists and ends with a non-error summary from `pull_deletion`.
- `render.done` marker count ≥ 1 and matches the number of accepted `del_*` edits across the 3 objects.

- [ ] **Step 3: Verify `SKIP_HOOKS=1` opts out**

```bash
LIMIT=1 SKIP_HOOKS=1 STAGES=del_mesh \
    bash scripts/tools/run_pipeline_v3_shard.sh shard06 \
         configs/pipeline_v3_shard06.yaml
```

Expected: banner contains `Hook pull_deletion_render SKIPPED (SKIP_HOOKS=1)`; no new `hook_pull_deletion_render.log` content is appended for this run.

- [ ] **Step 4: Record findings in AI_LOG**

Append a 2026-04-21 dated entry to `docs/AI_LOG.md` summarising: wall-time overlap observed vs the previous manual pull_deletion render, any unexpected hook failures, whether shard07/08 configs should adopt the hook next.

- [ ] **Step 5: Commit AI_LOG update**

```bash
git add docs/AI_LOG.md
git commit -m "docs(ai_log): record post-stage hooks landing on shard06"
```

---

## Self-Review Notes

1. **Spec coverage:**
   - §3.1 / §3.2 (YAML surface + schema) → Task 1 parser + Task 6 config example.
   - §3.3 (placeholder resolution) → Task 3.
   - §4.1 (trigger model, chain splice) → Task 2; §4.1 example output is asserted by `test_chain_splice_format`.
   - §4.2 (selection / skip) → `test_hook_dropped_when_after_stage_not_selected`, Task 5 `SKIP_HOOKS` branch, Task 8 Step 3.
   - §4.3 (failure semantics) → Task 5 reuse of `show_stage_errors` + Task 8 Step 2.
   - §5 (code layout) → Tasks 1–7 map row-by-row to the spec table.
   - §6.1 (unit tests) → Tasks 1–4 test classes match the spec's test list.
   - §6.2 (smoke) → Task 8 Steps 1–2.
   - §6.3 (parallelism verification) → Task 8 Step 4 records observed overlap in AI_LOG; full-shard quantitative verification is operational, not a plan task.
2. **Placeholder scan:** no TBD / TODO / "similar to Task N" — all code blocks are complete.
3. **Type consistency:** `Hook` fields (`name, after_stage, uses, command, env_passthrough`) are referenced identically across Tasks 1–5; `resolve_hook_command` kwargs (`py_pipe, cfg_path, shard, blender, h3d_dataset_root, h3d_encode_work_dir`) appear unchanged in Task 3 tests, Task 5 shell Python snippet, and the shell env variable names.
