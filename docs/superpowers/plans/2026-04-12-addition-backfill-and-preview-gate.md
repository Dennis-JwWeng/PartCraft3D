# Addition Backfill Inline + Preview-Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans.

**Goal:** Move add backfill inline into s5b, add 5-view preview render step (s6p) before VLM gate (sq3), move expensive encoding (s6b) after gate.

**Architecture:** s5b creates del PLY + add meta.json atomically. s6p renders 5xVIEW_INDICES preview frames (Blender for del/add, TRELLIS for mod/scl/mat/glb) using image_npz cameras. sq3 uses image_npz before (free) + preview_*.png after for 2x5 VLM collage covering all types. s6 reuses preview frames. s6b (survivors only) full encode + before.npz + hardlinks to add_*.

**Tech Stack:** Python 3.10+, cv2, numpy; Blender 3.5; TRELLIS render_one_view; pytest.

**Spec:** docs/superpowers/specs/2026-04-12-addition-backfill-and-preview-gate-design.md


---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| partcraft/pipeline_v2/addition_utils.py | CREATE | invert_delete_prompt with from->to fix |
| tests/test_addition_utils.py | CREATE | Unit tests for prompt inversion |
| partcraft/pipeline_v2/s5b_deletion.py | MODIFY | Remove rough NPZ; add _backfill_add; s6b before.npz + hardlinks |
| partcraft/pipeline_v2/s6_preview.py | CREATE | 5-view preview renders |
| partcraft/pipeline_v2/sq3_qc_e.py | MODIFY | 5-view collage; scan add_*/meta.json |
| partcraft/pipeline_v2/s6_render_3d.py | MODIFY | Reuse preview_{view_index}.png for after.png |
| partcraft/pipeline_v2/validators.py | MODIFY | Add check_s6p; neutralize check_s7 |
| partcraft/pipeline_v2/run.py | MODIFY | ALL_STEPS order; GPU_STEPS; _STATUS_KEYS; dispatch |
| configs/pipeline_v2_shard02.yaml | MODIFY | Add E_pre, move E after E_qc, remove F |
| partcraft/pipeline_v2/s7_addition_backfill.py | MODIFY | Convert run() to no-op |
| tests/test_pipeline_smoke.py | MODIFY | Update order assertion; add s6p smoke |

---

## Task 1: addition_utils.py — Improved Prompt Inversion

**Files:** Create `partcraft/pipeline_v2/addition_utils.py`, `tests/test_addition_utils.py`

- [ ] **Step 1.1: Write failing tests** (see spec for full test file)

Key test cases for `tests/test_addition_utils.py`:
- `"Remove the wheel"` → `"Add the wheel"`
- `"Remove the antenna from the robot"` → `"Add the antenna to the robot"` ← critical from→to fix
- `"Delete the handle from the door"` → `"Add the handle to the door"`
- `"Get rid of the bumper"` → `"Add the bumper"`
- `"Take away the wheel from the car"` → `"Add the wheel to the car"`
- `""` → `""`; `"Cut off"` → `"Add back cut off"`

- [ ] **Step 1.2: Run — expect ModuleNotFoundError**
```bash
cd /mnt/zsn/zsn_workspace/PartCraft3D
python -m pytest tests/test_addition_utils.py -v 2>&1 | head -5
```

- [ ] **Step 1.3: Create `partcraft/pipeline_v2/addition_utils.py`**

```python
from __future__ import annotations

_INVERT_VERBS: list[tuple[str, str]] = [
    ("get rid of", "add"), ("take away", "add"), ("delete", "add"),
    ("remove", "add"), ("strip", "add"), ("erase", "add"),
]

def invert_delete_prompt(prompt: str) -> str:
    """Deletion imperative -> addition imperative. 'from' -> 'to'."""
    if not prompt:
        return prompt
    p = prompt.strip()
    low = p.lower()
    result = None
    for old, new in _INVERT_VERBS:
        if low.startswith(old):
            result = new.capitalize() + p[len(old):]
            break
        idx = low.find(" " + old + " ")
        if idx >= 0:
            result = p[:idx + 1] + new + p[idx + 1 + len(old):]
            break
    if result is None:
        result = "Add back " + p[0].lower() + p[1:]
    result = result.replace(" from ", " to ", 1)
    return result

__all__ = ["invert_delete_prompt"]
```

- [ ] **Step 1.4: Run — expect 13 PASSED**
```bash
python -m pytest tests/test_addition_utils.py -v
```

- [ ] **Step 1.5: Commit**
```bash
git add partcraft/pipeline_v2/addition_utils.py tests/test_addition_utils.py
git commit -m "feat: add addition_utils with improved invert_delete_prompt (from->to fix)"
```

---

## Task 2: s5b_deletion.py — Inline Add Backfill

**Files:** Modify `partcraft/pipeline_v2/s5b_deletion.py` lines 37-164

- [ ] **Step 2.1: Replace imports (lines 37-42)** — remove `_ensure_refiner`, add `import json` and `from .addition_utils import invert_delete_prompt`

- [ ] **Step 2.2: Add `_backfill_add` after DelMeshResult dataclass (after line 53)**

```python
def _backfill_add(ctx, del_spec, add_seq, *, force=False, logger=None):
    """Create add_*/meta.json. NPZ/PNG links deferred to s6b."""
    import logging as _l, json as _j
    log = logger or _l.getLogger("pipeline_v2.s5b")
    add_id = ctx.edit_id("addition", add_seq)
    add_dir = ctx.edit_3d_dir(add_id)
    meta_path = add_dir / "meta.json"
    if meta_path.is_file() and not force:
        return False
    try:
        add_dir.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(_j.dumps({
            "edit_id": add_id, "edit_type": "addition",
            "obj_id": ctx.obj_id, "shard": ctx.shard,
            "source_del_id": del_spec.edit_id,
            "selected_part_ids": list(del_spec.selected_part_ids),
            "view_index": del_spec.view_index,
            "prompt": invert_delete_prompt(del_spec.prompt),
            "target_part_desc": del_spec.target_part_desc,
            "rationale": f"inverse of {del_spec.edit_id}",
        }, ensure_ascii=False, indent=2))
        return True
    except Exception as e:
        log.warning("[s5b] add backfill seq=%d: %s", add_seq, e); return False
```

- [ ] **Step 2.3: Replace `run_mesh_delete_for_object` (lines 55-125)** — remove refiner/SLAT code, add `add_seq` counter, call `_backfill_add` after each successful PLY export, increment `add_seq` in all branches

Key structure of new function:
```python
def run_mesh_delete_for_object(ctx, *, dataset, force=False, logger=None):
    ...
    add_seq = 0
    for spec in specs:
        if is_gate_a_failed(...): res.n_skip += 1; add_seq += 1; continue
        if a_ply.is_file() and not force: res.n_skip += 1; add_seq += 1; continue
        try:
            TrellisRefiner.direct_delete_mesh(...)
            _backfill_add(ctx, spec, add_seq, ...)
            res.n_ok += 1
        except ...:
            res.n_fail += 1
        add_seq += 1
```

- [ ] **Step 2.4: Replace `run_mesh_delete` (lines 128-164)** — remove `use_refiner` param and refiner setup

- [ ] **Step 2.5: Verify**
```bash
python -c "from partcraft.pipeline_v2.s5b_deletion import run_mesh_delete, run_reencode; print('ok')"
```

- [ ] **Step 2.6: Commit**
```bash
git add partcraft/pipeline_v2/s5b_deletion.py
git commit -m "feat(s5b): remove rough NPZ, inline add meta.json backfill"
```

---

## Task 3: New s6_preview.py — 5-View Preview Renders

**Files:** Create `partcraft/pipeline_v2/s6_preview.py`

Camera source: `load_views_from_npz(ctx.image_npz, VIEW_INDICES)` → 5 frame dicts with `transform_matrix` + `camera_angle_x`. Same cameras as `overview.png`.

Render routes:
- `del` → `run_blender(tmp_with_part_0_ply, blender, 518, [[180,180,180]], frames)`
- `add` → same as del but uses `source_del_dir / "before.ply"` (original mesh = add's "after")
- `mod/scl/mat/glb` → `render_one_view(pipeline, load_slat(after.npz), frame, 518)` × 5
- `idn` → skip

Output: `edits_3d/<edit_id>/preview_{0..4}.png`

- [ ] **Step 3.1: Create the file** (full code in spec; key functions: `_render_ply_views`, `_render_slat_views`, `_save_previews`, `_previews_exist`, `run_for_object`, `_has_trellis_edits`, `run`)

- [ ] **Step 3.2: Verify**
```bash
python -c "from partcraft.pipeline_v2.s6_preview import run, PreviewResult; print('ok')"
```

- [ ] **Step 3.3: Commit**
```bash
git add partcraft/pipeline_v2/s6_preview.py
git commit -m "feat(s6p): new 5-view preview render step using VIEW_INDICES cameras"
```

---

## Task 4: sq3_qc_e.py — 5-View Collage + Addition Coverage

**Files:** Modify `partcraft/pipeline_v2/sq3_qc_e.py` (full rewrite)

Key changes vs. current 78-line file:
1. `_collage(b_path, a_path)` → `_make_collage(before_imgs_list, after_imgs_list)` building 2×5 grid
2. `_load_before_imgs(ctx)` reads from `image_npz[VIEW_INDICES]` (zero render cost)
3. `_load_after_previews(edit_dir)` reads 5 `preview_{i}.png` files
4. `_judge_one(...)` extracts the common judge logic
5. Main loop adds iteration over `_iter_add_edits(ctx)` (add_*/meta.json on disk)

- [ ] **Step 4.1: Verify _passes baseline**
```bash
python -m pytest tests/test_sq3_passes.py -v
```

- [ ] **Step 4.2: Rewrite sq3_qc_e.py** — `_passes` function body UNCHANGED; all new helpers around it

- [ ] **Step 4.3: Verify _passes tests still pass**
```bash
python -m pytest tests/test_sq3_passes.py -v
```

- [ ] **Step 4.4: Commit**
```bash
git add partcraft/pipeline_v2/sq3_qc_e.py
git commit -m "feat(sq3): 5-view collage from preview PNGs; cover addition edits"
```

---

## Task 5: s6_render_3d.py — Reuse Preview Frame

**Files:** Modify `partcraft/pipeline_v2/s6_render_3d.py`

- [ ] **Step 5.1: Add `_hardlink_or_copy` helper after imports**

```python
import shutil as _shutil

def _hardlink_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_file(): dst.unlink()
    try: os.link(src, dst)
    except OSError: _shutil.copy2(src, dst)
```

- [ ] **Step 5.2: Modify inner loop in `run_for_object` (lines 127-142)**

Insert before the `try:` block for `which == "after"`:
```python
            if which == "after":
                preview = edit_dir / f"preview_{spec.view_index}.png"
                if preview.is_file():
                    _hardlink_or_copy(preview, png); res.n_ok += 1; continue
```

- [ ] **Step 5.3: Verify import + Step 5.4: Commit**
```bash
python -c "from partcraft.pipeline_v2.s6_render_3d import run; print('ok')"
git add partcraft/pipeline_v2/s6_render_3d.py
git commit -m "feat(s6): reuse preview_{view_index}.png for after.png"
```

---

## Task 6: s5b_deletion.py (s6b section) — before.npz + Hardlinks

**Files:** Modify `partcraft/pipeline_v2/s5b_deletion.py` s6b section (~lines 167-239)

- [ ] **Step 6.1: Add 3 helpers before `run_reencode_for_object` (~line 177)**

```python
def _hardlink(src, dst):
    """os.link with shutil.copy2 fallback."""
    ...

def _write_before_npz(ctx, slat_dir: Path, before_npz: Path):
    """Load {slat_dir}/{shard}/{obj_id}_coords.pt + _feats.pt -> before.npz
    with keys slat_coords + slat_feats (matches load_slat() reads)."""
    import torch
    coords = torch.load(slat_dir/ctx.shard/f"{ctx.obj_id}_coords.pt", map_location="cpu").numpy()
    feats  = torch.load(slat_dir/ctx.shard/f"{ctx.obj_id}_feats.pt",  map_location="cpu").numpy()
    np.savez(str(before_npz), slat_coords=coords, slat_feats=feats)

def _link_add_pair(ctx, spec, add_seq, pair_dir, logger=None):
    """Hardlink del/{after,before}.{npz,png} -> add/{before,after}.{npz,png}."""
    add_id = ctx.edit_id("addition", add_seq)
    add_dir = ctx.edit_3d_dir(add_id)
    if not (add_dir / "meta.json").is_file(): return
    _hardlink(pair_dir/"after.npz",  add_dir/"before.npz")
    _hardlink(pair_dir/"before.npz", add_dir/"after.npz")
    for dn, an in [("after.png","before.png"), ("before.png","after.png")]:
        if (pair_dir/dn).is_file(): _hardlink(pair_dir/dn, add_dir/an)
```

- [ ] **Step 6.2: Update `run_reencode_for_object` signature** — add `slat_dir: Path` param

- [ ] **Step 6.3: Update loop body** — add `add_seq` counter; after `np.savez(a_npz, **payload)`:
```python
            before_npz = pair_dir / "before.npz"
            if not before_npz.is_file() or force:
                _write_before_npz(ctx, slat_dir, before_npz)
            _link_add_pair(ctx, spec, add_seq, pair_dir, logger=log)
```
Increment `add_seq += 1` at end of each loop iteration.

- [ ] **Step 6.4: Update `run_reencode`** — add `slat_dir = Path(cfg.get("data",{}).get("slat_dir",""))` and pass to `run_reencode_for_object`

- [ ] **Step 6.5: Verify + Commit**
```bash
python -c "from partcraft.pipeline_v2.s5b_deletion import run_reencode; print('ok')"
git add partcraft/pipeline_v2/s5b_deletion.py
git commit -m "feat(s6b): generate before.npz from SLAT .pt + hardlink NPZ/PNG to add_*"
```

---

## Task 7: validators.py — Add check_s6p

**Files:** Modify `partcraft/pipeline_v2/validators.py`

- [ ] **Step 7.1: Add `check_s6p` after `check_s6b` (line ~158)**

```python
def check_s6p(ctx: ObjectContext) -> StepCheck:
    gate = _require_phase1("s6p_preview", ctx)
    if gate is not None: return gate
    if not ctx.edits_3d_dir.is_dir():
        return StepCheck(step="s6p_preview", ok=True, expected=0, found=0)
    paths = [(f"{d.name}/preview_0.png", d/"preview_0.png")
             for d in sorted(ctx.edits_3d_dir.iterdir())
             if d.is_dir() and d.name.split("_")[0] != "idn"]
    return _check_files("s6p_preview", paths)
```

- [ ] **Step 7.2: Replace `check_s7` body with no-op**
```python
def check_s7(ctx: ObjectContext) -> StepCheck:
    return StepCheck(step="s7_add_backfill", ok=True, expected=0, found=0, skip=True)
```

- [ ] **Step 7.3: Add `"s6p": check_s6p` to VALIDATORS dict (after `"s5b"`).**

- [ ] **Step 7.4: Verify + Commit**
```bash
python -c "from partcraft.pipeline_v2.validators import VALIDATORS; assert 's6p' in VALIDATORS; print('ok')"
git add partcraft/pipeline_v2/validators.py
git commit -m "feat(validators): add check_s6p; neutralize check_s7"
```

---

## Task 8: run.py — Fix Stage Order + Dispatch

**Files:** Modify `partcraft/pipeline_v2/run.py`

- [ ] **Step 8.1: Lines 53-54** — new ALL_STEPS and GPU_STEPS:
```python
ALL_STEPS = ("s1", "s2", "sq1", "s4", "s5", "s5b", "s6p", "sq2", "sq3", "s6", "s6b")
GPU_STEPS = frozenset({"s5", "s6p", "s6", "s6b"})
```

- [ ] **Step 8.2: After s5b dispatch block** — add s6p:
```python
    elif step == "s6p":
        from .s6_preview import run as s6p_run
        blender = resolve_blender_executable(cfg)
        ckpt = psvc.image_edit_service(cfg).get("trellis_text_ckpt","checkpoints/TRELLIS-text-xlarge")
        s6p_run(ctxs, blender_path=blender, ckpt=ckpt, force=args.force, logger=log)
```

- [ ] **Step 8.3: s7 dispatch (lines 218-220)** — convert to:
```python
    elif step == "s7":
        log.info("[s7] no-op: addition backfill is now inline in s5b")
```

- [ ] **Step 8.4: _STATUS_KEYS (lines 475-480)**:
```python
_STATUS_KEYS = {
    "s1":"s1_phase1","s2":"s2_highlights","s4":"s4_flux_2d",
    "s5":"s5_trellis","s5b":"s5b_del_mesh","s6p":"s6p_preview",
    "s6":"s6_render_3d","s6b":"s6b_del_reencode",
    "sq1":"sq1_qc_A","sq2":"sq2_qc_C","sq3":"sq3_qc_E",
}
```

- [ ] **Step 8.5: Verify**
```bash
python -c "
from partcraft.pipeline_v2.run import ALL_STEPS, GPU_STEPS, _STATUS_KEYS
steps = list(ALL_STEPS)
assert steps.index('s6p') < steps.index('sq3')
assert steps.index('sq3') < steps.index('s6')
assert 's7' not in steps and 's6p' in GPU_STEPS
print('ok:', ALL_STEPS)
"
```

- [ ] **Step 8.6: Commit**
```bash
git add partcraft/pipeline_v2/run.py
git commit -m "fix(run): move sq3 before s6/s6b; add s6p dispatch; s7 no-op"
```

---

## Task 9: configs/pipeline_v2_shard02.yaml — Update Stage Order

**Files:** Modify `configs/pipeline_v2_shard02.yaml` lines 33-40

- [ ] **Step 9.1: Replace `stages:` block**

```yaml
  stages:
  - {name: A,     desc: "phase1 VLM + QC-A",      servers: vlm,  steps: [s1, sq1]}
  - {name: C,     desc: "FLUX 2D",                 servers: flux, steps: [s4]}
  - {name: D,     desc: "TRELLIS 3D edit",          servers: none, steps: [s5],      use_gpus: true}
  - {name: D2,    desc: "deletion PLY + add meta",  servers: none, steps: [s5b]}
  - {name: E_pre, desc: "5-view preview render",    servers: none, steps: [s6p],     use_gpus: true}
  - {name: E_qc,  desc: "QC-E final (all types)",   servers: vlm,  steps: [sq3]}
  - {name: E,     desc: "3D render + del encode",   servers: none, steps: [s6, s6b], use_gpus: true}
```

- [ ] **Step 9.2: Verify + Commit**
```bash
python -c "
import yaml; from pathlib import Path
cfg = yaml.safe_load(Path('configs/pipeline_v2_shard02.yaml').read_text())
names = [s['name'] for s in cfg['pipeline']['stages']]
assert 'E_pre' in names and 'F' not in names
assert names.index('E_pre') < names.index('E_qc') < names.index('E')
print('ok:', names)
"
git add configs/pipeline_v2_shard02.yaml
git commit -m "config(shard02): add E_pre; gate before encode; remove F"
```

---

## Task 10: s7_addition_backfill.py — No-Op Stub

**Files:** Modify `partcraft/pipeline_v2/s7_addition_backfill.py` lines 167-180

- [ ] **Step 10.1: Replace `run()` function**
```python
def run(ctxs, *, force=False, logger=None):
    """No-op: addition backfill is now inline in s5b."""
    import logging as _l
    log = logger or _l.getLogger("pipeline_v2.s7")
    log.info("[s7] no-op: addition backfill moved to s5b")
    return []
```

- [ ] **Step 10.2: Verify + Commit**
```bash
python -c "from partcraft.pipeline_v2.s7_addition_backfill import run; assert run([]) == []; print('ok')"
git add partcraft/pipeline_v2/s7_addition_backfill.py
git commit -m "feat(s7): convert to no-op stub"
```

---

## Task 11: tests/test_pipeline_smoke.py — Update for New Order

**Files:** Modify `tests/test_pipeline_smoke.py`

- [ ] **Step 11.1: Fix `test_all_steps_tuple_order` (lines 100-106)**

```python
    def test_all_steps_tuple_order(self):
        """s6p before sq3; sq3 before s6 (gate before encode); s7 removed."""
        from partcraft.pipeline_v2.run import ALL_STEPS
        steps = list(ALL_STEPS)
        assert steps.index("sq1") < steps.index("s4"),  "sq1 before s4"
        assert steps.index("sq2") > steps.index("s4"),  "sq2 after s4"
        assert steps.index("s6p") < steps.index("sq3"), "s6p before sq3"
        assert steps.index("sq3") < steps.index("s6"),  "sq3 before s6"
        assert "s7" not in steps, "s7 must not be in ALL_STEPS"
```

- [ ] **Step 11.2: Add to `TestImports`** (after `test_sq3_importable`)

```python
    def test_s6p_importable(self):
        from partcraft.pipeline_v2 import s6_preview
        assert hasattr(s6_preview, "run") and hasattr(s6_preview, "PreviewResult")

    def test_addition_utils_importable(self):
        from partcraft.pipeline_v2 import addition_utils
        assert hasattr(addition_utils, "invert_delete_prompt")

    def test_s7_is_noop(self):
        from partcraft.pipeline_v2 import s7_addition_backfill
        assert s7_addition_backfill.run([]) == []
```

- [ ] **Step 11.3: Update `test_validators_importable`** — add `assert "s6p" in validators.VALIDATORS`

- [ ] **Step 11.4: Update `_MINIMAL_YAML` stages** — add `E_pre` stage with `s6p`, remove `F`, ensure `E_qc` before `E`

- [ ] **Step 11.5: Run all tests**
```bash
python -m pytest tests/test_pipeline_smoke.py tests/test_sq3_passes.py tests/test_addition_utils.py -v
```
Expected: all pass.

- [ ] **Step 11.6: Commit**
```bash
git add tests/test_pipeline_smoke.py
git commit -m "test: update smoke tests for new stage order (s6p, sq3->s6, no s7)"
```

---

## Self-Review

**Spec coverage (all 15 requirements):**

| Requirement | Task |
|---|---|
| add backfill inline in s5b | 2 |
| improved invert_delete_prompt (from->to) | 1 |
| remove rough NPZ from s5b | 2 |
| new s6p 5-view preview renders | 3 |
| same VIEW_INDICES cameras | 3 |
| before images from image_npz free | 4 |
| sq3 covers addition edits | 4 |
| sq3 5-view collage | 4 |
| s6 reuses preview for after.png | 5 |
| s6b before.npz from SLAT .pt | 6 |
| s6b hardlinks NPZ/PNG to add_* | 6 |
| ALL_STEPS sq3 before s6/s6b | 8 |
| s7 no-op | 10 |
| validators check_s6p | 7 |
| config E_pre F removed | 9 |

**Type consistency:** _passes body unchanged; VIEW_INDICES single source from specs.py; _write_before_npz saves slat_coords+slat_feats matching load_slat() reads; run_reencode passes slat_dir to run_reencode_for_object.
