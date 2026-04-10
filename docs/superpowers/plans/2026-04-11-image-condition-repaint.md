# Image-Condition Repaint Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Switch the TRELLIS repaint pipeline from fixed interleaved text+image steps to configurable mode (image-only by default), with mask alignment safeguards.

**Architecture:** Add `mode` param to `interweave_Trellis_TI` driving step selection via `_use_text_step`; wire through `TrellisRefiner.edit(repaint_mode)` and YAML config. Inversion steps unchanged (text model, cfg=0).

**Tech Stack:** Python, PyTorch, TRELLIS flow models (text + image), DINOv2 image encoder, YAML pipeline config.

---

## File Map

| File | Change |
|---|---|
| `third_party/interweave_Trellis.py` | Add `_use_text_step`, `mode` param, replace 2× `cnt%2==0` |
| `partcraft/trellis/refiner.py` | Add `repaint_mode` to `edit()`; pass to `interweave_Trellis_TI`; mask inflation guard |
| `partcraft/pipeline_v2/s5_trellis_3d.py` | Read `repaint_mode` from config; pass to `refiner.edit()` |
| `configs/pipeline_v2_shard00.yaml` | Add `repaint_mode: image` |
| `configs/pipeline_v2_shard02.yaml` | Add `repaint_mode: image` |
| `configs/pipeline_v2_gpu01.yaml` | Add `repaint_mode: image` |
| `tests/test_qc_io.py` | Add tests for `is_gate_a_failed` |

---

## Task 1: Add `mode` param to `interweave_Trellis_TI`

**Files:**
- Modify: `third_party/interweave_Trellis.py:329-340` (before function def + signature)
- Modify: `third_party/interweave_Trellis.py:424` (S1 repaint if-branch)
- Modify: `third_party/interweave_Trellis.py:477` (S2 repaint if-branch)

- [ ] **Step 1: Add `_use_text_step` helper and `mode` param**

Insert the helper function immediately before `def interweave_Trellis_TI` and add the `mode` parameter:

```python
# Insert just before line 331 (def interweave_Trellis_TI):
def _use_text_step(cnt: int, mode: str) -> bool:
    """Return True if this forward-repaint step should use the text flow model.

    mode choices:
      'interleaved' - alternate text/image every step (original behaviour)
      'text'        - always use the text model
      'image'       - always use the image model
    """
    if mode == 'text':
        return True
    if mode == 'image':
        return False
    return cnt % 2 == 0  # 'interleaved'


# Change signature from:
def interweave_Trellis_TI(args, trellis_text, trellis_img,
    slat, mask,
    prompts,
    img_new,
    seed):

# To:
def interweave_Trellis_TI(args, trellis_text, trellis_img,
    slat, mask,
    prompts,
    img_new,
    seed,
    mode: str = 'interleaved'):
```

- [ ] **Step 2: Replace S1 `cnt%2 == 0` with `_use_text_step`**

Find the S1 forward-repaint block (currently line ~424). Change:
```python
# BEFORE
                if cnt%2 == 0:
                    s1_text_sampler_params['cfg_strength'] = args['cfg_strength']
                    x_t_1 = RF_sample_once(text_s1_flow_model, ...)
                else:
                    s1_img_sampler_params['cfg_strength'] = 5.0
                    x_t_1 = RF_sample_once(img_s1_flow_model, ...)

# AFTER
                if _use_text_step(cnt, mode):
                    s1_text_sampler_params['cfg_strength'] = args['cfg_strength']
                    x_t_1 = RF_sample_once(text_s1_flow_model, ...)
                else:
                    s1_img_sampler_params['cfg_strength'] = 5.0
                    x_t_1 = RF_sample_once(img_s1_flow_model, ...)
```

- [ ] **Step 3: Replace S2 `cnt%2 == 0` with `_use_text_step`**

Find the S2 forward-repaint block (currently line ~477). Change:
```python
# BEFORE
                if cnt%2 == 0:
                    s2_text_sampler_params['cfg_strength'] = args['cfg_strength']
                    x_t_1 = RF_sample_once(text_s2_flow_model, ...)
                else:
                    s2_img_sampler_params['cfg_strength'] = 5.0
                    x_t_1 = RF_sample_once(img_s2_flow_model, ...)

# AFTER
                if _use_text_step(cnt, mode):
                    s2_text_sampler_params['cfg_strength'] = args['cfg_strength']
                    x_t_1 = RF_sample_once(text_s2_flow_model, ...)
                else:
                    s2_img_sampler_params['cfg_strength'] = 5.0
                    x_t_1 = RF_sample_once(img_s2_flow_model, ...)
```

- [ ] **Step 4: Smoke-test the import**

```bash
cd /mnt/zsn/zsn_workspace/PartCraft3D
python3 -c "
import sys; sys.path.insert(0, 'third_party')
from interweave_Trellis import interweave_Trellis_TI, _use_text_step
import inspect
sig = inspect.signature(interweave_Trellis_TI)
assert 'mode' in sig.parameters, 'mode param missing'
assert _use_text_step(0, 'image') is False
assert _use_text_step(0, 'text') is True
assert _use_text_step(0, 'interleaved') is True
assert _use_text_step(1, 'interleaved') is False
print('OK')
"
```
Expected: `OK`

- [ ] **Step 5: Commit**

```bash
cd /mnt/zsn/zsn_workspace/PartCraft3D
git add third_party/interweave_Trellis.py
git commit -m "feat: add mode param to interweave_Trellis_TI (image/text/interleaved)"
```

---

## Task 2: Wire `repaint_mode` through `TrellisRefiner.edit()`

**Files:**
- Modify: `partcraft/trellis/refiner.py:1219-1228` (edit signature)
- Modify: `partcraft/trellis/refiner.py:1280-1296` (image conditioning guard)
- Modify: `partcraft/trellis/refiner.py:1311-1314` (interweave call)
- Modify: `partcraft/trellis/refiner.py:610-618` (mask inflation warning)

- [ ] **Step 1: Add `repaint_mode` to `edit()` signature**

Change the `edit()` method signature from:
```python
    def edit(
        self,
        slat,
        mask: torch.Tensor,
        prompts: dict,
        img_cond: torch.Tensor | None = None,
        img_new: Image.Image | None = None,
        seed: int = 1,
        combinations: list[dict] | None = None,
    ) -> list[dict]:
```
To:
```python
    def edit(
        self,
        slat,
        mask: torch.Tensor,
        prompts: dict,
        img_cond: torch.Tensor | None = None,
        img_new: Image.Image | None = None,
        seed: int = 1,
        combinations: list[dict] | None = None,
        repaint_mode: str = 'interleaved',
    ) -> list[dict]:
```

- [ ] **Step 2: Add fallback guard for image-only mode without conditioning**

In the image conditioning setup section (around line 1280, just before `if img_cond is not None:`), add:

```python
        # Guard: image-only mode requires actual conditioning.
        # Fall back to interleaved if none is available so we don't
        # feed a blank-white image through all repaint steps.
        effective_mode = repaint_mode
        if effective_mode == 'image' and img_cond is None and img_new is None:
            logger.warning(
                "repaint_mode='image' but no img_cond/img_new provided "
                "— falling back to 'interleaved'")
            effective_mode = 'interleaved'
```

- [ ] **Step 3: Pass `mode` to `interweave_Trellis_TI`**

Change the call from:
```python
                result = interweave_Trellis_TI(
                    args, self.trellis_text, self.trellis_img,
                    slat, mask, prompts, effective_img, seed=seed)
```
To:
```python
                result = interweave_Trellis_TI(
                    args, self.trellis_text, self.trellis_img,
                    slat, mask, prompts, effective_img, seed=seed,
                    mode=effective_mode)
```

- [ ] **Step 4: Add mask inflation warning in `build_part_mask`**

The existing diagnostic block around line 610 already logs `slat_in_edit/slat_total`. Add one more check after:
```python
        if slat_in_edit == 0 and edit_type != "Addition":
            logger.warning("WARNING: No SLAT voxels overlap with mask! "
                           "Coordinate space may be misaligned.")
```
Add immediately after:
```python
        if slat_total > 0 and slat_in_edit / slat_total > 0.95 and edit_type != "Global":
            logger.warning(
                f"WARNING: mask covers {slat_in_edit/slat_total*100:.1f}%% of SLAT voxels "
                f"— possible mask inflation (expected <95%% for non-Global edits)")
```

- [ ] **Step 5: Verify import and signature**

```bash
cd /mnt/zsn/zsn_workspace/PartCraft3D
python3 -c "
import sys; sys.path.insert(0, '.')
import inspect
from partcraft.trellis.refiner import TrellisRefiner
sig = inspect.signature(TrellisRefiner.edit)
assert 'repaint_mode' in sig.parameters, 'repaint_mode param missing'
assert sig.parameters['repaint_mode'].default == 'interleaved'
print('OK')
"
```
Expected: `OK`

- [ ] **Step 6: Commit**

```bash
cd /mnt/zsn/zsn_workspace/PartCraft3D
git add partcraft/trellis/refiner.py
git commit -m "feat: add repaint_mode to TrellisRefiner.edit(), mask inflation guard"
```

---

## Task 3: Wire config → s5 → refiner, update YAML files

**Files:**
- Modify: `partcraft/pipeline_v2/s5_trellis_3d.py:147-200` (read config + pass to edit)
- Modify: `configs/pipeline_v2_shard00.yaml`
- Modify: `configs/pipeline_v2_shard02.yaml`
- Modify: `configs/pipeline_v2_gpu01.yaml`

- [ ] **Step 1: Read `repaint_mode` from config in `run_for_object`**

After the existing `scale_large` lines (~line 150), add:
```python
        repaint_mode = str(p25_cfg.get("repaint_mode", "interleaved"))
```

- [ ] **Step 2: Pass `repaint_mode` to `refiner.edit()`**

Change:
```python
            edit_results = refiner.edit(
                ori_slat, mask, prompts,
                img_cond=img_cond, seed=seed, combinations=None,
            )
```
To:
```python
            edit_results = refiner.edit(
                ori_slat, mask, prompts,
                img_cond=img_cond, seed=seed, combinations=None,
                repaint_mode=repaint_mode,
            )
```

- [ ] **Step 3: Add `repaint_mode` to the three YAML configs**

In each config file, inside the `services.image_edit` block, add `repaint_mode: image`.

`configs/pipeline_v2_shard00.yaml` — find the `image_edit:` block and add the key:
```yaml
  image_edit:
    enabled: true
    trellis_text_ckpt: /root/workspace/PartCraft3D/checkpoints/TRELLIS-text-xlarge
    image_edit_backend: local_diffusers
    workers_per_server: 2
    export_ply: false
    export_ply_for_deletion: true
    large_part_threshold: 0.35
    repaint_mode: image        # ← add this line
```

Apply the same change to `configs/pipeline_v2_shard02.yaml` and `configs/pipeline_v2_gpu01.yaml`.

- [ ] **Step 4: Verify config is read correctly**

```bash
cd /mnt/zsn/zsn_workspace/PartCraft3D
python3 -c "
import yaml
from partcraft.pipeline_v2 import services_cfg as psvc
cfg = yaml.safe_load(open('configs/pipeline_v2_shard00.yaml'))
p25 = psvc.trellis_image_edit_flat(cfg)
print('repaint_mode =', p25.get('repaint_mode'))
assert p25.get('repaint_mode') == 'image', f'got {p25.get(\"repaint_mode\")}'
print('OK')
"
```
Expected:
```
repaint_mode = image
OK
```

- [ ] **Step 5: Commit**

```bash
cd /mnt/zsn/zsn_workspace/PartCraft3D
git add partcraft/pipeline_v2/s5_trellis_3d.py \
        configs/pipeline_v2_shard00.yaml \
        configs/pipeline_v2_shard02.yaml \
        configs/pipeline_v2_gpu01.yaml
git commit -m "feat: read repaint_mode from config, pass to refiner.edit(); set image in all configs"
```

---

## Task 4: Tests for `is_gate_a_failed`

**Files:**
- Modify: `tests/test_qc_io.py` (append new test class)

- [ ] **Step 1: Add tests for `is_gate_a_failed`**

Append to `tests/test_qc_io.py`:
```python
class TestGateAFailed(unittest.TestCase):
    """is_gate_a_failed only blocks on Gate A, never on Gate C."""

    def setUp(self):
        self._t = tempfile.TemporaryDirectory()
        self.p = Path(self._t.name)

    def tearDown(self):
        self._t.cleanup()

    def test_no_qc_file_returns_false(self):
        from partcraft.pipeline_v2.qc_io import is_gate_a_failed
        self.assertFalse(is_gate_a_failed(_ctx(self.p), "del_001"))

    def test_gate_c_fail_does_not_block(self):
        """Gate C failure must NOT cause is_gate_a_failed to return True."""
        from partcraft.pipeline_v2.qc_io import update_edit_gate, is_gate_a_failed
        ctx = _ctx(self.p)
        update_edit_gate(ctx, "mod_001", "modification", "C",
                         vlm_result={"pass": False, "reason": "wrong region"})
        self.assertFalse(is_gate_a_failed(ctx, "mod_001"))

    def test_gate_a_fail_blocks(self):
        """Gate A failure must cause is_gate_a_failed to return True."""
        from partcraft.pipeline_v2.qc_io import update_edit_gate, is_gate_a_failed
        ctx = _ctx(self.p)
        update_edit_gate(ctx, "mod_002", "modification", "A",
                         vlm_result={"pass": False, "reason": "wrong part"})
        self.assertTrue(is_gate_a_failed(ctx, "mod_002"))

    def test_gate_a_pass_gate_c_fail_does_not_block(self):
        """Gate A pass + Gate C fail => is_gate_a_failed False."""
        from partcraft.pipeline_v2.qc_io import update_edit_gate, is_gate_a_failed
        ctx = _ctx(self.p)
        update_edit_gate(ctx, "mod_003", "modification", "A",
                         vlm_result={"pass": True})
        update_edit_gate(ctx, "mod_003", "modification", "C",
                         vlm_result={"pass": False, "reason": "global edit"})
        self.assertFalse(is_gate_a_failed(ctx, "mod_003"))

    def test_unknown_edit_id_returns_false(self):
        from partcraft.pipeline_v2.qc_io import update_edit_gate, is_gate_a_failed
        ctx = _ctx(self.p)
        update_edit_gate(ctx, "mod_004", "modification", "A",
                         vlm_result={"pass": False})
        self.assertFalse(is_gate_a_failed(ctx, "nonexistent_edit"))
```

- [ ] **Step 2: Run the new tests**

```bash
cd /mnt/zsn/zsn_workspace/PartCraft3D
python3 -m pytest tests/test_qc_io.py -v 2>&1 | tail -20
```
Expected: all `TestGateAFailed` tests PASS.

- [ ] **Step 3: Commit**

```bash
cd /mnt/zsn/zsn_workspace/PartCraft3D
git add tests/test_qc_io.py
git commit -m "test: add is_gate_a_failed tests — Gate C never blocks s5"
```

---

## Self-Review Checklist

- [x] Spec §1 (`_use_text_step` + `mode` param): Task 1
- [x] Spec §2 (`repaint_mode` in `edit()`): Task 2
- [x] Spec §3 (config + s5 wiring): Task 3
- [x] Spec §4a (mask inflation guard): Task 2 Step 4
- [x] Spec §4b (`npz_view` guard): `resolve_2d_conditioning` already has `if spec.npz_view >= 0` check at line ~63 — no change needed
- [x] Gate C bypass (already committed): no task needed
- [x] `is_gate_a_failed` tests: Task 4
- [x] Default `repaint_mode='interleaved'` preserves old behaviour: Task 2 Step 1 default value
- [x] Fallback guard for missing `img_cond` with `mode='image'`: Task 2 Step 2
