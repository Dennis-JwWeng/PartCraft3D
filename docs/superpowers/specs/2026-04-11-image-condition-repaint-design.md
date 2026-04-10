# Image-Condition Repaint Pipeline — Design Spec

**Date:** 2026-04-11  
**Branch:** feature/prompt-driven-part-selection  
**Status:** Ready for implementation

---

## Background & Motivation

The TRELLIS repaint pipeline (`interweave_Trellis_TI`) currently runs in **interleaved mode**: forward repaint steps alternate between the text flow model (`cnt%2==0`) and the image flow model (`cnt%2==1`).

Vinedresser3D benchmark results show **image-only mode is 8–10 s faster per edit** and produces comparable or better quality. The user has validated this independently. The goal of this spec is to switch the production pipeline to image-only conditioning while:

1. Ensuring the 3D voxel mask correctly aligns to SLAT occupancy.
2. Verifying that the single-view 2D edited image is a valid conditioning signal for the target part.
3. Keeping the interleaved mode accessible via a config flag (don't break existing behaviour).

---

## Scope

| In scope | Out of scope |
|---|---|
| Add `mode` param to `interweave_Trellis_TI` | Changing inversion to use image model |
| Wire `mode` through `refiner.edit()` and YAML config | Multiview conditioning (already single-view) |
| Review & assert mask alignment in `build_part_mask` | New VLM quality gates |
| Gate C bypass already done (committed) | Changing FLUX s4 step |

---

## Design

### 1. `third_party/interweave_Trellis.py` — Add `mode` parameter

Add the `_use_text_step` helper from Vinedresser3D and a `mode` parameter to `interweave_Trellis_TI`.

Replace all `cnt%2 == 0` forward-repaint checks (S1 loop, S2 loop) with `_use_text_step(cnt, mode)`.

Inversion steps and backward re-inversion steps are NOT changed — they always use the text model (correct by design; SLAT is the source of truth, not any 2D image).

### 2. `partcraft/trellis/refiner.py` — Wire `mode` through `edit()`

Add `repaint_mode: str = 'interleaved'` parameter to `TrellisRefiner.edit()`.
Pass it to `interweave_Trellis_TI(..., mode=repaint_mode)`.

Guard: if `repaint_mode == 'image'` and `img_cond is None` and `img_new is None`, log a warning and fall back to `'interleaved'` to avoid meaningless blank-image conditioning.

### 3. Pipeline config — Expose `repaint_mode`

In the `trellis` config block (YAML):
```yaml
trellis:
  repaint_mode: image   # 'image' | 'text' | 'interleaved'
```

In `s5_trellis_3d.py :: run_for_object`, read `p25_cfg.get("repaint_mode", "interleaved")` and pass to `refiner.edit()`. Default stays `interleaved` so old configs don't break.

### 4. Mask alignment review

**4a — 3D voxel mask ↔ SLAT**

Read `build_part_mask` in `refiner.py`. Verify `_align_masks_to_slat` is called on the raw part mesh mask before returning. Add a `logger.warning` if the aligned mask occupies >95% of SLAT voxels (inflation bug indicator).

**4b — 2D edited image ↔ 3D edit region**

`resolve_2d_conditioning` uses `spec.npz_view` for the single-view render. Add an explicit guard: if `spec.npz_view < 0`, log a warning and skip image conditioning (fall back to `img_cond=None`). Log view index + edit_id at INFO level for traceability.

---

## Data Flow (image-only mode)

```
EditSpec.npz_view
    └── s4: render frame → FLUX edit → edits_2d/{edit_id}_edited.png
                                          |
s5: resolve_2d_conditioning               |
    +-- load {edit_id}_input.png  (original render)
    +-- load {edit_id}_edited.png (FLUX output)
        +-- encode_multiview_cond([edited], [original])
            +-- DINOv2 encode → img_cond [1, 1369, 1024]

refiner.edit(ori_slat, mask, prompts, img_cond=img_cond, repaint_mode='image')
    +-- monkey-patch trellis_img.get_cond → returns img_cond
    +-- interweave_Trellis_TI(..., mode='image')
        +-- S2 inversion:   text_model + cfg=0  (UNCHANGED)
        +-- S1 inversion:   text_model + cfg=0  (UNCHANGED)
        +-- S1 forward repaint: _use_text_step → False → img_s1_flow_model + img_cond
        +-- S2 forward repaint: _use_text_step → False → img_s2_flow_model + img_cond
        +-- preserved voxels: inject from inverse_dict + soft mask blend (UNCHANGED)
```

---

## Files to Change

| File | Change |
|---|---|
| `third_party/interweave_Trellis.py` | Add `_use_text_step`, `mode` param, replace 2x `cnt%2==0` |
| `partcraft/trellis/refiner.py` | Add `repaint_mode` to `edit()`, pass to `interweave_Trellis_TI`; mask alignment log in `build_part_mask` |
| `partcraft/pipeline_v2/s5_trellis_3d.py` | Read `repaint_mode` from config, pass to `refiner.edit()` |
| `configs/pipeline_v2_gpu01.yaml` + `_shard00.yaml` + `_shard02.yaml` | Add `trellis.repaint_mode: image` |

Already done (not in this plan):
- `qc_io.py` — `is_gate_a_failed()` added
- `s5_trellis_3d.py` + `s5b_deletion.py` — use `is_gate_a_failed`

---

## Risks & Mitigations

| Risk | Mitigation |
|---|---|
| `img_cond=None` with `mode='image'` → blank image → bad output | Guard in `refiner.edit()`: fall back to `interleaved` + warn |
| Config key missing on old YAML | Default `repaint_mode='interleaved'` in code |
| noise-space mismatch (text inversion + image forward) | Mitigated by existing soft mask sigma blending; no change needed |

---

## Success Criteria

1. `interweave_Trellis_TI(mode='image')` runs S1/S2 forward repaint with image model only.
2. Test object `00a2502bda5c47bf8018307efb3b6d5c`: all edits with `_edited.png` complete (previously 15 blocked by Gate C, now unblocked).
3. Per-edit wall time measurably lower than `interleaved`.
4. No "No SLAT voxels overlap" warnings in mask alignment logs.
