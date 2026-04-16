# Mode E: Text → Edits → Alignment Gate (Design Spec)

**Date:** 2026-04-15  
**Status:** Approved, ready for implementation  
**Output folder tag:** `mode_e_text_align`

---

## Motivation

Existing Mode A (image + semantic captions) combines visual grounding and edit generation in a single VLM call. PartVerse captions are noisy, causing the VLM to select semantically incorrect parts. This experiment decouples the two concerns:

- **Edit generation** — cheap, text-only, uses PartVerse captions (Mode B)
- **Alignment gate** — targeted image call per edit, with selected parts highlighted explicitly in red

The gate also produces a reliable `best_view` (original view index) for downstream FLUX image editing, which Mode B cannot determine without image access.

---

## Architecture

```
Per object — two async phases:

PHASE 1  (text only, one VLM call per object)
  build_semantic_list(mesh_npz, img_npz)      <- PartVerse text_captions
        |
  call_vlm_text_async(SYSTEM_PROMPT_B, ...)   <- Mode B edit generation
        |
  validate_simple()                           <- schema + imperative-verb check
        |
  raw.txt  parsed.json                        <- saved to test folder per object

PHASE 2  (image + text, one VLM call per edit)
  For each edit in parsed.json:
    [1] check_rules(edit, parts_by_id)        <- rule gate (no VLM)
         parts_by_id built from pids in mesh_npz (same source as Phase 1)
         -> fails -> gate_a=fail, stop
    [2] pixel_counts[v] for v in 0..4         <- count_part_pixels_in_overview
         best_col_view  = argmax(pixel_counts)
         other_views    = [v for v in 0..4 if v != best_col_view]
         column_map     = [best_col_view] + other_views
    [3] build_gate_image(ov_img, selected_part_ids, column_map)
         -> 5x2 grid (same layout as overview.png):
           col 0: top = RGB of best_col_view
                  bot = red/grey highlight of best_col_view
           col 1-4: top = RGB of other_views[col-1]
                    bot = normal palette of other_views[col-1]  (unmodified)
    [4] call_vlm_async(SYSTEM_PROMPT_ALIGN_GATE, image=gate_img, user=edit_text)
         -> {aligned: bool, reason: str, best_view: 0-4}  <- col index in gate image
    [5] best_view_original = column_map[gate_vlm.best_view]
         -> write to edit_status.json
```

---

## File Layout

```
bench_shard08/mode_e_text_align/
  objects/{shard}/{obj_id}/phase1/
    raw.txt               <- Phase 1 VLM raw output
    parsed.json           <- parsed edits (all, before gate)
    edit_status.json      <- gate result per edit
  _summary.json           <- overall yield stats
```

---

## VLM Prompts

### Phase 1 — Edit Generation

Reuses existing `SYSTEM_PROMPT_B` + `build_semantic_list()` from `s1_vlm_core.py`. No changes.

### Phase 2 — Alignment Gate (`SYSTEM_PROMPT_ALIGN_GATE`)

Added to `partcraft/pipeline_v3/s1_vlm_core.py`:

```
You are a 3D-edit alignment judge.

INPUT:
  - A 5x2 grid image (same layout as a standard overview):
      TOP row    = 5 RGB photos (views 0-4, left to right)
      BOTTOM row = 5 part-coloured renders
      IMPORTANT: Column 0 bottom is special -- selected (target) parts are RED,
                 all other parts are GREY.
                 Columns 1-4 bottom use normal palette colours (unmodified).
  - Edit instruction and type in text.

TASK:
  1. Find the RED region in column 0 (bottom row) -- these are the target parts.
  2. Examine the matching region in column 0 top-row RGB for visual context.
  3. Use columns 1-4 for additional object context.
  4. Judge whether the instruction makes semantic sense for the red-highlighted parts.
  5. Choose which column (0-4) gives the clearest view of the target parts.

OUTPUT: ONE valid JSON object -- no prose, no markdown fences.
{
  "aligned":   <true|false>,
  "reason":    "<1-2 sentences>",
  "best_view": <0-4, column index in THIS image where target parts are clearest>
}

RULES:
  R1. aligned=true  iff the red parts match the instruction's stated target AND
      the edit type is appropriate for those parts.
  R2. aligned=false if no red is visible in column 0 (parts fully occluded), or
      the highlighted parts clearly do not match the instruction's intent.
  R3. best_view = column where red coverage is largest AND top-row RGB shows the
      target parts most clearly for editing purposes.
  R4. For global edits the image will be all-grey (no red). Always output
      aligned=true, best_view=0 for global edits.
```

**User prompt per edit:**
```
Edit type: {edit_type}
Instruction: "{prompt}"
Selected parts: {comma-separated part_N labels}
```

---

## Gate Decision Logic

| Condition | gate_a status | VLM called? |
|---|---|---|
| check_rules returns any failure | fail | No |
| aligned=true from VLM | pass | Yes |
| aligned=false from VLM | fail | Yes |
| VLM call throws / bad JSON | fail | Yes (failed) |
| overview.png missing | pass (auto) for all edits, best_view=0 | No |
| Global edit (selected_part_ids=[]) | pass (auto), best_view=0 | No |

---

## edit_status.json Schema (per object)

```json
{
  "obj_id": "<obj_id>",
  "mode": "text_align",
  "schema_version": 1,
  "updated": "<iso timestamp>",
  "edits": {
    "<edit_id>": {
      "edit_type": "<deletion|modification|scale|material|color|global>",
      "stages": {
        "gate_a": { "status": "pass|fail", "ts": "<iso>" }
      },
      "gates": {
        "A": {
          "rule": { "pass": true, "checks": {} },
          "vlm": {
            "pass": true,
            "score": 1.0,
            "reason": "...",
            "best_view": 2,
            "best_view_col": 0,
            "pixel_counts": [0, 0, 412, 38, 0],
            "column_map": [2, 0, 1, 3, 4]
          }
        }
      }
    }
  }
}
```

Field provenance in `gates.A.vlm`:

| Field | Source | Meaning |
|---|---|---|
| pass / score | VLM aligned field | Gate decision (score 1.0 or 0.0) |
| reason | VLM | Explanation |
| best_view | column_map[best_view_col] | Original view index (0-4), used by downstream FLUX |
| best_view_col | VLM output | Column index in the gate image (0-4) |
| pixel_counts | count_part_pixels_in_overview | Pixel count per original view [v0..v4] |
| column_map | Computed | column_map[col] = original view index for that column |

---

## _summary.json Schema

```json
{
  "n_objects": 20,
  "n_p1_fail": 2,
  "n_edits_total": 147,
  "n_gate_pass": 112,
  "n_gate_fail": 35,
  "yield_rate": 0.76,
  "by_type": {
    "deletion":     { "total": 30, "pass": 26 },
    "modification": { "total": 28, "pass": 22 },
    "scale":        { "total": 15, "pass": 12 },
    "material":     { "total": 24, "pass": 18 },
    "color":        { "total": 20, "pass": 16 },
    "global":       { "total": 30, "pass": 18 }
  }
}
```

---

## Modules Touched

| File | Change |
|---|---|
| partcraft/pipeline_v3/s1_vlm_core.py | Add SYSTEM_PROMPT_ALIGN_GATE, build_align_gate_user_prompt(), parse_align_gate_output() |
| partcraft/pipeline_v3/qc_rules.py | Read-only (reuse check_rules, count_part_pixels_in_overview) |
| scripts/tools/run_text_align_gate_test.py | New standalone experiment script |

No changes to pipeline_v2/, pipeline_v3/run.py, or any production pipeline step.

---

## New Helper Functions

### build_gate_image(ov_img, selected_part_ids, column_map) -> bytes (local to script)

1. Extract each column's top and bottom cells using the same geometry constants as qc_rules.py (_N_VIEWS, _COL_SEP, _ROW_SEP, _W_IMG, _H_IMG).
2. Column 0 (= column_map[0] = best_col_view):
   - Top cell: RGB photo (unchanged)
   - Bottom cell: recolour — nearest palette match; if pid in selected_part_ids: red (220,45,45); else: grey (65,65,65)
3. Columns 1-4 (= column_map[1..4]):
   - Top cell: RGB photo (unchanged)
   - Bottom cell: original palette cell (unchanged)
4. Stitch 5 columns with same separators as original overview.
5. Return PNG bytes (cv2.imencode).

### build_align_gate_user_prompt(edit_type, prompt, selected_part_ids) -> str (in s1_vlm_core.py)

Returns:
  "Edit type: {edit_type}\nInstruction: \"{prompt}\"\nSelected parts: {part_N, ...}"

### parse_align_gate_output(raw) -> dict | None (in s1_vlm_core.py)

Uses extract_json_object(raw). Returns dict if "aligned" key is present and is bool; otherwise None.

---

## Success Criteria for the Experiment

1. Yield rate: what fraction of Mode B edits pass Gate A? Hypothesis: 70-85%.
2. Failure pattern: which edit types fail most? Expectation: material/color fail more (captions describe visual appearance, not material properties).
3. best_view quality: does pixel-count argmax match visual intuition? Verifiable from HTML report.
4. Speed: Phase 1 + Phase 2 wall time vs Mode A single call.
