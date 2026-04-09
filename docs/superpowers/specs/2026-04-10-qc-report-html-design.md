# QC Pipeline HTML Report Generator — Design Spec

**Date**: 2026-04-10  
**Branch**: feature/prompt-driven-part-selection  
**Status**: Approved, pending implementation  
**Related spec**: `2026-04-09-qc-pipeline-design.md`

---

## Goal

Generate a single self-contained static HTML file that visualises the full QC pipeline outcome for a shard. The report is object-centric: each 3D object is a collapsible card containing its edits, and each edit shows the three QC gate results with supporting images. The report is opened in a local browser — no server required.

---

## New File

```
scripts/vis/generate_qc_report.py
```

No modifications to existing files.

---

## Invocation

```bash
python scripts/vis/generate_qc_report.py \
    --config configs/pipeline_v2_shard02.yaml \
    [--output <path/to/report.html>]   # default: <output_root>/pipeline_v2_shard{NN}/qc_report.html
    [--no-embed]                        # use relative image paths instead of base64
    [--limit N]                         # process only first N objects (debug)
    [--filter-fail]                     # include only objects with at least one QC failure
```

`--config` is required. `output_root` and `shard` are derived from the config using the same `partcraft.utils.config` loader used by the pipeline.

---

## Data Sources

Script scans `<output_root>/pipeline_v2_shard{NN}/objects/<shard>/<obj_id>/` for every object directory present on disk.

`edit_id` is constructed as `{type_prefix}_{index:03d}` (e.g. `del_000`, `mod_001`) by enumerating edits in `parsed.json` order and grouping by type — matching the convention used in `qc.json`.

Per object, the following files are read (all optional except `status.json` and `phase1/parsed.json`):

| File | Required | Content |
|------|----------|---------|
| `status.json` | Yes | Step completion status, edit counts per type |
| `phase1/parsed.json` | Yes | Edit list with prompt, type, selected_part_ids, target_part_desc, etc. |
| `phase1/overview.png` | No | 5x2 VLM render grid (object overview) |
| `qc.json` | No | Per-edit gate results (written by sq1/sq2/sq3 steps) |
| `highlights/e{idx:02d}.png` | No | Gate C input: magenta highlight of target part |
| `edits_2d/{edit_id}_edited.png` | No | Gate C input: FLUX 2D edit result |
| `edits_3d/{edit_id}/before.png` | No | Gate E input: 3D render before edit |
| `edits_3d/{edit_id}/after.png` | No | Gate E input: 3D render after edit |

Missing files (stage not yet run) render as a grey placeholder — not an error.

---

## Image Handling

**Default (embed mode)**:
- Images are resized with Pillow before base64-encoding: `overview.png` -> max 600 px wide; gate images -> max 400 px wide.
- Resized base64 strings are stored in a JS `const IMAGES = {...}` block inside `<script>`.
- Images are **not** injected into the DOM until the relevant section is expanded (lazy injection). This prevents the browser from decoding thousands of images on load.
- If Pillow is unavailable, images are embedded at original size with a console warning.

**`--no-embed` mode**:
- `<img src>` attributes use relative paths from the HTML file's location back to the object directories.
- HTML file must remain co-located with the `objects/` directory tree.

---

## HTML Structure

Single HTML page, zero external JS/CSS dependencies.

### Page header
- Report title, shard ID, generation timestamp, total object count.

### Summary section
Two panels side-by-side:

**Funnel chart (SVG)**  
Horizontal funnel showing absolute counts and pass-rates at each gate:
```
Total edits   N
  -> Gate A   N  (xx.x%)   [drop: N]
  -> Gate C   N  (xx.x%)   [drop: N]
  -> Gate E   N  (xx.x%)   [drop: N]
  Final DB    N
```
Gate C and E counts exclude edits for which the gate is N/A (e.g. deletion has no Gate C).

**Per-type bar chart (CSS bars)**  
One row per edit type (deletion / modification / scale / material / global / addition), showing pass count as a horizontal bar with the ratio labelled.

### Filter bar
- Text search on obj_id (prefix match)
- Dropdown: edit type (all / deletion / modification / scale / material / global)
- Dropdown: QC status (all / has_fail / all_pass / incomplete)
- All filters are applied client-side with plain JS; no page reload.

### Object list
One collapsible card per object, sorted by number of QC failures descending (objects with failures at top).

**Collapsed state** (one row):
```
> <obj_id short>  <N edits>  PASS:<N> FAIL:<N>  [step badges: s1 sq1 s4 sq2 s5 sq3]
```

**Expanded state**:
- `phase1/overview.png` thumbnail (left) + object full description text (right)
- Edit table below (one collapsible row per edit)

### Edit rows (inside expanded object)
**Collapsed edit row**:
```
> <edit_id>  [type badge]  <prompt truncated>  PASS / FAIL / pending
```

**Expanded edit row** — three gate sub-sections:

**Gate A** (sq1 — instruction & structure check):
- Rule layer: list of rule codes that fired (red) or all-pass
- VLM layer: score (0-1), pass/fail badge, reason text
- Image: `phase1/overview.png`
- Skipped states: "VLM skipped (rule failed)" / "Not yet run"

**Gate C** (sq2 — 2D region alignment):
- `region_match`: pass/fail badge + reason text
- Images: `highlights/e{idx}.png` (left) and `edits_2d/{edit_id}_edited.png` (right)
- N/A for: deletion, addition

**Gate E** (sq3 — final 3D quality):
- 8-dim score table: edit_executed, correct_region, preserve_other, visual_quality, artifact_free, prompt_quality
- Pass/fail per threshold (per-type thresholds shown)
- Images: `edits_3d/{edit_id}/before.png` (left) and `after.png` (right)

### Lightbox
Clicking any image opens it full-screen with a dark overlay. Click overlay or press Esc to close. Pure JS, no library.

---

## Colour Semantics

| State | Colour | When |
|-------|--------|------|
| Pass | Green #22c55e | Gate passed / step ok |
| Fail | Red #ef4444 | Gate failed / step error |
| N/A | Grey #94a3b8 | Gate not applicable for this edit type |
| Incomplete | Amber #f59e0b | qc.json absent or gate not yet run |

---

## Python Module Structure

```python
# generate_qc_report.py

@dataclass
class GateResult:
    applicable: bool          # False -> render as N/A
    ran: bool                 # False -> render as incomplete
    rule_pass: bool | None
    rule_checks: dict         # failing rule codes
    vlm_pass: bool | None
    vlm_score: float | None
    vlm_reason: str
    final_pass: bool | None

@dataclass
class EditReport:
    edit_id: str
    edit_type: str
    prompt: str
    target_part_desc: str
    selected_part_ids: list[int]
    gate_a: GateResult
    gate_c: GateResult
    gate_e: GateResult
    final_pass: bool | None   # None = incomplete

@dataclass
class ObjectReport:
    obj_id: str
    shard: str
    status: dict              # raw status.json steps dict
    full_desc: str
    parts: list[dict]
    edits: list[EditReport]
    images: dict[str, str]    # key -> base64 or relative path

def collect_shard_data(output_root, shard, limit, filter_fail) -> list[ObjectReport]: ...
def build_summary(objects: list[ObjectReport]) -> dict: ...
def render_html(summary, objects, meta, no_embed) -> str: ...
def main(): ...
```

---

## Summary Statistics Definitions

- **Total edits**: sum of all edits across all objects (from parsed.json counts)
- **Gate A applicable**: all edits
- **Gate A pass**: gates.A.rule.pass == true AND gates.A.vlm.pass == true (rule pass + vlm null = incomplete, not counted as pass)
- **Gate C applicable**: edits where edit_type in FLUX_TYPES (modification, scale, material, global)
- **Gate C pass**: gates.C.vlm.region_match == true
- **Gate E applicable**: all edits (sq3 runs independently; Gate E applicability is per-type, same as QC spec)
- **Gate E pass**: final 3D quality thresholds met per type
- **Final DB**: edits where final_pass == true

---

## Out of Scope

- Automatic shard-level aggregation across multiple shards (run once per shard)
- Live reload / polling (static only)
- Export to CSV or JSON from the report UI
- Editing or re-running QC from the report
