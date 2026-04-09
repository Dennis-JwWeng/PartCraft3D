# QC Pipeline HTML Report Generator - Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans

**Goal:** Create scripts/vis/generate_qc_report.py - a CLI generating a self-contained static HTML QC report with embedded images, funnel summary, and per-edit gate drill-down.

**Spec:** docs/superpowers/specs/2026-04-10-qc-report-html-design.md

---

## File Structure

| File | Action |
|------|--------|
| scripts/vis/generate_qc_report.py | Create - all logic |
| tests/test_vis_qc_report.py | Create - unit tests |

No existing files modified.

---

## Task 1: Dataclasses and qc.json parser

_Create scripts/vis/generate_qc_report.py and tests/test_vis_qc_report.py_

- [ ] **Step 1.1: Write failing tests (TestGateResult, TestParseEditGates) in tests/test_vis_qc_report.py**

- [ ] **Step 1.2: Run: python -m pytest tests/test_vis_qc_report.py -v 2>&1 | head -20 -- Expected: FileNotFoundError**

- [ ] **Step 1.3: Implement GateResult dataclass with not_applicable()/incomplete()/from_qc_gate() classmethods; EditReport, ObjectReport dataclasses; parse_edit_gates() function; FLUX_TYPES and TYPE_PREFIX constants**

- [ ] **Step 1.4: Run: python -m pytest tests/test_vis_qc_report.py::TestGateResult tests/test_vis_qc_report.py::TestParseEditGates -v -- Expected: 7 PASS**

- [ ] **Step 1.5: git add scripts/vis/generate_qc_report.py tests/test_vis_qc_report.py && git commit -m "feat(vis): add qc report dataclasses and gate parser"**

## Task 2: Object data collector

_Append to both files_

- [ ] **Step 2.1: Append TestCollectObject to tests/test_vis_qc_report.py - tests: test_no_qc_json_yields_incomplete (no qc.json -> final_pass=None, gate_a.ran=False), test_qc_pass_reflected (qc.json with del_000 pass -> n_pass=1)**

- [ ] **Step 2.2: Run: python -m pytest tests/test_vis_qc_report.py::TestCollectObject -v 2>&1 | head -10 -- Expected: AttributeError (collect_object not found)**

- [ ] **Step 2.3: Implement _make_edit_id(edit_type, type_counts), _load_image_b64(path, max_width) using Pillow resize + base64, _collect_images(obj_dir, edits, no_embed) reading overview/highlight/flux/before/after, collect_object(obj_dir, no_embed) reading status.json+parsed.json+qc.json, collect_shard_data(output_root, shard, limit, filter_fail, no_embed)**

- [ ] **Step 2.4: Run: python -m pytest tests/test_vis_qc_report.py -v -- Expected: all PASS**

- [ ] **Step 2.5: git add ... && git commit -m "feat(vis): add object data collector and image loader"**

## Task 3: Summary statistics

_Append to both files_

- [ ] **Step 3.1: Append TestBuildSummary to tests - tests: test_funnel_counts (3 edits -> correct gate counts), test_type_breakdown (2 deletions -> by_type counts)**

- [ ] **Step 3.2: Run pytest TestBuildSummary -- Expected: AttributeError (build_summary not found)**

- [ ] **Step 3.3: Implement ALL_TYPES list and build_summary(objects) returning {total, n_objects, gate_a_pass, gate_c_applicable, gate_c_pass, gate_e_applicable, gate_e_pass, final_pass, by_type}**

- [ ] **Step 3.4: Run: python -m pytest tests/test_vis_qc_report.py -v -- Expected: all PASS**

- [ ] **Step 3.5: git add ... && git commit -m "feat(vis): add summary statistics builder"**

## Task 4: HTML renderer

_Append to both files - largest task_

- [ ] **Step 4.1: Append TestRenderHtml with 4 tests: contains_obj_id, funnel_labels (Gate A/C/E/Final DB), edit_prompt_and_fail_reason (prompt_too_short code visible), valid_html (starts with <!DOCTYPE html>, has const REPORT, openLightbox)**

- [ ] **Step 4.2: Run pytest TestRenderHtml -- Expected: AttributeError (render_html not found)**

- [ ] **Step 4.3: Implement helpers: _pct(), _esc(), _gate_badge(), _gate_section() (renders gate row with rule codes + VLM + 8-dim table + lazy images via data-src), _objs_json(), _imgs_json(); CSS constant _CSS (dark theme: --bg #0f172a, pass green, fail red, na grey, warn amber); JS constant _JS (openLightbox/closeLightbox with Esc key support, lazy data-src injection on details toggle, applyFilters for search+type+status, expandAll/collapseAll); render_html() assembling funnel table + type bars + object cards with nested edit rows**

- [ ] **Step 4.4: Run: python -m pytest tests/test_vis_qc_report.py -v -- Expected: all PASS**

- [ ] **Step 4.5: git add ... && git commit -m "feat(vis): add HTML renderer with funnel, type chart, gate detail, lightbox"**

## Task 5: CLI wiring and end-to-end test

_Append to both files_

- [ ] **Step 5.1: Append TestEndToEnd.test_cli_on_shard02_smoke - runs subprocess: python scripts/vis/generate_qc_report.py --config configs/pipeline_v2_shard02.yaml --output /tmp/r.html --limit 5 --no-embed; asserts returncode=0, file exists, starts with <!DOCTYPE html>, len > 5000; skipTest if config absent**

- [ ] **Step 5.2: Run pytest TestEndToEnd -- Expected: non-zero exit (no main)**

- [ ] **Step 5.3: Implement main() - argparse with --config(required), --output, --no-embed, --limit, --filter-fail; load_config from partcraft.utils.config; derive output_root and shard from cfg; call collect_shard_data, build_summary, render_html; write to out_path**

- [ ] **Step 5.4: Run: python -m pytest tests/test_vis_qc_report.py -v -- Expected: all pass**

- [ ] **Step 5.5: Manual smoke: python scripts/vis/generate_qc_report.py --config configs/pipeline_v2_shard02.yaml --limit 10 --no-embed --output /tmp/qc_report_test.html -- verify in browser: dark theme, funnel, object cards collapsed, N/A gates grey, lightbox, filters**

- [ ] **Step 5.6: git add ... && git commit -m "feat(vis): add CLI entry point for qc html report generator"**

---

## Spec Coverage

| Requirement | Covered by |
|---|---|
| Static HTML offline | Tasks 4-5 CSS/JS inline |
| Object-centric hierarchy | Task 4 render_html object cards |
| Funnel chart A/C/E/Final | Tasks 3+4 build_summary + funnel table |
| Per-type bar chart | Task 4 tbars |
| Base64 images via Pillow 600/400px | Task 2 _load_image_b64 |
| --no-embed relative paths | Task 2 _collect_images no_embed |
| Lazy data-src injection | Task 4 JS details toggle |
| Lightbox click-to-zoom | Task 4 openLightbox/closeLightbox |
| Gate A rule codes + VLM | Task 4 _gate_section rule/vlm rows |
| Gate C region_match + images | Task 4 _gate_section highlight/flux |
| Gate E 8-dim table + images | Task 4 _gate_section vlm_8dim |
| N/A gates greyed | GateResult.not_applicable() + .na CSS |
| Pending gates amber | GateResult.incomplete() + .incomplete CSS |
| Filter bar search/type/status | Task 4 JS applyFilters() |
| --limit N, --filter-fail | Task 2+5 collect_shard_data + main |
| edit_id TYPE_PREFIX convention | Task 2 _make_edit_id() |
| Config-driven output_root/shard | Task 5 main() via load_config |
| Failures-first sort | Task 2 collect_shard_data sort |
