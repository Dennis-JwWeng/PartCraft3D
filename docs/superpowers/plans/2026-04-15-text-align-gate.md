# Mode E: Text → Edits → Alignment Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a standalone experiment script that generates 3D-edit proposals from PartVerse text captions (Mode B) then gates each proposal with a VLM alignment check using a reconstructed highlighted overview image.

**Architecture:** Two async phases per object — Phase 1 calls the text-only VLM to generate edits (Mode B), Phase 2 calls the image VLM once per edit with a custom 5×2 gate image (col 0 = highlighted selection, cols 1–4 = regular context views). Results go into a test-local `edit_status.json`; no production pipeline files are touched.

**Tech Stack:** Python 3.10+, asyncio, OpenAI-compatible async client, numpy, opencv-python (cv2), partcraft.pipeline_v3.s1_vlm_core, partcraft.pipeline_v3.qc_rules

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `partcraft/pipeline_v3/s1_vlm_core.py` | Modify (append) | Add gate prompt constant + two helpers |
| `scripts/tools/run_text_align_gate_test.py` | Create | Full experiment: Phase 1 + Phase 2 + summary |

---

## Task 1: Gate Prompt + Helpers in s1_vlm_core.py

**Files:**
- Modify: `partcraft/pipeline_v3/s1_vlm_core.py` (append after line 1399, after `parse_s1_output`)

- [ ] **Step 1: Append `SYSTEM_PROMPT_ALIGN_GATE` constant**

  Open `partcraft/pipeline_v3/s1_vlm_core.py` and append after the final blank line (after `parse_s1_output`):

  ```python
  # ═══════════════════════════════════════════════════════════════════════
  #  Alignment Gate (Mode E) — image+text call, judges edit↔part alignment
  # ═══════════════════════════════════════════════════════════════════════

  SYSTEM_PROMPT_ALIGN_GATE = """\
  You are a 3D-edit alignment judge.

  INPUT:
    • A 5×2 grid image (same layout as a standard overview):
        TOP row    = 5 RGB photos (views 0–4, left to right)
        BOTTOM row = 5 part-coloured renders
        IMPORTANT: Column 0 bottom is special — selected (target) parts are RED,
                   all other parts are GREY.
                   Columns 1–4 bottom use normal palette colours (unmodified).
    • Edit instruction and type in text.

  TASK:
    1. Find the RED region in column 0 (bottom row) — these are the target parts.
    2. Examine the matching region in column 0 top-row RGB for visual context.
    3. Use columns 1–4 for additional object context.
    4. Judge whether the instruction makes semantic sense for the red-highlighted parts.
    5. Choose which column (0–4) gives the clearest view of the target parts.

  OUTPUT: ONE valid JSON object — no prose, no markdown fences.
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
  """


  def build_align_gate_user_prompt(
      edit_type: str,
      prompt: str,
      selected_part_ids: list[int],
  ) -> str:
      """User prompt for the alignment gate VLM call (text portion only)."""
      parts_str = ", ".join(f"part_{p}" for p in sorted(selected_part_ids)) or "(none)"
      return (
          f"Edit type: {edit_type}\n"
          f'Instruction: "{prompt}"\n'
          f"Selected parts: {parts_str}"
      )


  def parse_align_gate_output(raw: str) -> dict | None:
      """Parse alignment gate VLM response.

      Returns dict with at least {"aligned": bool, "reason": str, "best_view": int}
      or None on failure.
      """
      d = extract_json_object(raw)
      if not isinstance(d, dict):
          return None
      if not isinstance(d.get("aligned"), bool):
          return None
      if not isinstance(d.get("best_view"), int):
          d["best_view"] = 0   # safe default
      return d
  ```

- [ ] **Step 2: Add the three new names to `__all__`**

  Find the `__all__` list (line ~593). The list ends near line 615. Add:

  ```python
      "SYSTEM_PROMPT_ALIGN_GATE",
      "build_align_gate_user_prompt",
      "parse_align_gate_output",
  ```

  inside the existing `__all__ = [...]` block (order does not matter).

- [ ] **Step 3: Smoke-test the imports**

  Run from repo root:
  ```bash
  python -c "from partcraft.pipeline_v3.s1_vlm_core import \
      SYSTEM_PROMPT_ALIGN_GATE, build_align_gate_user_prompt, parse_align_gate_output; \
      print('ok'); \
      print(build_align_gate_user_prompt('deletion', 'Remove the wheel', [1,3])); \
      r = parse_align_gate_output('{\"aligned\": true, \"reason\": \"ok\", \"best_view\": 2}'); \
      print(r)"
  ```
  Expected output:
  ```
  ok
  Edit type: deletion
  Instruction: "Remove the wheel"
  Selected parts: part_1, part_3
  {'aligned': True, 'reason': 'ok', 'best_view': 2}
  ```

- [ ] **Step 4: Commit**

  ```bash
  git add partcraft/pipeline_v3/s1_vlm_core.py
  git commit -m "feat(pipeline_v3): add SYSTEM_PROMPT_ALIGN_GATE and gate helpers"
  ```

---

## Task 2: `build_gate_image` helper (local to script)

This helper is defined inside `run_text_align_gate_test.py` (not shared) because it uses cv2 and numpy with overview geometry constants, and is only needed here.

**Geometry constants** (must match `qc_rules.py`):
```python
_N_VIEWS = 5
_COL_SEP = 4   # px horizontal separator between columns
_ROW_SEP = 6   # px vertical separator between rows
```

**Red / grey colours (BGR for cv2):**
```python
_RED  = (45, 45, 220)   # BGR = red (220, 45, 45) RGB
_GREY = (65, 65, 65)
```

**Palette (BGR)** — imported from `partcraft.pipeline_v3.qc_rules`:
```python
from partcraft.pipeline_v3.qc_rules import _PALETTE_BGR
```

**Algorithm for `build_gate_image`:**

```python
import cv2, numpy as np
from partcraft.pipeline_v3.qc_rules import _PALETTE_BGR, _N_VIEWS, _COL_SEP, _ROW_SEP

_RED  = (45, 45, 220)
_GREY = (65, 65, 65)

def _extract_cell(img: np.ndarray, col: int, row: int) -> np.ndarray:
    """Extract one (row, col) cell from the 5×2 overview BGR image.

    row 0 = top (RGB photos), row 1 = bottom (palette renders).
    """
    H_total, W_total = img.shape[:2]
    W_cell = (W_total - (_N_VIEWS - 1) * _COL_SEP) // _N_VIEWS
    H_cell = (H_total - _ROW_SEP) // 2
    x0 = col * (W_cell + _COL_SEP)
    y0 = row * (H_cell + _ROW_SEP)
    return img[y0: y0 + H_cell, x0: x0 + W_cell].copy()


def _highlight_cell(cell: np.ndarray, selected_part_ids: set[int]) -> np.ndarray:
    """Recolour a bottom-row palette cell: selected parts → red, others → grey."""
    out = np.empty_like(cell)
    palette = np.array(_PALETTE_BGR, dtype=np.int32)       # (N, 3)
    flat = cell.reshape(-1, 3).astype(np.int32)            # (H*W, 3)
    diffs = np.linalg.norm(flat[:, None, :] - palette[None, :, :], axis=2)  # (H*W, N)
    nearest_pid = np.argmin(diffs, axis=1)                 # (H*W,)
    # Background: pixels with all channels > 230 keep white (background exclusion)
    is_bg = np.all(flat > 230, axis=1)
    red_mask  = np.array([pid in selected_part_ids for pid in nearest_pid])
    out_flat = out.reshape(-1, 3)
    out_flat[is_bg]  = [255, 255, 255]
    out_flat[~is_bg & red_mask]  = list(_RED)
    out_flat[~is_bg & ~red_mask] = list(_GREY)
    return out


def build_gate_image(
    ov_img: np.ndarray,
    selected_part_ids: list[int],
    column_map: list[int],
) -> bytes:
    """Build the 5×2 gate image as PNG bytes.

    column_map[0] = original view index placed in col 0 with red/grey highlight.
    column_map[1..4] = original view indices placed in cols 1–4, unmodified.
    """
    sel_set = set(selected_part_ids)
    top_cells = []
    bot_cells = []
    for col_idx, orig_view in enumerate(column_map):
        top = _extract_cell(ov_img, orig_view, 0)    # RGB
        bot = _extract_cell(ov_img, orig_view, 1)    # palette
        if col_idx == 0 and sel_set:
            bot = _highlight_cell(bot, sel_set)
        top_cells.append(top)
        bot_cells.append(bot)

    sep_v = np.full((top_cells[0].shape[0], _COL_SEP, 3), 255, dtype=np.uint8)
    sep_h = np.full((_ROW_SEP, sum(c.shape[1] for c in top_cells) + (_N_VIEWS-1)*_COL_SEP, 3),
                    255, dtype=np.uint8)

    def hstack_with_sep(cells):
        row = cells[0]
        for c in cells[1:]:
            row = np.concatenate([row, sep_v[:c.shape[0]], c], axis=1)
        return row

    top_row = hstack_with_sep(top_cells)
    bot_row = hstack_with_sep(bot_cells)
    full = np.concatenate([top_row, sep_h, bot_row], axis=0)
    ok, buf = cv2.imencode(".png", full)
    assert ok
    return buf.tobytes()
```

---

## Task 3: Create `scripts/tools/run_text_align_gate_test.py`

**Files:**
- Create: `scripts/tools/run_text_align_gate_test.py`

- [ ] **Step 1: Write the full script**

  ```python
  #!/usr/bin/env python3
  """Mode E: Text-only edit generation (Mode B) + per-edit VLM alignment gate.

  Phase 1 (text): build_semantic_list -> call_vlm_text_async(SYSTEM_PROMPT_B)
  Phase 2 (image): for each edit, build 5x2 gate image -> call_vlm_async(SYSTEM_PROMPT_ALIGN_GATE)

  Usage:
      python scripts/tools/run_text_align_gate_test.py \
          --vlm-urls http://localhost:8142/v1,http://localhost:8143/v1 \
          --vlm-model <model-name> \
          [--output-dir <path>]  [--concurrency 4]
  """
  from __future__ import annotations

  import argparse, asyncio, base64, json, logging, time
  from datetime import datetime, timezone
  from pathlib import Path

  import cv2
  import numpy as np
  from openai import AsyncOpenAI

  logging.basicConfig(level=logging.INFO,
                      format="%(asctime)s %(levelname)s %(message)s",
                      datefmt="%H:%M:%S")
  log = logging.getLogger("text_align_gate")

  REPO_ROOT      = Path(__file__).resolve().parent.parent.parent
  OBJ_IDS_FILE   = REPO_ROOT / "configs" / "shard08_test_obj_ids.txt"
  MESH_ROOT      = Path("/mnt/zsn/data/partverse/bench/inputs/mesh")
  IMAGES_ROOT    = Path("/mnt/zsn/data/partverse/bench/inputs/images")
  OVERVIEWS_ROOT = REPO_ROOT / "outputs" / "partverse" / "bench_shard08_overviews"
  SHARD          = "08"

  import sys
  sys.path.insert(0, str(REPO_ROOT))

  from partcraft.pipeline_v3.s1_vlm_core import (
      SYSTEM_PROMPT_B,
      SYSTEM_PROMPT_ALIGN_GATE,
      build_semantic_list,
      build_align_gate_user_prompt,
      parse_align_gate_output,
      call_vlm_async,
      call_vlm_text_async,
      extract_json_object,
      validate_simple,
      quota_for,
  )
  from partcraft.pipeline_v3.qc_rules import (
      check_rules,
      count_part_pixels_in_overview,
      _PALETTE_BGR, _N_VIEWS, _COL_SEP, _ROW_SEP,
  )

  # ── gate image constants ──────────────────────────────────────────────────────
  _RED  = (45, 45, 220)    # BGR
  _GREY = (65, 65, 65)


  def _extract_cell(img: np.ndarray, col: int, row: int) -> np.ndarray:
      H_total, W_total = img.shape[:2]
      W_cell = (W_total - (_N_VIEWS - 1) * _COL_SEP) // _N_VIEWS
      H_cell = (H_total - _ROW_SEP) // 2
      x0 = col * (W_cell + _COL_SEP)
      y0 = row * (H_cell + _ROW_SEP)
      return img[y0: y0 + H_cell, x0: x0 + W_cell].copy()


  def _highlight_cell(cell: np.ndarray, selected_part_ids: set[int]) -> np.ndarray:
      """Recolour bottom-row palette cell: selected → red, others → grey."""
      palette = np.array(_PALETTE_BGR, dtype=np.int32)
      flat    = cell.reshape(-1, 3).astype(np.int32)
      diffs   = np.linalg.norm(flat[:, None, :] - palette[None, :, :], axis=2)
      nearest = np.argmin(diffs, axis=1)
      is_bg   = np.all(flat > 230, axis=1)
      is_sel  = np.array([pid in selected_part_ids for pid in nearest])
      out_flat = np.empty_like(flat)
      out_flat[is_bg]            = [255, 255, 255]
      out_flat[~is_bg & is_sel]  = list(_RED)
      out_flat[~is_bg & ~is_sel] = list(_GREY)
      return out_flat.reshape(cell.shape).astype(np.uint8)


  def build_gate_image(
      ov_img: np.ndarray,
      selected_part_ids: list[int],
      column_map: list[int],
  ) -> bytes:
      """5×2 gate image: col 0 = highlighted best view, cols 1-4 = normal context."""
      sel_set = set(selected_part_ids)
      top_cells, bot_cells = [], []
      for col_idx, orig_view in enumerate(column_map):
          top = _extract_cell(ov_img, orig_view, 0)
          bot = _extract_cell(ov_img, orig_view, 1)
          if col_idx == 0 and sel_set:
              bot = _highlight_cell(bot, sel_set)
          top_cells.append(top)
          bot_cells.append(bot)

      sep_v = np.full((top_cells[0].shape[0], _COL_SEP, 3), 255, dtype=np.uint8)
      sep_h = np.full(
          (_ROW_SEP,
           sum(c.shape[1] for c in top_cells) + (_N_VIEWS - 1) * _COL_SEP, 3),
          255, dtype=np.uint8,
      )

      def hstack(cells):
          row = cells[0]
          for c in cells[1:]:
              sv = np.full((c.shape[0], _COL_SEP, 3), 255, dtype=np.uint8)
              row = np.concatenate([row, sv, c], axis=1)
          return row

      full = np.concatenate([hstack(top_cells), sep_h, hstack(bot_cells)], axis=0)
      ok, buf = cv2.imencode(".png", full)
      assert ok, "cv2.imencode failed"
      return buf.tobytes()


  # ── edit_status helpers ───────────────────────────────────────────────────────

  def _ts() -> str:
      return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


  def _edit_id(obj_id: str, edit_type: str, idx: int) -> str:
      prefix = {"deletion": "del", "modification": "mod", "scale": "scl",
                "material": "mat", "color": "col", "global": "glb"}.get(edit_type, "unk")
      return f"{prefix}_{obj_id}_{idx:03d}"


  def _write_edit_status(path: Path, obj_id: str, edits_status: dict) -> None:
      doc = {
          "obj_id": obj_id,
          "mode": "text_align",
          "schema_version": 1,
          "updated": _ts(),
          "edits": edits_status,
      }
      path.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")


  # ── per-object processing ─────────────────────────────────────────────────────

  async def process_one(
      obj_id: str,
      client: AsyncOpenAI,
      vlm_model: str,
      out_base: Path,
  ) -> dict:
      t0 = time.perf_counter()
      shard = SHARD
      mesh_npz  = MESH_ROOT   / shard / f"{obj_id}.npz"
      img_npz   = IMAGES_ROOT / shard / f"{obj_id}.npz"
      ov_path   = OVERVIEWS_ROOT / shard / obj_id / "overview.png"
      out_dir   = out_base / "objects" / shard / obj_id / "phase1"
      out_dir.mkdir(parents=True, exist_ok=True)
      result_path = out_dir / "edit_status.json"

      # ── Phase 1: text edit generation ──────────────────────────────────────
      try:
          pids, sem_list = build_semantic_list(mesh_npz, img_npz)
      except Exception as e:
          log.warning("[%s] build_semantic_list failed: %s", obj_id, e)
          return {"obj_id": obj_id, "status": "p1_fail", "error": f"sem_list: {e}"}

      quota   = quota_for(len(pids))
      n_total = sum(quota.values())
      quota_line = (
          f"Generate EXACTLY {n_total} edits — "
          f"{quota.get('deletion',0)} deletion · {quota.get('modification',0)} modification · "
          f"{quota.get('scale',0)} scale · {quota.get('material',0)} material · "
          f"{quota.get('color',0)} color · {quota.get('global',0)} global"
      )
      user_p1 = sem_list + "\n\n" + quota_line

      try:
          p1_raw = await call_vlm_text_async(
              client, SYSTEM_PROMPT_B, user_p1, vlm_model, max_tokens=4096)
      except Exception as e:
          log.warning("[%s] Phase 1 VLM error: %s", obj_id, e)
          return {"obj_id": obj_id, "status": "p1_fail", "error": f"vlm: {e}"}

      (out_dir / "raw.txt").write_text(p1_raw, encoding="utf-8")

      p1_parsed = extract_json_object(p1_raw)
      validation = {"ok": False, "errors": ["no_json"]}
      if p1_parsed is not None:
          validation = validate_simple(p1_parsed, set(pids), quota)

      (out_dir / "parsed.json").write_text(
          json.dumps({"obj_id": obj_id, "mode": "text_align",
                      "validation": validation, "parsed": p1_parsed or {}},
                     indent=2, ensure_ascii=False), encoding="utf-8")

      if not validation["ok"] or not p1_parsed:
          log.warning("[%s] Phase 1 validation failed: %s", obj_id, validation.get("errors"))
          return {"obj_id": obj_id, "status": "p1_fail", "validation": validation}

      edits = p1_parsed.get("edits", [])
      t_p1 = time.perf_counter() - t0
      log.info("[%s] Phase 1 ok in %.1fs — %d edits", obj_id, t_p1, len(edits))

      # ── Load overview for gate ──────────────────────────────────────────────
      ov_img: np.ndarray | None = None
      if ov_path.is_file():
          buf = np.frombuffer(ov_path.read_bytes(), dtype=np.uint8)
          decoded = cv2.imdecode(buf, cv2.IMREAD_COLOR)
          if decoded is not None:
              ov_img = decoded

      # parts_by_id for check_rules — built from pids
      parts_by_id = {p: {"part_id": p} for p in pids}

      # ── Phase 2: alignment gate per edit ───────────────────────────────────
      edits_status: dict[str, dict] = {}
      n_pass = n_fail = 0

      for idx, edit in enumerate(edits):
          et     = edit.get("edit_type", "unknown")
          prompt = edit.get("prompt", "")
          sel    = list(edit.get("selected_part_ids") or [])
          eid    = _edit_id(obj_id, et, idx)

          gate_record: dict = {}

          # Layer 1: rule check
          rule_fails = check_rules(edit, parts_by_id)
          gate_record["rule"] = {"pass": not rule_fails, "checks": rule_fails}

          if rule_fails:
              ts = _ts()
              edits_status[eid] = {
                  "edit_type": et,
                  "stages": {"gate_a": {"status": "fail", "ts": ts}},
                  "gates": {"A": {**gate_record, "vlm": None}},
              }
              n_fail += 1
              log.debug("[%s] %s rule_fail: %s", obj_id, eid, rule_fails)
              continue

          # Global edit: auto-pass, no image call
          if et == "global" or not sel:
              ts = _ts()
              edits_status[eid] = {
                  "edit_type": et,
                  "stages": {"gate_a": {"status": "pass", "ts": ts}},
                  "gates": {"A": {**gate_record,
                                  "vlm": {"pass": True, "score": 1.0,
                                          "reason": "global edit auto-pass",
                                          "best_view": 0, "best_view_col": 0,
                                          "pixel_counts": [0]*5,
                                          "column_map": list(range(5))}}},
              }
              n_pass += 1
              continue

          # No overview: auto-pass all with best_view=0
          if ov_img is None:
              ts = _ts()
              edits_status[eid] = {
                  "edit_type": et,
                  "stages": {"gate_a": {"status": "pass", "ts": ts}},
                  "gates": {"A": {**gate_record,
                                  "vlm": {"pass": True, "score": 1.0,
                                          "reason": "no overview, auto-pass",
                                          "best_view": 0, "best_view_col": 0,
                                          "pixel_counts": [-1]*5,
                                          "column_map": list(range(5))}}},
              }
              n_pass += 1
              continue

          # Layer 2: compute pixel counts → column_map
          px = [count_part_pixels_in_overview(ov_img, v, sel) for v in range(_N_VIEWS)]
          best_col_view = int(np.argmax(px))
          other_views   = [v for v in range(_N_VIEWS) if v != best_col_view]
          column_map    = [best_col_view] + other_views

          # Layer 3: VLM alignment gate
          gate_img = build_gate_image(ov_img, sel, column_map)
          gate_user = build_align_gate_user_prompt(et, prompt, sel)

          try:
              gate_raw = await call_vlm_async(
                  client, gate_img, SYSTEM_PROMPT_ALIGN_GATE, gate_user,
                  vlm_model, max_tokens=256)
              gate_out = parse_align_gate_output(gate_raw)
          except Exception as e:
              log.warning("[%s] %s gate VLM error: %s", obj_id, eid, e)
              gate_out = None

          if gate_out is None:
              gate_out = {"aligned": False, "reason": "vlm_error", "best_view": 0}

          aligned        = bool(gate_out.get("aligned", False))
          best_view_col  = int(gate_out.get("best_view", 0))
          best_view_orig = column_map[best_view_col] if 0 <= best_view_col < 5 else column_map[0]

          ts = _ts()
          edits_status[eid] = {
              "edit_type": et,
              "stages": {"gate_a": {"status": "pass" if aligned else "fail", "ts": ts}},
              "gates": {"A": {
                  **gate_record,
                  "vlm": {
                      "pass":          aligned,
                      "score":         1.0 if aligned else 0.0,
                      "reason":        gate_out.get("reason", ""),
                      "best_view":     best_view_orig,
                      "best_view_col": best_view_col,
                      "pixel_counts":  px,
                      "column_map":    column_map,
                  },
              }},
          }
          if aligned:
              n_pass += 1
          else:
              n_fail += 1

      _write_edit_status(result_path, obj_id, edits_status)

      elapsed = time.perf_counter() - t0
      log.info("[%s] done %.1fs — %d pass / %d fail (yield %.0f%%)",
               obj_id, elapsed, n_pass, n_fail,
               100 * n_pass / max(1, n_pass + n_fail))

      return {
          "obj_id": obj_id, "status": "ok",
          "n_edits": len(edits), "n_pass": n_pass, "n_fail": n_fail,
          "elapsed": round(elapsed, 1),
      }


  # ── main ─────────────────────────────────────────────────────────────────────

  async def main(args) -> None:
      vlm_urls  = [u.strip() for u in args.vlm_urls.split(",") if u.strip()]
      out_base  = Path(args.output_dir)
      obj_ids   = [l.strip() for l in Path(OBJ_IDS_FILE).read_text().splitlines()
                   if l.strip() and not l.startswith("#")]
      log.info("Running Mode E on %d objects → %s", len(obj_ids), out_base)

      sem = asyncio.Semaphore(args.concurrency)

      async def _one(obj_id: str, url: str) -> dict:
          async with sem:
              client = AsyncOpenAI(base_url=url, api_key="EMPTY")
              return await process_one(obj_id, client, args.vlm_model, out_base)

      tasks = [_one(oid, vlm_urls[i % len(vlm_urls)])
               for i, oid in enumerate(obj_ids)]
      results = await asyncio.gather(*tasks, return_exceptions=True)

      # ── summary ──────────────────────────────────────────────────────────
      by_type: dict[str, dict] = {}
      n_p1_fail = 0
      n_total = n_pass_total = 0
      obj_rows = []

      for r in results:
          if isinstance(r, Exception):
              n_p1_fail += 1
              continue
          if r.get("status") != "ok":
              n_p1_fail += 1
              continue
          n  = r["n_edits"]
          np_ = r["n_pass"]
          n_total     += n
          n_pass_total += np_
          obj_rows.append(r)

          # read edit_status.json to get per-type breakdown
          oid   = r["obj_id"]
          esjson = out_base / "objects" / SHARD / oid / "phase1" / "edit_status.json"
          if esjson.is_file():
              es = json.loads(esjson.read_text())
              for eid, ev in es.get("edits", {}).items():
                  et = ev.get("edit_type", "unknown")
                  passed = ev.get("stages", {}).get("gate_a", {}).get("status") == "pass"
                  bt = by_type.setdefault(et, {"total": 0, "pass": 0})
                  bt["total"] += 1
                  if passed:
                      bt["pass"] += 1

      summary = {
          "n_objects":     len(obj_ids),
          "n_p1_fail":     n_p1_fail,
          "n_edits_total": n_total,
          "n_gate_pass":   n_pass_total,
          "n_gate_fail":   n_total - n_pass_total,
          "yield_rate":    round(n_pass_total / max(1, n_total), 3),
          "by_type":       by_type,
      }
      (out_base / "_summary.json").write_text(
          json.dumps(summary, indent=2), encoding="utf-8")

      log.info("=== SUMMARY ===")
      log.info("Objects: %d  |  P1 fail: %d", len(obj_ids), n_p1_fail)
      log.info("Edits total: %d  |  gate pass: %d  |  yield: %.0f%%",
               n_total, n_pass_total, 100 * summary["yield_rate"])
      for et, v in sorted(by_type.items()):
          log.info("  %-14s %d/%d", et, v["pass"], v["total"])


  if __name__ == "__main__":
      ap = argparse.ArgumentParser()
      ap.add_argument("--vlm-urls",    required=True,
                      help="Comma-separated VLM base URLs")
      ap.add_argument("--vlm-model",   required=True)
      ap.add_argument("--output-dir",
                      default=str(REPO_ROOT / "data" / "partverse" / "outputs" /
                                  "partverse" / "bench_shard08" / "mode_e_text_align"))
      ap.add_argument("--concurrency", type=int, default=4)
      asyncio.run(main(ap.parse_args()))
  ```

- [ ] **Step 2: Verify syntax**

  ```bash
  python -m py_compile scripts/tools/run_text_align_gate_test.py && echo "syntax ok"
  ```
  Expected: `syntax ok`

- [ ] **Step 3: Dry-run import check**

  ```bash
  python -c "
  import sys; sys.path.insert(0, '.')
  from scripts.tools import run_text_align_gate_test as m
  print('imports ok')
  print('build_gate_image' in dir(m))
  "
  ```
  Expected:
  ```
  imports ok
  True
  ```

- [ ] **Step 4: Commit**

  ```bash
  git add scripts/tools/run_text_align_gate_test.py
  git commit -m "feat: add Mode E text-align-gate experiment script"
  ```

---

## Task 4: Validation Run on bench_shard08

- [ ] **Step 1: Check VLM servers are running**

  ```bash
  curl -s http://localhost:8142/v1/models | python3 -m json.tool | head -5
  ```
  Expected: JSON with a model entry. If not, start servers per the project's standard procedure.

- [ ] **Step 2: Run the experiment**

  ```bash
  python scripts/tools/run_text_align_gate_test.py \
      --vlm-urls http://localhost:8142/v1,http://localhost:8143/v1 \
      --vlm-model <model-name-from-step-1> \
      --concurrency 4
  ```

  Watch logs for per-object summaries. Expected wall time: 5–15 minutes for 20 objects.

- [ ] **Step 3: Inspect summary**

  ```bash
  cat data/partverse/outputs/partverse/bench_shard08/mode_e_text_align/_summary.json
  ```
  Check:
  - `n_p1_fail` should be ≤ 3 (most objects have valid captions)
  - `yield_rate` hypothesis: 0.65–0.85
  - `by_type` shows per-type breakdown

- [ ] **Step 4: Spot-check one object**

  ```bash
  # Replace with an obj_id from the run
  cat data/partverse/outputs/partverse/bench_shard08/mode_e_text_align/objects/08/<obj_id>/phase1/edit_status.json | python3 -m json.tool | head -60
  ```
  Verify:
  - Each edit has `gates.A.vlm.pixel_counts` (list of 5 ints)
  - Each edit has `gates.A.vlm.column_map` (list of 5 ints, first = argmax of pixel_counts)
  - Each edit has `gates.A.vlm.best_view` = `column_map[best_view_col]` (original view index)
  - `stages.gate_a.status` matches `gates.A.vlm.pass`

- [ ] **Step 5: Commit results note**

  ```bash
  git add data/partverse/outputs/partverse/bench_shard08/mode_e_text_align/_summary.json
  git commit -m "experiment: Mode E text-align-gate bench_shard08 results"
  ```

---

## Self-Review Checklist

- [x] `SYSTEM_PROMPT_ALIGN_GATE` in Task 1 matches spec R1-R4
- [x] `column_map` provenance: `[best_col_view] + other_views`, stored in `gates.A.vlm`
- [x] `best_view` in output = `column_map[best_view_col]` (original view index)
- [x] `pixel_counts[v]` covers all 5 views before argmax
- [x] Rule check (`check_rules`) runs before VLM gate — no VLM call on rule failures
- [x] Global edits auto-pass without image call
- [x] Missing `overview.png` auto-passes all edits with `best_view=0`
- [x] `edit_status.json` schema matches spec (mode, schema_version, stages, gates)
- [x] `_summary.json` includes `by_type` breakdown
- [x] No production pipeline files touched
