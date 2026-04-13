# Phase-1 Part Menu: Peer Group & Structural Hints from anno_info

**Date:** 2026-04-13
**Files:** `s1_vlm_core.py`, `s1_phase1_vlm.py`
**Status:** Approved

## Problem

Phase-1 VLM frequently edits one member of a symmetric/peer pair in isolation
(e.g. deletes one leg, changes one arm) because the flat part menu gives no
indication that certain parts are semantically equivalent peers.

Symmetry detection from PLY geometry is fragile and assumes symmetric objects.
The source `_info.json` annotation already provides `ordered_part_level` and
`weights` which, without any geometric assumptions, identify peer parts:

  **Same level + weight ratio ≥ 0.75 = peer group** (semantically equivalent parts)

## Design

### Change 1: `build_part_menu` enrichment (s1_vlm_core.py)

Add optional `anno_obj_dir: Path | None = None`. When present:
- Load `{obj_id}_info.json` from the dir
- Extract `ordered_part_level[pid]` and `weights[pid]` per part
- Compute peer groups: parts sharing the same level with weight_ratio ≥ 0.75
- Mark level=0 parts with weight > 2× median as `[STRUCTURAL BODY]`
- Append `level=N  peer_group=[i,j,k]` (or `[STRUCTURAL BODY]`) to each row

Fallback: if anno_obj_dir is None or _info.json missing → current behavior unchanged.

### Change 2: P5 rewrite (USER_PROMPT_TEMPLATE in s1_vlm_core.py)

Replace symmetry-assumption-based P5 with peer-group-reference P5:

  P5. PEER GROUPS → GROUP EDITS. The part menu marks `peer_group=[...]` for
      parts that are semantically equivalent (same structural tier, similar size).
      When editing any part in a peer group, you MUST produce a single group edit
      whose `selected_part_ids` includes EVERY member of that peer_group.
      Example: if parts 2 and 3 share peer_group=[2,3], never edit just part 2
      alone — always edit [2,3] together.
      Parts marked [STRUCTURAL BODY] must never appear in deletion or
      extreme-scale edits (R3 reinforcement).

### Change 3: Pass anno_obj_dir through prerender pipeline (s1_phase1_vlm.py)

- `prerender()`: compute `anno_obj_dir` from ctx and pass to `build_part_menu`
- `_prerender_worker(args)`: extend tuple to include `anno_obj_dir_str` (5th element)
- `render_one` in `run_many_streaming`: pass `str(ctx.root.anno_object_dir(ctx.obj_id) or "")`

## Files Changed

- `partcraft/pipeline_v2/s1_vlm_core.py` — build_part_menu + P5
- `partcraft/pipeline_v2/s1_phase1_vlm.py` — prerender + _prerender_worker + render_one

## Success Criteria

- Part menus with anno_info show `peer_group=` annotations for qualifying parts
- VLM outputs show group edits (multi-part `selected_part_ids`) for peer-group parts
- No regression when anno_info is absent (pure fallback to old menu format)
