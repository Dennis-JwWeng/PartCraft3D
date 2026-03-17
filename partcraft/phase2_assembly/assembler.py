"""Phase 2: Mesh assembly — execute edit specs to produce before/after mesh pairs.

Operates on watertight part meshes from HY3D-Part.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh
from tqdm import tqdm

from partcraft.io.hy3d_loader import HY3DPartDataset, ObjectRecord
from partcraft.phase1_planning.planner import EditSpec
from partcraft.phase2_assembly.alignment import (
    align_part_to_target, compute_penetration_ratio, compute_gap_distance,
)


@dataclass
class AssembledPair:
    """Result of assembling a before/after mesh pair."""
    edit_spec: EditSpec
    before_mesh: trimesh.Trimesh
    after_mesh: trimesh.Trimesh
    success: bool = True
    error: str = ""


def _load_spec(line: str) -> EditSpec:
    d = json.loads(line)
    return EditSpec(**d)


def execute_deletion(spec: EditSpec, dataset: HY3DPartDataset,
                     obj: ObjectRecord | None = None) -> AssembledPair:
    """Remove parts from an object."""
    own_obj = obj is None
    if own_obj:
        obj = dataset.load_object(spec.shard, spec.obj_id)
    try:
        before_mesh = obj.get_full_mesh()
        after_mesh = obj.get_assembled_mesh(spec.keep_part_ids)
        return AssembledPair(spec, before_mesh, after_mesh)
    except Exception as e:
        return AssembledPair(spec, trimesh.Trimesh(), trimesh.Trimesh(), False, str(e))
    finally:
        if own_obj:
            obj.close()


def execute_addition(spec: EditSpec, dataset: HY3DPartDataset,
                     obj: ObjectRecord | None = None) -> AssembledPair:
    """Add parts to produce the full object (reverse of deletion)."""
    own_obj = obj is None
    if own_obj:
        obj = dataset.load_object(spec.shard, spec.obj_id)
    try:
        before_mesh = obj.get_assembled_mesh(spec.base_part_ids)
        after_mesh = obj.get_full_mesh()
        return AssembledPair(spec, before_mesh, after_mesh)
    except Exception as e:
        return AssembledPair(spec, trimesh.Trimesh(), trimesh.Trimesh(), False, str(e))
    finally:
        if own_obj:
            obj.close()


def execute_swap(spec: EditSpec, dataset: HY3DPartDataset, cfg: dict) -> AssembledPair:
    """Replace one part with a same-category part from another object.

    Uses contact-aware alignment and validates the result with
    penetration + gap checks.
    """
    try:
        src_obj = dataset.load_object(spec.shard, spec.obj_id)
        donor_obj = dataset.load_object(spec.new_shard, spec.new_obj_id)
    except Exception as e:
        return AssembledPair(spec, trimesh.Trimesh(), trimesh.Trimesh(), False, str(e))
    try:
        before_mesh = src_obj.get_full_mesh(colored=True)
        old_part = src_obj.get_part_mesh(spec.old_part_id, colored=False)  # alignment only
        new_part = donor_obj.get_part_mesh(spec.new_part_id, colored=True)
        base_mesh = src_obj.get_assembled_mesh(spec.keep_part_ids, colored=True)

        max_scale = cfg["phase2"].get("max_scale_ratio", 3.0)
        strategy = cfg["phase2"].get("alignment_strategy", "bbox")

        # Pass the body mesh for contact-aware alignment
        new_part_aligned, scale_ratio = align_part_to_target(
            new_part, old_part, body=base_mesh, strategy=strategy,
        )

        if scale_ratio > max_scale or scale_ratio < 1.0 / max_scale:
            return AssembledPair(
                spec, before_mesh, trimesh.Trimesh(), False,
                f"Scale mismatch: {scale_ratio:.2f}x (limit: {max_scale}x)",
            )

        # --- Geometric validation ---
        max_penetration = cfg["phase2"].get("max_penetration", 0.3)
        max_gap = cfg["phase2"].get("max_gap_distance", -1.0)

        # Check penetration: new part shouldn't be mostly inside the body
        penetration = compute_penetration_ratio(new_part_aligned, base_mesh)
        if penetration > max_penetration:
            return AssembledPair(
                spec, before_mesh, trimesh.Trimesh(), False,
                f"Penetration too high: {penetration:.1%} (limit: {max_penetration:.0%})",
            )

        # Check gap: new part shouldn't be floating far from the body
        if max_gap > 0:
            gap = compute_gap_distance(new_part_aligned, base_mesh)
            body_extent = base_mesh.bounding_box.extents.max()
            # Normalize gap by body size
            if gap / body_extent > max_gap:
                return AssembledPair(
                    spec, before_mesh, trimesh.Trimesh(), False,
                    f"Gap too large: {gap:.4f} ({gap/body_extent:.1%} of body, "
                    f"limit: {max_gap:.0%})",
                )

        after_mesh = trimesh.util.concatenate([base_mesh, new_part_aligned])
        return AssembledPair(spec, before_mesh, after_mesh)
    except Exception as e:
        return AssembledPair(spec, trimesh.Trimesh(), trimesh.Trimesh(), False, str(e))
    finally:
        src_obj.close()
        donor_obj.close()


def execute_graft(spec: EditSpec, dataset: HY3DPartDataset, cfg: dict) -> AssembledPair:
    """Transplant a part from a donor object onto a target object."""
    try:
        target_obj = dataset.load_object(spec.shard, spec.obj_id)
        donor_obj = dataset.load_object(spec.new_shard, spec.new_obj_id)
    except Exception as e:
        return AssembledPair(spec, trimesh.Trimesh(), trimesh.Trimesh(), False, str(e))
    try:
        before_mesh = target_obj.get_full_mesh()
        graft_part = donor_obj.get_part_mesh(spec.new_part_id)

        # Position the grafted part relative to the target object's bounding box
        target_center = before_mesh.bounding_box.centroid
        target_extents = before_mesh.bounding_box.extents
        graft_center = graft_part.bounding_box.centroid
        graft_extents = graft_part.bounding_box.extents

        # Scale the graft to be proportional to the target
        max_target_dim = target_extents.max()
        max_graft_dim = graft_extents.max()
        if max_graft_dim > 0:
            # Graft should be at most 40% of target's size
            desired_scale = (max_target_dim * 0.4) / max_graft_dim
            if desired_scale < 1.0:
                graft_part.apply_scale(desired_scale)

        # Center graft on top of target (simple heuristic — can be refined)
        graft_part.apply_translation(target_center - graft_part.bounding_box.centroid)
        # Shift up by half of target + half of graft
        graft_part.apply_translation([0, 0, (target_extents[2] + graft_part.bounding_box.extents[2]) / 2])

        after_mesh = trimesh.util.concatenate([before_mesh, graft_part])
        return AssembledPair(spec, before_mesh, after_mesh)
    except Exception as e:
        return AssembledPair(spec, trimesh.Trimesh(), trimesh.Trimesh(), False, str(e))
    finally:
        target_obj.close()
        donor_obj.close()


def execute_spec(spec: EditSpec, dataset: HY3DPartDataset, cfg: dict,
                 obj: ObjectRecord | None = None) -> AssembledPair:
    """Dispatch to the correct assembly function based on edit type."""
    if spec.edit_type == "deletion":
        return execute_deletion(spec, dataset, obj=obj)
    elif spec.edit_type == "addition":
        if spec.new_obj_id:
            return execute_graft(spec, dataset, cfg)
        return execute_addition(spec, dataset, obj=obj)
    elif spec.edit_type == "modification":
        # Style modifications (mod_*) are handled by Phase 2.5 (TRELLIS), not mesh assembly.
        # Only swap modifications (swap_*) have a donor object.
        if not spec.new_obj_id:
            return AssembledPair(spec, trimesh.Trimesh(), trimesh.Trimesh(), False,
                                 "Style modification — skipped (handled by phase2_5)")
        return execute_swap(spec, dataset, cfg)
    else:
        return AssembledPair(spec, trimesh.Trimesh(), trimesh.Trimesh(), False,
                             f"Unknown edit type: {spec.edit_type}")


def _find_addition_partner(del_spec: dict, all_specs: list[dict]) -> dict | None:
    """Find the matching addition spec for a deletion spec.

    Matches by: same obj_id, addition type, and the deleted part_ids == added part_ids.
    """
    del_parts = set(del_spec.get("remove_part_ids", []))
    for sd in all_specs:
        if sd["edit_type"] != "addition":
            continue
        if sd.get("new_obj_id"):
            continue  # graft, not a simple reverse
        add_parts = set(sd.get("add_part_ids", []))
        if add_parts == del_parts:
            return sd
    return None


def _make_result_dict(spec: EditSpec, before_path: str, after_path: str) -> dict:
    return {
        "ok": True,
        "edit_id": spec.edit_id,
        "edit_type": spec.edit_type,
        "obj_id": spec.obj_id,
        "object_desc": spec.object_desc,
        "before_desc": spec.before_desc or spec.object_desc,
        "before_mesh": before_path,
        "after_mesh": after_path,
        "remove_labels": spec.remove_labels,
        "add_labels": spec.add_labels,
        "old_label": spec.old_label,
        "new_label": spec.new_label,
    }


def _is_pair_done(eid: str, mesh_out: Path) -> bool:
    """Check if a pair has already been assembled (both PLY files exist)."""
    pair_dir = mesh_out / eid
    return (pair_dir / "before.ply").exists() and (pair_dir / "after.ply").exists()


def _process_object_group(obj_id: str, spec_dicts: list[dict],
                          cfg: dict, done_ids: set[str] | None = None) -> list[dict]:
    """Process all specs for one object. Runs in a worker process.

    Optimization: for deletion/addition pairs, only assemble once —
    the addition reuses the deletion's meshes with before/after swapped.
    Skips edit_ids already in done_ids (checkpoint resume).

    Loads the NPZ once, executes all specs, exports PLY files.
    Returns list of result dicts (for manifest / error log).
    """
    dataset = HY3DPartDataset(
        cfg["data"]["image_npz_dir"],
        cfg["data"]["mesh_npz_dir"],
        cfg["data"]["shards"],
    )
    mesh_out = Path(cfg["data"]["output_dir"]) / "mesh_pairs"
    if done_ids is None:
        done_ids = set()
    results = []

    # Separate deletion specs from others; track which additions are paired
    del_specs = [sd for sd in spec_dicts if sd["edit_type"] == "deletion"]
    paired_add_ids: set[str] = set()

    # Load the object ONCE for this entire group
    shard = spec_dicts[0]["shard"] if spec_dicts else None
    obj = dataset.load_object(shard, obj_id) if shard else None

    try:
        # First pass: process deletions and their paired additions together
        for dsd in del_specs:
            del_spec = EditSpec(**dsd)
            del_eid = del_spec.edit_id
            add_sd = _find_addition_partner(dsd, spec_dicts)
            add_eid = add_sd["edit_id"] if add_sd else None

            # Check if both deletion and its paired addition are already done
            del_done = del_eid in done_ids or _is_pair_done(del_eid, mesh_out)
            add_done = (add_eid is None) or add_eid in done_ids or _is_pair_done(add_eid, mesh_out)

            if add_sd is not None:
                paired_add_ids.add(add_sd["edit_id"])

            if del_done and add_done:
                continue

            # Need to assemble — reuse the already-loaded obj
            pair = execute_spec(del_spec, dataset, cfg, obj=obj)

            if not pair.success:
                results.append({"ok": False, "edit_id": del_eid, "error": pair.error})
                continue

            # Save deletion meshes
            if not del_done:
                del_dir = mesh_out / del_eid
                del_dir.mkdir(parents=True, exist_ok=True)
                del_before = str(del_dir / "before.ply")
                del_after = str(del_dir / "after.ply")
                pair.before_mesh.export(del_before)
                pair.after_mesh.export(del_after)
                results.append(_make_result_dict(del_spec, del_before, del_after))

            # Save paired addition (swapped)
            if add_sd is not None and not add_done:
                add_spec = EditSpec(**add_sd)
                add_dir = mesh_out / add_eid
                add_dir.mkdir(parents=True, exist_ok=True)
                add_before = str(add_dir / "before.ply")
                add_after = str(add_dir / "after.ply")
                pair.after_mesh.export(add_before)
                pair.before_mesh.export(add_after)
                results.append(_make_result_dict(add_spec, add_before, add_after))

        # Second pass: remaining specs (unpaired additions, modifications, etc.)
        for sd in spec_dicts:
            if sd["edit_type"] == "deletion":
                continue
            if sd["edit_id"] in paired_add_ids:
                continue

            spec = EditSpec(**sd)
            eid = spec.edit_id

            if eid in done_ids or _is_pair_done(eid, mesh_out):
                continue

            try:
                pair = execute_spec(spec, dataset, cfg, obj=obj)
            except Exception as e:
                results.append({"ok": False, "edit_id": eid, "error": str(e)})
                continue

            if pair.success:
                pair_dir = mesh_out / eid
                pair_dir.mkdir(parents=True, exist_ok=True)
                before_path = str(pair_dir / "before.ply")
                after_path = str(pair_dir / "after.ply")
                pair.before_mesh.export(before_path)
                pair.after_mesh.export(after_path)
                results.append(_make_result_dict(spec, before_path, after_path))
            else:
                results.append({"ok": False, "edit_id": eid, "error": pair.error})

    finally:
        if obj is not None:
            obj.close()

    return results


def run_phase2(cfg: dict, specs: list[EditSpec] | None = None,
               dataset: HY3DPartDataset | None = None,
               limit: int | None = None,
               max_workers: int | None = None) -> list[AssembledPair]:
    """Run Phase 2: execute all edit specs and produce mesh pairs.

    Groups specs by object to avoid redundant NPZ loading, then
    processes object groups in parallel with multiprocessing.

    Returns list of successful AssembledPairs. Failed ones are logged and skipped.
    """
    cache_dir = Path(cfg["phase2"]["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)

    if dataset is None:
        dataset = HY3DPartDataset(
            cfg["data"]["image_npz_dir"],
            cfg["data"]["mesh_npz_dir"],
            cfg["data"]["shards"],
        )

    if specs is None:
        specs_path = Path(cfg["phase1"]["cache_dir"]) / "edit_specs.jsonl"
        specs = []
        with open(specs_path) as f:
            for line in f:
                if line.strip():
                    specs.append(_load_spec(line))

    if limit:
        specs = specs[:limit]

    mesh_out = Path(cfg["data"]["output_dir"]) / "mesh_pairs"
    mesh_out.mkdir(parents=True, exist_ok=True)

    error_log = cache_dir / "assembly_errors.jsonl"
    manifest_path = cache_dir / "assembled_pairs.jsonl"

    # --- Checkpoint resume: load already-done edit_ids from manifest ---
    done_ids: set[str] = set()
    if manifest_path.exists():
        with open(manifest_path) as f:
            for line in f:
                if line.strip():
                    try:
                        done_ids.add(json.loads(line)["edit_id"])
                    except (json.JSONDecodeError, KeyError):
                        pass
    if done_ids:
        print(f"  Resuming: {len(done_ids)} pairs already done, skipping them.")

    # Group specs by object ID for NPZ reuse
    by_obj: dict[str, list[dict]] = defaultdict(list)
    for spec in specs:
        by_obj[spec.obj_id].append(spec.to_dict())

    # Filter out fully-done object groups to avoid unnecessary work
    pending_by_obj: dict[str, list[dict]] = {}
    skipped_specs = 0
    for obj_id, spec_dicts in by_obj.items():
        pending = [sd for sd in spec_dicts if sd["edit_id"] not in done_ids]
        if pending:
            # Pass full group so del/add pairing still works, worker skips done ones
            pending_by_obj[obj_id] = spec_dicts
        else:
            skipped_specs += len(spec_dicts)

    # Split large object groups into chunks to avoid long-tail stalls.
    # Each chunk still loads the NPZ once, but parallelism improves.
    CHUNK_SIZE = 20  # max del/add pairs per chunk (each deletion = 2 specs with pair)
    task_list: list[tuple[str, list[dict]]] = []
    for obj_id, spec_dicts in pending_by_obj.items():
        del_specs_in_group = [sd for sd in spec_dicts if sd["edit_type"] == "deletion"]
        if len(del_specs_in_group) <= CHUNK_SIZE:
            task_list.append((obj_id, spec_dicts))
        else:
            # Chunk the deletion specs; each chunk carries its paired additions
            del_part_ids_all = {sd["edit_id"] for sd in del_specs_in_group}
            add_specs_in_group = [sd for sd in spec_dicts if sd["edit_type"] != "deletion"]
            for i in range(0, len(del_specs_in_group), CHUNK_SIZE):
                chunk_dels = del_specs_in_group[i:i + CHUNK_SIZE]
                chunk_del_ids = {sd["edit_id"] for sd in chunk_dels}
                # Include additions that pair with this chunk's deletions
                chunk_del_parts = set()
                for sd in chunk_dels:
                    chunk_del_parts.update(sd.get("remove_part_ids", []))
                chunk_adds = [sd for sd in add_specs_in_group
                              if set(sd.get("add_part_ids", [])) & chunk_del_parts]
                task_list.append((obj_id, chunk_dels + chunk_adds))

    n_tasks = len(task_list)
    n_workers = max_workers or min(max(n_tasks, 1), os.cpu_count() or 4, 8)
    print(f"Phase 2: Assembling {len(specs) - skipped_specs} new pairs "
          f"from {len(pending_by_obj)} objects ({n_tasks} tasks, {n_workers} workers), "
          f"{skipped_specs} already done...")

    results = []
    success, fail, skipped = 0, 0, 0

    def _handle_result(r: dict, manifest_fp, err_fp):
        nonlocal success, fail, skipped
        if r.get("skipped"):
            skipped += 1
            return
        if r["ok"]:
            manifest_fp.write(json.dumps(
                {k: v for k, v in r.items() if k not in ("ok", "skipped")},
                ensure_ascii=False) + "\n")
            manifest_fp.flush()
            success += 1
        else:
            # Style mod skips are not real failures — just count them
            if "handled by phase2_5" in r.get("error", ""):
                skipped += 1
            else:
                err_fp.write(json.dumps(
                    {"edit_id": r["edit_id"], "error": r["error"]}
                ) + "\n")
                fail += 1

    # Append to manifest (not overwrite), overwrite error log
    with open(error_log, "w") as err_fp, \
         open(manifest_path, "a") as manifest_fp:
        if n_workers <= 1 or n_tasks <= 1:
            for obj_id, spec_dicts in tqdm(task_list, desc="Phase 2: Assembly"):
                for r in _process_object_group(obj_id, spec_dicts, cfg, done_ids):
                    _handle_result(r, manifest_fp, err_fp)
        else:
            futures = {}
            with ProcessPoolExecutor(max_workers=n_workers) as pool:
                for obj_id, spec_dicts in task_list:
                    fut = pool.submit(_process_object_group,
                                      obj_id, spec_dicts, cfg, done_ids)
                    futures[fut] = obj_id

                for fut in tqdm(as_completed(futures), total=len(futures),
                                desc="Phase 2: Assembly"):
                    try:
                        group_results = fut.result()
                    except Exception as e:
                        obj_id = futures[fut]
                        err_fp.write(json.dumps(
                            {"edit_id": f"obj_{obj_id[:12]}", "error": str(e)}
                        ) + "\n")
                        fail += 1
                        continue

                    for r in group_results:
                        _handle_result(r, manifest_fp, err_fp)

    total_done = len(done_ids) + success
    parts = [f"{success} new + {len(done_ids)} cached = {total_done} assembled"]
    if skipped:
        parts.append(f"{skipped} skipped (style mod → phase2_5)")
    if fail:
        parts.append(f"{fail} failed")
    print(f"Phase 2 complete: {', '.join(parts)} → {mesh_out}")
    return results
