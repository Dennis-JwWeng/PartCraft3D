"""Phase 1: Edit planning — generate edit specifications from the Part Catalog.

Four edit strategies:
  1. Deletion     (remove a part from an object)
  2. Addition     (add a part to an object — reverse of deletion)
  3. Modification (swap a part's shape via VLM + TRELLIS)
  4. Global       (change whole-object style/theme via VLM + TRELLIS)

Deletion specs are generated first, then additions (as their reverse),
then modifications (swap only), then global edits.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

from partcraft.phase0_semantic.catalog import PartCatalog, CatalogEntry


@dataclass
class EditSpec:
    """Specification for a single edit operation."""
    edit_id: str
    edit_type: str                     # "deletion" | "addition" | "modification"
    obj_id: str
    shard: str
    object_desc: str
    before_desc: str = ""

    # Deletion: which parts to remove
    remove_part_ids: list[int] = field(default_factory=list)
    remove_labels: list[str] = field(default_factory=list)
    keep_part_ids: list[int] = field(default_factory=list)

    # Addition: which parts to add (reverse of deletion)
    add_part_ids: list[int] = field(default_factory=list)
    add_labels: list[str] = field(default_factory=list)
    base_part_ids: list[int] = field(default_factory=list)

    # Modification: which part to change
    old_part_id: int = -1
    old_label: str = ""

    # Addition: link to source deletion (addition uses deletion output as input)
    source_del_id: str = ""

    # VLM-generated editing instructions (from Phase 0)
    edit_prompt: str = ""
    after_desc: str = ""
    before_part_desc: str = ""
    after_part_desc: str = ""
    mod_type: str = ""                 # "style" or "swap" (for modification)

    # Best view for 2D editing (NPZ view index, from Step 1 orthogonal selection)
    best_view: int = -1                # -1 = not set, fallback to mask-based

    def to_dict(self) -> dict:
        return asdict(self)


def _find_edit_by_type(edits: list[dict], edit_type: str) -> dict | None:
    for e in edits:
        if e.get("type") == edit_type:
            return e
    return None


def plan_edits(catalog: PartCatalog, cfg: dict) -> list[EditSpec]:
    """Generate all edit specs: deletion → addition → modification → global.

    For each non-core part:
      - One deletion spec (remove this part)
      - One addition spec (add this part back — reverse)
      - N swap modification specs (shape replacement only)
    For groups of same-category parts:
      - One group deletion + one group addition
    For each object:
      - 2-3 global style/theme edits (whole-object)
    """
    min_cluster = cfg["phase1"].get("min_cluster_size", 50)
    core_cats = set(cfg["phase1"].get("core_categories", []))
    min_parts = cfg["phase1"].get("min_parts_per_object", 2)
    max_global = cfg["phase1"].get("max_global_edits_per_object", 3)

    del_specs = []
    add_specs = []
    mod_specs = []
    glb_specs = []
    del_counter = add_counter = mod_counter = glb_counter = 0

    # Objects with group_edits are handled separately below — skip per-part generation
    group_edit_obj_ids = set(catalog.object_group_edits.keys())

    for obj_id, indices in catalog.by_object.items():
        if obj_id in group_edit_obj_ids:
            continue  # will be handled by group edits section

        entries = [catalog.entries[i] for i in indices]
        all_pids = [e.part_id for e in entries]
        obj_desc = catalog.object_descs.get(obj_id, "")

        if len(entries) < min_parts:
            continue

        for entry in entries:
            # Skip core parts and tiny parts
            if entry.core or entry.category in core_cats:
                pass  # core parts can still have modifications
            else:
                if entry.cluster_size > 0 and entry.cluster_size < min_cluster:
                    continue

                keep_pids = [p for p in all_pids if p != entry.part_id]
                if not keep_pids:
                    continue

                del_edit = _find_edit_by_type(entry.edits, "deletion")
                add_edit = _find_edit_by_type(entry.edits, "addition")

                # --- Deletion ---
                del_id = f"del_{del_counter:06d}"
                del_specs.append(EditSpec(
                    edit_id=del_id,
                    edit_type="deletion",
                    obj_id=obj_id,
                    shard=entry.shard,
                    object_desc=obj_desc,
                    before_desc=obj_desc,
                    remove_part_ids=[entry.part_id],
                    remove_labels=[entry.label],
                    keep_part_ids=keep_pids,
                    edit_prompt=del_edit.get("prompt", "") if del_edit else "",
                    after_desc=del_edit.get("after_desc", "") if del_edit else "",
                ))
                del_counter += 1

                # --- Addition (reverse of deletion) ---
                # Uses deletion's output SLAT as starting point,
                # original rendering as 2D condition to add the part back.
                add_specs.append(EditSpec(
                    edit_id=f"add_{add_counter:06d}",
                    edit_type="addition",
                    obj_id=obj_id,
                    shard=entry.shard,
                    object_desc=obj_desc,
                    before_desc=entry.desc_without or obj_desc,
                    add_part_ids=[entry.part_id],
                    add_labels=[entry.label],
                    base_part_ids=keep_pids,
                    source_del_id=del_id,
                    edit_prompt=add_edit.get("prompt", "") if add_edit else "",
                    after_desc=obj_desc,  # after adding back = original
                ))
                add_counter += 1

            # --- Modifications (all parts, including core) ---
            if entry.cluster_size > 0 and entry.cluster_size < min_cluster:
                continue

            # Only swap modifications (shape replacement), skip style/color
            mod_edits = [e for e in entry.edits
                         if e.get("type") == "modification"
                         and e.get("mod_type") == "swap"
                         and e.get("prompt")]
            if not mod_edits:
                continue

            keep_pids_mod = [p for p in all_pids if p != entry.part_id]
            for mod_edit in mod_edits:
                mod_specs.append(EditSpec(
                    edit_id=f"mod_{mod_counter:06d}",
                    edit_type="modification",
                    obj_id=obj_id,
                    shard=entry.shard,
                    object_desc=obj_desc,
                    before_desc=obj_desc,
                    old_part_id=entry.part_id,
                    old_label=entry.label,
                    keep_part_ids=keep_pids_mod,
                    edit_prompt=mod_edit.get("prompt", ""),
                    after_desc=mod_edit.get("after_desc", ""),
                    before_part_desc=mod_edit.get("before_part_desc", entry.desc),
                    after_part_desc=mod_edit.get("after_part_desc", ""),
                    mod_type=mod_edit.get("mod_type", "style"),
                ))
                mod_counter += 1

        # --- Group deletion/addition: same-category parts ---
        by_cat: dict[str, list[CatalogEntry]] = {}
        for e in entries:
            if (not e.core and e.category not in core_cats
                    and (e.cluster_size == 0 or e.cluster_size >= min_cluster)):
                by_cat.setdefault(e.category, []).append(e)

        for cat, cat_entries in by_cat.items():
            if len(cat_entries) < 2:
                continue
            remove_ids = [e.part_id for e in cat_entries]
            keep_ids = [p for p in all_pids if p not in remove_ids]
            if not keep_ids:
                continue

            cat_label = cat.replace("_", " ")
            gdel_id = f"gdel_{del_counter:06d}"
            del_specs.append(EditSpec(
                edit_id=gdel_id,
                edit_type="deletion",
                obj_id=obj_id,
                shard=entries[0].shard,
                object_desc=obj_desc,
                before_desc=obj_desc,
                remove_part_ids=remove_ids,
                remove_labels=[e.label for e in cat_entries],
                keep_part_ids=keep_ids,
                edit_prompt=f"Remove all {cat_label}s",
            ))
            del_counter += 1

            add_specs.append(EditSpec(
                edit_id=f"gadd_{add_counter:06d}",
                edit_type="addition",
                obj_id=obj_id,
                shard=entries[0].shard,
                object_desc=obj_desc,
                before_desc=cat_entries[0].desc_without or obj_desc,
                add_part_ids=remove_ids,
                add_labels=[e.label for e in cat_entries],
                base_part_ids=keep_ids,
                source_del_id=gdel_id,
                edit_prompt=f"Add {cat_label}s",
                after_desc=obj_desc,
            ))
            add_counter += 1

    # --- Group edits (from orthogonal 4-view enrichment) ---
    for obj_id, group_edits in catalog.object_group_edits.items():
        indices = catalog.by_object.get(obj_id, [])
        entries = [catalog.entries[i] for i in indices]
        if len(entries) < min_parts:
            continue

        obj_desc = catalog.object_descs.get(obj_id, "")
        all_pids = [e.part_id for e in entries]
        shard = entries[0].shard if entries else "00"

        for grp in group_edits:
            grp_pids = grp.get("part_ids", [])
            grp_best_view = grp.get("best_view", -1)
            grp_desc = grp.get("desc", "")
            grp_labels = [entries[p].label if p < len(entries) else f"part_{p}"
                          for p in grp_pids]
            keep_pids = [p for p in all_pids if p not in grp_pids]
            if not keep_pids:
                continue

            for edit in grp.get("edits", []):
                etype = edit.get("type")
                if etype == "deletion":
                    del_id = f"del_{del_counter:06d}"
                    del_specs.append(EditSpec(
                        edit_id=del_id,
                        edit_type="deletion",
                        obj_id=obj_id,
                        shard=shard,
                        object_desc=obj_desc,
                        before_desc=obj_desc,
                        remove_part_ids=grp_pids,
                        remove_labels=grp_labels,
                        keep_part_ids=keep_pids,
                        edit_prompt=edit.get("prompt", ""),
                        after_desc=edit.get("after_desc", ""),
                        best_view=grp_best_view,
                    ))
                    del_counter += 1
                elif etype == "addition":
                    add_specs.append(EditSpec(
                        edit_id=f"add_{add_counter:06d}",
                        edit_type="addition",
                        obj_id=obj_id,
                        shard=shard,
                        object_desc=obj_desc,
                        before_desc=edit.get("after_desc", obj_desc),
                        add_part_ids=grp_pids,
                        add_labels=grp_labels,
                        base_part_ids=keep_pids,
                        source_del_id=f"del_{del_counter - 1:06d}",
                        edit_prompt=edit.get("prompt", ""),
                        after_desc=obj_desc,
                        best_view=grp_best_view,
                    ))
                    add_counter += 1
                elif etype == "modification":
                    mod_specs.append(EditSpec(
                        edit_id=f"mod_{mod_counter:06d}",
                        edit_type="modification",
                        obj_id=obj_id,
                        shard=shard,
                        object_desc=obj_desc,
                        before_desc=obj_desc,
                        old_part_id=grp_pids[0] if grp_pids else -1,
                        old_label=grp_labels[0] if grp_labels else "",
                        remove_part_ids=grp_pids,
                        keep_part_ids=keep_pids,
                        edit_prompt=edit.get("prompt", ""),
                        after_desc=edit.get("after_desc", ""),
                        before_part_desc=edit.get("before_part_desc", grp_desc),
                        after_part_desc=edit.get("after_part_desc", ""),
                        mod_type=edit.get("mod_type", "swap"),
                        best_view=grp_best_view,
                    ))
                    mod_counter += 1

    # --- Global edits (whole-object style/theme changes) ---
    for obj_id, indices in catalog.by_object.items():
        entries = [catalog.entries[i] for i in indices]
        if len(entries) < min_parts:
            continue

        obj_desc = catalog.object_descs.get(obj_id, "")
        shard = entries[0].shard

        for ge in catalog.object_global_edits.get(obj_id, [])[:max_global]:
            prompt = ge.get("prompt", "")
            if not prompt:
                continue
            glb_specs.append(EditSpec(
                edit_id=f"glb_{glb_counter:06d}",
                edit_type="global",
                obj_id=obj_id,
                shard=shard,
                object_desc=obj_desc,
                before_desc=obj_desc,
                edit_prompt=prompt,
                after_desc=ge.get("after_desc", ""),
            ))
            glb_counter += 1

    # Order: deletion → addition → modification → global
    all_specs = del_specs + add_specs + mod_specs + glb_specs
    return all_specs


def plan_edits_for_record(record: dict, cfg: dict,
                          counters: dict | None = None) -> list[EditSpec]:
    """Plan edits from a single semantic_labels JSONL record.

    Works without building a full PartCatalog — suitable for streaming mode
    where each object is planned immediately after enrichment.

    Args:
        record: one line from semantic_labels.jsonl (dict with obj_id, parts,
                group_edits, global_edits, etc.)
        cfg: pipeline config
        counters: mutable dict with keys del/add/mod/glb for global ID counters.
                  If None, starts from 0.

    Returns:
        List of EditSpec for this single object.
    """
    if counters is None:
        counters = {"del": 0, "add": 0, "mod": 0, "glb": 0}

    min_parts = cfg["phase1"].get("min_parts_per_object", 2)
    max_global = cfg["phase1"].get("max_global_edits_per_object", 3)

    obj_id = record["obj_id"]
    shard = record.get("shard", "00")
    obj_desc = record.get("object_desc", "")
    parts = record.get("parts", [])

    if len(parts) < min_parts:
        return []

    all_pids = [p["part_id"] for p in parts]
    specs: list[EditSpec] = []

    # --- Group edits ---
    for grp in record.get("group_edits", []):
        grp_pids = grp.get("part_ids", [])
        grp_best_view = grp.get("best_view", -1)
        grp_desc = grp.get("desc", "")
        grp_labels = []
        for pid in grp_pids:
            found = False
            for p in parts:
                if p["part_id"] == pid:
                    grp_labels.append(p.get("label", f"part_{pid}"))
                    found = True
                    break
            if not found:
                grp_labels.append(f"part_{pid}")

        keep_pids = [p for p in all_pids if p not in grp_pids]
        if not keep_pids:
            continue

        for edit in grp.get("edits", []):
            etype = edit.get("type")
            if etype == "deletion":
                del_id = f"del_{counters['del']:06d}"
                specs.append(EditSpec(
                    edit_id=del_id,
                    edit_type="deletion",
                    obj_id=obj_id, shard=shard,
                    object_desc=obj_desc, before_desc=obj_desc,
                    remove_part_ids=grp_pids, remove_labels=grp_labels,
                    keep_part_ids=keep_pids,
                    edit_prompt=edit.get("prompt", ""),
                    after_desc=edit.get("after_desc", ""),
                    best_view=grp_best_view,
                ))
                counters["del"] += 1
            elif etype == "addition":
                specs.append(EditSpec(
                    edit_id=f"add_{counters['add']:06d}",
                    edit_type="addition",
                    obj_id=obj_id, shard=shard,
                    object_desc=obj_desc,
                    before_desc=edit.get("after_desc", obj_desc),
                    add_part_ids=grp_pids, add_labels=grp_labels,
                    base_part_ids=keep_pids,
                    source_del_id=del_id if counters["del"] > 0
                        else f"del_{counters['del']:06d}",
                    edit_prompt=edit.get("prompt", ""),
                    after_desc=obj_desc,
                    best_view=grp_best_view,
                ))
                counters["add"] += 1
            elif etype == "modification":
                specs.append(EditSpec(
                    edit_id=f"mod_{counters['mod']:06d}",
                    edit_type="modification",
                    obj_id=obj_id, shard=shard,
                    object_desc=obj_desc, before_desc=obj_desc,
                    old_part_id=grp_pids[0] if grp_pids else -1,
                    old_label=grp_labels[0] if grp_labels else "",
                    remove_part_ids=grp_pids, keep_part_ids=keep_pids,
                    edit_prompt=edit.get("prompt", ""),
                    after_desc=edit.get("after_desc", ""),
                    before_part_desc=edit.get("before_part_desc", grp_desc),
                    after_part_desc=edit.get("after_part_desc", ""),
                    mod_type=edit.get("mod_type", "swap"),
                    best_view=grp_best_view,
                ))
                counters["mod"] += 1

    # --- Global edits ---
    for ge in record.get("global_edits", [])[:max_global]:
        prompt = ge.get("prompt", "")
        if not prompt:
            continue
        specs.append(EditSpec(
            edit_id=f"glb_{counters['glb']:06d}",
            edit_type="global",
            obj_id=obj_id, shard=shard,
            object_desc=obj_desc, before_desc=obj_desc,
            edit_prompt=prompt,
            after_desc=ge.get("after_desc", ""),
        ))
        counters["glb"] += 1

    return specs


def run_phase1(cfg: dict, catalog: PartCatalog,
               output_suffix: str = "") -> list[EditSpec]:
    """Run Phase 1: generate all edit specifications."""
    cache_dir = Path(cfg["phase1"]["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_path = cache_dir / f"edit_specs{output_suffix}.jsonl"

    print("Phase 1: Planning edits...")
    print(f"  Catalog: {catalog.summary()}")

    all_specs = plan_edits(catalog, cfg)

    n_del = sum(1 for s in all_specs if s.edit_type == "deletion")
    n_add = sum(1 for s in all_specs if s.edit_type == "addition")
    n_mod = sum(1 for s in all_specs if s.edit_type == "modification")
    n_glb = sum(1 for s in all_specs if s.edit_type == "global")
    print(f"  Deletion: {n_del}, Addition: {n_add}, "
          f"Modification(swap): {n_mod}, Global: {n_glb}")
    print(f"  Total: {len(all_specs)} edit specs")

    with open(output_path, "w") as f:
        for spec in all_specs:
            f.write(json.dumps(spec.to_dict(), ensure_ascii=False) + "\n")

    print(f"Phase 1 complete -> {output_path}")
    return all_specs
