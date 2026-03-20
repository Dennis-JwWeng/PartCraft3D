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


def _make_edit_id(prefix: str, obj_id: str, seq: int) -> str:
    """Create a globally-unique, human-readable edit ID.

    Format: {prefix}_{obj_id}_{seq:03d}
    Examples: del_abc12345_000, add_abc12345_001, glb_abc12345_000

    Full obj_id is embedded so you can directly trace any edit back to its
    source object without a lookup table.
    """
    return f"{prefix}_{obj_id}_{seq:03d}"


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

    # Build set of part IDs covered by group_edits (per object),
    # so per-part generation skips only those parts — not the whole object.
    group_edit_part_ids: dict[str, set[int]] = {}
    for obj_id, group_edits in catalog.object_group_edits.items():
        pids = set()
        for grp in group_edits:
            pids.update(grp.get("part_ids", []))
        group_edit_part_ids[obj_id] = pids

    for obj_id, indices in catalog.by_object.items():
        entries = [catalog.entries[i] for i in indices]
        all_pids = [e.part_id for e in entries]
        obj_desc = catalog.object_descs.get(obj_id, "")
        skip_pids = group_edit_part_ids.get(obj_id, set())
        # Default best_view for per-part edits: front orthogonal view
        _ortho = catalog.object_ortho_views.get(obj_id, [])
        obj_best_view = _ortho[0] if _ortho else 0

        if len(entries) < min_parts:
            continue

        # Per-object counters for unique edit IDs
        obj_del = obj_add = obj_mod = 0

        for entry in entries:
            # Skip parts already covered by group edits
            if entry.part_id in skip_pids:
                continue

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
                del_id = _make_edit_id("del", obj_id, obj_del)
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
                    best_view=obj_best_view,
                ))
                obj_del += 1

                # --- Addition (reverse of deletion) ---
                add_specs.append(EditSpec(
                    edit_id=_make_edit_id("add", obj_id, obj_add),
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
                    after_desc=obj_desc,
                    best_view=obj_best_view,
                ))
                obj_add += 1

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
                    edit_id=_make_edit_id("mod", obj_id, obj_mod),
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
                    best_view=obj_best_view,
                ))
                obj_mod += 1

        # --- Group deletion/addition: same-category parts ---
        # Skip auto-grouping for objects that already have enriched group_edits
        # (those groups are handled in the dedicated section below).
        if obj_id in group_edit_part_ids:
            continue
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
            gdel_id = _make_edit_id("gdel", obj_id, obj_del)
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
            obj_del += 1

            add_specs.append(EditSpec(
                edit_id=_make_edit_id("gadd", obj_id, obj_add),
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
            obj_add += 1

    # --- Group edits (from orthogonal 4-view enrichment) ---
    for obj_id, group_edits in catalog.object_group_edits.items():
        indices = catalog.by_object.get(obj_id, [])
        entries = [catalog.entries[i] for i in indices]
        if len(entries) < min_parts:
            continue

        obj_desc = catalog.object_descs.get(obj_id, "")
        all_pids = [e.part_id for e in entries]
        shard = entries[0].shard if entries else "00"

        # Count existing specs for this object to continue numbering
        obj_del = sum(1 for s in del_specs if s.obj_id == obj_id)
        obj_add = sum(1 for s in add_specs if s.obj_id == obj_id)
        obj_mod = sum(1 for s in mod_specs if s.obj_id == obj_id)

        pid_to_entry = {e.part_id: e for e in entries}
        for grp in group_edits:
            grp_pids = grp.get("part_ids", [])
            grp_best_view = grp.get("best_view", -1)
            grp_desc = grp.get("desc", "")
            grp_labels = [pid_to_entry[p].label if p in pid_to_entry
                          else f"part_{p}" for p in grp_pids]
            keep_pids = [p for p in all_pids if p not in grp_pids]
            if not keep_pids:
                continue

            grp_del_id = None  # Track this group's deletion ID
            for edit in grp.get("edits", []):
                etype = edit.get("type")
                if etype == "deletion":
                    grp_del_id = _make_edit_id("del", obj_id, obj_del)
                    del_specs.append(EditSpec(
                        edit_id=grp_del_id,
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
                    obj_del += 1
                elif etype == "addition" and grp_del_id is not None:
                    add_specs.append(EditSpec(
                        edit_id=_make_edit_id("add", obj_id, obj_add),
                        edit_type="addition",
                        obj_id=obj_id,
                        shard=shard,
                        object_desc=obj_desc,
                        before_desc=edit.get("after_desc", obj_desc),
                        add_part_ids=grp_pids,
                        add_labels=grp_labels,
                        base_part_ids=keep_pids,
                        source_del_id=grp_del_id,
                        edit_prompt=edit.get("prompt", ""),
                        after_desc=obj_desc,
                        best_view=grp_best_view,
                    ))
                    obj_add += 1
                elif etype == "modification":
                    mod_specs.append(EditSpec(
                        edit_id=_make_edit_id("mod", obj_id, obj_mod),
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
                    obj_mod += 1

    # --- Global edits (whole-object style/theme changes) ---
    for obj_id, indices in catalog.by_object.items():
        entries = [catalog.entries[i] for i in indices]
        if len(entries) < min_parts:
            continue

        obj_desc = catalog.object_descs.get(obj_id, "")
        shard = entries[0].shard

        # Use front orthogonal view as best_view for global edits
        ortho_views = catalog.object_ortho_views.get(obj_id, [])
        global_best_view = ortho_views[0] if ortho_views else 0

        obj_glb = 0
        for ge in catalog.object_global_edits.get(obj_id, [])[:max_global]:
            prompt = ge.get("prompt", "")
            if not prompt:
                continue
            glb_specs.append(EditSpec(
                edit_id=_make_edit_id("glb", obj_id, obj_glb),
                edit_type="global",
                obj_id=obj_id,
                shard=shard,
                object_desc=obj_desc,
                before_desc=obj_desc,
                edit_prompt=prompt,
                after_desc=ge.get("after_desc", ""),
                best_view=global_best_view,
            ))
            obj_glb += 1

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
        counters: DEPRECATED, ignored. Edit IDs are per-object unique via
                  obj_id hash — no global counter needed.

    Returns:
        List of EditSpec for this single object.
    """
    # Per-object counters (reset per object, namespaced by obj_id hash)
    _counters = {"del": 0, "add": 0, "mod": 0, "glb": 0}

    min_parts = cfg["phase1"].get("min_parts_per_object", 2)
    max_global = cfg["phase1"].get("max_global_edits_per_object", 3)

    obj_id = record["obj_id"]
    shard = record.get("shard", "00")
    obj_desc = record.get("object_desc", "")
    parts = record.get("parts", [])

    if len(parts) < min_parts:
        return []

    core_cats = set(cfg["phase1"].get("core_categories", []))

    # Default best_view for per-part edits: front orthogonal view
    ortho_views = record.get("orthogonal_views", [])
    obj_best_view = ortho_views[0] if ortho_views else 0

    all_pids = [p["part_id"] for p in parts]
    specs: list[EditSpec] = []

    has_group_edits = bool(record.get("group_edits"))

    # --- Group edits (if present, skip per-part for those parts) ---
    group_part_ids: set[int] = set()
    for grp in record.get("group_edits", []):
        grp_pids = grp.get("part_ids", [])
        group_part_ids.update(grp_pids)
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

        grp_del_id = None  # Track this group's deletion ID
        for edit in grp.get("edits", []):
            etype = edit.get("type")
            if etype == "deletion":
                grp_del_id = _make_edit_id("del", obj_id, _counters["del"])
                specs.append(EditSpec(
                    edit_id=grp_del_id,
                    edit_type="deletion",
                    obj_id=obj_id, shard=shard,
                    object_desc=obj_desc, before_desc=obj_desc,
                    remove_part_ids=grp_pids, remove_labels=grp_labels,
                    keep_part_ids=keep_pids,
                    edit_prompt=edit.get("prompt", ""),
                    after_desc=edit.get("after_desc", ""),
                    best_view=grp_best_view,
                ))
                _counters["del"] += 1
            elif etype == "addition" and grp_del_id is not None:
                specs.append(EditSpec(
                    edit_id=_make_edit_id("add", obj_id, _counters["add"]),
                    edit_type="addition",
                    obj_id=obj_id, shard=shard,
                    object_desc=obj_desc,
                    before_desc=edit.get("after_desc", obj_desc),
                    add_part_ids=grp_pids, add_labels=grp_labels,
                    base_part_ids=keep_pids,
                    source_del_id=grp_del_id,
                    edit_prompt=edit.get("prompt", ""),
                    after_desc=obj_desc,
                    best_view=grp_best_view,
                ))
                _counters["add"] += 1
            elif etype == "modification":
                specs.append(EditSpec(
                    edit_id=_make_edit_id("mod", obj_id, _counters["mod"]),
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
                _counters["mod"] += 1

    # --- Per-part edits (deletion / addition / modification) ---
    for part in parts:
        pid = part["part_id"]
        label = part.get("label", f"part_{pid}")
        is_core = part.get("core", False) or label in core_cats
        desc = part.get("desc", "")
        desc_without = part.get("desc_without", "")
        part_edits = part.get("edits", [])

        # Skip parts already handled by group edits
        if pid in group_part_ids:
            continue

        keep_pids = [p for p in all_pids if p != pid]
        if not keep_pids:
            continue

        # Non-core parts: deletion + addition
        if not is_core:
            del_edit = _find_edit_by_type(part_edits, "deletion")
            add_edit = _find_edit_by_type(part_edits, "addition")

            del_id = _make_edit_id("del", obj_id, _counters["del"])
            specs.append(EditSpec(
                edit_id=del_id,
                edit_type="deletion",
                obj_id=obj_id, shard=shard,
                object_desc=obj_desc, before_desc=obj_desc,
                remove_part_ids=[pid], remove_labels=[label],
                keep_part_ids=keep_pids,
                edit_prompt=del_edit.get("prompt", "") if del_edit else "",
                after_desc=del_edit.get("after_desc", "") if del_edit else "",
                best_view=obj_best_view,
            ))
            _counters["del"] += 1

            specs.append(EditSpec(
                edit_id=_make_edit_id("add", obj_id, _counters["add"]),
                edit_type="addition",
                obj_id=obj_id, shard=shard,
                object_desc=obj_desc,
                before_desc=desc_without or obj_desc,
                add_part_ids=[pid], add_labels=[label],
                base_part_ids=keep_pids,
                source_del_id=del_id,
                edit_prompt=add_edit.get("prompt", "") if add_edit else "",
                after_desc=obj_desc,
                best_view=obj_best_view,
            ))
            _counters["add"] += 1

        # All parts (including core): swap modifications
        mod_edits = [e for e in part_edits
                     if e.get("type") == "modification"
                     and e.get("mod_type") == "swap"
                     and e.get("prompt")]
        for mod_edit in mod_edits:
            specs.append(EditSpec(
                edit_id=_make_edit_id("mod", obj_id, _counters["mod"]),
                edit_type="modification",
                obj_id=obj_id, shard=shard,
                object_desc=obj_desc, before_desc=obj_desc,
                old_part_id=pid, old_label=label,
                keep_part_ids=keep_pids,
                edit_prompt=mod_edit.get("prompt", ""),
                after_desc=mod_edit.get("after_desc", ""),
                before_part_desc=mod_edit.get("before_part_desc", desc),
                after_part_desc=mod_edit.get("after_part_desc", ""),
                mod_type=mod_edit.get("mod_type", "swap"),
                best_view=obj_best_view,
            ))
            _counters["mod"] += 1

    # --- Global edits ---
    for ge in record.get("global_edits", [])[:max_global]:
        prompt = ge.get("prompt", "")
        if not prompt:
            continue
        specs.append(EditSpec(
            edit_id=_make_edit_id("glb", obj_id, _counters["glb"]),
            edit_type="global",
            obj_id=obj_id, shard=shard,
            object_desc=obj_desc, before_desc=obj_desc,
            edit_prompt=prompt,
            after_desc=ge.get("after_desc", ""),
            best_view=obj_best_view,
        ))
        _counters["glb"] += 1

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
