"""Phase 1: Edit planning — generate edit specifications from the Part Catalog.

Seven edit strategies (see partcraft/edit_types.py for taxonomy):
  1. Deletion     — remove a part from an object (GT mesh)
  2. Addition     — add a part to an object (reverse of deletion)
  3. Modification — swap a part's shape (TRELLIS S1+S2)
  4. Scale        — anisotropic part scaling (TRELLIS S1+S2)
  5. Material     — part-level material/texture change (TRELLIS S2 only)
  6. Global       — change whole-object style/theme (TRELLIS S2 only)
  7. Identity     — no-op, irrelevant instruction (anti-hallucination)

Order: deletion → addition → modification → scale → material → global → identity
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field, asdict
from pathlib import Path

from partcraft.edit_types import (
    DELETION, ADDITION, MODIFICATION, SCALE, MATERIAL, GLOBAL, IDENTITY,
    ID_PREFIX, SCALE_TEMPLATES, MATERIAL_TEMPLATES, IDENTITY_PROMPTS,
)
from partcraft.phase0_semantic.catalog import PartCatalog, CatalogEntry


@dataclass
class EditSpec:
    """Specification for a single edit operation."""
    edit_id: str
    edit_type: str                     # see partcraft.edit_types for constants
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


def _record_part_prompt_name(part: dict) -> str:
    """Natural-language phrase for prompts and labels in EditSpecs.

    Prefer ``desc`` (e.g. PartVerse caption from enrichment); else un-slug
    ``label`` (snake_case → spaces). Internal ``label`` is still used for
    ``core_categories`` matching elsewhere.
    """
    d = (part.get("desc") or "").strip()
    if d:
        return d
    lab = (part.get("label") or "").strip()
    if lab:
        return lab.replace("_", " ")
    pid = part.get("part_id", -1)
    return f"part {pid}"


def plan_edits(catalog: PartCatalog, cfg: dict) -> list[EditSpec]:
    """Generate all edit specs using the unified 7-type planner.

    Batch mode now reuses :func:`plan_edits_for_record` (same as streaming),
    so both execution modes produce a consistent taxonomy:
    deletion/addition/modification/scale/material/global/identity.
    """
    all_specs: list[EditSpec] = []
    for obj_id in sorted(catalog.by_object.keys()):
        indices = catalog.by_object.get(obj_id, [])
        entries = [catalog.entries[i] for i in indices]
        if not entries:
            continue

        record = {
            "obj_id": obj_id,
            "shard": entries[0].shard,
            "object_desc": catalog.object_descs.get(obj_id, ""),
            "orthogonal_views": catalog.object_ortho_views.get(obj_id, []),
            "group_edits": catalog.object_group_edits.get(obj_id, []),
            "global_edits": catalog.object_global_edits.get(obj_id, []),
            "parts": [
                {
                    "part_id": e.part_id,
                    "label": e.label,
                    "core": e.core,
                    "desc": e.desc,
                    "desc_without": e.desc_without,
                    "edits": e.edits,
                }
                for e in entries
            ],
        }
        all_specs.extend(plan_edits_for_record(record, cfg))
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
    _counters = {v: 0 for v in ID_PREFIX.values()}

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

    # Track which parts already have VLM-generated material/scale edits
    vlm_material_pids: set[int] = set()
    vlm_scale_pids: set[int] = set()

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
                    grp_labels.append(_record_part_prompt_name(p))
                    found = True
                    break
            if not found:
                grp_labels.append(f"part {pid}")

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
                    edit_type=MODIFICATION,
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
            elif etype == "material":
                specs.append(EditSpec(
                    edit_id=_make_edit_id("mat", obj_id, _counters["mat"]),
                    edit_type=MATERIAL,
                    obj_id=obj_id, shard=shard,
                    object_desc=obj_desc, before_desc=obj_desc,
                    old_part_id=grp_pids[0] if grp_pids else -1,
                    old_label=grp_labels[0] if grp_labels else "",
                    remove_part_ids=grp_pids, keep_part_ids=keep_pids,
                    edit_prompt=edit.get("prompt", ""),
                    after_desc=edit.get("after_desc", ""),
                    before_part_desc=edit.get("before_part_desc", grp_desc),
                    after_part_desc=edit.get("after_part_desc", ""),
                    mod_type="material",
                    best_view=grp_best_view,
                ))
                _counters["mat"] += 1
                vlm_material_pids.update(grp_pids)
            elif etype == "scale":
                scale_after_desc = (
                    edit.get("after_desc", "")
                    or edit.get("after_part_desc", "")
                    or obj_desc
                )
                specs.append(EditSpec(
                    edit_id=_make_edit_id("scl", obj_id, _counters["scl"]),
                    edit_type=SCALE,
                    obj_id=obj_id, shard=shard,
                    object_desc=obj_desc, before_desc=obj_desc,
                    old_part_id=grp_pids[0] if grp_pids else -1,
                    old_label=grp_labels[0] if grp_labels else "",
                    remove_part_ids=grp_pids, remove_labels=grp_labels,
                    keep_part_ids=keep_pids,
                    edit_prompt=edit.get("prompt", ""),
                    after_desc=scale_after_desc,
                    before_part_desc=edit.get("before_part_desc", grp_desc),
                    after_part_desc=edit.get("after_part_desc", ""),
                    mod_type="scale",
                    best_view=grp_best_view,
                ))
                _counters["scl"] += 1
                vlm_scale_pids.update(grp_pids)

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
            part_phrase = _record_part_prompt_name(part)
            specs.append(EditSpec(
                edit_id=del_id,
                edit_type="deletion",
                obj_id=obj_id, shard=shard,
                object_desc=obj_desc, before_desc=obj_desc,
                remove_part_ids=[pid], remove_labels=[part_phrase],
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
                add_part_ids=[pid], add_labels=[part_phrase],
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
                old_part_id=pid, old_label=_record_part_prompt_name(part),
                keep_part_ids=keep_pids,
                edit_prompt=mod_edit.get("prompt", ""),
                after_desc=mod_edit.get("after_desc", ""),
                before_part_desc=mod_edit.get("before_part_desc", desc),
                after_part_desc=mod_edit.get("after_part_desc", ""),
                mod_type=mod_edit.get("mod_type", "swap"),
                best_view=obj_best_view,
            ))
            _counters["mod"] += 1

        # Per-part VLM-generated material edits
        mat_edits = [e for e in part_edits
                     if e.get("type") == "material" and e.get("prompt")]
        for mat_edit in mat_edits:
            part_phrase = _record_part_prompt_name(part)
            specs.append(EditSpec(
                edit_id=_make_edit_id("mat", obj_id, _counters["mat"]),
                edit_type=MATERIAL,
                obj_id=obj_id, shard=shard,
                object_desc=obj_desc, before_desc=obj_desc,
                old_part_id=pid, old_label=part_phrase,
                keep_part_ids=keep_pids,
                edit_prompt=mat_edit.get("prompt", ""),
                after_desc=mat_edit.get("after_desc", ""),
                before_part_desc=mat_edit.get("before_part_desc", desc),
                after_part_desc=mat_edit.get("after_part_desc", ""),
                mod_type="material",
                best_view=obj_best_view,
            ))
            _counters["mat"] += 1
            vlm_material_pids.add(pid)

        # Per-part VLM-generated scale edits
        scl_edits = [e for e in part_edits
                     if e.get("type") == "scale" and e.get("prompt")]
        for scl_edit in scl_edits:
            part_phrase = _record_part_prompt_name(part)
            scale_after_desc = (
                scl_edit.get("after_desc", "")
                or scl_edit.get("after_part_desc", "")
                or obj_desc
            )
            specs.append(EditSpec(
                edit_id=_make_edit_id("scl", obj_id, _counters["scl"]),
                edit_type=SCALE,
                obj_id=obj_id, shard=shard,
                object_desc=obj_desc, before_desc=obj_desc,
                old_part_id=pid, old_label=part_phrase,
                keep_part_ids=keep_pids,
                edit_prompt=scl_edit.get("prompt", ""),
                after_desc=scale_after_desc,
                before_part_desc=scl_edit.get("before_part_desc", desc),
                after_part_desc=scl_edit.get("after_part_desc", ""),
                mod_type="scale",
                best_view=obj_best_view,
            ))
            _counters["scl"] += 1
            vlm_scale_pids.add(pid)

    # --- Global edits ---
    for ge in record.get("global_edits", [])[:max_global]:
        prompt = ge.get("prompt", "")
        if not prompt:
            continue
        specs.append(EditSpec(
            edit_id=_make_edit_id("glb", obj_id, _counters["glb"]),
            edit_type=GLOBAL,
            obj_id=obj_id, shard=shard,
            object_desc=obj_desc, before_desc=obj_desc,
            edit_prompt=prompt,
            after_desc=ge.get("after_desc", ""),
            best_view=obj_best_view,
        ))
        _counters["glb"] += 1

    # --- Scale edits (template fallback for parts without VLM scale) ---
    max_scale = cfg["phase1"].get("max_scale_edits_per_part", 1)
    rng = random.Random(hash(obj_id))
    for part in parts:
        pid = part["part_id"]
        if pid in vlm_scale_pids:
            continue  # VLM already generated scale edits for this part
        label = part.get("label", f"part_{pid}")
        is_core = part.get("core", False) or label in core_cats

        keep_pids = [p for p in all_pids if p != pid]
        if not keep_pids:
            continue

        part_phrase = _record_part_prompt_name(part)
        templates = rng.sample(SCALE_TEMPLATES,
                               min(max_scale, len(SCALE_TEMPLATES)))
        for tmpl_prompt, tmpl_before, tmpl_after in templates:
            tmpl_after_desc = tmpl_after.format(part=part_phrase)
            specs.append(EditSpec(
                edit_id=_make_edit_id("scl", obj_id, _counters["scl"]),
                edit_type=SCALE,
                obj_id=obj_id, shard=shard,
                object_desc=obj_desc, before_desc=obj_desc,
                old_part_id=pid, old_label=part_phrase,
                keep_part_ids=keep_pids,
                edit_prompt=tmpl_prompt.format(part=part_phrase),
                after_desc=tmpl_after_desc,
                before_part_desc=tmpl_before.format(part=part_phrase),
                after_part_desc=tmpl_after_desc,
                mod_type="scale",
                best_view=obj_best_view,
            ))
            _counters["scl"] += 1

    # --- Material edits (template fallback for parts without VLM material) ---
    max_material = cfg["phase1"].get("max_material_edits_per_part", 1)
    for part in parts:
        pid = part["part_id"]
        if pid in vlm_material_pids:
            continue  # VLM already generated material edits for this part

        keep_pids = [p for p in all_pids if p != pid]
        if not keep_pids:
            continue

        part_phrase = _record_part_prompt_name(part)
        templates = rng.sample(MATERIAL_TEMPLATES,
                               min(max_material, len(MATERIAL_TEMPLATES)))
        for tmpl_prompt, tmpl_after in templates:
            specs.append(EditSpec(
                edit_id=_make_edit_id("mat", obj_id, _counters["mat"]),
                edit_type=MATERIAL,
                obj_id=obj_id, shard=shard,
                object_desc=obj_desc, before_desc=obj_desc,
                old_part_id=pid, old_label=part_phrase,
                keep_part_ids=keep_pids,
                edit_prompt=tmpl_prompt.format(part=part_phrase),
                after_desc=obj_desc,
                before_part_desc=part_phrase,
                after_part_desc=tmpl_after.format(part=part_phrase),
                mod_type="material",
                best_view=obj_best_view,
            ))
            _counters["mat"] += 1

    # --- Identity edits (no-op, anti-hallucination) ---
    max_identity = cfg["phase1"].get("max_identity_edits_per_object", 1)
    id_prompts = rng.sample(IDENTITY_PROMPTS,
                            min(max_identity, len(IDENTITY_PROMPTS)))
    for prompt in id_prompts:
        specs.append(EditSpec(
            edit_id=_make_edit_id("idt", obj_id, _counters["idt"]),
            edit_type=IDENTITY,
            obj_id=obj_id, shard=shard,
            object_desc=obj_desc, before_desc=obj_desc,
            edit_prompt=prompt,
            after_desc=obj_desc,
            best_view=obj_best_view,
        ))
        _counters["idt"] += 1

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

    from collections import Counter
    counts = Counter(s.edit_type for s in all_specs)
    parts = [f"{t}: {counts.get(t, 0)}"
             for t in [DELETION, ADDITION, MODIFICATION, SCALE, MATERIAL,
                       GLOBAL, IDENTITY]
             if counts.get(t, 0) > 0]
    print(f"  {', '.join(parts)}")
    print(f"  Total: {len(all_specs)} edit specs")

    with open(output_path, "w") as f:
        for spec in all_specs:
            f.write(json.dumps(spec.to_dict(), ensure_ascii=False) + "\n")

    print(f"Phase 1 complete -> {output_path}")
    return all_specs
