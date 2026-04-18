"""Adapter: pipeline_v2 ``edit_status.json`` -> ``PromotionRecord`` stream."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

from .canonical_record import PromotionRecord, PassResult
from .layout import parse_edit_id


def _read_json(p: Path) -> dict[str, Any]:
    return json.loads(p.read_text())


def _gate_e_to_pass(gate_e: dict[str, Any] | None) -> PassResult | None:
    if not gate_e:
        return None
    vlm = gate_e.get("vlm") or {}
    if not vlm:
        return None
    extra: dict[str, Any] = {}
    if isinstance(vlm.get("metrics"), dict):
        extra["metrics"] = vlm["metrics"]
    return PassResult(
        passed=bool(vlm.get("pass")),
        score=vlm.get("score"),
        producer="v3.gate_quality (backfilled into v2)",
        reason=vlm.get("reason", ""),
        ts=vlm.get("ts", ""),
        extra=extra,
    )


def _gate_a_to_pass(gate_a: dict[str, Any] | None) -> PassResult | None:
    if not gate_a:
        return None
    vlm = gate_a.get("vlm") or {}
    rule = gate_a.get("rule") or {}
    rule_pass = bool(rule.get("pass", False))
    vlm_pass = bool(vlm.get("pass", False)) if vlm else rule_pass
    return PassResult(
        passed=rule_pass and vlm_pass,
        score=vlm.get("score") if vlm else None,
        producer="v2.s1_phase1 / sq1_qc_A",
        reason=vlm.get("reason", "") if vlm else "",
        ts=vlm.get("ts", "") if vlm else "",
    )


def _addition_source_del_map(obj_dir: Path) -> dict[str, str]:
    """Return ``{add_edit_id: source_del_id}`` by scanning ``edits_3d/add_*/meta.json``.

    Pipeline-v3 (``mesh_deletion._write_addition_meta``) records this link when
    an addition edit is created as the inverse of a deletion.  v2 reuses the
    same on-disk layout, so we read it the same way.
    """
    out: dict[str, str] = {}
    edits_root = obj_dir / "edits_3d"
    if not edits_root.is_dir():
        return out
    for sub in edits_root.iterdir():
        if not sub.is_dir() or not sub.name.startswith("add_"):
            continue
        meta = sub / "meta.json"
        if not meta.is_file():
            continue
        try:
            d = json.loads(meta.read_text())
        except Exception:
            continue
        src = d.get("source_del_id")
        add_id = d.get("edit_id") or sub.name
        if src:
            out[add_id] = src
    return out


def _inherited_pass(src: PassResult, *, source_del_id: str) -> PassResult:
    """Synthetic ``gate_text_align`` verdict for an addition, inherited from
    its source deletion.  Additions are the prompt-flipped inverse of a
    deletion (see ``pipeline_v3.addition_utils.invert_delete_prompt``); v3
    handles the same idea implicitly via Gate E (`_inherit_gate_e`).
    """
    base_reason = (src.reason or "").strip()
    suffix = f"inherited from {source_del_id}"
    new_reason = f"{base_reason} | {suffix}" if base_reason else suffix
    return PassResult(
        passed=src.passed,
        score=src.score,
        producer=f"{src.producer} (inherited:add->del)",
        reason=new_reason,
        ts=src.ts,
        extra={**(src.extra or {}), "inherited_from": source_del_id},
    )


def _spec_subset_for_edit(
    edit_id: str, edit_type: str, parsed_edits: list[dict[str, Any]],
    parts_by_id: dict[int, str],
) -> dict[str, Any]:
    _, _, idx = parse_edit_id(edit_id)
    same_type = [e for e in parsed_edits if e.get("edit_type") == edit_type]
    src = same_type[idx] if idx < len(same_type) else {}
    pids = list(src.get("selected_part_ids") or [])
    return {
        "edit_id": edit_id,
        "edit_type": edit_type,
        "prompt": src.get("prompt", ""),
        "selected_part_ids": pids,
        "part_labels": [parts_by_id.get(p, "") for p in pids],
        "target_part_desc": src.get("target_part_desc", ""),
        # v2 uses ``new_parts_desc``; v3 uses ``after_desc``; fall back to the
        # target part description (or the edit prompt) when neither is set.
        "new_parts_desc": (
            src.get("new_parts_desc")
            or src.get("after_desc")
            or src.get("after_desc_full")
            or src.get("target_part_desc", "")
        ),
        "edit_params": dict(src.get("edit_params") or {}),
    }


def iter_records_from_v2_obj(
    obj_dir: Path, *, run_tag: str,
) -> Iterator[PromotionRecord]:
    es = _read_json(obj_dir / "edit_status.json")
    obj_id = es.get("obj_id") or obj_dir.name
    shard = es.get("shard") or obj_dir.parent.name

    parsed_p = obj_dir / "phase1" / "parsed.json"
    parsed = _read_json(parsed_p) if parsed_p.is_file() else {}
    from ._parsed import extract_edits_and_parts
    parsed_edits, parts_by_id = extract_edits_and_parts(parsed)

    edits = es.get("edits") or {}

    # Pre-pass: collect del Gate A so we can inherit it for additions.
    del_gta: dict[str, PassResult] = {}
    for edit_id, entry in edits.items():
        if entry.get("edit_type") != "deletion":
            continue
        p = _gate_a_to_pass((entry.get("gates") or {}).get("A"))
        if p is not None:
            del_gta[edit_id] = p

    # Map add_id -> source_del_id from on-disk meta.json.
    add_src_map = _addition_source_del_map(obj_dir)

    for edit_id, entry in edits.items():
        edit_type = entry.get("edit_type", "")
        gates = entry.get("gates") or {}
        passes: dict[str, PassResult] = {}
        gta = _gate_a_to_pass(gates.get("A"))
        if gta is None and edit_type == "addition":
            src_del = add_src_map.get(edit_id)
            if src_del and src_del in del_gta:
                gta = _inherited_pass(del_gta[src_del], source_del_id=src_del)
        if gta is not None:
            passes["gate_text_align"] = gta
        gte = _gate_e_to_pass(gates.get("E"))
        if gte is not None:
            passes["gate_quality"] = gte

        edits_3d_dir = obj_dir / "edits_3d" / edit_id
        after_glb_p = edits_3d_dir / "after_new.glb"
        after_npz_p = edits_3d_dir / "after.npz"
        preview_pngs = [edits_3d_dir / f"preview_{k}.png" for k in range(5)]

        spec = _spec_subset_for_edit(edit_id, edit_type, parsed_edits, parts_by_id)

        yield PromotionRecord(
            obj_id=obj_id, shard=shard,
            edit_id=edit_id, edit_type=edit_type,
            source_pipeline="v2", source_run_tag=run_tag,
            source_run_dir=obj_dir,
            spec=spec, passes=passes,
            after_glb=after_glb_p if after_glb_p.is_file() else None,
            after_npz=after_npz_p if after_npz_p.is_file() else None,
            preview_pngs=preview_pngs,
        )


def iter_records_from_v2_run(
    run_root: Path, *, run_tag: str | None = None,
) -> Iterator[PromotionRecord]:
    tag = run_tag or run_root.name
    objects_root = run_root / "objects"
    if not objects_root.is_dir():
        return
    for shard_dir in sorted(objects_root.iterdir()):
        if not shard_dir.is_dir():
            continue
        for obj_dir in sorted(shard_dir.iterdir()):
            if not (obj_dir / "edit_status.json").is_file():
                continue
            yield from iter_records_from_v2_obj(obj_dir, run_tag=tag)
