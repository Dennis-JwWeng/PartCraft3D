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
        "new_parts_desc": (src.get("new_parts_desc") or src.get("target_part_desc", "")),
        "edit_params": dict(src.get("edit_params") or {}),
    }


def iter_records_from_v2_obj(
    obj_dir: Path, *, run_tag: str,
) -> Iterator[PromotionRecord]:
    es = _read_json(obj_dir / "edit_status.json")
    obj_id = es.get("obj_id") or obj_dir.name
    shard = es.get("shard") or obj_dir.parent.name

    parsed_p = obj_dir / "phase1" / "parsed.json"
    parsed = _read_json(parsed_p) if parsed_p.is_file() else {"edits": [], "parts": []}
    parsed_edits = parsed.get("edits") or []
    parts_by_id = {int(p["id"]): p.get("name", "") for p in (parsed.get("parts") or [])}

    edits = es.get("edits") or {}
    for edit_id, entry in edits.items():
        edit_type = entry.get("edit_type", "")
        gates = entry.get("gates") or {}
        passes: dict[str, PassResult] = {}
        gta = _gate_a_to_pass(gates.get("A"))
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
