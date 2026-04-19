"""Adapter: pipeline_v3 ``edit_status.json`` -> ``PromotionRecord`` stream."""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

from .canonical_record import PromotionRecord, PassResult
from .source_v2 import _read_json, _gate_a_to_pass, _spec_subset_for_edit


def _gate_e_v3_to_pass(gate_e: dict | None) -> PassResult | None:
    if not gate_e:
        return None
    vlm = gate_e.get("vlm") or {}
    if not vlm:
        return None
    extra: dict = {}
    if isinstance(vlm.get("metrics"), dict):
        extra["metrics"] = vlm["metrics"]
    return PassResult(
        passed=bool(vlm.get("pass")),
        score=vlm.get("score"),
        producer="v3.gate_quality (native)",
        reason=vlm.get("reason", ""),
        ts=vlm.get("ts", ""),
        extra=extra,
    )


def iter_records_from_v3_obj(
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
    for edit_id, entry in edits.items():
        edit_type = entry.get("edit_type", "")
        gates = entry.get("gates") or {}
        passes: dict[str, PassResult] = {}
        gta = _gate_a_to_pass(gates.get("A"))
        if gta is not None:
            passes["gate_text_align"] = gta
        gte = _gate_e_v3_to_pass(gates.get("E"))
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
            source_pipeline="v3", source_run_tag=run_tag,
            source_run_dir=obj_dir,
            spec=spec, passes=passes,
            after_glb=after_glb_p if after_glb_p.is_file() else None,
            after_npz=after_npz_p if after_npz_p.is_file() else None,
            preview_pngs=preview_pngs,
        )


def iter_records_from_v3_run(
    run_root: Path, *, run_tag: str | None = None,
) -> Iterator[PromotionRecord]:
    """Walk ``<run_root>/[<mode>/]objects/<NN>/<obj_id>/`` and yield records."""
    tag = run_tag or run_root.name
    candidates: list[Path] = []
    if (run_root / "objects").is_dir():
        candidates = [run_root]
    else:
        for child in sorted(run_root.iterdir()):
            if child.is_dir() and (child / "objects").is_dir():
                candidates.append(child)
    for mode_dir in candidates:
        objects_root = mode_dir / "objects"
        for shard_dir in sorted(objects_root.iterdir()):
            if not shard_dir.is_dir():
                continue
            for obj_dir in sorted(shard_dir.iterdir()):
                if not (obj_dir / "edit_status.json").is_file():
                    continue
                yield from iter_records_from_v3_obj(obj_dir, run_tag=tag)
