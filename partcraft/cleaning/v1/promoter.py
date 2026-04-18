"""Apply promotion rule + materialize a v1 directory tree."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .canonical_record import PromotionRecord, evaluate_rule
from .layout import V1Layout
from .linker import LinkMode, link_one
from .pending import DelLatentPending, PendingEntry


@dataclass
class PromoterConfig:
    rule: dict
    link_mode: LinkMode
    img_enc_root: Path
    slat_root: Path
    view_indices: list[int]
    force: bool = False
    promoter_version: str = "1.0.0"


@dataclass
class PromotionSummary:
    promoted: int = 0
    skipped_existing: int = 0
    deferred: int = 0
    failed: int = 0
    filtered: int = 0
    fallback_count: int = 0
    notes: list[str] = field(default_factory=list)


def _now_z() -> str:
    return datetime.now(tz=timezone.utc).isoformat(timespec="seconds")


def _pack_pt_to_npz(pt_path: Path, npz_path: Path) -> None:
    import numpy as np
    import torch
    obj = torch.load(pt_path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict):
        arrs = {k: (v.cpu().numpy() if hasattr(v, "cpu") else v) for k, v in obj.items()}
    elif hasattr(obj, "cpu"):
        arrs = {"data": obj.cpu().numpy()}
    else:
        arrs = {"data": obj}
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(npz_path, **arrs)


def _materialize_before(
    *, shard: str, obj_id: str, layout: V1Layout, cfg: PromoterConfig,
    summary: PromotionSummary,
) -> bool:
    img_enc_dir = cfg.img_enc_root / obj_id
    if not img_enc_dir.is_dir():
        summary.notes.append(f"missing img_Enc dir: {img_enc_dir}")
        return False
    for k, idx in enumerate(cfg.view_indices):
        src = img_enc_dir / f"{idx:03d}.png"
        dst = layout.before_view_paths(shard, obj_id)[k]
        if not src.is_file():
            summary.notes.append(f"missing view {src}")
            return False
        res = link_one(src, dst, mode=cfg.link_mode, force=cfg.force)
        if res.fell_back:
            summary.fallback_count += 1
    feats_pt = cfg.slat_root / shard / f"{obj_id}_feats.pt"
    coords_pt = cfg.slat_root / shard / f"{obj_id}_coords.pt"
    ss_pt = cfg.slat_root / shard / f"{obj_id}_ss.pt"
    if not feats_pt.is_file() or not coords_pt.is_file():
        summary.notes.append(f"missing SLAT .pt for {obj_id}")
        return False
    slat_dst = layout.before_slat_npz(shard, obj_id)
    if cfg.force or not slat_dst.is_file():
        import numpy as np
        import torch
        feats = torch.load(feats_pt, map_location="cpu", weights_only=False)
        coords = torch.load(coords_pt, map_location="cpu", weights_only=False)
        feats_arr = feats.cpu().numpy() if hasattr(feats, "cpu") else feats
        coords_arr = coords.cpu().numpy() if hasattr(coords, "cpu") else coords
        slat_dst.parent.mkdir(parents=True, exist_ok=True)
        np.savez(slat_dst, feats=feats_arr, coords=coords_arr)
    ss_dst = layout.before_ss_npz(shard, obj_id)
    if cfg.force or not ss_dst.is_file():
        if ss_pt.is_file():
            _pack_pt_to_npz(ss_pt, ss_dst)
        else:
            ss_dst.parent.mkdir(parents=True, exist_ok=True)
            ss_dst.with_suffix(".missing.json").write_text(json.dumps({
                "reason": "ss_pt_not_found", "expected": str(ss_pt), "ts": _now_z(),
            }))
    return True


def _write_meta_json(layout: V1Layout, rec: PromotionRecord, cfg: PromoterConfig) -> None:
    p = layout.meta_json(rec.shard, rec.obj_id)
    if p.is_file() and not cfg.force:
        return
    parsed_p = rec.source_run_dir / "phase1" / "parsed.json"
    parsed = json.loads(parsed_p.read_text()) if parsed_p.is_file() else {}
    raw_caption = ""
    raw_p = rec.source_run_dir / "phase1" / "raw.txt"
    if raw_p.is_file():
        raw_caption = raw_p.read_text(errors="replace").strip()
    meta = {
        "obj_id": rec.obj_id, "shard": rec.shard,
        "source_dataset": "partverse",
        "caption": raw_caption,
        "part_list": [
            {"part_id": int(p["id"]), "name": p.get("name", "")}
            for p in (parsed.get("parts") or [])
        ],
        "promoted_at": _now_z(),
        "promoter_version": cfg.promoter_version,
    }
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(meta, indent=2))


def _resolve_suffix(layout: V1Layout, rec: PromotionRecord) -> str | None:
    base = layout.edit_dir(rec.shard, rec.obj_id, rec.edit_id, suffix="")
    if not base.exists():
        return ""
    qc_p = layout.qc_json(rec.shard, rec.obj_id, rec.edit_id, suffix="")
    if qc_p.is_file():
        try:
            qc = json.loads(qc_p.read_text())
            if qc.get("source", {}).get("run_tag") == rec.source_run_tag:
                return None
        except Exception:
            pass
    n = 2
    while layout.edit_dir(rec.shard, rec.obj_id, rec.edit_id, suffix=f"__r{n}").exists():
        n += 1
    return f"__r{n}"


def _materialize_edit(
    rec: PromotionRecord, suffix: str, *, layout: V1Layout, cfg: PromoterConfig,
    summary: PromotionSummary, pending: DelLatentPending,
) -> None:
    edit_dir = layout.edit_dir(rec.shard, rec.obj_id, rec.edit_id, suffix=suffix)
    edit_dir.mkdir(parents=True, exist_ok=True)
    layout.spec_json(rec.shard, rec.obj_id, rec.edit_id, suffix=suffix).write_text(
        json.dumps(rec.spec, indent=2))
    layout.qc_json(rec.shard, rec.obj_id, rec.edit_id, suffix=suffix).write_text(
        json.dumps(rec.to_qc_json(promoted_at=_now_z()), indent=2))
    for k, src in enumerate(rec.preview_pngs):
        dst = layout.after_view_paths(rec.shard, rec.obj_id, rec.edit_id, suffix=suffix)[k]
        if src.is_file():
            res = link_one(src, dst, mode=cfg.link_mode, force=cfg.force)
            if res.fell_back:
                summary.fallback_count += 1
    if rec.is_deletion():
        marker = layout.after_pending_marker(rec.shard, rec.obj_id, rec.edit_id, suffix=suffix)
        marker.write_text(json.dumps({
            "edit_id": rec.edit_id, "suffix": suffix,
            "after_glb": str(rec.after_glb) if rec.after_glb else None,
            "ts": _now_z(),
        }))
        pending.append(PendingEntry(rec.shard, rec.obj_id, rec.edit_id, suffix))
    else:
        if rec.after_npz and rec.after_npz.is_file():
            dst = layout.after_npz_path(rec.shard, rec.obj_id, rec.edit_id, suffix=suffix)
            res = link_one(rec.after_npz, dst, mode=cfg.link_mode, force=cfg.force)
            if res.fell_back:
                summary.fallback_count += 1
        else:
            summary.notes.append(f"missing after.npz for {rec.edit_id} ({rec.source_run_tag})")


def promote_records(
    recs: Iterable[PromotionRecord], *,
    layout: V1Layout, cfg: PromoterConfig, pending: DelLatentPending,
) -> PromotionSummary:
    summary = PromotionSummary()
    seen_objs: set[tuple[str, str]] = set()
    for rec in recs:
        ok, reason = evaluate_rule(rec.passes, cfg.rule, edit_type=rec.edit_type)
        if not ok:
            if reason.startswith("missing"):
                summary.deferred += 1
            elif reason.startswith("disallowed_type"):
                summary.filtered += 1
            else:
                summary.failed += 1
            continue
        if (rec.shard, rec.obj_id) not in seen_objs:
            need_before = (cfg.force or
                           not layout.before_slat_npz(rec.shard, rec.obj_id).is_file())
            if need_before:
                if not _materialize_before(shard=rec.shard, obj_id=rec.obj_id,
                                            layout=layout, cfg=cfg, summary=summary):
                    summary.failed += 1
                    continue
            _write_meta_json(layout, rec, cfg)
            seen_objs.add((rec.shard, rec.obj_id))
        suffix = _resolve_suffix(layout, rec)
        if suffix is None:
            summary.skipped_existing += 1
            continue
        _materialize_edit(rec, suffix, layout=layout, cfg=cfg,
                          summary=summary, pending=pending)
        summary.promoted += 1
    return summary
