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
    slat_root: Path
    view_indices: list[int]
    # Preferred source for ``before/views/``: the per-object NPZ that the
    # pipeline itself loads via ``load_views_from_npz`` for Gate-A judging.
    # Layout: ``<image_npz_root>/<shard>/<obj_id>.npz`` with PNG-byte arrays
    # keyed by ``"{idx:03d}.png"``.  This guarantees the v1 ``before`` views
    # match the pipeline's judged-against pixels (background already
    # composited).  When None we fall back to per-frame PNGs under
    # ``img_enc_root/<obj_id>/{idx:03d}.png`` (RGBA, no bg compositing).
    image_npz_root: Path | None = None
    img_enc_root: Path | None = None
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


def _to_np(t):
    """Convert a torch tensor (possibly with grad) to numpy; passthrough otherwise."""
    if hasattr(t, "detach"):
        return t.detach().cpu().numpy()
    if hasattr(t, "cpu"):
        return t.cpu().numpy()
    return t


def _pack_pt_to_npz(pt_path: Path, npz_path: Path) -> None:
    import numpy as np
    import torch
    obj = torch.load(pt_path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict):
        arrs = {k: _to_np(v) for k, v in obj.items()}
    elif hasattr(obj, "cpu") or hasattr(obj, "detach"):
        arrs = {"data": _to_np(obj)}
    else:
        arrs = {"data": obj}
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(npz_path, **arrs)


def _decode_views_from_npz(
    npz_path: Path, view_indices: list[int], dsts: list[Path], *, force: bool,
) -> tuple[bool, str]:
    """Extract canonical-view PNGs from a per-object image NPZ.

    The NPZ stores each rendered frame as 1-D ``uint8`` PNG bytes under key
    ``"{idx:03d}.png"``.  Those PNGs are RGBA with a transparent background;
    naively writing the raw bytes leaves alpha intact and viewers render the
    transparent regions as a checkerboard, which is not what the pipeline
    actually judges.

    ``partcraft.render.overview.load_views_from_npz`` (the function every
    downstream judge uses) decodes each PNG with ``cv2.IMREAD_UNCHANGED``
    and, when alpha is present, composites onto a solid-white background
    (``rgb*a + 255*(1-a)``) before returning BGR.  We replicate that exact
    step here and re-encode as RGB PNG so the on-disk ``before/views/``
    pixels match the model-facing pixels byte-for-byte after the same
    composite that Gate A applied.
    """
    import cv2
    import numpy as np
    if not npz_path.is_file():
        return False, f"missing image NPZ: {npz_path}"
    try:
        z = np.load(npz_path)
    except Exception as e:
        return False, f"failed to open NPZ {npz_path}: {e}"
    try:
        for idx, dst in zip(view_indices, dsts):
            key = f"{idx:03d}.png"
            if key not in z.files:
                return False, f"NPZ {npz_path.name} missing view {key}"
            if dst.is_file() and not force:
                continue
            buf = np.frombuffer(bytes(z[key].tobytes()), dtype=np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
            if img is None:
                return False, f"NPZ {npz_path.name} view {key} failed to decode"
            if img.ndim == 3 and img.shape[2] == 4:
                a = img[:, :, 3:4].astype(np.float32) / 255.0
                rgb = img[:, :, :3].astype(np.float32)
                bg = np.full_like(rgb, 255)
                img = (rgb * a + bg * (1 - a)).astype(np.uint8)
            elif img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            dst.parent.mkdir(parents=True, exist_ok=True)
            ok, enc = cv2.imencode(".png", img)
            if not ok:
                return False, f"failed to encode {dst.name}"
            dst.write_bytes(bytes(enc))
    finally:
        try:
            z.close()
        except Exception:
            pass
    return True, ""


def _materialize_before(
    *, shard: str, obj_id: str, layout: V1Layout, cfg: PromoterConfig,
    summary: PromotionSummary,
) -> bool:
    dsts = layout.before_view_paths(shard, obj_id)
    if cfg.image_npz_root is not None:
        npz = cfg.image_npz_root / shard / f"{obj_id}.npz"
        ok, err = _decode_views_from_npz(npz, cfg.view_indices, dsts, force=cfg.force)
        if not ok:
            summary.notes.append(err)
            return False
    else:
        if cfg.img_enc_root is None:
            summary.notes.append("neither image_npz_root nor img_enc_root configured")
            return False
        img_enc_dir = cfg.img_enc_root / obj_id
        if not img_enc_dir.is_dir():
            summary.notes.append(f"missing img_Enc dir: {img_enc_dir}")
            return False
        for k, idx in enumerate(cfg.view_indices):
            src = img_enc_dir / f"{idx:03d}.png"
            if not src.is_file():
                summary.notes.append(f"missing view {src}")
                return False
            res = link_one(src, dsts[k], mode=cfg.link_mode, force=cfg.force)
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
        feats_arr = feats.detach().cpu().numpy() if hasattr(feats, "detach") else (
                       feats.cpu().numpy() if hasattr(feats, "cpu") else feats)
        coords_arr = coords.detach().cpu().numpy() if hasattr(coords, "detach") else (
                        coords.cpu().numpy() if hasattr(coords, "cpu") else coords)
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
    from ._parsed import extract_edits_and_parts
    _, parts_by_id = extract_edits_and_parts(parsed)
    meta = {
        "obj_id": rec.obj_id, "shard": rec.shard,
        "source_dataset": "partverse",
        "caption": raw_caption,
        "part_list": [
            {"part_id": pid, "name": name}
            for pid, name in sorted(parts_by_id.items())
        ],
        "promoted_at": _now_z(),
        "promoter_version": cfg.promoter_version,
    }
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(meta, indent=2))


def _resolve_suffix(layout: V1Layout, rec: PromotionRecord, *,
                    force: bool = False) -> str | None:
    """Pick the destination suffix for ``rec`` (``""`` for fresh, ``__rN`` for
    a different-run collision).  Returns ``None`` iff the same run_tag has
    already been promoted to ``edit_id`` (and ``force=False``)."""
    base = layout.edit_dir(rec.shard, rec.obj_id, rec.edit_id, suffix="")
    if not base.exists():
        return ""
    qc_p = layout.qc_json(rec.shard, rec.obj_id, rec.edit_id, suffix="")
    if qc_p.is_file():
        try:
            qc = json.loads(qc_p.read_text())
            if qc.get("source", {}).get("run_tag") == rec.source_run_tag:
                # Same source already here. Either skip (default) or overwrite
                # in place when the caller asked us to force.
                return "" if force else None
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
        suffix = _resolve_suffix(layout, rec, force=cfg.force)
        if suffix is None:
            summary.skipped_existing += 1
            continue
        _materialize_edit(rec, suffix, layout=layout, cfg=cfg,
                          summary=summary, pending=pending)
        summary.promoted += 1
    return summary
