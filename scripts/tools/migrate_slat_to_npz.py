#!/usr/bin/env python3
"""Migrate legacy pipeline outputs to the unified NPZ format (SLAT + SS + DINOv2).

Converts all edit pair artifacts produced by the old pipeline into the new
``before.npz`` / ``after.npz`` format with keys ``slat_feats``, ``slat_coords``,
``ss`` (sparse-structure VAE latent), and ``dino_voxel_mean`` (multi-view averaged
DINOv2 features projected onto voxels, ``[N, 1024]`` float16).

Five processing phases (run in order):

  Phase 1 — **Simple conversion** (modification / scale / material / global)
      Pairs that already have ``*_slat/`` directories: load feats.pt + coords.pt,
      compute SS via the sparse-structure encoder, write ``*.npz``.

  Phase 2 — **Deletion backfill**
      Pairs that only have PLY output (no SLAT): load the pre-encoded source SLAT
      from ``slat_dir``, build the part mask (mesh voxelization), filter the SLAT
      to obtain before/after, compute SS, write ``*.npz``.

  Phase 3 — **Addition backfill**
      Addition is the reverse of its source deletion: swap the source deletion
      pair's ``before.npz`` / ``after.npz``.

  Phase 4 — **Identity backfill**
      Identity copies ``before.npz`` as both before *and* after for the same
      object.  Finds the first migrated pair of that object.

  Phase 5 — **DINOv2 voxel feature extraction** (all edit types)
      For each edit pair that has ``after.ply``: render via Blender Cycles GPU
      (40 views), voxelize, extract DINOv2 features, and write
      ``dino_voxel_mean`` into the existing ``after.npz``.
      For ``before``: load pre-saved ``{obj_id}_dino_voxel_mean.pt`` from
      ``slat_dir`` (produced by ``encode_into_SLAT`` with ``save_dino_voxel_mean=True``).

Existing ``*.npz`` files are never overwritten (idempotent).

Usage
-----
Full migration (needs GPU, loads SS encoder + dataset):

    python scripts/tools/migrate_slat_to_npz.py \\
        --config  configs/partverse_H200_shard00.yaml \\
        --mesh-pairs /mnt/zsn/data/partverse/outputs/partverse/mesh_pairs_shard00 \\
        --specs-jsonl /mnt/zsn/data/partverse/outputs/partverse/cache/phase1/edit_specs_shard00.jsonl

Dry run (no GPU, no writes):

    python scripts/tools/migrate_slat_to_npz.py \\
        --config  configs/partverse_H200_shard00.yaml \\
        --mesh-pairs /path/to/mesh_pairs \\
        --specs-jsonl /path/to/edit_specs.jsonl \\
        --dry-run

Phase 1 only (no dataset or specs needed — just has *_slat/ dirs):

    python scripts/tools/migrate_slat_to_npz.py \\
        --ckpt-root /path/to/checkpoints \\
        --mesh-pairs /path/to/mesh_pairs \\
        --phase 1
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "third_party"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("migrate")


# ───────────────────────────── data structures ──────────────────────────────

@dataclass
class SpecInfo:
    """Minimal fields extracted from EditSpec JSONL for migration."""
    edit_id: str
    edit_type: str
    obj_id: str
    shard: str
    remove_part_ids: list[int] = field(default_factory=list)
    source_del_id: str = ""


# ──────────────────────────── SS encoder loader ─────────────────────────────

def _load_ss_encoder(ckpt_root: Path, device: str = "cuda"):
    """Load only the sparse-structure VAE encoder (lightweight)."""
    from trellis.pipelines import TrellisTextTo3DPipeline
    import trellis.models as trellis_models

    text_ckpt = str(ckpt_root / "TRELLIS-text-xlarge")
    log.info("Loading TRELLIS text pipeline from %s …", text_ckpt)
    pipeline = TrellisTextTo3DPipeline.from_pretrained(text_ckpt)

    if "sparse_structure_encoder" not in pipeline.models:
        ss_enc_path = str(
            ckpt_root / "TRELLIS-text-xlarge" / "ckpts" / "ss_enc_conv3d_16l8_fp16"
        )
        ss_encoder = trellis_models.from_pretrained(ss_enc_path)
        pipeline.models["sparse_structure_encoder"] = ss_encoder

    pipeline.cuda() if device == "cuda" else pipeline.cpu()
    encoder = pipeline.models["sparse_structure_encoder"]
    log.info("SS encoder ready on %s", device)
    return encoder


@torch.no_grad()
def encode_ss(encoder, coords: torch.Tensor, device: str = "cuda") -> torch.Tensor:
    """coords [N,4] → z_s [C, R, R, R]."""
    occ = torch.zeros(1, 1, 64, 64, 64, device=device)
    occ[0, 0, coords[:, 1], coords[:, 2], coords[:, 3]] = 1
    z_s = encoder(occ)
    return z_s.squeeze(0)


def _save_npz(path: Path, feats, coords, z_s, dino_voxel_mean=None):
    data = {
        "slat_feats": feats.detach().cpu().float().numpy(),
        "slat_coords": coords.detach().cpu().int().numpy(),
        "ss": z_s.detach().cpu().float().numpy(),
    }
    if dino_voxel_mean is not None:
        if isinstance(dino_voxel_mean, torch.Tensor):
            dino_voxel_mean = dino_voxel_mean.detach().cpu().numpy()
        data["dino_voxel_mean"] = dino_voxel_mean.astype(np.float16)
    np.savez(path, **data)


# ──────────────────────────── Phase 1: simple conversion ────────────────────

def _obj_id_from_edit_id(edit_id: str) -> str:
    """Extract object UUID from edit_id like ``mod_<uuid>_003``."""
    parts = edit_id.split("_", 1)
    if len(parts) < 2:
        return edit_id
    rest = parts[1]
    idx = rest.rfind("_")
    return rest[:idx] if idx != -1 else rest


def phase1_convert(
    pair_dirs: list[Path],
    encoder,
    device: str,
    *,
    dry_run: bool,
) -> dict[str, int]:
    """Convert existing ``*_slat/`` directories to ``*.npz`` + SS.

    Groups pair dirs by object so that the shared ``before`` SLAT
    (identical across all edits of the same object) is only encoded
    once and then hard-linked to subsequent pair dirs.
    """
    stats = {"converted": 0, "skipped": 0, "no_src": 0, "error": 0,
             "hardlinked": 0}

    obj_groups: dict[str, list[Path]] = defaultdict(list)
    for d in pair_dirs:
        obj_id = _obj_id_from_edit_id(d.name)
        obj_groups[obj_id].append(d)

    done = 0
    total = len(pair_dirs)
    for obj_id, dirs in obj_groups.items():
        canonical_before: Path | None = None

        for d in dirs:
            # ── before side (shared across edits of the same object) ──
            b_npz = d / "before.npz"
            if b_npz.exists():
                if canonical_before is None:
                    canonical_before = b_npz
                stats["skipped"] += 1
            else:
                b_slat = d / "before_slat"
                if not b_slat.exists():
                    stats["no_src"] += 1
                elif dry_run:
                    stats["converted"] += 1
                elif canonical_before is not None:
                    try:
                        os.link(str(canonical_before), str(b_npz))
                        stats["hardlinked"] += 1
                    except OSError:
                        shutil.copy2(str(canonical_before), str(b_npz))
                        stats["hardlinked"] += 1
                else:
                    resolved = b_slat.resolve()
                    feats_f, coords_f = resolved / "feats.pt", resolved / "coords.pt"
                    if not feats_f.exists() or not coords_f.exists():
                        stats["no_src"] += 1
                    else:
                        try:
                            feats = torch.load(feats_f, weights_only=True).to(device)
                            coords = torch.load(coords_f, weights_only=True).to(device)
                            z_s = encode_ss(encoder, coords, device)
                            _save_npz(b_npz, feats, coords, z_s)
                            canonical_before = b_npz
                            stats["converted"] += 1
                        except Exception as e:
                            log.warning("Phase1 error %s/before: %s", d.name, e)
                            stats["error"] += 1

            # ── after side (unique per edit) ──
            a_npz = d / "after.npz"
            if a_npz.exists():
                stats["skipped"] += 1
            else:
                a_slat = d / "after_slat"
                if not a_slat.exists():
                    stats["no_src"] += 1
                elif dry_run:
                    stats["converted"] += 1
                else:
                    resolved = a_slat.resolve()
                    feats_f, coords_f = resolved / "feats.pt", resolved / "coords.pt"
                    if not feats_f.exists() or not coords_f.exists():
                        stats["no_src"] += 1
                    else:
                        try:
                            feats = torch.load(feats_f, weights_only=True).to(device)
                            coords = torch.load(coords_f, weights_only=True).to(device)
                            z_s = encode_ss(encoder, coords, device)
                            _save_npz(a_npz, feats, coords, z_s)
                            stats["converted"] += 1
                        except Exception as e:
                            log.warning("Phase1 error %s/after: %s", d.name, e)
                            stats["error"] += 1

            done += 1
            if done % 500 == 0:
                log.info("  Phase1 progress: %d / %d dirs  (hardlinked %d)",
                         done, total, stats["hardlinked"])

    return stats


# ──────────────────────────── Phase 2: deletion backfill ────────────────────

def phase2_deletion(
    specs_by_type: dict[str, list[SpecInfo]],
    mesh_pairs: Path,
    refiner,
    dataset,
    device: str,
    *,
    dry_run: bool,
) -> dict[str, int]:
    """Backfill SLAT + SS for deletion pairs that only have PLY.

    ``z_s_before`` and ``before.npz`` are computed once per object and
    hard-linked to subsequent deletion dirs of the same object.
    """
    from interweave_Trellis import get_coords_mask
    from trellis.modules import sparse as sp

    del_specs = specs_by_type.get("deletion", [])
    stats = {"converted": 0, "skipped": 0, "no_dir": 0, "error": 0,
             "hardlinked": 0}

    obj_groups: dict[str, list[SpecInfo]] = defaultdict(list)
    for s in del_specs:
        obj_groups[s.obj_id].append(s)

    obj_done = 0
    for obj_id, obj_specs in obj_groups.items():
        pair_dirs_need_work = []
        for s in obj_specs:
            d = mesh_pairs / s.edit_id
            if not d.is_dir():
                stats["no_dir"] += 1
                continue
            if (d / "before.npz").exists() and (d / "after.npz").exists():
                stats["skipped"] += 1
                continue
            pair_dirs_need_work.append(s)

        if not pair_dirs_need_work:
            obj_done += 1
            continue

        if dry_run:
            stats["converted"] += len(pair_dirs_need_work)
            obj_done += 1
            continue

        try:
            ori_slat = refiner.encode_object(None, obj_id)
            shard = pair_dirs_need_work[0].shard
            obj_record = dataset.load_object(shard, obj_id)
        except Exception as e:
            log.warning("Phase2: cannot load object %s: %s", obj_id, e)
            stats["error"] += len(pair_dirs_need_work)
            obj_done += 1
            continue

        ss_enc = refiner.trellis_text.models["sparse_structure_encoder"]
        z_s_before = encode_ss(ss_enc, ori_slat.coords, device)
        canonical_before: Path | None = None

        for s in pair_dirs_need_work:
            d = mesh_pairs / s.edit_id
            try:
                d.mkdir(parents=True, exist_ok=True)
                b = d / "before.npz"
                a = d / "after.npz"

                # ── before.npz (identical for all deletions of the same object) ──
                if not b.exists():
                    if canonical_before is not None:
                        try:
                            os.link(str(canonical_before), str(b))
                        except OSError:
                            shutil.copy2(str(canonical_before), str(b))
                        stats["hardlinked"] += 1
                    else:
                        _save_npz(b, ori_slat.feats, ori_slat.coords, z_s_before)
                        canonical_before = b

                # ── after.npz (unique per deletion — different parts removed) ──
                if not a.exists():
                    mask, _ = refiner.build_part_mask(
                        obj_id, obj_record,
                        s.remove_part_ids, ori_slat, "Deletion",
                    )
                    keep = get_coords_mask(ori_slat.coords, mask)
                    after_feats = ori_slat.feats[keep]
                    after_coords = ori_slat.coords[keep]
                    z_s_after = encode_ss(ss_enc, after_coords, device)
                    _save_npz(a, after_feats, after_coords, z_s_after)

                stats["converted"] += 1
            except Exception as e:
                log.warning("Phase2 error %s: %s", s.edit_id, e)
                stats["error"] += 1

        try:
            obj_record.close()
        except Exception:
            pass

        obj_done += 1
        if obj_done % 100 == 0:
            log.info("  Phase2 progress: %d / %d objects  (hardlinked %d)",
                     obj_done, len(obj_groups), stats["hardlinked"])

    return stats


# ──────────────────────────── Phase 3: addition backfill ────────────────────

def _link_or_copy(src: Path, dst: Path) -> None:
    """Hard-link *src* → *dst*; fall back to copy on cross-device."""
    try:
        os.link(str(src), str(dst))
    except OSError:
        shutil.copy2(str(src), str(dst))


def phase3_addition(
    specs_by_type: dict[str, list[SpecInfo]],
    mesh_pairs: Path,
    *,
    dry_run: bool,
) -> dict[str, int]:
    """Backfill addition pairs by swapping the source deletion pair's npz.

    ``add.after.npz`` == ``del.before.npz`` (the original object) — shared
    across all additions of the same object, so we hard-link instead of copy.
    """
    add_specs = specs_by_type.get("addition", [])
    stats = {"converted": 0, "skipped": 0, "no_source": 0, "hardlinked": 0}

    obj_groups: dict[str, list[SpecInfo]] = defaultdict(list)
    for s in add_specs:
        obj_groups[s.obj_id].append(s)

    for obj_id, obj_specs in obj_groups.items():
        canonical_after: Path | None = None

        for s in obj_specs:
            add_dir = mesh_pairs / s.edit_id
            if (add_dir / "before.npz").exists() and (add_dir / "after.npz").exists():
                if canonical_after is None:
                    canonical_after = add_dir / "after.npz"
                stats["skipped"] += 1
                continue

            del_dir = mesh_pairs / s.source_del_id
            if not (del_dir / "before.npz").exists() or not (del_dir / "after.npz").exists():
                stats["no_source"] += 1
                continue

            if dry_run:
                stats["converted"] += 1
                continue

            add_dir.mkdir(parents=True, exist_ok=True)

            # before.npz ← del's after.npz (unique per deletion edit)
            dst_b = add_dir / "before.npz"
            if not dst_b.exists():
                _link_or_copy(del_dir / "after.npz", dst_b)

            # after.npz ← del's before.npz (= original object, shared)
            dst_a = add_dir / "after.npz"
            if not dst_a.exists():
                if canonical_after is not None:
                    _link_or_copy(canonical_after, dst_a)
                    stats["hardlinked"] += 1
                else:
                    _link_or_copy(del_dir / "before.npz", dst_a)
                    canonical_after = dst_a

            stats["converted"] += 1

    return stats


# ──────────────────────────── Phase 4: identity backfill ────────────────────

def phase4_identity(
    specs_by_type: dict[str, list[SpecInfo]],
    all_specs: list[SpecInfo],
    mesh_pairs: Path,
    *,
    dry_run: bool,
) -> dict[str, int]:
    """Backfill identity pairs: same before.npz used as both before and after.

    All identity files for the same object are hard-linked to a single
    canonical ``before.npz`` from any already-migrated edit of that object.
    """
    idt_specs = specs_by_type.get("identity", [])
    stats = {"converted": 0, "skipped": 0, "no_source": 0, "hardlinked": 0}

    obj_to_first_pair: dict[str, Path | None] = {}
    for s in all_specs:
        if s.edit_type == "identity":
            continue
        if s.obj_id in obj_to_first_pair:
            continue
        d = mesh_pairs / s.edit_id
        if (d / "before.npz").exists():
            obj_to_first_pair[s.obj_id] = d

    for s in idt_specs:
        idt_dir = mesh_pairs / s.edit_id
        if (idt_dir / "before.npz").exists() and (idt_dir / "after.npz").exists():
            stats["skipped"] += 1
            continue

        src_dir = obj_to_first_pair.get(s.obj_id)
        if src_dir is None or not (src_dir / "before.npz").exists():
            stats["no_source"] += 1
            continue

        if dry_run:
            stats["converted"] += 1
            continue

        idt_dir.mkdir(parents=True, exist_ok=True)
        src_npz = src_dir / "before.npz"
        for tag in ("before.npz", "after.npz"):
            dst = idt_dir / tag
            if not dst.exists():
                _link_or_copy(src_npz, dst)
                stats["hardlinked"] += 1
        stats["converted"] += 1

    return stats


# ──────────────────────────── PLY render+encode ───────────────────────────────

def _render_and_extract_dino(
    ply_path: Path,
    name: str,
    work_dir: Path,
    num_views: int = 40,
    blender_path: str | None = None,
) -> np.ndarray:
    """Render a PLY with Blender Cycles, voxelize, extract DINOv2 features.

    Returns ``dino_voxel_mean [N, 1024]`` float16.
    """
    import tempfile

    render_out = work_dir / name
    render_out.mkdir(parents=True, exist_ok=True)

    # ── render ──
    from encode_asset.render_img_for_enc import render, voxelize

    if blender_path:
        os.environ["BLENDER_PATH"] = blender_path
    ret = render(str(ply_path), name, str(work_dir) + os.sep, num_views=num_views)
    if ret != 0:
        raise RuntimeError(f"Blender render failed (exit {ret}) for {ply_path}")

    mesh_ply = render_out / "mesh.ply"
    if not mesh_ply.exists():
        raise FileNotFoundError(f"Blender produced no mesh.ply for {name}")

    # ── voxelize ──
    voxelize(str(mesh_ply), name, str(render_out))

    # ── DINOv2 feature extraction ──
    from encode_asset.encode_into_SLAT import extract_dino_voxel_mean

    dino_voxel_mean, _indices = extract_dino_voxel_mean(str(render_out), num_views)
    return dino_voxel_mean


def _inject_dino_into_npz(npz_path: Path, dino_voxel_mean: np.ndarray) -> None:
    """Add or update ``dino_voxel_mean`` key in an existing NPZ file."""
    existing = dict(np.load(npz_path))
    existing["dino_voxel_mean"] = dino_voxel_mean.astype(np.float16)
    np.savez(npz_path, **existing)


# ──────────────────────────── Phase 5: DINOv2 feature extraction ──────────

def phase5_dino_features(
    pair_dirs: list[Path],
    specs_by_type: dict[str, list[SpecInfo]],
    slat_dir: Path | None,
    work_dir: Path,
    *,
    num_views: int = 40,
    blender_path: str | None = None,
    dry_run: bool,
) -> dict[str, int]:
    """Extract DINOv2 voxel features for all edit pairs via PLY rendering.

    For **after** side: render ``after.ply`` → Blender → DINOv2 → inject into
    ``after.npz``.

    For **before** side: load ``{obj_id}_dino_voxel_mean.pt`` from ``slat_dir``
    (produced by ``encode_into_SLAT``). If not available, render ``before.ply``
    as fallback.

    Skips pairs whose NPZ already contains ``dino_voxel_mean``.
    """
    stats = {
        "after_rendered": 0, "after_skipped": 0, "after_no_ply": 0,
        "after_error": 0,
        "before_loaded": 0, "before_rendered": 0, "before_skipped": 0,
        "before_no_src": 0, "before_error": 0,
        "before_hardlinked": 0,
    }

    # Group by obj_id for before sharing
    obj_groups: dict[str, list[Path]] = defaultdict(list)
    for d in pair_dirs:
        obj_id = _obj_id_from_edit_id(d.name)
        obj_groups[obj_id].append(d)

    work_dir.mkdir(parents=True, exist_ok=True)
    total_dirs = len(pair_dirs)
    processed = 0

    for obj_id, dirs in obj_groups.items():
        # ── before side (shared per object) ──
        before_dino: np.ndarray | None = None
        canonical_before_npz: Path | None = None

        # Try loading pre-saved dino_voxel_mean from slat_dir
        if slat_dir is not None:
            dvm_candidates = [
                slat_dir / f"{obj_id}_dino_voxel_mean.pt",
            ]
            # Also search shard subdirectories
            if slat_dir.is_dir():
                for sub in sorted(slat_dir.iterdir()):
                    if sub.is_dir():
                        dvm_candidates.append(sub / f"{obj_id}_dino_voxel_mean.pt")
            for cand in dvm_candidates:
                if cand.exists():
                    if not dry_run:
                        t = torch.load(str(cand), map_location="cpu", weights_only=True)
                        before_dino = t.numpy().astype(np.float16)
                    stats["before_loaded"] += 1
                    break

        for d in dirs:
            # ── after side ──
            a_npz = d / "after.npz"
            if a_npz.exists() and not dry_run:
                try:
                    existing = np.load(a_npz)
                    if "dino_voxel_mean" in existing.files:
                        stats["after_skipped"] += 1
                        processed += 1
                        continue
                except Exception:
                    pass

            a_ply = d / "after.ply"
            if not a_ply.exists() or not a_npz.exists():
                stats["after_no_ply"] += 1
            elif dry_run:
                stats["after_rendered"] += 1
            else:
                try:
                    dvm = _render_and_extract_dino(
                        a_ply, f"after_{d.name}", work_dir,
                        num_views=num_views, blender_path=blender_path,
                    )
                    _inject_dino_into_npz(a_npz, dvm)
                    stats["after_rendered"] += 1
                except Exception as e:
                    log.warning("Phase5 after error %s: %s", d.name, e)
                    stats["after_error"] += 1

            # ── before side ──
            b_npz = d / "before.npz"
            if b_npz.exists() and not dry_run:
                try:
                    existing = np.load(b_npz)
                    if "dino_voxel_mean" in existing.files:
                        stats["before_skipped"] += 1
                        processed += 1
                        continue
                except Exception:
                    pass

            if not b_npz.exists():
                stats["before_no_src"] += 1
            elif before_dino is not None and not dry_run:
                _inject_dino_into_npz(b_npz, before_dino)
                stats["before_loaded"] += 1
            elif dry_run:
                stats["before_rendered"] += 1
            else:
                # Fallback: render before.ply
                b_ply = d / "before.ply"
                if b_ply.exists():
                    try:
                        before_dino = _render_and_extract_dino(
                            b_ply, f"before_{d.name}", work_dir,
                            num_views=num_views, blender_path=blender_path,
                        )
                        _inject_dino_into_npz(b_npz, before_dino)
                        stats["before_rendered"] += 1
                    except Exception as e:
                        log.warning("Phase5 before error %s: %s", d.name, e)
                        stats["before_error"] += 1
                else:
                    stats["before_no_src"] += 1

            processed += 1
            if processed % 100 == 0:
                log.info("  Phase5 progress: %d / %d dirs", processed, total_dirs)

    return stats


# ──────────────────────────── spec loading ──────────────────────────────────

def _load_specs(specs_path: Path) -> list[SpecInfo]:
    specs: list[SpecInfo] = []
    with open(specs_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            specs.append(SpecInfo(
                edit_id=d["edit_id"],
                edit_type=d.get("edit_type", ""),
                obj_id=d.get("obj_id", ""),
                shard=d.get("shard", ""),
                remove_part_ids=d.get("remove_part_ids", []),
                source_del_id=d.get("source_del_id", ""),
            ))
    return specs


def _group_by_type(specs: list[SpecInfo]) -> dict[str, list[SpecInfo]]:
    groups: dict[str, list[SpecInfo]] = defaultdict(list)
    for s in specs:
        groups[s.edit_type].append(s)
    return groups


# ──────────────────────────── refiner setup ─────────────────────────────────

def _create_refiner(cfg: dict, ckpt_root: str, device: str):
    """Create a TrellisRefiner with only the SS encoder loaded (lightweight).

    The full text/image pipeline is NOT loaded — ``load_models()`` is not
    called.  Instead we load only the SS encoder and attach it as a minimal
    ``trellis_text`` object so that ``encode_ss()``, ``encode_object()``, and
    ``build_part_mask()`` work.
    """
    from partcraft.phase2_assembly.trellis_refine import TrellisRefiner
    from scripts.pipeline_common import resolve_data_dirs

    slat_dir, img_enc_dir = resolve_data_dirs(cfg)
    cache_dir = cfg.get("phase2_5", {}).get("cache_dir", "/tmp/migrate_cache")

    refiner = TrellisRefiner(
        cache_dir=cache_dir,
        device=device,
        ckpt_dir=ckpt_root,
        slat_dir=slat_dir,
        img_enc_dir=img_enc_dir,
    )

    encoder = _load_ss_encoder(Path(ckpt_root), device)

    class _MinimalPipeline:
        def __init__(self, enc):
            self.models = {"sparse_structure_encoder": enc}

    refiner.trellis_text = _MinimalPipeline(encoder)
    return refiner


# ──────────────────────────── main ──────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Migrate legacy pipeline outputs to NPZ format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Pipeline YAML config (derives ckpt_root, slat_dir, dataset paths)",
    )
    parser.add_argument(
        "--mesh-pairs", required=True,
        help="Root mesh_pairs directory to migrate",
    )
    parser.add_argument(
        "--specs-jsonl", type=str, default=None,
        help="edit_specs JSONL (needed for Phase 2–4; "
             "can be omitted for Phase 1 only)",
    )
    parser.add_argument(
        "--ckpt-root", type=str, default=None,
        help="Checkpoint root (overrides config; contains TRELLIS-text-xlarge)",
    )
    parser.add_argument(
        "--phase", type=str, default="all",
        help="Comma-separated phases to run: 1,2,3,4,5 or 'all' (default: all)",
    )
    parser.add_argument("--include-list", type=str, default=None,
                        help="Text file with edit_ids to process (one per line); "
                             "others are skipped. Phase 3/4 auto-includes "
                             "addition/identity whose source deletion is included.")
    parser.add_argument("--blender-path", type=str, default=None,
                        help="Blender executable path (Phase 5). "
                             "Reads BLENDER_PATH env if not set.")
    parser.add_argument("--dino-views", type=int, default=40,
                        help="Number of views for Phase 5 DINOv2 rendering (default: 40)")
    parser.add_argument("--dino-work-dir", type=str, default=None,
                        help="Working directory for Phase 5 render intermediates "
                             "(default: <mesh-pairs>/../_dino_render_tmp)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Count only, do not write files or load models")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    mesh_pairs = Path(args.mesh_pairs)
    if not mesh_pairs.is_dir():
        log.error("%s is not a directory", mesh_pairs)
        sys.exit(1)

    phases = (
        {1, 2, 3, 4, 5} if args.phase == "all"
        else {int(p) for p in args.phase.split(",")}
    )
    log.info("Phases to run: %s  dry_run=%s", sorted(phases), args.dry_run)

    # ── Resolve config and paths ──
    cfg = None
    ckpt_root = args.ckpt_root
    if args.config:
        from partcraft.utils.config import load_config
        cfg = load_config(args.config)
        if ckpt_root is None:
            ckpt_root = cfg.get("ckpt_root")

    if ckpt_root is None and not args.dry_run:
        log.error("--ckpt-root is required (or provide --config)")
        sys.exit(1)

    # ── Load include list (optional filter) ──
    include_set: set[str] | None = None
    if args.include_list:
        inc_path = Path(args.include_list)
        if not inc_path.exists():
            log.error("Include list not found: %s", inc_path)
            sys.exit(1)
        include_set = set()
        with open(inc_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    include_set.add(line)
        log.info("Include list: %d edit_ids from %s", len(include_set), inc_path)

    # ── Collect pair dirs ──
    pair_dirs = sorted(d for d in mesh_pairs.iterdir() if d.is_dir())
    if include_set is not None:
        pair_dirs = [d for d in pair_dirs if d.name in include_set]
    log.info("Found %d pair directories under %s (filtered=%s)",
             len(pair_dirs), mesh_pairs, include_set is not None)

    # ── Load specs (Phase 2–4) ──
    specs: list[SpecInfo] = []
    specs_by_type: dict[str, list[SpecInfo]] = {}
    if phases & {2, 3, 4, 5}:
        if not args.specs_jsonl:
            log.error("--specs-jsonl is required for Phase 2/3/4/5")
            sys.exit(1)
        specs_path = Path(args.specs_jsonl)
        if not specs_path.exists():
            log.error("Specs file not found: %s", specs_path)
            sys.exit(1)
        specs = _load_specs(specs_path)

        # Filter by include list: deletion must be in set;
        # addition/identity auto-included if their source deletion is included
        if include_set is not None:
            included_del_ids = {
                s.edit_id for s in specs
                if s.edit_type == "deletion" and s.edit_id in include_set
            }
            filtered = []
            for s in specs:
                if s.edit_id in include_set:
                    filtered.append(s)
                elif s.edit_type == "addition" and s.source_del_id in included_del_ids:
                    filtered.append(s)
                elif s.edit_type == "identity":
                    # Include identity if any edit of the same object is included
                    obj_included = any(
                        o.obj_id == s.obj_id and o.edit_id in include_set
                        for o in specs if o.edit_type != "identity"
                    )
                    if obj_included:
                        filtered.append(s)
            log.info("Specs filtered by include list: %d → %d",
                     len(specs), len(filtered))
            specs = filtered

        specs_by_type = _group_by_type(specs)
        log.info(
            "Loaded %d specs: %s",
            len(specs),
            {k: len(v) for k, v in specs_by_type.items()},
        )

    # ────────── Phase 1 ──────────
    if 1 in phases:
        log.info("=" * 60)
        log.info("Phase 1: Simple *_slat/ → npz conversion")
        encoder = None
        if not args.dry_run:
            encoder = _load_ss_encoder(Path(ckpt_root), args.device)
        s = phase1_convert(pair_dirs, encoder, args.device, dry_run=args.dry_run)
        log.info("Phase 1 done: %s", s)

    # ────────── Phase 2 ──────────
    if 2 in phases:
        log.info("=" * 60)
        log.info("Phase 2: Deletion backfill (SLAT + SS from source)")
        refiner = None
        dataset = None
        if not args.dry_run:
            if cfg is None:
                log.error("--config is required for Phase 2")
                sys.exit(1)
            from scripts.pipeline_common import create_dataset
            refiner = _create_refiner(cfg, ckpt_root, args.device)
            dataset = create_dataset(cfg)
        s = phase2_deletion(
            specs_by_type, mesh_pairs, refiner, dataset,
            args.device, dry_run=args.dry_run,
        )
        log.info("Phase 2 done: %s", s)

    # ────────── Phase 3 ──────────
    if 3 in phases:
        log.info("=" * 60)
        log.info("Phase 3: Addition backfill (swap from deletion)")
        s = phase3_addition(specs_by_type, mesh_pairs, dry_run=args.dry_run)
        log.info("Phase 3 done: %s", s)

    # ────────── Phase 4 ──────────
    if 4 in phases:
        log.info("=" * 60)
        log.info("Phase 4: Identity backfill")
        s = phase4_identity(
            specs_by_type, specs, mesh_pairs, dry_run=args.dry_run,
        )
        log.info("Phase 4 done: %s", s)

    # ────────── Phase 5 ──────────
    if 5 in phases:
        log.info("=" * 60)
        log.info("Phase 5: DINOv2 voxel feature extraction (PLY render)")
        slat_dir = None
        if cfg is not None:
            from scripts.pipeline_common import resolve_data_dirs
            slat_dir = Path(resolve_data_dirs(cfg)[0])
        dino_work = (
            Path(args.dino_work_dir) if args.dino_work_dir
            else mesh_pairs.parent / "_dino_render_tmp"
        )
        blender = args.blender_path or os.environ.get("BLENDER_PATH")
        s = phase5_dino_features(
            pair_dirs, specs_by_type, slat_dir, dino_work,
            num_views=args.dino_views,
            blender_path=blender,
            dry_run=args.dry_run,
        )
        log.info("Phase 5 done: %s", s)

    log.info("=" * 60)
    log.info("Migration complete.")


if __name__ == "__main__":
    main()
