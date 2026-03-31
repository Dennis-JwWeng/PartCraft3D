#!/usr/bin/env python3
"""Render before/after orbiting view gallery for object-centric edit pairs.

Reads the repacked ``partverse_pairs/shard_XX/{obj_id}/`` layout and produces
a comparison PNG per edit: top row = before views, bottom row = after views,
with prompt and metadata overlay.

Supports both the new object-centric format AND legacy flat ``mesh_pairs/``
layout.

Usage
-----
Render all edits in a shard (sample 5 per type)::

    python scripts/vis/render_edit_gallery.py \\
        --input-dir partverse_pairs \\
        --shards 00 \\
        --sample-per-type 5

Render specific edit types::

    python scripts/vis/render_edit_gallery.py \\
        --input-dir partverse_pairs \\
        --edit-types modification scale \\
        --sample-per-type 10

Only render edits that passed cleaning::

    python scripts/vis/render_edit_gallery.py \\
        --input-dir partverse_pairs \\
        --min-tier medium

Legacy flat mesh_pairs format::

    python scripts/vis/render_edit_gallery.py \\
        --pairs-dir outputs/partverse/shard_01/mesh_pairs_shard01 \\
        --edit-ids mod_xxx_000 scl_xxx_001

All outputs go to ``--output-dir`` (default: ``{input_dir}/vis_gallery/``).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# =========================================================================
# SLAT loading (supports .npz and legacy feats.pt/coords.pt)
# =========================================================================

def load_slat(path: Path | str, device: str = "cuda"):
    """Load SLAT from NPZ or legacy directory."""
    from trellis.modules import sparse as sp

    path = Path(path)
    npz = path if path.suffix == ".npz" else path.with_suffix(".npz")
    if npz.exists():
        data = np.load(str(npz))
        feats = torch.from_numpy(data["slat_feats"]).to(device)
        coords = torch.from_numpy(data["slat_coords"]).to(device)
        return sp.SparseTensor(feats=feats, coords=coords)
    # Legacy: directory with feats.pt + coords.pt
    if path.is_dir():
        feats = torch.load(path / "feats.pt", weights_only=True).to(device)
        coords = torch.load(path / "coords.pt", weights_only=True).to(device)
        return sp.SparseTensor(feats=feats, coords=coords)
    raise FileNotFoundError(f"No SLAT found at {path}")


# =========================================================================
# Rendering
# =========================================================================

def render_orbit_views(
    pipeline,
    slat,
    num_views: int = 8,
    pitch: float = 0.4,
) -> list[np.ndarray]:
    """Decode SLAT → Gaussian → render orbiting views.

    Returns list of [H, W, 3] uint8 arrays.
    """
    from trellis.utils import render_utils

    outputs = pipeline.decode_slat(slat, ["gaussian"])
    gaussian = outputs["gaussian"][0]
    yaws = torch.linspace(0, 2 * np.pi, num_views + 1)[:-1]
    pitches = torch.tensor([pitch] * num_views)
    imgs = render_utils.Trellis_render_multiview_images(
        gaussian, yaws.tolist(), pitches.tolist()
    )["color"]
    return imgs


# =========================================================================
# Image composition
# =========================================================================

def _get_font(size: int = 16):
    """Get a font, with fallback."""
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def compose_gallery(
    before_imgs: list[np.ndarray],
    after_imgs: list[np.ndarray],
    *,
    edit_id: str = "",
    edit_type: str = "",
    prompt: str = "",
    obj_id: str = "",
    tier: str = "",
    score: float = 0.0,
) -> np.ndarray:
    """Compose a comparison gallery PNG.

    Layout::

        ┌──────────────────────────────────────────┐
        │ [TYPE] edit_id  prompt...       tier/score│  <- header bar
        ├──────────────────────────────────────────┤
        │ Before: view0  view1  view2  ...  viewN  │
        ├──────────────────────────────────────────┤
        │ After:  view0  view1  view2  ...  viewN  │
        └──────────────────────────────────────────┘

    Returns [H, W, 3] uint8 array.
    """
    n = min(len(before_imgs), len(after_imgs))
    if n == 0:
        return np.zeros((64, 256, 3), dtype=np.uint8)

    h, w = before_imgs[0].shape[:2]
    label_w = 72  # width of "Before"/"After" label column
    header_h = 48
    row_label_h = 0  # integrated into label column

    canvas_w = label_w + w * n
    canvas_h = header_h + h * 2

    canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    # Header bar
    draw.rectangle([(0, 0), (canvas_w, header_h)], fill=(35, 35, 35))
    font_title = _get_font(14)
    font_small = _get_font(11)

    type_tag = f"[{edit_type.upper()}]" if edit_type else ""
    title = f"{type_tag} {edit_id}"
    draw.text((8, 4), title, fill=(255, 255, 255), font=font_title)

    # Prompt (truncated)
    prompt_text = prompt[:100] + ("..." if len(prompt) > 100 else "")
    draw.text((8, 24), prompt_text, fill=(200, 200, 200), font=font_small)

    # Tier/score badge
    if tier:
        tier_colors = {
            "high": (46, 160, 67),
            "medium": (210, 153, 34),
            "low": (210, 105, 30),
            "negative": (200, 50, 50),
            "rejected": (128, 128, 128),
        }
        badge_color = tier_colors.get(tier, (128, 128, 128))
        badge_text = f"{tier} {score:.2f}" if score else tier
        tw = draw.textlength(badge_text, font=font_small)
        bx = canvas_w - int(tw) - 16
        draw.rounded_rectangle(
            [(bx - 4, 6), (canvas_w - 8, 26)],
            radius=4, fill=badge_color,
        )
        draw.text((bx, 8), badge_text, fill=(255, 255, 255), font=font_small)

    # Object ID
    if obj_id:
        oid_text = f"obj: {obj_id[:16]}"
        draw.text((canvas_w - 140, 30), oid_text,
                  fill=(160, 160, 160), font=font_small)

    # Row labels
    font_label = _get_font(13)
    for row, label in enumerate(["Before", "After"]):
        y = header_h + row * h + h // 2 - 8
        draw.text((8, y), label, fill=(80, 80, 80), font=font_label)

    # Paste view images
    for i in range(n):
        x = label_w + i * w
        canvas.paste(Image.fromarray(before_imgs[i]), (x, header_h))
        canvas.paste(Image.fromarray(after_imgs[i]), (x, header_h + h))

    # Thin separator line between before/after rows
    draw.line(
        [(label_w, header_h + h), (canvas_w, header_h + h)],
        fill=(200, 200, 200), width=1,
    )

    return np.array(canvas)


# =========================================================================
# Edit pair discovery
# =========================================================================

def discover_object_centric(
    input_dir: Path,
    shards: list[str] | None = None,
    edit_types: set[str] | None = None,
    min_tier: str | None = None,
    sample_per_type: int | None = None,
) -> list[dict]:
    """Discover edit pairs from object-centric repacked layout.

    Returns list of dicts with keys:
        before_path, after_path, edit_id, edit_type, prompt, obj_id,
        tier, score
    """
    tier_order = {"high": 0, "medium": 1, "low": 2, "negative": 3, "rejected": 4}
    min_tier_val = tier_order.get(min_tier, 99) if min_tier else 99

    entries = []
    shard_dirs = sorted([
        d for d in input_dir.iterdir()
        if d.is_dir() and d.name.startswith("shard_")
    ])
    if shards:
        shard_set = set(shards)
        shard_dirs = [d for d in shard_dirs
                      if d.name.replace("shard_", "") in shard_set]

    for shard_dir in shard_dirs:
        for obj_dir in sorted(shard_dir.iterdir()):
            if not obj_dir.is_dir():
                continue
            meta_path = obj_dir / "metadata.json"
            if not meta_path.exists():
                continue

            with open(meta_path) as f:
                meta = json.load(f)

            # Load quality if available
            quality_map = {}
            quality_path = obj_dir / "quality.json"
            if quality_path.exists():
                with open(quality_path) as f:
                    q = json.load(f)
                for qe in q.get("edits", []):
                    quality_map[qe["edit_id"]] = {
                        "tier": qe.get("tier", ""),
                        "score": qe.get("score", 0.0),
                    }

            obj_id = meta["obj_id"]
            original = obj_dir / "original.npz"
            if not original.exists():
                continue

            for edit in meta.get("edits", []):
                etype = edit["type"]
                edit_id = edit["edit_id"]

                if edit_types and etype not in edit_types:
                    continue

                # Quality filtering
                qinfo = quality_map.get(edit_id, {})
                tier = qinfo.get("tier", "")
                score = qinfo.get("score", 0.0)
                if min_tier and tier:
                    if tier_order.get(tier, 4) > min_tier_val:
                        continue

                # Resolve before/after paths
                if etype == "identity":
                    before_path = str(original)
                    after_path = str(original)
                elif etype == "addition":
                    del_seq = edit.get("source_del_seq", -1)
                    if del_seq < 0:
                        continue
                    del_file = f"del_{del_seq:03d}.npz"
                    before_path = str(obj_dir / del_file)
                    after_path = str(original)
                else:
                    before_path = str(original)
                    fname = edit.get("file")
                    if not fname:
                        continue
                    after_path = str(obj_dir / fname)

                if not Path(before_path).exists() or not Path(after_path).exists():
                    continue

                entries.append({
                    "before_path": before_path,
                    "after_path": after_path,
                    "edit_id": edit_id,
                    "edit_type": etype,
                    "prompt": edit.get("prompt", ""),
                    "obj_id": obj_id,
                    "tier": tier,
                    "score": score,
                })

    # Sample per type
    if sample_per_type is not None and sample_per_type > 0:
        by_type: dict[str, list[dict]] = defaultdict(list)
        for e in entries:
            by_type[e["edit_type"]].append(e)
        sampled = []
        for etype in sorted(by_type.keys()):
            sampled.extend(by_type[etype][:sample_per_type])
        entries = sampled

    return entries


def discover_flat_pairs(
    pairs_dir: Path,
    edit_ids: list[str] | None = None,
) -> list[dict]:
    """Discover edit pairs from legacy flat mesh_pairs/ layout."""
    entries = []
    if edit_ids:
        dirs = [pairs_dir / eid for eid in edit_ids if (pairs_dir / eid).exists()]
    else:
        dirs = sorted([d for d in pairs_dir.iterdir() if d.is_dir()])

    for d in dirs:
        # Try NPZ first, then legacy slat dirs
        before = after = None
        for tag in ["before", "after"]:
            npz = d / f"{tag}.npz"
            slat_dir = d / f"{tag}_slat"
            if npz.exists():
                if tag == "before":
                    before = str(npz)
                else:
                    after = str(npz)
            elif slat_dir.is_dir() and (slat_dir / "feats.pt").exists():
                if tag == "before":
                    before = str(slat_dir)
                else:
                    after = str(slat_dir)

        if before and after:
            eid = d.name
            # Infer edit type from prefix
            etype = "unknown"
            for prefix, t in [("del", "deletion"), ("add", "addition"),
                               ("mod", "modification"), ("scl", "scale"),
                               ("mat", "material"), ("glb", "global"),
                               ("idt", "identity")]:
                if eid.startswith(prefix + "_"):
                    etype = t
                    break
            entries.append({
                "before_path": before,
                "after_path": after,
                "edit_id": eid,
                "edit_type": etype,
                "prompt": "",
                "obj_id": "",
                "tier": "",
                "score": 0.0,
            })

    return entries


# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Render before/after orbiting view gallery for edit pairs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Input source (choose one)
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--input-dir",
                     help="Object-centric repacked data root (shard_XX dirs)")
    grp.add_argument("--pairs-dir",
                     help="Legacy flat mesh_pairs directory")

    # Filtering
    parser.add_argument("--shards", nargs="*", default=None)
    parser.add_argument("--edit-types", nargs="*", default=None)
    parser.add_argument("--edit-ids", nargs="*", default=None)
    parser.add_argument("--min-tier", default=None,
                        choices=["high", "medium", "low", "negative"])
    parser.add_argument("--sample-per-type", type=int, default=None)

    # Rendering
    parser.add_argument("--num-views", type=int, default=8,
                        help="Number of orbiting views (default: 8)")
    parser.add_argument("--pitch", type=float, default=0.4,
                        help="Camera pitch in radians (default: 0.4)")
    parser.add_argument("--resolution", type=int, default=512,
                        help="Render resolution (default: 512)")

    # Output
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--force", action="store_true",
                        help="Re-render even if cached")

    # TRELLIS
    parser.add_argument("--ckpt-root", type=str, default=None,
                        help="Checkpoint root (default: project checkpoints/)")
    parser.add_argument("--config", type=str, default=None,
                        help="Pipeline config (for ckpt_root and attn_backend)")

    args = parser.parse_args()

    # ── Discover entries ─────────────────────────────────────────────
    if args.input_dir:
        input_dir = Path(args.input_dir)
        entries = discover_object_centric(
            input_dir,
            shards=args.shards,
            edit_types=set(args.edit_types) if args.edit_types else None,
            min_tier=args.min_tier,
            sample_per_type=args.sample_per_type,
        )
        default_output = input_dir / "vis_gallery"
    else:
        pairs_dir = Path(args.pairs_dir)
        entries = discover_flat_pairs(pairs_dir, args.edit_ids)
        default_output = pairs_dir.parent / "vis_gallery"

    if not entries:
        logger.error("No edit pairs found")
        sys.exit(1)

    # Type summary
    type_counts = defaultdict(int)
    for e in entries:
        type_counts[e["edit_type"]] += 1
    logger.info("Found %d edit pairs: %s", len(entries),
                ", ".join(f"{t}={c}" for t, c in sorted(type_counts.items())))

    output_dir = Path(args.output_dir) if args.output_dir else default_output
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load TRELLIS pipeline ────────────────────────────────────────
    ckpt_root = args.ckpt_root
    if not ckpt_root and args.config:
        import yaml
        with open(args.config) as f:
            cfg = yaml.safe_load(f) or {}
        ckpt_root = cfg.get("ckpt_root")
        # Set attn backend
        attn = cfg.get("pipeline", {}).get("attn_backend")
        if attn:
            os.environ["ATTN_BACKEND"] = attn

    if not ckpt_root:
        ckpt_root = str(_PROJECT_ROOT / "checkpoints")

    third_party = str(_PROJECT_ROOT / "third_party")
    if third_party not in sys.path:
        sys.path.insert(0, third_party)

    text_ckpt = str(Path(ckpt_root) / "TRELLIS-text-xlarge")
    logger.info("Loading TRELLIS from %s ...", text_ckpt)

    from trellis.pipelines import TrellisTextTo3DPipeline
    pipeline = TrellisTextTo3DPipeline.from_pretrained(text_ckpt)
    pipeline.cuda()
    logger.info("TRELLIS loaded")

    # ── Render loop ──────────────────────────────────────────────────
    rendered = 0
    skipped = 0
    errors = 0

    for entry in tqdm(entries, desc="Rendering"):
        edit_id = entry["edit_id"]
        out_path = output_dir / f"{edit_id}.png"

        if not args.force and out_path.exists():
            skipped += 1
            continue

        try:
            before_slat = load_slat(entry["before_path"])
            after_slat = load_slat(entry["after_path"])

            before_imgs = render_orbit_views(
                pipeline, before_slat, args.num_views, args.pitch)
            after_imgs = render_orbit_views(
                pipeline, after_slat, args.num_views, args.pitch)

            gallery = compose_gallery(
                before_imgs, after_imgs,
                edit_id=edit_id,
                edit_type=entry["edit_type"],
                prompt=entry["prompt"],
                obj_id=entry["obj_id"],
                tier=entry["tier"],
                score=entry["score"],
            )

            Image.fromarray(gallery).save(str(out_path))
            rendered += 1

        except Exception as e:
            logger.error("  %s: %s", edit_id, e)
            errors += 1

    logger.info(
        "Done: %d rendered, %d skipped (cached), %d errors → %s",
        rendered, skipped, errors, output_dir,
    )


if __name__ == "__main__":
    main()
