#!/usr/bin/env python3
"""Render before/after comparison turntable videos from Phase 2.5 edit pairs.

Loads SLAT files saved by export_pair, decodes to Gaussian via TRELLIS,
renders side-by-side turntable video with edit prompt overlay.

Usage:
    # Render all pairs as side-by-side comparison videos
    ATTN_BACKEND=xformers python scripts/vis/render_gs_pairs.py \
        --config configs/partobjaverse.yaml

    # Render specific tag (e.g. multiview experiment)
    ATTN_BACKEND=xformers python scripts/vis/render_gs_pairs.py \
        --config configs/partobjaverse.yaml --tag multiview

    # Render specific edit IDs (new format: {type}_{obj_id}_{seq})
    ATTN_BACKEND=xformers python scripts/vis/render_gs_pairs.py \
        --config configs/partobjaverse.yaml --edit-ids del_my-chair-001_000

    # Also save individual views (16 per model)
    ATTN_BACKEND=xformers python scripts/vis/render_gs_pairs.py \
        --config configs/partobjaverse.yaml --save-views --num-views 16

    # Skip comparison video, only render individual before/after videos
    ATTN_BACKEND=xformers python scripts/vis/render_gs_pairs.py \
        --config configs/partobjaverse.yaml --no-compare

    # PartVerse / sharded output: config matches run_streaming (shard_*/mesh_pairs).
    # If streaming used --tag 0326 → mesh_pairs_0326/ (pass --tag 0326, or omit tag
    # when that is the only mesh_pairs_* under the shard — it will be inferred).
    python scripts/vis/render_gs_pairs.py --config configs/partverse_local.yaml \
        --sample-per-type 2
"""

import argparse
import glob as _glob
import json
import os
import sys
from pathlib import Path

import cv2
import imageio
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from partcraft.utils.config import load_config
from partcraft.utils.logging import setup_logging

# Same output_dir / cache layout as run_streaming (shard_* + phase caches).
_SCRIPTS_DIR = str(_PROJECT_ROOT / "scripts")
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)
import pipeline_common as _pipeline_common  # noqa: E402


# ---------------------------------------------------------------------------
# Edit ID → type (fallback when jsonl has no edit_type)
# ---------------------------------------------------------------------------

_PREFIX_TO_EDIT_TYPE: tuple[tuple[str, str], ...] = (
    ("gdel", "deletion"),
    ("gadd", "addition"),
    ("del", "deletion"),
    ("add", "addition"),
    ("mod", "modification"),
    ("scl", "scale"),
    ("mat", "material"),
    ("glb", "global"),
    ("idt", "identity"),
)


def infer_edit_type_from_id(edit_id: str) -> str:
    """Infer PartCraft edit_type from edit_id prefix (see planner._make_edit_id)."""
    for prefix, etype in _PREFIX_TO_EDIT_TYPE:
        if edit_id.startswith(prefix + "_"):
            return etype
    return "unknown"


def _resolve_mesh_pairs_dir(
    output_base: Path, user_tag: str, logger,
) -> tuple[Path, str]:
    """Return ``(mesh_pairs_dir, effective_tag)`` for edit_results / vis_compare.

    Looks for ``mesh_pairs`` or ``mesh_pairs_<user_tag>``. With an empty
    ``user_tag``, falls back to a *single* ``mesh_pairs_*`` directory under the
    shard (or its parent), inferring the tag from the folder name — same naming
    as ``run_streaming.py`` / ``run_phase2_5.py``.
    """
    ut = (user_tag or "").strip()
    tag_suffix = f"_{ut}" if ut else ""
    rel = f"mesh_pairs{tag_suffix}"
    primary = output_base / rel
    if primary.is_dir():
        return primary, ut

    tried: list[Path] = [primary]
    if output_base.name.startswith("shard_"):
        alt = output_base.parent / rel
        tried.append(alt)
        if alt.is_dir():
            logger.warning(
                "mesh_pairs not under %s; using %s (export may predate shard layout).",
                output_base,
                alt,
            )
            return alt, ut

    # No explicit tag: exactly one mesh_pairs_* → infer tag (e.g. mesh_pairs_0326)
    if not ut:
        scan_roots = [output_base]
        if output_base.name.startswith("shard_"):
            scan_roots.append(output_base.parent)
        for root in scan_roots:
            if not root.is_dir():
                continue
            tagged = sorted(
                p for p in root.iterdir()
                if p.is_dir()
                and p.name.startswith("mesh_pairs_")
                and len(p.name) > len("mesh_pairs_")
            )
            if len(tagged) == 1:
                inferred = tagged[0].name[len("mesh_pairs_") :]
                logger.warning(
                    "Using %s (inferred --tag %r for edit_results; pass explicitly to silence).",
                    tagged[0],
                    inferred,
                )
                return tagged[0], inferred
            if len(tagged) > 1:
                logger.error(
                    "Multiple tagged mesh_pairs dirs under %s: %s — pass --tag to pick one.",
                    root,
                    [p.name for p in tagged],
                )
                sys.exit(1)

    logger.error("Pairs directory not found. Checked:")
    for p in tried:
        logger.error("  %s", p)
    if output_base.exists():
        subs = sorted(p.name for p in output_base.iterdir() if p.is_dir())
        logger.error("Directories under %s: %s", output_base, subs or "(none)")
    else:
        logger.error("Output directory does not exist: %s", output_base)
    logger.error(
        "Run Phase 2.5 until mesh_pairs/ or mesh_pairs_<tag>/ appears, "
        "or pass --pairs-dir. If you used streaming --tag MYTAG, pass --tag MYTAG here."
    )
    sys.exit(1)


def sample_pairs_by_type(
    pair_dirs: list[Path],
    prompts: dict[str, dict],
    per_type: int,
) -> list[Path]:
    """Keep up to ``per_type`` pairs per edit_type (stable sort by edit_id)."""
    from collections import defaultdict

    by_type: dict[str, list[Path]] = defaultdict(list)
    for d in sorted(pair_dirs, key=lambda p: p.name):
        info = prompts.get(d.name, {})
        et = (info.get("edit_type") or info.get("effective_edit_type") or "").strip()
        if not et:
            et = infer_edit_type_from_id(d.name)
        by_type[et].append(d)
    out: list[Path] = []
    for et in sorted(by_type.keys()):
        chunk = by_type[et][:per_type]
        out.extend(chunk)
    return out


# ---------------------------------------------------------------------------
# SLAT / Gaussian helpers
# ---------------------------------------------------------------------------

def load_slat(slat_dir: Path, device: str = "cuda"):
    """Load SLAT from feats.pt + coords.pt (follows symlinks)."""
    from trellis.modules import sparse as sp
    resolved = slat_dir.resolve()
    feats = torch.load(resolved / "feats.pt", weights_only=True)
    coords = torch.load(resolved / "coords.pt", weights_only=True)
    return sp.SparseTensor(feats=feats.to(device), coords=coords.to(device))


def render_gaussian_turntable(gaussian, n_frames: int = 120,
                              pitch: float = 0.45) -> list[np.ndarray]:
    """Render a smooth turntable video from a Gaussian."""
    from trellis.utils import render_utils
    yaws = torch.linspace(0, 2 * np.pi, n_frames + 1)[:-1]
    pitches = torch.tensor([pitch] * n_frames)
    imgs = render_utils.Trellis_render_multiview_images(
        gaussian, yaws.tolist(), pitches.tolist())['color']
    return imgs


def render_gaussian_views(gaussian, num_views: int = 16,
                          pitch: float = 0.45) -> list[np.ndarray]:
    """Render multiview images from a Gaussian."""
    from trellis.utils import render_utils
    yaws = torch.linspace(0, 2 * np.pi, num_views + 1)[:-1]
    pitches = torch.tensor([pitch] * num_views)
    return render_utils.Trellis_render_multiview_images(
        gaussian, yaws.tolist(), pitches.tolist())['color']


# ---------------------------------------------------------------------------
# Video composition
# ---------------------------------------------------------------------------

def wrap_text(text: str, max_chars: int = 60) -> list[str]:
    """Word-wrap text into lines."""
    lines = []
    for raw_line in text.split("\n"):
        remaining = raw_line
        while remaining:
            if len(remaining) <= max_chars:
                lines.append(remaining)
                break
            split = remaining[:max_chars].rfind(" ")
            if split <= 0:
                split = max_chars
            lines.append(remaining[:split])
            remaining = remaining[split:].strip()
    return lines


def make_text_bar(text: str, width: int, bar_height: int = 60,
                  bg_color: tuple = (30, 30, 30),
                  fg_color: tuple = (255, 255, 255)) -> np.ndarray:
    """Create a text bar image with prompt text."""
    bar = np.full((bar_height, width, 3), bg_color, dtype=np.uint8)
    lines = wrap_text(text, max_chars=width // 8)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = 1
    y = 20
    for line in lines[:3]:  # max 3 lines
        cv2.putText(bar, line, (10, y), font, font_scale,
                    fg_color, thickness, cv2.LINE_AA)
        y += 18
    return bar


def make_label_bar(label: str, width: int, height: int = 32,
                   bg_color: tuple = (240, 240, 240),
                   fg_color: tuple = (40, 40, 40)) -> np.ndarray:
    """Create a 'Before' / 'After' label bar."""
    bar = np.full((height, width, 3), bg_color, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    tw = cv2.getTextSize(label, font, 0.7, 2)[0][0]
    cv2.putText(bar, label, ((width - tw) // 2, 24),
                font, 0.7, fg_color, 2, cv2.LINE_AA)
    return bar


def compose_comparison_frame(before_img: np.ndarray, after_img: np.ndarray,
                             prompt: str, edit_type: str = "") -> np.ndarray:
    """Compose a single comparison frame: prompt bar + [Before | After]."""
    h, w, _ = before_img.shape

    # Label bars
    before_label = make_label_bar("Before", w)
    after_label = make_label_bar("After", w)

    # Stack label + image
    before_col = np.vstack([before_label, before_img])
    after_col = np.vstack([after_label, after_img])

    # Separator
    sep = np.full((before_col.shape[0], 3, 3), 180, dtype=np.uint8)

    # Side by side
    combined = np.hstack([before_col, sep, after_col])

    # Prompt bar on top
    tag = f"[{edit_type.upper()}] " if edit_type else ""
    prompt_bar = make_text_bar(f"{tag}{prompt}", combined.shape[1])
    return np.vstack([prompt_bar, combined])


def build_comparison_video(before_frames: list[np.ndarray],
                           after_frames: list[np.ndarray],
                           prompt: str, edit_type: str = "",
                           output_path: str = "compare.mp4",
                           fps: int = 30):
    """Build and save a side-by-side comparison video."""
    frames = []
    for bf, af in zip(before_frames, after_frames):
        frame = compose_comparison_frame(bf, af, prompt, edit_type)
        frames.append(frame)

    writer = imageio.get_writer(output_path, format="FFMPEG", fps=fps,
                                codec="libx264", quality=8,
                                pixelformat="yuv420p")
    for frame in frames:
        writer.append_data(frame)
    writer.close()


# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------

def _resolve_path(p: str | Path) -> Path:
    """Resolve a path: absolute stays, relative resolved against project root."""
    pp = Path(p)
    return pp if pp.is_absolute() else _PROJECT_ROOT / pp


def load_edit_prompts(cache_dir: Path, tag: str = "") -> dict[str, dict]:
    """Load edit prompts from edit_results jsonl.

    Handles both single-worker (edit_results_{tag}.jsonl) and multi-worker
    (edit_results_{tag}_w0.jsonl, _w1.jsonl, ...) output files.
    """
    tag_suffix = f"_{tag}" if tag else ""
    prompts = {}

    # Collect all matching results files (base + per-worker)
    candidates = []
    base = cache_dir / f"edit_results{tag_suffix}.jsonl"
    if base.exists():
        candidates.append(base)
    # Multi-worker files: edit_results_{tag}_w0.jsonl, _w1.jsonl, ...
    worker_pattern = str(cache_dir / f"edit_results{tag_suffix}_w*.jsonl")
    candidates.extend(Path(p) for p in sorted(_glob.glob(worker_pattern)))
    # Fallback: try without tag
    if not candidates:
        fallback = cache_dir / "edit_results.jsonl"
        if fallback.exists():
            candidates.append(fallback)

    for results_path in candidates:
        with open(results_path) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                    if rec.get("status") == "success":
                        prompts[rec["edit_id"]] = {
                            "edit_prompt": rec.get("edit_prompt", ""),
                            "edit_type": rec.get("edit_type",
                                                 rec.get("effective_edit_type",
                                                         "")),
                            "object_desc": rec.get("object_desc", ""),
                            "after_desc": rec.get("after_desc", ""),
                            "obj_id": rec.get("obj_id", ""),
                        }
                except (json.JSONDecodeError, KeyError):
                    pass
    return prompts


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Render before/after comparison from Phase 2.5 pairs")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--tag", type=str, default=None,
                        help="Experiment tag (matches --tag from run_phase2_5.py)")
    parser.add_argument("--pairs-dir", type=str, default=None,
                        help="Override pairs directory")
    parser.add_argument("--edit-ids", nargs="*", default=None,
                        help="Specific edit IDs to render")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for videos (default: {output}/vis_compare[_tag])")
    parser.add_argument("--n-frames", type=int, default=120,
                        help="Number of turntable frames")
    parser.add_argument("--fps", type=int, default=30,
                        help="Video FPS")
    parser.add_argument("--pitch", type=float, default=0.45,
                        help="Camera pitch in radians")
    parser.add_argument("--no-compare", action="store_true",
                        help="Skip comparison video, only render individual videos")
    parser.add_argument("--save-views", action="store_true",
                        help="Also save individual multiview PNG images")
    parser.add_argument("--num-views", type=int, default=16,
                        help="Number of view images (only with --save-views)")
    parser.add_argument("--force", action="store_true",
                        help="Re-render even if cached")
    parser.add_argument("--sample-per-type", type=int, default=None,
                        metavar="N",
                        help="After SLAT validation, keep at most N pairs per "
                             "edit_type (from edit_results jsonl or edit_id prefix)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    _pipeline_common.normalize_cache_dirs(cfg)
    _pipeline_common.set_attn_backend(cfg)
    logger = setup_logging(cfg, "render_gs_pairs")

    output_base = Path(cfg["data"]["output_dir"])
    user_tag = (args.tag or "").strip()

    # Locate pairs directory
    if args.pairs_dir:
        pairs_dir = Path(args.pairs_dir)
        if not pairs_dir.is_dir():
            logger.error("Pairs directory not found or not a directory: %s", pairs_dir)
            sys.exit(1)
        effective_tag = user_tag
        if not effective_tag and pairs_dir.name.startswith("mesh_pairs_") and len(
                pairs_dir.name) > len("mesh_pairs_"):
            effective_tag = pairs_dir.name[len("mesh_pairs_") :]
            logger.info("Inferred --tag %r from pairs directory name.", effective_tag)
    else:
        pairs_dir, effective_tag = _resolve_mesh_pairs_dir(output_base, user_tag, logger)

    tag_suffix = f"_{effective_tag}" if effective_tag else ""

    # Output directory for comparison videos
    if args.output_dir:
        vis_dir = Path(args.output_dir)
    else:
        vis_dir = output_base / f"vis_compare{tag_suffix}"
    vis_dir.mkdir(parents=True, exist_ok=True)

    # Find valid pair directories
    if args.edit_ids:
        pair_dirs = [pairs_dir / eid for eid in args.edit_ids
                     if (pairs_dir / eid).exists()]
    else:
        pair_dirs = sorted([d for d in pairs_dir.iterdir() if d.is_dir()])

    valid_pairs = []
    for d in pair_dirs:
        before_slat = d / "before_slat"
        after_slat = d / "after_slat"
        # Resolve symlinks (before_slat may be a relative symlink to shared dir)
        has_before = (before_slat.resolve() / "feats.pt").exists()
        has_after = (after_slat.resolve() / "feats.pt").exists()
        if has_before and has_after:
            valid_pairs.append(d)
        else:
            logger.warning(f"Skipping {d.name}: missing SLAT files"
                           f" (before={has_before}, after={has_after})")

    if not valid_pairs:
        logger.error("No valid pairs found with SLAT files")
        sys.exit(1)

    # Load edit prompts — phase2_5 cache_dir already under output_dir after
    # normalize_cache_dirs (same path streaming wrote edit_results*.jsonl to).
    raw_cache = cfg.get("phase2_5", {}).get("cache_dir", "cache/phase2_5")
    cache_dir = Path(raw_cache) if os.path.isabs(raw_cache) else _resolve_path(
        raw_cache)
    edit_prompts = load_edit_prompts(cache_dir, effective_tag)

    if args.sample_per_type is not None:
        if args.sample_per_type <= 0:
            logger.error("--sample-per-type must be positive")
            sys.exit(1)
        if args.edit_ids:
            logger.error("Use either --edit-ids or --sample-per-type, not both")
            sys.exit(1)
        before_n = len(valid_pairs)
        valid_pairs = sample_pairs_by_type(valid_pairs, edit_prompts,
                                           args.sample_per_type)
        logger.info(f"Sampled {len(valid_pairs)} / {before_n} pairs "
                    f"(<= {args.sample_per_type} per edit_type)")

    logger.info(f"Rendering {len(valid_pairs)} pairs -> {vis_dir}")

    # Load TRELLIS decoder
    p25_cfg = cfg.get("phase2_5", {})
    project_root = Path(__file__).resolve().parents[2]
    third_party = str(project_root / "third_party")
    if third_party not in sys.path:
        sys.path.insert(0, third_party)

    ckpt_dir = Path(cfg.get("ckpt_root") or p25_cfg.get(
        "ckpt_dir", str(project_root / "checkpoints")))
    text_ckpt = str(ckpt_dir / "TRELLIS-text-xlarge")

    from trellis.pipelines import TrellisTextTo3DPipeline
    logger.info(f"Loading TRELLIS decoder from {text_ckpt}...")
    pipeline = TrellisTextTo3DPipeline.from_pretrained(text_ckpt)
    pipeline.cuda()
    logger.info("TRELLIS decoder loaded")

    # Process each pair
    for pair_dir in tqdm(valid_pairs, desc="Rendering"):
        edit_id = pair_dir.name
        compare_path = vis_dir / f"{edit_id}.mp4"

        # Skip if cached
        if not args.force and compare_path.exists() and not args.no_compare:
            logger.info(f"  {edit_id}: cached, skipping")
            continue

        # Get prompt info
        info = edit_prompts.get(edit_id, {})
        prompt = info.get("edit_prompt", edit_id)
        edit_type = info.get("edit_type", "")
        obj_id = info.get("obj_id", "")

        logger.info(f"  {edit_id} [{edit_type}]: {prompt[:80]}"
                     + (f"  (obj: {obj_id})" if obj_id else ""))

        # Decode both SLATs to Gaussian
        gaussians = {}
        for tag in ['before', 'after']:
            slat = load_slat(pair_dir / f"{tag}_slat")
            outputs = pipeline.decode_slat(slat, ['gaussian'])
            gaussians[tag] = outputs['gaussian'][0]

        # Render turntable frames
        before_frames = render_gaussian_turntable(
            gaussians['before'], args.n_frames, args.pitch)
        after_frames = render_gaussian_turntable(
            gaussians['after'], args.n_frames, args.pitch)

        # Save comparison video
        if not args.no_compare:
            build_comparison_video(
                before_frames, after_frames,
                prompt, edit_type,
                str(compare_path), args.fps)
            logger.info(f"    -> {compare_path}")

        # Save individual videos
        for tag, frames in [('before', before_frames), ('after', after_frames)]:
            video_path = pair_dir / f"{tag}.mp4"
            if args.force or not video_path.exists():
                imageio.mimsave(str(video_path), frames, fps=args.fps,
                                codec="libx264", quality=8,
                                pixelformat="yuv420p")

        # Save view images
        if args.save_views:
            for tag in ['before', 'after']:
                views_dir = pair_dir / f"{tag}_views"
                if args.force or not views_dir.exists():
                    views_dir.mkdir(parents=True, exist_ok=True)
                    imgs = render_gaussian_views(
                        gaussians[tag], args.num_views, args.pitch)
                    for j, img in enumerate(imgs):
                        Image.fromarray(img).save(
                            str(views_dir / f"{j:03d}.png"))

    logger.info(f"\nDone! Comparison videos saved to: {vis_dir}")
    logger.info(f"Usage: open {vis_dir}/<edit_id>.mp4")


if __name__ == "__main__":
    main()
