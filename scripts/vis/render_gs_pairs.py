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

    # Render specific edit IDs
    ATTN_BACKEND=xformers python scripts/vis/render_gs_pairs.py \
        --config configs/partobjaverse.yaml --edit-ids mod_000001 del_000002

    # Also save individual views (16 per model)
    ATTN_BACKEND=xformers pytho--n scripts/vis/render_gs_pairs.py \
        --config configs/partobjaverse.yaml --save-views --num-views 16

    # Skip comparison video, only render individual before/after videos
    ATTN_BACKEND=xformers python scripts/vis/render_gs_pairs.py \
        --config configs/partobjaverse.yaml --no-compare
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import imageio
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from partcraft.utils.config import load_config
from partcraft.utils.logging import setup_logging


# ---------------------------------------------------------------------------
# SLAT / Gaussian helpers
# ---------------------------------------------------------------------------

def load_slat(slat_dir: Path, device: str = "cuda"):
    """Load SLAT from feats.pt + coords.pt."""
    from trellis.modules import sparse as sp
    feats = torch.load(slat_dir / "feats.pt", weights_only=True)
    coords = torch.load(slat_dir / "coords.pt", weights_only=True)
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

    imageio.mimsave(output_path, frames, fps=fps, codec="libx264",
                    quality=8, pixelformat="yuv420p")


# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------

def load_edit_prompts(cache_dir: Path, tag: str = "") -> dict[str, dict]:
    """Load edit prompts from edit_results jsonl."""
    tag_suffix = f"_{tag}" if tag else ""
    results_path = cache_dir / f"edit_results{tag_suffix}.jsonl"
    prompts = {}

    if not results_path.exists():
        # Fallback: try without tag
        results_path = cache_dir / "edit_results.jsonl"

    if results_path.exists():
        with open(results_path) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                    if rec.get("status") == "success":
                        prompts[rec["edit_id"]] = {
                            "edit_prompt": rec.get("edit_prompt", ""),
                            "edit_type": rec.get("edit_type", ""),
                            "object_desc": rec.get("object_desc", ""),
                            "after_desc": rec.get("after_desc", ""),
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
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = setup_logging(cfg, "render_gs_pairs")

    output_base = Path(cfg["data"]["output_dir"])
    tag_suffix = f"_{args.tag}" if args.tag else ""

    # Locate pairs directory
    if args.pairs_dir:
        pairs_dir = Path(args.pairs_dir)
    else:
        pairs_dir = output_base / f"mesh_pairs{tag_suffix}"

    if not pairs_dir.exists():
        logger.error(f"Pairs directory not found: {pairs_dir}")
        sys.exit(1)

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
        has_before = (d / "before_slat" / "feats.pt").exists()
        has_after = (d / "after_slat" / "feats.pt").exists()
        if has_before and has_after:
            valid_pairs.append(d)
        else:
            logger.warning(f"Skipping {d.name}: missing SLAT files")

    if not valid_pairs:
        logger.error("No valid pairs found with SLAT files")
        sys.exit(1)

    # Load edit prompts
    # NOTE: load_config() already resolves relative cache_dir against output_dir
    cache_dir = Path(cfg.get("phase2_5", {}).get("cache_dir", "cache/phase2_5"))
    edit_prompts = load_edit_prompts(cache_dir, args.tag or "")

    logger.info(f"Rendering {len(valid_pairs)} pairs -> {vis_dir}")

    # Load TRELLIS decoder
    p25_cfg = cfg.get("phase2_5", {})
    vinedresser_path = p25_cfg.get(
        "vinedresser_path", "/Node11_nvme/wjw/3D_Editing/Vinedresser3D-main")
    sys.path.insert(0, vinedresser_path)

    ckpt_dir = Path(p25_cfg.get("ckpt_dir",
                                str(Path(__file__).resolve().parents[2] / "checkpoints")))
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

        logger.info(f"  {edit_id} [{edit_type}]: {prompt[:80]}")

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
