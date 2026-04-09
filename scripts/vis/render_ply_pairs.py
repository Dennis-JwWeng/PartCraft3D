#!/usr/bin/env python3
"""Render before/after 4-view comparison images from PLY mesh pairs.

Uses Blender Cycles GPU rendering for accurate textures and colors.
Outputs a single comparison PNG per edit pair (4 views x before/after).

Usage:
    # Render all pairs for a tag
    python scripts/vis/render_ply_pairs.py \
        --config configs/partobjaverse_test.yaml --tag test3

    # Render specific edit IDs
    python scripts/vis/render_ply_pairs.py \
        --config configs/partobjaverse_test.yaml --tag test3 \
        --edit-ids del_00200996b8f34f55a2dd2f44d316d107_000

    # Custom pairs directory
    python scripts/vis/render_ply_pairs.py \
        --pairs-dir outputs/partobjaverse_tiny/mesh_pairs_test3
"""

import argparse
import glob as _glob
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

_BLENDER_RENDER_SCRIPT = _PROJECT_ROOT / "scripts" / "blender_render.py"

# 4 orthogonal views: front, right, back, left
_FOUR_VIEWS = [
    {"yaw": 0,        "pitch": 0.3, "radius": 2.5, "fov": 1.047},
    {"yaw": math.pi/2,"pitch": 0.3, "radius": 2.5, "fov": 1.047},
    {"yaw": math.pi,  "pitch": 0.3, "radius": 2.5, "fov": 1.047},
    {"yaw": 3*math.pi/2, "pitch": 0.3, "radius": 2.5, "fov": 1.047},
]

# 3-view optimal coverage: front + right-back + high overview.
# - View 1 (0°, 0.45 rad ≈ 26°): front + sides, slight overhead
# - View 2 (120°, 0.45 rad):      right-back, same elevation
# - View 3 (240°, 1.1 rad ≈ 63°): left-back, high overhead → top surface
_THREE_VIEWS = [
    {"yaw": 0,              "pitch": 0.45, "radius": 2.5, "fov": 1.047},
    {"yaw": 2*math.pi/3,    "pitch": 0.45, "radius": 2.5, "fov": 1.047},
    {"yaw": 4*math.pi/3,    "pitch": 1.1,  "radius": 2.5, "fov": 1.047},
]


# ---------------------------------------------------------------------------
# Blender multi-view rendering
# ---------------------------------------------------------------------------

def _render_views(mesh_path: str, views: list[dict], resolution: int = 512,
                  blender_path: str = "blender",
                  bg_color: tuple = (1.0, 1.0, 1.0),
                  ref_mesh_path: str | None = None) -> list[np.ndarray]:
    """Render arbitrary views of a mesh via Blender Cycles.

    If ``ref_mesh_path`` is given, normalization (scale + offset) is computed
    from that reference mesh and applied to ``mesh_path``, so a sequence of
    renders sharing the same reference end up at a consistent scale.

    Returns list of (H, W, 3) uint8 numpy arrays (RGB).
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        cmd = [
            blender_path, "-b", "-P", str(_BLENDER_RENDER_SCRIPT), "--",
            "--object", str(mesh_path),
            "--output_folder", tmp_dir,
            "--views", json.dumps(views),
            "--resolution", str(resolution),
        ]
        if ref_mesh_path is not None:
            cmd += ["--ref_object", str(ref_mesh_path)]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            print(f"  Blender stdout:\n{result.stdout[-2000:]}")
            print(f"  Blender stderr:\n{result.stderr[-2000:]}")
            raise RuntimeError(
                f"Blender render failed (exit {result.returncode}) for {mesh_path}")

        images = []
        for i in range(len(views)):
            png_path = os.path.join(tmp_dir, f"{i:03d}.png")
            if not os.path.exists(png_path):
                raise FileNotFoundError(f"Render output not found: {png_path}")
            img = cv2.imread(png_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise RuntimeError(f"Failed to read: {png_path}")
            # RGBA -> RGB with white background
            if img.shape[2] == 4:
                alpha = img[:, :, 3:4].astype(np.float32) / 255.0
                rgb = img[:, :, :3].astype(np.float32)
                bg = np.full_like(rgb, [bg_color[2] * 255, bg_color[1] * 255,
                                        bg_color[0] * 255])
                composited = (rgb * alpha + bg * (1 - alpha)).astype(np.uint8)
                img = cv2.cvtColor(composited, cv2.COLOR_BGR2RGB)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)

    return images


def render_3views(mesh_path: str, resolution: int = 512,
                  blender_path: str = "blender",
                  bg_color: tuple = (1.0, 1.0, 1.0),
                  ref_mesh_path: str | None = None) -> list[np.ndarray]:
    """Render 3 optimal-coverage views of a mesh via Blender Cycles."""
    return _render_views(mesh_path, _THREE_VIEWS, resolution, blender_path,
                         bg_color, ref_mesh_path=ref_mesh_path)


def render_4views(mesh_path: str, resolution: int = 512,
                  blender_path: str = "blender",
                  bg_color: tuple = (1.0, 1.0, 1.0),
                  ref_mesh_path: str | None = None) -> list[np.ndarray]:
    """Render 4 orthogonal views of a mesh via Blender Cycles."""
    return _render_views(mesh_path, _FOUR_VIEWS, resolution, blender_path,
                         bg_color, ref_mesh_path=ref_mesh_path)


# ---------------------------------------------------------------------------
# Image composition
# ---------------------------------------------------------------------------

from _vis_common import make_text_bar, make_label_bar  # noqa: E402


def compose_comparison(before_views: list[np.ndarray],
                       after_views: list[np.ndarray],
                       prompt: str, edit_type: str = "") -> np.ndarray:
    """Compose a comparison image:
        [prompt bar]
        [Before label] [view0] [view1] [view2] [view3]
        [After  label] [view0] [view1] [view2] [view3]
    """
    h, w, _ = before_views[0].shape
    sep_w = 2  # separator width between views

    # Build rows of 4 views
    def make_row(views):
        sep = np.full((h, sep_w, 3), 220, dtype=np.uint8)
        parts = []
        for i, v in enumerate(views):
            if i > 0:
                parts.append(sep)
            parts.append(v)
        return np.hstack(parts)

    before_row = make_row(before_views)
    after_row = make_row(after_views)
    row_w = before_row.shape[1]

    # Add row labels on left
    label_w = 70
    def make_side_label(text, row_h):
        label = np.full((row_h, label_w, 3), 240, dtype=np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX
        tw = cv2.getTextSize(text, font, 0.55, 2)[0]
        x = (label_w - tw[0]) // 2
        y = (row_h + tw[1]) // 2
        cv2.putText(label, text, (x, y), font, 0.55, (40, 40, 40), 2, cv2.LINE_AA)
        return label

    before_labeled = np.hstack([make_side_label("Before", h), before_row])
    after_labeled = np.hstack([make_side_label("After", h), after_row])
    total_w = before_labeled.shape[1]

    # Horizontal separator between before/after
    h_sep = np.full((2, total_w, 3), 180, dtype=np.uint8)

    # Prompt bar
    tag = f"[{edit_type.upper()}] " if edit_type else ""
    prompt_bar = make_text_bar(f"{tag}{prompt}", total_w)

    return np.vstack([prompt_bar, before_labeled, h_sep, after_labeled])


def compose_after_only(after_views: list[np.ndarray],
                       prompt: str, edit_type: str = "") -> np.ndarray:
    """Compose image for after-only (no before mesh)."""
    h, w, _ = after_views[0].shape
    sep_w = 2
    parts = []
    for i, v in enumerate(after_views):
        if i > 0:
            parts.append(np.full((h, sep_w, 3), 220, dtype=np.uint8))
        parts.append(v)
    row = np.hstack(parts)
    total_w = row.shape[1]
    tag = f"[{edit_type.upper()}] " if edit_type else ""
    prompt_bar = make_text_bar(f"{tag}{prompt}", total_w)
    return np.vstack([prompt_bar, row])


# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------

def load_edit_prompts_from_results(cache_dir: Path, tag: str = "") -> dict[str, dict]:
    tag_suffix = f"_{tag}" if tag else ""
    prompts = {}
    candidates = []
    base = cache_dir / f"edit_results{tag_suffix}.jsonl"
    if base.exists():
        candidates.append(base)
    worker_pattern = str(cache_dir / f"edit_results{tag_suffix}_w*.jsonl")
    candidates.extend(Path(p) for p in sorted(_glob.glob(worker_pattern)))
    if not candidates:
        fallback = cache_dir / "edit_results.jsonl"
        if fallback.exists():
            candidates.append(fallback)

    for path in candidates:
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                    if rec.get("status") == "success":
                        prompts[rec["edit_id"]] = {
                            "edit_prompt": rec.get("edit_prompt", ""),
                            "edit_type": rec.get("edit_type",
                                                 rec.get("effective_edit_type", "")),
                            "obj_id": rec.get("obj_id", ""),
                        }
                except (json.JSONDecodeError, KeyError):
                    pass
    return prompts


def load_edit_prompts_from_pairs_dir(pairs_dir: Path) -> dict[str, dict]:
    prompts = {}
    for d in pairs_dir.iterdir():
        if not d.is_dir():
            continue
        meta = d / "metadata.json"
        if meta.exists():
            try:
                rec = json.loads(meta.read_text())
                prompts[d.name] = {
                    "edit_prompt": rec.get("edit_prompt", ""),
                    "edit_type": rec.get("edit_type", ""),
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
        description="Render before/after 4-view comparison images (Blender Cycles)")
    parser.add_argument("--config", type=str, default=None,
                        help="Pipeline config YAML (optional if --pairs-dir given)")
    parser.add_argument("--tag", type=str, default=None,
                        help="Experiment tag (e.g. test3)")
    parser.add_argument("--pairs-dir", type=str, default=None,
                        help="Override pairs directory containing edit_id subdirs")
    parser.add_argument("--edit-ids", nargs="*", default=None,
                        help="Specific edit IDs to render")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for images")
    parser.add_argument("--resolution", type=int, default=512,
                        help="Render resolution per view (default: 512)")
    parser.add_argument("--blender", type=str,
                        default="/home/artgen/software/blender-3.3.1-linux-x64/blender",
                        help="Path to Blender executable")
    parser.add_argument("--force", action="store_true",
                        help="Re-render even if output exists")
    args = parser.parse_args()

    # Resolve pairs directory
    if args.pairs_dir:
        pairs_dir = Path(args.pairs_dir)
        output_base = pairs_dir.parent
    elif args.config:
        from partcraft.utils.config import load_config
        cfg = load_config(args.config)
        output_base = Path(cfg["data"]["output_dir"])
        tag_suffix = f"_{args.tag}" if args.tag else ""
        pairs_dir = output_base / f"mesh_pairs{tag_suffix}"
    else:
        parser.error("Provide --config or --pairs-dir")
        return

    if not pairs_dir.exists():
        print(f"ERROR: Pairs directory not found: {pairs_dir}")
        sys.exit(1)

    # Output directory
    if args.output_dir:
        vis_dir = Path(args.output_dir)
    else:
        tag_suffix = f"_{args.tag}" if args.tag else ""
        vis_dir = output_base / f"vis_ply{tag_suffix}"
    vis_dir.mkdir(parents=True, exist_ok=True)

    # Discover valid pairs
    if args.edit_ids:
        candidate_dirs = [pairs_dir / eid for eid in args.edit_ids]
    else:
        candidate_dirs = sorted(d for d in pairs_dir.iterdir() if d.is_dir())

    valid_pairs = []
    for d in candidate_dirs:
        before_ply = d / "before.ply"
        after_ply = d / "after.ply"
        if before_ply.exists() and after_ply.exists():
            valid_pairs.append(d)
        elif after_ply.exists():
            valid_pairs.append(d)
        else:
            print(f"  SKIP {d.name}: no PLY files found")

    if not valid_pairs:
        print("ERROR: No valid pairs found with PLY files")
        sys.exit(1)

    # Load edit prompts
    edit_prompts: dict[str, dict] = {}
    if args.config:
        from partcraft.utils.config import load_config
        cfg = load_config(args.config)
        raw_cache = (cfg.get("services") or {}).get("image_edit", {}).get("cache_dir", "cache/phase2_5")
        cache_dir = Path(raw_cache) if Path(raw_cache).is_absolute() else Path(cfg["data"]["output_dir"]) / raw_cache
        edit_prompts = load_edit_prompts_from_results(cache_dir, args.tag or "")

    if not edit_prompts:
        edit_prompts = load_edit_prompts_from_pairs_dir(pairs_dir)

    print(f"Rendering {len(valid_pairs)} pairs -> {vis_dir}")
    print(f"  resolution={args.resolution}, views=4, renderer=Blender Cycles")
    print()

    for idx, pair_dir in enumerate(valid_pairs):
        edit_id = pair_dir.name
        out_path = vis_dir / f"{edit_id}.png"

        if not args.force and out_path.exists():
            print(f"[{idx+1}/{len(valid_pairs)}] {edit_id}: cached, skipping")
            continue

        info = edit_prompts.get(edit_id, {})
        prompt = info.get("edit_prompt", edit_id)
        edit_type = info.get("edit_type", "")

        before_ply = pair_dir / "before.ply"
        after_ply = pair_dir / "after.ply"
        has_before = before_ply.exists()

        print(f"[{idx+1}/{len(valid_pairs)}] {edit_id} [{edit_type}]")
        print(f"  prompt: {prompt[:100]}")

        try:
            print(f"  rendering after...")
            after_views = render_4views(
                str(after_ply), args.resolution, blender_path=args.blender)

            if has_before:
                print(f"  rendering before...")
                before_views = render_4views(
                    str(before_ply), args.resolution, blender_path=args.blender)
            else:
                before_views = None
        except (RuntimeError, FileNotFoundError) as e:
            print(f"  ERROR: {e}")
            print()
            continue

        # Compose and save
        if before_views is not None:
            comp = compose_comparison(before_views, after_views, prompt, edit_type)
        else:
            comp = compose_after_only(after_views, prompt, edit_type)

        cv2.imwrite(str(out_path), cv2.cvtColor(comp, cv2.COLOR_RGB2BGR))
        print(f"  -> {out_path}")
        print()

    print(f"Done! {len(valid_pairs)} pairs rendered -> {vis_dir}")


if __name__ == "__main__":
    main()
