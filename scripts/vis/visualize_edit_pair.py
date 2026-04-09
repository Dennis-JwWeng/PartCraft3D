#!/usr/bin/env python3
"""Visualize before/after 3D edit pairs as side-by-side rotation videos.

Uses Open3D GPU offscreen rendering for fast headless rendering.

Usage:
    python scripts/visualize_edit_pair.py --limit 5
    python scripts/visualize_edit_pair.py --edit_id del_000000
    python scripts/visualize_edit_pair.py --limit 3 --fps 30 --frames 120
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import imageio
import numpy as np
import open3d as o3d
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))



def trimesh_to_o3d(mesh: trimesh.Trimesh) -> o3d.geometry.TriangleMesh:
    """Convert trimesh mesh to Open3D, preserving vertex colors."""
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)
    o3d_mesh.compute_vertex_normals()

    if mesh.visual.kind == "vertex" and mesh.visual.vertex_colors is not None:
        colors = mesh.visual.vertex_colors[:, :3].astype(np.float64) / 255.0
        o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    elif mesh.visual.kind == "face" and mesh.visual.face_colors is not None:
        # Convert face colors to vertex colors by averaging
        fc = mesh.visual.face_colors[:, :3].astype(np.float64) / 255.0
        vc = np.zeros((len(mesh.vertices), 3))
        counts = np.zeros(len(mesh.vertices))
        for i, face in enumerate(mesh.faces):
            for vi in face:
                vc[vi] += fc[i]
                counts[vi] += 1
        counts = np.maximum(counts, 1)
        vc /= counts[:, None]
        o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(vc)

    return o3d_mesh


def render_rotation(mesh_path: str, n_frames: int = 60, resolution: int = 512,
                    bg_color: tuple = (1.0, 1.0, 1.0)) -> list[np.ndarray]:
    """Render a mesh rotating 360 degrees using Open3D offscreen renderer."""
    # Load mesh
    tm = trimesh.load(mesh_path, process=False)
    o3d_mesh = trimesh_to_o3d(tm)

    # Normalize to unit sphere centered at origin
    center = o3d_mesh.get_center()
    o3d_mesh.translate(-center)
    bbox = o3d_mesh.get_axis_aligned_bounding_box()
    scale = 1.0 / max(bbox.get_max_bound() - bbox.get_min_bound())
    o3d_mesh.scale(scale, center=[0, 0, 0])

    # Setup renderer
    renderer = o3d.visualization.rendering.OffscreenRenderer(resolution, resolution)
    renderer.scene.set_background(np.array([*bg_color, 1.0]))

    # Material with vertex colors
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultLit"

    renderer.scene.add_geometry("mesh", o3d_mesh, mat)

    # Lighting
    renderer.scene.scene.set_sun_light([0.5, -0.5, -1.0], [1.0, 1.0, 1.0], 60000)
    renderer.scene.scene.enable_sun_light(True)

    # Camera distance
    cam_dist = 1.8

    images = []
    for i in range(n_frames):
        angle = 2 * np.pi * i / n_frames
        elev = 0.4  # slight elevation
        eye = [
            cam_dist * np.cos(angle) * np.cos(elev),
            cam_dist * np.sin(angle) * np.cos(elev),
            cam_dist * np.sin(elev),
        ]
        renderer.setup_camera(45.0, [0, 0, 0], eye, [0, 0, 1])
        img = np.asarray(renderer.render_to_image())
        images.append(img)

    del renderer
    return images


def build_prompt_text(entry: dict) -> str:
    """Build a human-readable edit instruction.

    For addition: strips the added part from object_desc so the prompt
    describes the 'before' state (without the part).
    """
    etype = entry["edit_type"]
    obj_desc = entry.get("object_desc", "")

    if etype == "deletion":
        parts = ", ".join(entry.get("remove_labels", []))
        return f"[DELETE] Remove {parts} from: {obj_desc}"
    elif etype == "addition":
        parts = ", ".join(entry.get("add_labels", []))
        before_desc = entry.get("before_desc", obj_desc)
        return f"[ADD] Add {parts} to: {before_desc}"
    elif etype == "modification":
        # Phase 2.5 entries use edit_prompt + old_part_label
        edit_prompt = entry.get("edit_prompt", "")
        if edit_prompt:
            return f"[MOD] {edit_prompt}"
        old = entry.get("old_label") or entry.get("old_part_label", "?")
        new = entry.get("new_label") or entry.get("after_part_desc", "?")
        return f"[MOD] Replace {old} with {new} on: {obj_desc}"
    else:
        return f"[{etype.upper()}] {obj_desc}"


from _vis_common import make_text_bar, make_label_bar  # noqa: E402


def add_text_bar(frame: np.ndarray, text: str, bar_height: int = 50) -> np.ndarray:
    """Add a prompt text bar on top of the frame."""
    _, w, _ = frame.shape
    bar = make_text_bar(text, w, bar_height=bar_height)
    return np.vstack([bar, frame])


def add_labels(before_frame: np.ndarray, after_frame: np.ndarray,
               label_height: int = 30) -> tuple[np.ndarray, np.ndarray]:
    """Add 'Before' / 'After' labels at the top."""
    _, w, _ = before_frame.shape
    b_bar = make_label_bar("Before", w, height=label_height)
    a_bar = make_label_bar("After", w, height=label_height)
    return np.vstack([b_bar, before_frame]), np.vstack([a_bar, after_frame])


def generate_video(entry: dict, output_path: str,
                   n_frames: int = 60, fps: int = 20, resolution: int = 512):
    """Generate a side-by-side rotation comparison video."""
    prompt = build_prompt_text(entry)
    print(f"  Rendering before mesh...")
    before_imgs = render_rotation(entry["before_mesh"], n_frames, resolution)
    print(f"  Rendering after mesh...")
    after_imgs = render_rotation(entry["after_mesh"], n_frames, resolution)

    # Compose video frames
    frames = []
    for bf, af in zip(before_imgs, after_imgs):
        bf, af = add_labels(bf, af)
        # Thin separator line
        sep = np.ones((bf.shape[0], 3, 3), dtype=np.uint8) * 180
        combined = np.hstack([bf, sep, af])
        combined = add_text_bar(combined, prompt)
        frames.append(combined)

    imageio.mimsave(output_path, frames, fps=fps, codec="libx264",
                    quality=8, pixelformat="yuv420p")


def main():
    parser = argparse.ArgumentParser(description="Visualize 3D edit pairs as rotation videos")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--manifest", type=str, default=None)
    parser.add_argument("--edit_id", type=str, default=None)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    from partcraft.utils.config import load_config
    cfg = load_config(args.config)

    # Collect manifest files: Phase 2 + Phase 2.5 (unless explicit --manifest)
    if args.manifest:
        manifest_paths = [args.manifest]
    else:
        manifest_paths = []
        p2 = Path(cfg["phase2"]["cache_dir"]) / "assembled_pairs.jsonl"
        if p2.exists():
            manifest_paths.append(str(p2))
        p25 = Path(cfg["data"]["output_dir"]) / (cfg.get("services") or {}).get("image_edit", {}).get("cache_dir", "cache/phase2_5") / "modification_pairs.jsonl"
        if p25.exists():
            manifest_paths.append(str(p25))
        if not manifest_paths:
            print("No manifest files found (assembled_pairs.jsonl or modification_pairs.jsonl)")
            return

    output_dir = Path(args.output_dir or str(
        Path(cfg["data"]["output_dir"]) / "vis_videos"))
    output_dir.mkdir(parents=True, exist_ok=True)

    entries = []
    seen_ids: set[str] = set()
    for manifest_path in manifest_paths:
        with open(manifest_path) as f:
            for line in f:
                if not line.strip():
                    continue
                entry = json.loads(line)
                if entry.get("status") == "failed":
                    continue
                eid = entry["edit_id"]
                if eid in seen_ids:
                    continue
                if args.edit_id and eid != args.edit_id:
                    continue
                seen_ids.add(eid)
                entries.append(entry)
                if not args.edit_id and len(entries) >= args.limit:
                    break
        if not args.edit_id and len(entries) >= args.limit:
            break

    if not entries:
        print(f"No entries found (edit_id={args.edit_id})")
        return

    print(f"Generating {len(entries)} comparison videos → {output_dir}")

    for entry in entries:
        eid = entry["edit_id"]
        out_path = str(output_dir / f"{eid}.mp4")
        print(f"\n[{eid}] {build_prompt_text(entry)}")
        generate_video(entry, out_path,
                       n_frames=args.frames, fps=args.fps,
                       resolution=args.resolution)
        print(f"  → {out_path}")

    print(f"\nDone! {len(entries)} videos saved to {output_dir}")


if __name__ == "__main__":
    main()
