#!/usr/bin/env python3
"""Match PartObjaverse-Tiny objects to their original textured Objaverse sources.

This script:
1. Reads PartObjaverse-Tiny_semantic.json to get all 200 UIDs
2. Uses objaverse API to resolve download URLs for textured GLBs
3. Reports which objects are available and which are missing
4. Optionally downloads the textured GLBs

Usage (vinedresser3d env):
    # Check availability only (no download)
    python scripts/match_tiny_to_objaverse.py --check-only

    # Download all textured GLBs
    python scripts/match_tiny_to_objaverse.py --download

    # Download first N objects
    python scripts/match_tiny_to_objaverse.py --download --limit 10

    # Specify output directory
    python scripts/match_tiny_to_objaverse.py --download --output data/objaverse_glbs
"""
import argparse
import json
import os
import sys
from pathlib import Path
from collections import defaultdict


def load_tiny_uids(semantic_json_path: str) -> dict[str, tuple[str, list[str]]]:
    """Load PartObjaverse-Tiny semantic.json, return {uid: (category, labels)}."""
    with open(semantic_json_path) as f:
        data = json.load(f)
    uid_map = {}
    for category, objects in data.items():
        for uid, labels in objects.items():
            uid_map[uid] = (category, labels)
    return uid_map


def check_objaverse_availability(uids: list[str]) -> dict:
    """Check which UIDs are available in Objaverse and get their paths."""
    import objaverse

    print(f"Querying Objaverse for {len(uids)} objects...")

    # objaverse.load_objects returns {uid: local_path} for downloaded objects
    # First, let's check the annotations/metadata to see if UIDs exist
    annotations = objaverse.load_annotations(uids)
    print(f"  Found annotations for {len(annotations)} / {len(uids)} objects")

    return annotations


def download_glbs(uids: list[str], output_dir: str) -> dict[str, str]:
    """Download original textured GLBs from Objaverse."""
    import objaverse

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Check which are already downloaded
    existing = {}
    missing_uids = []
    for uid in uids:
        local_path = out / f"{uid}.glb"
        if local_path.exists() and local_path.stat().st_size > 0:
            existing[uid] = str(local_path)
        else:
            missing_uids.append(uid)

    if existing:
        print(f"  Already downloaded: {len(existing)}")

    if not missing_uids:
        print("  All objects already downloaded!")
        return existing

    print(f"  Downloading {len(missing_uids)} GLBs from Objaverse...")
    downloaded = objaverse.load_objects(uids=missing_uids)

    # Copy/symlink to output directory with consistent naming
    for uid, src_path in downloaded.items():
        dst_path = out / f"{uid}.glb"
        if not dst_path.exists():
            try:
                os.symlink(os.path.abspath(src_path), str(dst_path))
            except OSError:
                import shutil
                shutil.copy2(src_path, str(dst_path))
        existing[uid] = str(dst_path)

    print(f"  Total available: {len(existing)} / {len(uids)}")
    return existing


def verify_glb_has_texture(glb_path: str) -> dict:
    """Check if a GLB file contains textures/colors."""
    import trimesh

    try:
        scene = trimesh.load(glb_path)
        info = {
            "exists": True,
            "file_size_mb": os.path.getsize(glb_path) / (1024 * 1024),
        }

        if isinstance(scene, trimesh.Scene):
            geoms = list(scene.geometry.values())
            info["num_geometries"] = len(geoms)
            info["total_faces"] = sum(len(g.faces) for g in geoms if hasattr(g, "faces"))
            info["total_vertices"] = sum(len(g.vertices) for g in geoms if hasattr(g, "vertices"))

            has_texture = False
            has_vertex_color = False
            for g in geoms:
                if not hasattr(g, "visual"):
                    continue
                if g.visual.kind == "texture":
                    mat = g.visual.material
                    if hasattr(mat, "baseColorTexture") and mat.baseColorTexture is not None:
                        has_texture = True
                    elif hasattr(mat, "image") and mat.image is not None:
                        has_texture = True
                elif g.visual.kind == "vertex":
                    has_vertex_color = True

            info["has_texture"] = has_texture
            info["has_vertex_color"] = has_vertex_color
            info["has_color"] = has_texture or has_vertex_color
        else:
            info["num_geometries"] = 1
            info["total_faces"] = len(scene.faces)
            info["total_vertices"] = len(scene.vertices)
            info["has_texture"] = scene.visual.kind == "texture"
            info["has_vertex_color"] = scene.visual.kind == "vertex"
            info["has_color"] = info["has_texture"] or info["has_vertex_color"]

        return info
    except Exception as e:
        return {"exists": True, "error": str(e), "has_color": False}


def main():
    parser = argparse.ArgumentParser(
        description="Match PartObjaverse-Tiny to original textured Objaverse GLBs")
    parser.add_argument("--semantic-json",
                        default=os.environ.get(
                            "PARTOBJAVERSE_SEMANTIC_JSON",
                            "data/partobjaverse_tiny/source/semantic.json"),
                        help="Path to PartObjaverse-Tiny_semantic.json")
    parser.add_argument("--output", default="data/objaverse_glbs",
                        help="Output directory for downloaded GLBs")
    parser.add_argument("--check-only", action="store_true",
                        help="Only check availability, don't download")
    parser.add_argument("--download", action="store_true",
                        help="Download GLBs from Objaverse")
    parser.add_argument("--verify", action="store_true",
                        help="Verify downloaded GLBs have textures")
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit to first N objects (0=all)")
    parser.add_argument("--save-mapping", default="data/partobjaverse_tiny/tiny_to_objaverse_mapping.json",
                        help="Save UID mapping to JSON")
    args = parser.parse_args()

    # Step 1: Load all Tiny UIDs
    print("=" * 60)
    print("Step 1: Loading PartObjaverse-Tiny UIDs")
    print("=" * 60)
    uid_map = load_tiny_uids(args.semantic_json)
    all_uids = sorted(uid_map.keys())
    print(f"  Total: {len(all_uids)} objects across {len(set(c for c,_ in uid_map.values()))} categories")

    # Category breakdown
    cat_counts = defaultdict(int)
    for uid, (cat, _) in uid_map.items():
        cat_counts[cat] += 1
    for cat, cnt in sorted(cat_counts.items()):
        print(f"    {cat}: {cnt}")

    if args.limit > 0:
        all_uids = all_uids[:args.limit]
        print(f"  Limited to: {len(all_uids)} objects")

    # Step 2: Check / Download
    if args.download:
        print("\n" + "=" * 60)
        print("Step 2: Downloading textured GLBs from Objaverse")
        print("=" * 60)
        glb_paths = download_glbs(all_uids, args.output)
    elif args.check_only:
        print("\n" + "=" * 60)
        print("Step 2: Checking Objaverse availability")
        print("=" * 60)
        annotations = check_objaverse_availability(all_uids)

        found = [uid for uid in all_uids if uid in annotations]
        missing = [uid for uid in all_uids if uid not in annotations]
        print(f"  Available: {len(found)} / {len(all_uids)}")
        if missing:
            print(f"  Missing ({len(missing)}):")
            for uid in missing[:10]:
                cat, labels = uid_map[uid]
                print(f"    {uid} ({cat})")
            if len(missing) > 10:
                print(f"    ... and {len(missing) - 10} more")
        glb_paths = {}
    else:
        # Just check if GLBs exist locally
        out = Path(args.output)
        glb_paths = {}
        for uid in all_uids:
            p = out / f"{uid}.glb"
            if p.exists():
                glb_paths[uid] = str(p)
        print(f"\n  Local GLBs found: {len(glb_paths)} / {len(all_uids)}")

    # Step 3: Verify textures
    if args.verify and glb_paths:
        print("\n" + "=" * 60)
        print("Step 3: Verifying textures in downloaded GLBs")
        print("=" * 60)

        stats = {"textured": 0, "vertex_color": 0, "no_color": 0, "error": 0}
        results = {}

        for i, (uid, path) in enumerate(sorted(glb_paths.items())):
            cat, labels = uid_map[uid]
            info = verify_glb_has_texture(path)
            results[uid] = info

            status = "?"
            if "error" in info:
                stats["error"] += 1
                status = "ERROR"
            elif info.get("has_texture"):
                stats["textured"] += 1
                status = "TEXTURED"
            elif info.get("has_vertex_color"):
                stats["vertex_color"] += 1
                status = "VERTEX_COLOR"
            else:
                stats["no_color"] += 1
                status = "NO_COLOR"

            if (i + 1) % 20 == 0 or status in ("NO_COLOR", "ERROR"):
                print(f"  [{i+1}/{len(glb_paths)}] {uid} ({cat}): {status}"
                      f"  faces={info.get('total_faces', '?')}"
                      f"  size={info.get('file_size_mb', 0):.1f}MB")

        print(f"\n  Summary:")
        print(f"    Textured (PBR):    {stats['textured']}")
        print(f"    Vertex colors:     {stats['vertex_color']}")
        print(f"    No color:          {stats['no_color']}")
        print(f"    Errors:            {stats['error']}")
        print(f"    Total with color:  {stats['textured'] + stats['vertex_color']} / {len(glb_paths)}")

    # Step 4: Save mapping
    mapping = {
        "total_tiny_objects": len(uid_map),
        "categories": dict(cat_counts),
        "objects": {},
    }
    for uid in all_uids:
        cat, labels = uid_map[uid]
        entry = {
            "category": cat,
            "labels": labels,
            "num_labels": len(labels),
        }
        if uid in glb_paths:
            entry["objaverse_glb"] = glb_paths[uid]
            entry["has_glb"] = True
        else:
            entry["has_glb"] = False
        mapping["objects"][uid] = entry

    mapping["matched"] = sum(1 for e in mapping["objects"].values() if e["has_glb"])
    mapping["unmatched"] = len(mapping["objects"]) - mapping["matched"]

    Path(args.save_mapping).parent.mkdir(parents=True, exist_ok=True)
    with open(args.save_mapping, "w") as f:
        json.dump(mapping, f, indent=2)
    print(f"\n  Mapping saved: {args.save_mapping}")
    print(f"  Matched: {mapping['matched']} / {len(mapping['objects'])}")


if __name__ == "__main__":
    main()
