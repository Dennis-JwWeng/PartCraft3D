#!/usr/bin/env python3
"""Download original textured Objaverse GLBs for PartObjaverse-Tiny objects.

Uses the local object-paths.json.gz (from objaverse cache) to resolve the
correct HuggingFace shard paths, then downloads via huggingface_hub.

Usage (vinedresser3d env):
    # Download all missing GLBs
    python scripts/download_objaverse_glbs.py

    # Limit downloads
    python scripts/download_objaverse_glbs.py --limit 10

    # Verify textures after download
    python scripts/download_objaverse_glbs.py --verify

    # Just check status, no download
    python scripts/download_objaverse_glbs.py --check-only
"""
import argparse
import gzip
import json
import os
import sys
from pathlib import Path


OBJAVERSE_CACHE = Path.home() / ".objaverse" / "hf-objaverse-v1"
OBJECT_PATHS_FILE = OBJAVERSE_CACHE / "object-paths.json.gz"
SEMANTIC_JSON = os.environ.get(
    "PARTOBJAVERSE_SEMANTIC_JSON",
    "/Node11_nvme/wjw/3D_Editing/SAMPart3D/PartObjaverse-Tiny/PartObjaverse-Tiny_semantic.json"
)


def load_object_paths() -> dict[str, str]:
    """Load uid -> hf_relative_path mapping from objaverse cache."""
    if not OBJECT_PATHS_FILE.exists():
        print(f"ERROR: {OBJECT_PATHS_FILE} not found.")
        print("Run: python -c \"import objaverse; objaverse.load_annotations()\" first.")
        sys.exit(1)
    with gzip.open(OBJECT_PATHS_FILE, "rb") as f:
        return json.load(f)


def load_tiny_uids() -> dict[str, tuple[str, list[str]]]:
    """Load {uid: (category, labels)} from PartObjaverse-Tiny."""
    with open(SEMANTIC_JSON) as f:
        data = json.load(f)
    uid_map = {}
    for cat, objs in data.items():
        for uid, labels in objs.items():
            uid_map[uid] = (cat, labels)
    return uid_map


def find_local_glb(uid: str, obj_paths: dict[str, str]) -> str | None:
    """Check if GLB exists locally in objaverse cache."""
    if uid not in obj_paths:
        return None
    rel_path = obj_paths[uid]
    local = OBJAVERSE_CACHE / rel_path
    if local.exists() and local.stat().st_size > 0:
        return str(local)
    return None


def download_via_hf_hub(uid: str, rel_path: str) -> str | None:
    """Download a single GLB using huggingface_hub."""
    from huggingface_hub import hf_hub_download
    try:
        local = hf_hub_download(
            repo_id="allenai/objaverse",
            filename=rel_path,
            repo_type="dataset",
        )
        return local
    except Exception as e:
        print(f"    FAILED {uid}: {e}")
        return None


def verify_glb_texture(glb_path: str) -> dict:
    """Check if GLB has textures/colors."""
    import trimesh
    try:
        scene = trimesh.load(glb_path)
        info = {"size_mb": os.path.getsize(glb_path) / (1024 * 1024)}
        geoms = []
        if isinstance(scene, trimesh.Scene):
            geoms = [g for g in scene.geometry.values() if hasattr(g, "visual")]
            info["faces"] = sum(len(g.faces) for g in geoms if hasattr(g, "faces"))
        else:
            geoms = [scene]
            info["faces"] = len(scene.faces)
        info["has_texture"] = any(g.visual.kind == "texture" for g in geoms)
        info["has_vertex_color"] = any(g.visual.kind == "vertex" for g in geoms)
        info["has_color"] = info["has_texture"] or info["has_vertex_color"]
        return info
    except Exception as e:
        return {"error": str(e), "has_color": False}


def main():
    parser = argparse.ArgumentParser(
        description="Download textured Objaverse GLBs for PartObjaverse-Tiny")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--save-mapping", default="data/partobjaverse_tiny/tiny_to_objaverse_mapping.json")
    args = parser.parse_args()

    # Step 1: Load mappings
    print("Loading object paths and Tiny UIDs...")
    obj_paths = load_object_paths()
    uid_map = load_tiny_uids()
    all_uids = sorted(uid_map.keys())
    print(f"  Objaverse total: {len(obj_paths)}")
    print(f"  Tiny objects: {len(all_uids)}")

    # Step 2: Check availability
    resolved = {}  # uid -> local_path
    need_download = []
    not_in_objaverse = []

    for uid in all_uids:
        if uid not in obj_paths:
            not_in_objaverse.append(uid)
            continue
        local = find_local_glb(uid, obj_paths)
        if local:
            resolved[uid] = local
        else:
            need_download.append(uid)

    print(f"\n  Already cached locally: {len(resolved)}")
    print(f"  Need download: {len(need_download)}")
    if not_in_objaverse:
        print(f"  Not in Objaverse: {len(not_in_objaverse)}")

    if args.check_only:
        # Print category breakdown
        from collections import Counter
        cat_cached = Counter()
        cat_total = Counter()
        for uid in all_uids:
            cat = uid_map[uid][0]
            cat_total[cat] += 1
            if uid in resolved:
                cat_cached[cat] += 1
        print(f"\n  Per-category:")
        for cat in sorted(cat_total.keys()):
            print(f"    {cat:25s}: {cat_cached.get(cat,0):3d} / {cat_total[cat]:3d}")
        return

    # Step 3: Download
    if need_download:
        if args.limit > 0:
            need_download = need_download[:args.limit]
        print(f"\nDownloading {len(need_download)} GLBs from HuggingFace...")

        for i, uid in enumerate(need_download):
            rel_path = obj_paths[uid]
            local = download_via_hf_hub(uid, rel_path)
            if local:
                resolved[uid] = local
            if (i + 1) % 10 == 0:
                print(f"    [{i+1}/{len(need_download)}] downloaded")

        print(f"\n  Total available: {len(resolved)} / {len(all_uids)}")

    # Step 4: Save mapping
    mapping = {
        "total": len(all_uids),
        "matched": len(resolved),
        "missing": len(all_uids) - len(resolved),
        "objaverse_cache": str(OBJAVERSE_CACHE),
        "objects": {},
    }
    for uid in all_uids:
        cat, labels = uid_map[uid]
        entry = {
            "category": cat,
            "labels": labels,
            "num_labels": len(labels),
        }
        if uid in resolved:
            entry["objaverse_glb"] = resolved[uid]
            entry["hf_path"] = obj_paths.get(uid, ""),
            entry["status"] = "matched"
        else:
            entry["objaverse_glb"] = None
            entry["status"] = "missing"
        mapping["objects"][uid] = entry

    Path(args.save_mapping).parent.mkdir(parents=True, exist_ok=True)
    with open(args.save_mapping, "w") as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)
    print(f"\n  Mapping saved: {args.save_mapping}")

    # Step 5: Verify
    if args.verify and resolved:
        print(f"\nVerifying textures for {len(resolved)} GLBs...")
        stats = {"textured": 0, "vertex_color": 0, "no_color": 0, "error": 0}
        no_color_list = []
        for uid, path in sorted(resolved.items()):
            info = verify_glb_texture(path)
            if info.get("has_texture"):
                stats["textured"] += 1
            elif info.get("has_vertex_color"):
                stats["vertex_color"] += 1
            elif "error" in info:
                stats["error"] += 1
            else:
                stats["no_color"] += 1
                no_color_list.append((uid, uid_map[uid][0]))

        print(f"\n  Texture verification:")
        print(f"    PBR textured:   {stats['textured']}")
        print(f"    Vertex colors:  {stats['vertex_color']}")
        print(f"    No color:       {stats['no_color']}")
        print(f"    Errors:         {stats['error']}")
        if no_color_list:
            print(f"  Objects without color:")
            for uid, cat in no_color_list:
                print(f"    {uid} ({cat})")


if __name__ == "__main__":
    main()
