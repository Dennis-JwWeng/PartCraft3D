#!/usr/bin/env python3
"""Repack flat ``mesh_pairs/{edit_id}/`` output into object-centric directories.

Reads the pipeline's flat edit-pair layout and produces a training-ready
structure grouped by object:

    {output_dir}/
      shard_{SS}/
        {obj_id}/
          original.npz        # shared before state (slat_feats, slat_coords, ss)
          mod_000.npz          # after state for modification edit 0
          scl_000.npz          # after state for scale edit 0
          del_000.npz          # after state for deletion edit 0
          ...
          metadata.json        # object + all edit metadata
      manifest.jsonl           # global flat index (one line per edit)

Addition and identity edits produce *no* NPZ file — they are fully
described by references in ``metadata.json`` (addition swaps the paired
deletion's before/after, identity maps original → original).

Usage
-----
Single shard:

    python scripts/tools/repack_to_object_dirs.py \\
        --mesh-pairs /data/outputs/shard_00/mesh_pairs_shard00 \\
        --specs-jsonl /data/outputs/shard_00/cache/phase1/edit_specs_shard00.jsonl \\
        --output-dir /data/partverse_pairs \\
        --shard 00

Dry run (count only):

    python scripts/tools/repack_to_object_dirs.py \\
        --mesh-pairs /data/outputs/shard_00/mesh_pairs_shard00 \\
        --specs-jsonl /data/outputs/shard_00/cache/phase1/edit_specs_shard00.jsonl \\
        --output-dir /data/partverse_pairs \\
        --shard 00 --dry-run

Generate manifest only (after all shards are repacked):

    python scripts/tools/repack_to_object_dirs.py \\
        --output-dir /data/partverse_pairs \\
        --manifest-only
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("repack")

# Edit type prefixes used in edit_id naming (same as partcraft.edit_types.ID_PREFIX)
_PREFIX_TO_TYPE = {
    "del": "deletion",
    "add": "addition",
    "mod": "modification",
    "scl": "scale",
    "mat": "material",
    "glb": "global",
    "idt": "identity",
}

_TYPE_TO_PREFIX = {v: k for k, v in _PREFIX_TO_TYPE.items()}

# Types that produce a physical after-state NPZ file
_FILE_TYPES = {"modification", "scale", "material", "global", "deletion"}


def _link_or_copy(src: Path, dst: Path) -> None:
    """Hard-link *src* to *dst*, falling back to copy on cross-device."""
    try:
        os.link(str(src), str(dst))
    except OSError:
        shutil.copy2(str(src), str(dst))


# ─────────────────────────── spec loading ─────────────────────────────────

def _load_specs(path: Path) -> list[dict]:
    specs: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            specs.append(json.loads(line))
    return specs


def _group_by_object(specs: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for s in specs:
        groups[s["obj_id"]].append(s)
    return groups


# ─────────────────────────── per-object repack ────────────────────────────

def _build_edit_entry(spec: dict, seq: int, prefix: str) -> dict:
    """Build a metadata.json edit entry from a pipeline EditSpec record."""
    etype = spec["edit_type"]
    entry: dict = {
        "edit_id": spec["edit_id"],
        "type": etype,
        "seq": seq,
        "prompt": spec.get("edit_prompt", ""),
        "after_desc": spec.get("after_desc", ""),
    }

    if etype in _FILE_TYPES:
        entry["file"] = f"{prefix}_{seq:03d}.npz"
    else:
        entry["file"] = None

    if etype == "deletion":
        entry["remove_part_ids"] = spec.get("remove_part_ids", [])
        entry["remove_labels"] = spec.get("remove_labels", [])
    elif etype == "addition":
        source_del_id = spec.get("source_del_id", "")
        entry["source_del_id"] = source_del_id
        # source_del_seq will be resolved after all deletion seqs are assigned
    elif etype == "modification":
        entry["mod_type"] = spec.get("mod_type", "")
        entry["target_part_ids"] = (
            [spec["old_part_id"]] if spec.get("old_part_id", -1) >= 0 else []
        )
        entry["target_part_labels"] = (
            [spec["old_label"]] if spec.get("old_label") else []
        )
    elif etype == "scale":
        entry["target_part_ids"] = (
            [spec["old_part_id"]] if spec.get("old_part_id", -1) >= 0 else []
        )
        entry["target_part_labels"] = (
            [spec["old_label"]] if spec.get("old_label") else []
        )
    elif etype == "material":
        entry["target_part_ids"] = (
            [spec["old_part_id"]] if spec.get("old_part_id", -1) >= 0 else []
        )
    elif etype == "global":
        pass  # no extra fields
    elif etype == "identity":
        pass

    return entry


def repack_object(
    obj_id: str,
    shard: str,
    specs: list[dict],
    mesh_pairs: Path,
    obj_dir: Path,
    *,
    dry_run: bool,
) -> dict:
    """Repack all edits for one object. Returns stats dict."""
    stats = {"files_linked": 0, "files_skipped": 0, "files_missing": 0,
             "edits_total": 0}

    # Assign per-type sequence numbers
    type_seq: dict[str, int] = defaultdict(int)
    entries: list[dict] = []
    del_id_to_seq: dict[str, int] = {}

    for spec in specs:
        etype = spec["edit_type"]
        prefix = _TYPE_TO_PREFIX.get(etype, etype[:3])
        seq = type_seq[etype]
        type_seq[etype] += 1

        entry = _build_edit_entry(spec, seq, prefix)
        entries.append(entry)

        if etype == "deletion":
            del_id_to_seq[spec["edit_id"]] = seq

    # Resolve addition source_del_seq references
    for entry in entries:
        if entry["type"] == "addition":
            src_del_id = entry.pop("source_del_id", "")
            entry["source_del_seq"] = del_id_to_seq.get(src_del_id, -1)

    stats["edits_total"] = len(entries)

    if dry_run:
        for e in entries:
            if e["file"] is not None:
                stats["files_linked"] += 1
        return stats

    obj_dir.mkdir(parents=True, exist_ok=True)

    # ── original.npz ──
    original_dst = obj_dir / "original.npz"
    if not original_dst.exists():
        placed = False
        for spec in specs:
            src = mesh_pairs / spec["edit_id"] / "before.npz"
            if src.exists():
                _link_or_copy(src, original_dst)
                placed = True
                break
        if not placed:
            stats["files_missing"] += 1

    # ── per-edit after NPZ ──
    for entry, spec in zip(entries, specs):
        fname = entry.get("file")
        if fname is None:
            continue
        dst = obj_dir / fname
        if dst.exists():
            stats["files_skipped"] += 1
            continue
        src = mesh_pairs / spec["edit_id"] / "after.npz"
        if src.exists():
            _link_or_copy(src, dst)
            stats["files_linked"] += 1
        else:
            stats["files_missing"] += 1

    # ── metadata.json ──
    object_desc = ""
    for spec in specs:
        if spec.get("object_desc"):
            object_desc = spec["object_desc"]
            break

    meta = {
        "obj_id": obj_id,
        "shard": shard,
        "object_desc": object_desc,
        "num_edits": len(entries),
        "edits": entries,
    }
    meta_path = obj_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return stats


# ─────────────────────────── manifest generation ──────────────────────────

def generate_manifest(output_dir: Path) -> int:
    """Scan all shard dirs and write a single ``manifest.jsonl``."""
    manifest_path = output_dir / "manifest.jsonl"
    count = 0
    with open(manifest_path, "w") as out:
        for shard_dir in sorted(output_dir.iterdir()):
            if not shard_dir.is_dir() or not shard_dir.name.startswith("shard_"):
                continue
            shard = shard_dir.name.replace("shard_", "")
            for obj_dir in sorted(shard_dir.iterdir()):
                if not obj_dir.is_dir():
                    continue
                meta_path = obj_dir / "metadata.json"
                if not meta_path.exists():
                    continue
                with open(meta_path) as f:
                    meta = json.load(f)
                obj_id = meta["obj_id"]
                for idx, edit in enumerate(meta["edits"]):
                    record = {
                        "shard": shard,
                        "obj_id": obj_id,
                        "edit_idx": idx,
                        "type": edit["type"],
                        "file": edit.get("file"),
                        "prompt": edit.get("prompt", ""),
                    }
                    if edit["type"] == "addition":
                        record["source_del_seq"] = edit.get("source_del_seq", -1)
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    count += 1
    return count


# ─────────────────────────── main ─────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Repack flat mesh_pairs to object-centric directories",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--mesh-pairs", type=str, default=None,
                        help="Root mesh_pairs directory to read from")
    parser.add_argument("--specs-jsonl", type=str, default=None,
                        help="Path to edit_specs JSONL file")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Root output directory for repacked data")
    parser.add_argument("--shard", type=str, default=None,
                        help="Shard identifier (e.g. 00)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Count only, do not create files")
    parser.add_argument("--manifest-only", action="store_true",
                        help="Only regenerate manifest.jsonl from existing metadata")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    if args.manifest_only:
        log.info("Generating manifest.jsonl from %s ...", output_dir)
        n = generate_manifest(output_dir)
        log.info("Manifest written: %d entries", n)
        return

    if not args.mesh_pairs or not args.specs_jsonl or not args.shard:
        parser.error("--mesh-pairs, --specs-jsonl, and --shard are required "
                      "(unless --manifest-only)")

    mesh_pairs = Path(args.mesh_pairs)
    if not mesh_pairs.is_dir():
        log.error("mesh_pairs directory not found: %s", mesh_pairs)
        sys.exit(1)

    specs_path = Path(args.specs_jsonl)
    if not specs_path.exists():
        log.error("specs file not found: %s", specs_path)
        sys.exit(1)

    shard = args.shard
    shard_dir = output_dir / f"shard_{shard}"

    log.info("Loading specs from %s ...", specs_path)
    all_specs = _load_specs(specs_path)
    obj_groups = _group_by_object(all_specs)
    log.info("Loaded %d specs across %d objects (shard %s)",
             len(all_specs), len(obj_groups), shard)

    total_stats = defaultdict(int)
    for i, (obj_id, specs) in enumerate(obj_groups.items()):
        obj_out = shard_dir / obj_id
        s = repack_object(
            obj_id, shard, specs, mesh_pairs, obj_out, dry_run=args.dry_run,
        )
        for k, v in s.items():
            total_stats[k] += v
        if (i + 1) % 200 == 0:
            log.info("  Progress: %d / %d objects  (linked %d, skipped %d, missing %d)",
                     i + 1, len(obj_groups),
                     total_stats["files_linked"],
                     total_stats["files_skipped"],
                     total_stats["files_missing"])

    log.info("Shard %s repack complete: %s", shard, dict(total_stats))

    if not args.dry_run:
        log.info("Generating manifest.jsonl ...")
        n = generate_manifest(output_dir)
        log.info("Manifest written: %d entries", n)

    log.info("Done.")


if __name__ == "__main__":
    main()
