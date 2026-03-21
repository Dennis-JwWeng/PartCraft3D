#!/usr/bin/env python3
"""Build training dataset JSON from streaming pipeline outputs.

Structure: per-object grouping, one shared SLAT as before,
deletion/addition as reverse pairs with auto-generated prompts.

Usage:
    python scripts/build_dataset.py --tag v2
    python scripts/build_dataset.py --tag v2 --fix-slat
"""

import argparse
import glob
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Reverse prompt generation (rule-based)
# ---------------------------------------------------------------------------

_REMOVE_PATTERNS = [
    (re.compile(r"^Remove\s+", re.IGNORECASE), "Add "),
    (re.compile(r"^Delete\s+", re.IGNORECASE), "Add "),
    (re.compile(r"^Take away\s+", re.IGNORECASE), "Add "),
    (re.compile(r"^Take off\s+", re.IGNORECASE), "Put on "),
    (re.compile(r"^Get rid of\s+", re.IGNORECASE), "Add "),
    (re.compile(r"^Cut off\s+", re.IGNORECASE), "Attach "),
    (re.compile(r"^Detach\s+", re.IGNORECASE), "Attach "),
    (re.compile(r"^Eliminate\s+", re.IGNORECASE), "Add "),
    (re.compile(r"^Strip\s+", re.IGNORECASE), "Add "),
]


def reverse_deletion_prompt(prompt: str) -> str:
    """Generate addition prompt from deletion prompt via rule-based reversal."""
    if not prompt or not prompt.strip():
        return ""
    prompt = prompt.strip()
    for pattern, replacement in _REMOVE_PATTERNS:
        if pattern.match(prompt):
            return pattern.sub(replacement, prompt, count=1)
    # Fallback: prepend "Add back" and lowercase first char
    return "Add back " + prompt[0].lower() + prompt[1:]


# ---------------------------------------------------------------------------
# Fix missing before_slat
# ---------------------------------------------------------------------------

def fix_missing_before_slat(mesh_pairs_dir: Path, slat_dir: Path):
    """Copy before_slat from data/slat/ for mod/glb pairs missing it."""
    fixed = 0
    for pair_dir in sorted(mesh_pairs_dir.iterdir()):
        if not pair_dir.is_dir():
            continue
        name = pair_dir.name
        if not (name.startswith("mod_") or name.startswith("glb_")):
            continue

        before_slat = pair_dir / "before_slat"
        if before_slat.exists() and (before_slat / "feats.pt").exists():
            continue

        parts = name.split("_")
        obj_id = "_".join(parts[1:-1])
        feats = slat_dir / f"{obj_id}_feats.pt"
        coords = slat_dir / f"{obj_id}_coords.pt"

        if not feats.exists() or not coords.exists():
            print(f"  SKIP {name}: no source SLAT for {obj_id}")
            continue

        if before_slat.is_symlink():
            before_slat.unlink()

        before_slat.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(feats), str(before_slat / "feats.pt"))
        shutil.copy2(str(coords), str(before_slat / "coords.pt"))
        fixed += 1

    return fixed


# ---------------------------------------------------------------------------
# Build dataset
# ---------------------------------------------------------------------------

def relpath(p: Path) -> str:
    """Return path relative to PROJECT_ROOT (without resolving symlinks)."""
    try:
        # Use absolute() instead of resolve() to preserve symlinks
        return str(p.absolute().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(p)


def build_dataset(mesh_pairs_dir: Path, results_files: list[Path],
                  slat_dir: Path, output_path: Path,
                  specs_files: list[Path] | None = None):
    # Load edit specs (for addition prompts that may be missing from results)
    spec_prompts: dict[str, str] = {}
    if specs_files:
        for sf in specs_files:
            with open(sf) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    d = json.loads(line)
                    prompt = d.get("edit_prompt", "")
                    if prompt:
                        spec_prompts[d["edit_id"]] = prompt

    # Load all successful results
    all_results = {}
    for rf in results_files:
        with open(rf) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                if d.get("status") == "success":
                    all_results[d["edit_id"]] = d

    # Group by object
    by_obj: dict[str, list[dict]] = defaultdict(list)
    for eid, rec in all_results.items():
        by_obj[rec["obj_id"]].append(rec)

    objects = {}
    total_edits = 0
    total_slat_pairs = 0
    type_counts = defaultdict(int)

    for obj_id in sorted(by_obj.keys()):
        recs = by_obj[obj_id]

        # Object-level SLAT
        obj_feats = slat_dir / f"{obj_id}_feats.pt"
        obj_coords = slat_dir / f"{obj_id}_coords.pt"
        has_obj_slat = obj_feats.exists() and obj_coords.exists()

        obj_entry = {
            "slat_feats": relpath(obj_feats) if has_obj_slat else None,
            "slat_coords": relpath(obj_coords) if has_obj_slat else None,
            "edits": [],
        }

        # Build deletion lookup for add reverse pairing
        del_by_suffix: dict[str, dict] = {}
        for rec in recs:
            if rec["edit_type"] == "deletion":
                suffix = "_".join(rec["edit_id"].split("_")[1:])
                del_by_suffix[suffix] = rec

        for rec in sorted(recs, key=lambda r: r["edit_id"]):
            eid = rec["edit_id"]
            etype = rec["edit_type"]
            effective = rec.get("effective_edit_type", "")
            pair_dir = mesh_pairs_dir / eid

            if not pair_dir.exists():
                continue

            # Prompt priority: result > spec > empty
            prompt = rec.get("edit_prompt") or spec_prompts.get(eid, "")

            edit_entry: dict = {
                "edit_id": eid,
                "edit_type": etype,
                "effective_edit_type": effective,
                "prompt": prompt,
            }

            # --- Deletion ---
            if etype == "deletion":
                add_id = "add_" + "_".join(eid.split("_")[1:])
                edit_entry["reverse_id"] = add_id
                edit_entry["before_ply"] = relpath(pair_dir / "before.ply")
                edit_entry["after_ply"] = relpath(pair_dir / "after.ply")

            # --- Addition (reverse of deletion) ---
            elif etype == "addition":
                suffix = "_".join(eid.split("_")[1:])
                del_rec = del_by_suffix.get(suffix)
                del_id = "del_" + suffix
                edit_entry["reverse_id"] = del_id

                # Prompt: spec (VLM-generated) > rule-based reversal
                if not prompt:
                    del_prompt = (del_rec.get("edit_prompt") or
                                  spec_prompts.get(del_id, "")) if del_rec else ""
                    edit_entry["prompt"] = reverse_deletion_prompt(del_prompt)

                edit_entry["before_ply"] = relpath(pair_dir / "before.ply")
                edit_entry["after_ply"] = relpath(pair_dir / "after.ply")

            # --- Modification / Global ---
            else:
                edit_entry["before_ply"] = relpath(pair_dir / "before.ply")
                edit_entry["after_ply"] = relpath(pair_dir / "after.ply")

                # SLAT paths
                before_slat = pair_dir / "before_slat"
                after_slat = pair_dir / "after_slat"
                has_bs = before_slat.exists() and (before_slat / "feats.pt").exists()
                has_as = after_slat.exists() and (after_slat / "feats.pt").exists()

                if has_bs:
                    edit_entry["before_slat"] = relpath(before_slat)
                if has_as:
                    edit_entry["after_slat"] = relpath(after_slat)

                edit_entry["has_slat_pair"] = has_bs and has_as
                if has_bs and has_as:
                    total_slat_pairs += 1

            type_counts[etype] += 1
            total_edits += 1
            obj_entry["edits"].append(edit_entry)

        if obj_entry["edits"]:
            objects[obj_id] = obj_entry

    dataset = {
        "meta": {
            "total_objects": len(objects),
            "total_edits": total_edits,
            "type_counts": dict(type_counts),
            "slat_pairs": total_slat_pairs,
            "note": "All paths are relative to project root.",
        },
        "objects": objects,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    return dataset


def main():
    parser = argparse.ArgumentParser(description="Build training dataset JSON")
    parser.add_argument("--tag", default="v2")
    parser.add_argument("--output-dir", default="outputs/partobjaverse_tiny")
    parser.add_argument("--fix-slat", action="store_true",
                        help="Fix missing before_slat for mod/glb pairs")
    args = parser.parse_args()

    output_dir = PROJECT_ROOT / args.output_dir
    tag_suffix = f"_{args.tag}" if args.tag else ""
    mesh_pairs_dir = output_dir / f"mesh_pairs{tag_suffix}"
    slat_dir = PROJECT_ROOT / "data" / "slat"

    # Step 1: Fix missing before_slat
    if args.fix_slat:
        print(f"Fixing missing before_slat...")
        fixed = fix_missing_before_slat(mesh_pairs_dir, slat_dir)
        print(f"Fixed: {fixed}")

    # Step 2: Load result files and spec files
    def _find_jsonl(subdir, prefix):
        pattern = str(output_dir / "cache" / subdir / f"{prefix}{tag_suffix}_w*.jsonl")
        files = sorted(Path(p) for p in glob.glob(pattern))
        if not files:
            single = output_dir / "cache" / subdir / f"{prefix}{tag_suffix}.jsonl"
            if single.exists():
                files = [single]
        return files

    results_files = _find_jsonl("phase2_5", "edit_results")
    specs_files = _find_jsonl("phase1", "edit_specs")

    if not results_files:
        print(f"No result files found")
        return

    out_path = output_dir / f"dataset{tag_suffix}.json"
    dataset = build_dataset(mesh_pairs_dir, results_files, slat_dir, out_path,
                            specs_files=specs_files)

    meta = dataset["meta"]
    print(f"\nDataset: {out_path}")
    print(f"Objects: {meta['total_objects']}")
    print(f"Total edits: {meta['total_edits']}")
    for t, c in sorted(meta["type_counts"].items()):
        print(f"  {t}: {c}")
    print(f"SLAT pairs (mod+glb): {meta['slat_pairs']}")


if __name__ == "__main__":
    main()
