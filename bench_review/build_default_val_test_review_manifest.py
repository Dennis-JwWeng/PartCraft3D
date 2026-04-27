#!/usr/bin/env python3
"""Build review manifest for default H3D val+test candidates with local pipeline assets."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

EDIT_TYPES = ["deletion", "addition", "modification", "scale", "material", "color", "global"]
FLUX_TYPES = {"modification", "scale", "material", "color", "global"}
DEFAULT_REPO = Path("/mnt/zsn/zsn_workspace/PartCraft3D")
DEFAULT_OUT_DIR = DEFAULT_REPO / "bench_review/default_val_test_available"


@dataclass
class MissingRequirements:
    rows: list[dict] | None = None

    def __post_init__(self) -> None:
        if self.rows is None:
            self.rows = []

    def add(self, shard: str, obj_id: str, edit_id: str, edit_type: str) -> None:
        self.rows.append({
            "shard": shard,
            "obj_id": obj_id,
            "edit_id": edit_id,
            "edit_type": edit_type,
        })

    @property
    def by_shard_edit_count(self) -> dict[str, int]:
        counts = Counter(str(row["shard"]) for row in self.rows or [])
        return dict(sorted(counts.items()))

    @property
    def by_shard_obj_ids(self) -> dict[str, set[str]]:
        out: dict[str, set[str]] = defaultdict(set)
        for row in self.rows or []:
            out[str(row["shard"])].add(str(row["obj_id"]))
        return dict(sorted(out.items()))


@dataclass(frozen=True)
class OutputPaths:
    manifest_path: Path
    edit_ids_path: Path
    summary_path: Path


def load_obj_splits(split_dir: Path) -> dict[str, str]:
    split_of: dict[str, str] = {}
    for split in ("train", "val", "test"):
        path = split_dir / f"{split}.obj_ids.txt"
        if not path.is_file():
            continue
        for obj_id in path.read_text(encoding="utf-8").split():
            split_of[obj_id] = split
    return split_of


def available_pipeline_shards(out_root: Path) -> set[str]:
    if not out_root.is_dir():
        return set()
    return {
        path.name.replace("shard", "", 1)
        for path in out_root.iterdir()
        if path.is_dir() and path.name.startswith("shard")
    }


def iter_default_val_test_candidates(repo: Path) -> tuple[list[dict], Counter[str], MissingRequirements]:
    out_root = repo / "outputs/partverse"
    h3d_root = repo / "data/H3D_v1"
    manifest = h3d_root / "manifests/all.jsonl"
    split_dir = repo / "H3D_v1_hf/data/splits"

    split_of = load_obj_splits(split_dir)
    shard_set = available_pipeline_shards(out_root)
    records: list[dict] = []
    reject: Counter[str] = Counter()
    missing = MissingRequirements()

    with manifest.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            shard = str(row.get("shard", ""))
            edit_type = row.get("edit_type")
            obj_id = row.get("obj_id")
            edit_id = row.get("edit_id")
            source_split = split_of.get(obj_id or "", "unknown")
            if source_split not in {"val", "test"}:
                reject["not_val_or_test"] += 1
                continue
            if edit_type not in EDIT_TYPES or not obj_id or not edit_id:
                reject["bad_record"] += 1
                continue
            if shard not in shard_set:
                reject["no_pipeline_shard_dir"] += 1
                missing.add(shard, str(obj_id), str(edit_id), str(edit_type))
                continue

            out_obj = out_root / f"shard{shard}" / "mode_e_text_align" / "objects" / shard / obj_id
            out_edit = out_obj / "edits_3d" / edit_id
            if not out_edit.is_dir():
                reject["missing_pipeline_edit_dir"] += 1
                missing.add(shard, str(obj_id), str(edit_id), str(edit_type))
                continue

            h3d_dir = h3d_root / edit_type / shard / obj_id / edit_id
            before_png = h3d_dir / "before.png"
            after_png = h3d_dir / "after.png"
            before_npz = h3d_dir / "before.npz"
            after_npz = h3d_dir / "after.npz"
            meta_json = h3d_dir / "meta.json"
            if not all(p.is_file() for p in (before_png, after_png, before_npz, after_npz, meta_json)):
                reject["missing_h3d_core_asset"] += 1
                continue

            record = {
                "bench_split": "default_val_test_available",
                "source_hf_split": source_split,
                "edit_id": edit_id,
                "edit_type": edit_type,
                "obj_id": obj_id,
                "shard": shard,
                "h3d_before_png": str(before_png),
                "h3d_after_png": str(after_png),
                "h3d_before_npz": str(before_npz),
                "h3d_after_npz": str(after_npz),
                "h3d_meta_json": str(meta_json),
                "pipeline_edit_dir": str(out_edit),
                "two_d_input_png": "",
                "two_d_edited_png": "",
            }
            if edit_type in FLUX_TYPES:
                e2d = out_obj / "edits_2d"
                two_d_input = e2d / f"{edit_id}_input.png"
                two_d_edited = e2d / f"{edit_id}_edited.png"
                if not two_d_input.is_file():
                    reject["flux_missing_2d_input"] += 1
                    continue
                if not two_d_edited.is_file():
                    reject["flux_missing_2d_edited"] += 1
                    continue
                record["two_d_input_png"] = str(two_d_input)
                record["two_d_edited_png"] = str(two_d_edited)
            records.append(record)

    records.sort(key=lambda r: (r["shard"], r["obj_id"], r["edit_type"], r["edit_id"]))
    return records, reject, missing


def write_outputs(out_dir: Path, records: list[dict], reject: Counter[str] | dict[str, int], missing: MissingRequirements) -> OutputPaths:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "h3d_default_val_test_available_manifest.jsonl"
    edit_ids_path = out_dir / "h3d_default_val_test_available_edit_ids.txt"
    summary_path = out_dir / "summary.md"

    with manifest_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    edit_ids_path.write_text("".join(record["edit_id"] + "\n" for record in records), encoding="utf-8")
    write_missing_lists(out_dir / "missing", missing)
    summary_path.write_text(render_summary(records, Counter(reject), missing), encoding="utf-8")
    return OutputPaths(manifest_path=manifest_path, edit_ids_path=edit_ids_path, summary_path=summary_path)


def write_missing_lists(out_dir: Path, missing: MissingRequirements) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    object_dirs: list[str] = []
    edit_dirs: list[str] = []
    by_shard_obj = missing.by_shard_obj_ids
    by_shard_edit_ids: dict[str, list[str]] = defaultdict(list)
    for row in missing.rows or []:
        shard = str(row["shard"])
        obj_id = str(row["obj_id"])
        edit_id = str(row["edit_id"])
        base = f"/mnt/zsn/zsn_workspace/PartCraft3D/outputs/partverse/shard{shard}/mode_e_text_align/objects/{shard}/{obj_id}"
        object_dirs.append(base)
        edit_dirs.append(f"{base}/edits_3d/{edit_id}")
        by_shard_edit_ids[shard].append(edit_id)
    out_dir.joinpath("missing_object_dirs.txt").write_text("\n".join(sorted(set(object_dirs))) + ("\n" if object_dirs else ""), encoding="utf-8")
    out_dir.joinpath("missing_pipeline_edit_dirs.txt").write_text("\n".join(edit_dirs) + ("\n" if edit_dirs else ""), encoding="utf-8")
    for shard, obj_ids in by_shard_obj.items():
        out_dir.joinpath(f"shard{shard}_missing_obj_ids.txt").write_text("\n".join(sorted(obj_ids)) + "\n", encoding="utf-8")
    for shard, edit_ids in sorted(by_shard_edit_ids.items()):
        out_dir.joinpath(f"shard{shard}_missing_edit_ids.txt").write_text("\n".join(edit_ids) + "\n", encoding="utf-8")


def render_summary(records: list[dict], reject: Counter[str], missing: MissingRequirements) -> str:
    by_type = Counter(record["edit_type"] for record in records)
    by_shard = Counter(record["shard"] for record in records)
    by_source = Counter(record["source_hf_split"] for record in records)
    missing_obj_counts = {shard: len(ids) for shard, ids in missing.by_shard_obj_ids.items()}

    lines = [
        "# Default val+test Available Review Manifest",
        "",
        "This manifest contains default val+test H3D edits whose local pipeline outputs are ready for ZIP review.",
        "Flux edit types require both `edits_2d/<edit_id>_input.png` and `edits_2d/<edit_id>_edited.png`.",
        "",
        f"Total available records: `{len(records)}`",
        "",
        "## Available By Shard",
        "",
        "| shard | edits |",
        "|---|---:|",
    ]
    for shard, count in sorted(by_shard.items()):
        lines.append(f"| {shard} | {count} |")
    lines.extend(["", "## Available By Edit Type", "", "| edit_type | edits |", "|---|---:|"])
    for edit_type in EDIT_TYPES:
        lines.append(f"| {edit_type} | {by_type[edit_type]} |")
    lines.extend(["", "## Source Split Mix", "", "| split | edits |", "|---|---:|"])
    for split, count in sorted(by_source.items()):
        lines.append(f"| {split} | {count} |")
    lines.extend(["", "## Reject Counts", "", "| reason | count |", "|---|---:|"])
    for reason, count in sorted(reject.items()):
        lines.append(f"| {reason} | {count} |")
    lines.extend(["", "## Missing Pipeline Outputs", "", "| shard | objects | edits |", "|---|---:|---:|"])
    for shard, edit_count in missing.by_shard_edit_count.items():
        lines.append(f"| {shard} | {missing_obj_counts.get(shard, 0)} | {edit_count} |")
    lines.extend([
        "",
        "## Review Export",
        "",
        "Open `h3d_review_tool.html`, drag in one `*_assets.zip`, mark decisions, then use `导出 selected_edit_ids_*.txt`.",
        "Concatenate those selected-edit-id files across chunks to get the final benchmark candidate list.",
    ])
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records, reject, missing = iter_default_val_test_candidates(args.repo)
    outputs = write_outputs(args.out_dir, records, reject, missing)
    print(f"wrote {outputs.manifest_path}")
    print(f"wrote {outputs.edit_ids_path}")
    print(f"wrote {outputs.summary_path}")
    print("records", len(records))
    print("reject", dict(sorted(reject.items())))
    print("missing_by_shard", missing.by_shard_edit_count)


if __name__ == "__main__":
    main()
