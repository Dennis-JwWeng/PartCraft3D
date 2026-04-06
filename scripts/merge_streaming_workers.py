#!/usr/bin/env python3
"""Merge multi-worker ``run_streaming.py`` JSONL outputs into single files.

``run_streaming`` with ``--num-workers N`` writes:
  - ``semantic_labels_{tag}_w{i}.jsonl``
  - ``edit_specs_{tag}_w{i}.jsonl``
  - ``edit_results_{tag}_w{i}.jsonl``

2D cache ``2d_edits_{tag}/`` and ``mesh_pairs_{tag}/`` are already shared; no merge.

After merging, batch steps 5–6 work with the same ``--config`` / ``--tag``:

    python scripts/run_pipeline.py --config <cfg> --steps 5 6 --tag <tag>

Usage:
    python scripts/merge_streaming_workers.py --config configs/partverse_local.yaml \\
        --tag 0326 --num-workers 4
    python scripts/merge_streaming_workers.py ... --dry-run   # stats only
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import warnings
from collections import Counter
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from partcraft.utils.config import load_config  # noqa: E402


def _apply_output_shard_layout(cfg: dict) -> None:
    data = cfg.get("data", {})
    if not data.get("output_by_shard"):
        return
    shards = data.get("shards")
    if not shards:
        warnings.warn(
            "output_by_shard is true but data.shards is empty; "
            "skipping shard output subdirectory.",
            UserWarning,
            stacklevel=2,
        )
        return
    if len(shards) > 1:
        warnings.warn(
            "output_by_shard is true but data.shards has multiple entries; "
            "skipping shard subdirectory.",
            UserWarning,
            stacklevel=2,
        )
        return
    shard = str(shards[0]).strip()
    if not shard:
        return
    nested = f"shard_{shard}"
    out = data.get("output_dir", "outputs")
    root = Path(out)
    if root.name == nested:
        return
    data["output_dir"] = str(root / nested)


def _normalize_cache_dirs(cfg: dict) -> None:
    """Match ``run_streaming`` / ``run_pipeline`` path layout (incl. shard subdir)."""
    _apply_output_shard_layout(cfg)
    _out = cfg["data"].get("output_dir", "outputs")
    _out_norm = _out.replace("\\", "/")
    _markers = (
        "cache/phase0", "cache/phase1", "cache/phase2",
        "cache/phase2_5", "cache/phase3", "cache/phase4",
    )
    for _phase_key in ("phase0", "phase1", "phase2", "phase2_5", "phase3", "phase4"):
        _pcfg = cfg.get(_phase_key, {})
        _cd = _pcfg.get("cache_dir", "")
        if not _cd or os.path.isabs(_cd):
            continue
        cd_norm = _cd.replace("\\", "/")
        if cd_norm == _out_norm or cd_norm.startswith(_out_norm + "/"):
            continue
        rel = None
        for m in _markers:
            if m in cd_norm:
                rel = cd_norm[cd_norm.index(m):]
                break
        if rel is not None:
            _pcfg["cache_dir"] = os.path.join(_out, *rel.split("/"))
        else:
            _pcfg["cache_dir"] = os.path.join(_out, _cd)


def _resolve(p: str | Path) -> Path:
    pp = Path(p)
    return pp if pp.is_absolute() else _PROJECT_ROOT / pp


def _count_jsonl_lines(path: Path) -> int:
    if not path.exists():
        return 0
    n = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def _collect_merged_records(
    paths: list[Path],
    id_key: str,
) -> tuple[dict[str, dict], list[str], list[str], int]:
    """Parse worker JSONL files; last occurrence wins on duplicate ``id_key``."""
    merged: dict[str, dict] = {}
    order: list[str] = []
    warnings: list[str] = []

    for p in paths:
        if not p.exists():
            continue
        with open(p, encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError as e:
                    warnings.append(f"{p}:{lineno}: JSON decode error: {e}")
                    continue
                key = rec.get(id_key)
                if key is None:
                    warnings.append(f"{p}:{lineno}: missing {id_key!r}, skipped")
                    continue
                if key in merged and merged[key] != rec:
                    warnings.append(
                        f"duplicate {id_key}={key!r}: keeping last ({p.name})"
                    )
                if key not in merged:
                    order.append(key)
                merged[key] = rec

    n_sources = sum(1 for p in paths if p.exists())
    return merged, order, warnings, n_sources


def _write_merged_jsonl(
    merged: dict[str, dict],
    order: list[str],
    out_path: Path,
    *,
    backup: bool,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if backup and out_path.exists():
        bak = out_path.with_suffix(out_path.suffix + ".bak")
        shutil.copy2(out_path, bak)

    with open(out_path, "w", encoding="utf-8") as fp:
        for k in order:
            fp.write(json.dumps(merged[k], ensure_ascii=False) + "\n")


def _mesh_pair_stats(mesh_pairs_dir: Path) -> tuple[int, int]:
    """Return (dirs_with_after_artifact, total_subdirs)."""
    if not mesh_pairs_dir.is_dir():
        return 0, 0
    subdirs = [d for d in mesh_pairs_dir.iterdir() if d.is_dir()]
    with_after = sum(
        1 for d in subdirs
        if (d / "after.npz").is_file()
        or (d / "after.ply").is_file()
        or (d / "after_slat").is_dir()
    )
    return with_after, len(subdirs)


def _count_edited_png(edit_dir: Path) -> int:
    if not edit_dir.is_dir():
        return 0
    return sum(1 for p in edit_dir.glob("*_edited.png") if p.is_file())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--tag", type=str, required=True)
    parser.add_argument(
        "--num-workers",
        type=int,
        required=True,
        help="Same value as run_streaming --num-workers",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print statistics only; do not write merged files",
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Before writing, copy existing targets to *.jsonl.bak",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    _normalize_cache_dirs(cfg)

    tag_suffix = f"_{args.tag}"
    n_w = args.num_workers

    p0 = cfg["phase0"]["cache_dir"]
    p1 = cfg["phase1"]["cache_dir"]
    p25 = cfg["phase2_5"]["cache_dir"]
    out_dir = cfg["data"]["output_dir"]

    base_labels = _resolve(p0) / f"semantic_labels{tag_suffix}"
    base_specs = _resolve(p1) / f"edit_specs{tag_suffix}"
    base_results = _resolve(p25) / f"edit_results{tag_suffix}"
    edit_2d_dir = _resolve(p25) / f"2d_edits{tag_suffix}"
    mesh_pairs_dir = _resolve(out_dir) / f"mesh_pairs{tag_suffix}"

    if n_w < 1:
        print("num-workers must be >= 1", file=sys.stderr)
        sys.exit(1)

    worker_label_paths: list[Path] = []
    worker_spec_paths: list[Path] = []
    worker_res_paths: list[Path] = []
    for i in range(n_w):
        suf = f"_w{i}" if n_w > 1 else ""
        worker_label_paths.append(Path(f"{base_labels}{suf}.jsonl"))
        worker_spec_paths.append(Path(f"{base_specs}{suf}.jsonl"))
        worker_res_paths.append(Path(f"{base_results}{suf}.jsonl"))

    print("=" * 60)
    print("Streaming merge / progress report")
    print("=" * 60)
    print(f"tag={args.tag!r}  num_workers={n_w}")
    print(f"labels dir:  {base_labels.parent}")
    print(f"specs dir:   {base_specs.parent}")
    print(f"results dir: {base_results.parent}")
    print(f"mesh_pairs:  {mesh_pairs_dir}")
    print(f"2d_edits:    {edit_2d_dir}")
    print()

    print("--- Per-worker file sizes (lines) ---")
    for i in range(n_w):
        lp, sp, rp = worker_label_paths[i], worker_spec_paths[i], worker_res_paths[i]
        print(
            f"  w{i}: labels={_count_jsonl_lines(lp):5d}  "
            f"specs={_count_jsonl_lines(sp):5d}  "
            f"results={_count_jsonl_lines(rp):5d}  "
            f"[exists: L={lp.exists()} S={sp.exists()} R={rp.exists()}]"
        )

    # Result status breakdown per worker
    print()
    print("--- edit_results status (per worker) ---")
    for i, rp in enumerate(worker_res_paths):
        if not rp.exists():
            print(f"  w{i}: (missing)")
            continue
        st: Counter[str] = Counter()
        with open(rp, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    st[json.loads(line).get("status", "?")] += 1
                except json.JSONDecodeError:
                    st["<bad json>"] += 1
        print(f"  w{i}: {dict(st)}")

    merged_labels_path = Path(f"{base_labels}.jsonl")
    merged_specs_path = Path(f"{base_specs}.jsonl")
    merged_results_path = Path(f"{base_results}.jsonl")

    m_l, o_l, w1, src_l = _collect_merged_records(worker_label_paths, "obj_id")
    m_s, o_s, w2, src_s = _collect_merged_records(worker_spec_paths, "edit_id")
    m_r, o_r, w3, src_r = _collect_merged_records(worker_res_paths, "edit_id")
    all_warn = w1 + w2 + w3

    if not args.dry_run:
        _write_merged_jsonl(m_l, o_l, merged_labels_path, backup=args.backup)
        _write_merged_jsonl(m_s, o_s, merged_specs_path, backup=args.backup)
        _write_merged_jsonl(m_r, o_r, merged_results_path, backup=args.backup)

    n_l, n_s, n_r = len(m_l), len(m_s), len(m_r)
    st_merged = Counter(r.get("status", "?") for r in m_r.values())

    print()
    print("--- Merged (unique keys) ---")
    mode = "(dry-run; files not written)" if args.dry_run else "(written)"
    print(
        f"  semantic_labels: {n_l} unique obj_id  "
        f"from {src_l}/{n_w} worker files  {mode}"
    )
    print(
        f"  edit_specs:      {n_s} unique edit_id  "
        f"from {src_s}/{n_w} worker files  {mode}"
    )
    print(
        f"  edit_results:    {n_r} unique edit_id  "
        f"from {src_r}/{n_w} worker files  {mode}"
    )
    print(f"  edit_results status (merged): {dict(st_merged)}")
    if not args.dry_run:
        print(f"\n  → {merged_labels_path}")
        print(f"  → {merged_specs_path}")
        print(f"  → {merged_results_path}")

    ma, mt = _mesh_pair_stats(mesh_pairs_dir)
    png_n = _count_edited_png(edit_2d_dir)
    print()
    print("--- Shared artifacts (not merged) ---")
    print(f"  mesh_pairs/{mesh_pairs_dir.name}: {ma} dirs with after.ply / {mt} subdirs")
    print(f"  2d_edits: {png_n} *_edited.png")

    if all_warn:
        print()
        print("--- Warnings ---")
        for w in all_warn[:50]:
            print(f"  {w}")
        if len(all_warn) > 50:
            print(f"  ... and {len(all_warn) - 50} more")

    print()
    print("=" * 60)


if __name__ == "__main__":
    main()
