#!/usr/bin/env python3
"""Extract passed deletion edit_ids from VLM cleaning scores.

Reads the JSONL produced by ``run_vlm_cleaning.py`` (one record per edit with
``edit_id``, ``edit_type``, ``quality_tier``) and writes a flat text file of
deletion ``edit_id`` whose tier meets ``--min-tier``.  The output is the
``--include-list`` argument expected by
``scripts/tools/migrate_slat_to_npz.py --phase 5``.

Tier order: high > medium > low > negative > rejected.

Usage
-----
    python scripts/tools/extract_passed_deletion_ids.py \\
        --scores  outputs/partverse/partverse_pairs/vlm_scores.jsonl \\
        --min-tier medium \\
        --out     outputs/partverse/shard_03/passed_deletion_ids.txt

    # Then feed it to Phase 5 re-encode:
    python scripts/tools/migrate_slat_to_npz.py \\
        --config configs/partverse_node39_shard03.yaml \\
        --mesh-pairs outputs/partverse/shard_03/mesh_pairs_shard03 \\
        --specs-jsonl outputs/partverse/shard_03/cache/phase1/edit_specs_shard03.jsonl \\
        --phase 5 --dino-views 40 \\
        --include-list outputs/partverse/shard_03/passed_deletion_ids.txt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_TIER_ORDER = {"high": 4, "medium": 3, "low": 2, "negative": 1, "rejected": 0}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scores", required=True, type=Path,
                   help="vlm_scores.jsonl produced by run_vlm_cleaning.py")
    p.add_argument("--out", required=True, type=Path,
                   help="Output text file (one deletion edit_id per line)")
    p.add_argument("--min-tier", default="medium",
                   choices=list(_TIER_ORDER.keys()),
                   help="Minimum tier to include (default: medium)")
    p.add_argument("--edit-type", default="deletion",
                   help="Filter by edit_type (default: deletion). "
                        "Use 'all' to skip type filtering.")
    p.add_argument("--shard", default=None,
                   help="Optional shard filter — keep only edit_ids whose "
                        "record has a matching 'shard' field.")
    args = p.parse_args()

    if not args.scores.is_file():
        print(f"[ERROR] scores file not found: {args.scores}", file=sys.stderr)
        return 1

    threshold = _TIER_ORDER[args.min_tier]
    type_filter = None if args.edit_type == "all" else args.edit_type

    seen: set[str] = set()
    n_total = 0
    n_kept = 0
    n_skip_type = 0
    n_skip_tier = 0
    n_skip_shard = 0
    tier_counts: dict[str, int] = {}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.scores.open() as fi, args.out.open("w") as fo:
        for line in fi:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            n_total += 1

            etype = rec.get("edit_type", "")
            if type_filter is not None and etype != type_filter:
                n_skip_type += 1
                continue

            if args.shard is not None and str(rec.get("shard", "")) != args.shard:
                n_skip_shard += 1
                continue

            tier = rec.get("quality_tier", rec.get("tier", "rejected"))
            tier_counts[tier] = tier_counts.get(tier, 0) + 1
            if _TIER_ORDER.get(tier, 0) < threshold:
                n_skip_tier += 1
                continue

            eid = rec.get("edit_id")
            if not eid or eid in seen:
                continue
            seen.add(eid)
            fo.write(eid + "\n")
            n_kept += 1

    print(f"[OK] read {n_total} records from {args.scores}")
    print(f"     wrote {n_kept} edit_ids to {args.out}")
    print(f"     filtered out: type={n_skip_type}  shard={n_skip_shard}  "
          f"below {args.min_tier}={n_skip_tier}")
    if tier_counts:
        ordered = sorted(tier_counts.items(),
                         key=lambda kv: -_TIER_ORDER.get(kv[0], 0))
        print("     tier distribution (post type/shard filter): "
              + ", ".join(f"{k}={v}" for k, v in ordered))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
