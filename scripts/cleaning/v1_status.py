#!/usr/bin/env python
"""Inspect the partverse v1 dataset and tell you what to run next.

Reports
-------
1. PROMOTED counts per (shard, edit_type): how many edit dirs exist in v1,
   how many have ``after.npz`` / ``before.npz``, and (for additions)
   how many have an ``add_npz_link`` recorded in qc.json.
2. PENDING queues: ``_pending/del_latent.txt`` line count, and
   ``_before_pending.json`` markers (additions waiting on a deletion to be
   encoded).
3. VERDICT: per edit-type traffic-light status with the specific NEXT
   COMMAND the agent should run.

This script is **read-only**. Use it before deciding which stage to launch
next; use it after each stage to confirm progress.

Exit code: 0 when nothing actionable remains, 1 otherwise (so it can be
used in CI / automation loops).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from partcraft.cleaning.v1.layout import (  # noqa: E402
    V1Layout, parse_edit_id,
)
from partcraft.cleaning.v1.pending import DelLatentPending  # noqa: E402

EDIT_TYPES = (
    "deletion", "addition", "modification",
    "scale", "material", "color", "glb",
)

# Edit types that need a per-edit before.npz hardlink to be considered
# "fully linked" in v1.  (Currently the promoter does not yet materialise
# per-edit before.npz for non-addition edits — see RUNBOOK_partverse_v1.md
# follow-ups.  The status reporter still counts whether it exists, which is
# how the agent learns if the future promoter extension has rolled out.)
ALL_NEED_PER_EDIT_BEFORE = True


@dataclass
class EditTypeStats:
    promoted: int = 0
    after_npz: int = 0
    before_npz: int = 0
    add_link_recorded: int = 0
    after_pending_marker: int = 0
    before_pending_marker: int = 0


@dataclass
class ShardStats:
    by_type: dict[str, EditTypeStats] = field(
        default_factory=lambda: {t: EditTypeStats() for t in EDIT_TYPES}
    )

    def add(self, et: str, *, after: bool, before: bool, link_qc: bool,
            after_pending: bool, before_pending: bool) -> None:
        s = self.by_type.setdefault(et, EditTypeStats())
        s.promoted += 1
        s.after_npz += int(after)
        s.before_npz += int(before)
        s.add_link_recorded += int(link_qc)
        s.after_pending_marker += int(after_pending)
        s.before_pending_marker += int(before_pending)


def _scan(layout: V1Layout, *, only_shard: str | None) -> dict[str, ShardStats]:
    out: dict[str, ShardStats] = defaultdict(ShardStats)
    objects_root = layout.root / "objects"
    if not objects_root.is_dir():
        return out
    shard_iter: Iterable[Path] = (
        [objects_root / only_shard] if only_shard else sorted(objects_root.iterdir())
    )
    for shard_dir in shard_iter:
        if not shard_dir.is_dir():
            continue
        sname = shard_dir.name
        for obj_dir in sorted(shard_dir.iterdir()):
            if not obj_dir.is_dir():
                continue
            edits_root = obj_dir / "edits"
            if not edits_root.is_dir():
                continue
            for ed in sorted(edits_root.iterdir()):
                if not ed.is_dir():
                    continue
                name = ed.name
                if "__r" in name:
                    base, _, n = name.rpartition("__r")
                    eid = base if n.isdigit() else name
                else:
                    eid = name
                try:
                    edit_type, _obj, _idx = parse_edit_id(eid)
                except ValueError:
                    continue
                qc_p = ed / "qc.json"
                link_qc = False
                if qc_p.is_file():
                    try:
                        qc = json.loads(qc_p.read_text())
                        link_qc = bool(
                            qc.get("passes", {}).get("add_npz_link", {}).get("pass")
                        )
                    except Exception:
                        pass
                out[sname].add(
                    edit_type,
                    after=(ed / "after.npz").is_file(),
                    before=(ed / "before.npz").is_file(),
                    link_qc=link_qc,
                    after_pending=(ed / "_after_pending.json").is_file(),
                    before_pending=(ed / "_before_pending.json").is_file(),
                )
    return out


# ───────────────────────── reporting ─────────────────────────────────────


def _fmt_int(n: int) -> str:
    return f"{n:>5d}"


def _print_promoted(stats: dict[str, ShardStats]) -> None:
    print("PROMOTED (per shard / per edit type):")
    if not stats:
        print("  <empty: no objects under objects/>")
        return
    for shard in sorted(stats):
        print(f"  shard {shard}:")
        sd = stats[shard]
        for et in EDIT_TYPES:
            s = sd.by_type.get(et, EditTypeStats())
            if s.promoted == 0:
                continue
            line = (
                f"    {et:<14s}  promoted={_fmt_int(s.promoted)}  "
                f"after.npz={_fmt_int(s.after_npz)}/{_fmt_int(s.promoted)}  "
                f"before.npz={_fmt_int(s.before_npz)}/{_fmt_int(s.promoted)}"
            )
            if et == "addition":
                line += (
                    f"  add_npz_link={_fmt_int(s.add_link_recorded)}/"
                    f"{_fmt_int(s.promoted)}"
                )
            if s.after_pending_marker:
                line += f"  [after_pending={s.after_pending_marker}]"
            if s.before_pending_marker:
                line += f"  [before_pending={s.before_pending_marker}]"
            print(line)


def _print_pending(layout: V1Layout) -> tuple[int, list[str]]:
    print("\nPENDING QUEUES:")
    notes: list[str] = []
    pending = DelLatentPending(layout.pending_del_latent_file())
    n_del = sum(1 for _ in pending.iter_entries())
    print(
        f"  _pending/del_latent.txt:  {n_del} entries  "
        f"({layout.pending_del_latent_file()})"
    )
    if n_del:
        notes.append("del_latent.txt drained")
    return n_del, notes


def _print_verdict(
    layout: V1Layout,
    stats: dict[str, ShardStats],
    n_del_pending: int,
    *, slat_root_hint: str,
) -> int:
    print("\nVERDICT:")
    nonzero_actions = 0
    next_cmds: list[str] = []

    # Aggregate across shards (the agent usually drives one shard at a time
    # via --shard, so this aggregate corresponds to the requested scope).
    agg: dict[str, EditTypeStats] = {t: EditTypeStats() for t in EDIT_TYPES}
    for sd in stats.values():
        for et in EDIT_TYPES:
            s = sd.by_type.get(et, EditTypeStats())
            a = agg[et]
            a.promoted += s.promoted
            a.after_npz += s.after_npz
            a.before_npz += s.before_npz
            a.add_link_recorded += s.add_link_recorded
            a.after_pending_marker += s.after_pending_marker
            a.before_pending_marker += s.before_pending_marker

    # ── deletion ────────────────────────────────────────────────────────
    s = agg["deletion"]
    if s.promoted == 0:
        print("  ·  deletion       no records yet (nothing promoted)")
    else:
        missing_after = s.promoted - s.after_npz
        if missing_after == 0 and n_del_pending == 0:
            print(f"  OK deletion       all {s.promoted} encoded")
        else:
            nonzero_actions += 1
            print(
                f"  ⚠  deletion       {s.promoted} promoted, "
                f"{missing_after} missing after.npz, "
                f"{n_del_pending} in pending queue"
            )
            print("     next: drain del_latent.txt via encode_del_latent.py")
            next_cmds.append(
                "python -m scripts.cleaning.encode_del_latent \\\n"
                f"    --v1-root {layout.root} \\\n"
                "    --rules configs/cleaning/promote_v1.yaml \\\n"
                "    --ckpt-root <trellis_ss_ckpt_root> \\\n"
                "    --num-gpus 8"
            )

    # ── addition ────────────────────────────────────────────────────────
    s = agg["addition"]
    if s.promoted == 0:
        print("  ·  addition       no records yet (nothing promoted)")
    else:
        missing_before = s.promoted - s.before_npz
        missing_after = s.promoted - s.after_npz
        if missing_before == 0 and missing_after == 0:
            print(
                f"  OK addition       all {s.promoted} fully linked "
                f"({s.add_link_recorded}/{s.promoted} have add_npz_link in qc)"
            )
        else:
            nonzero_actions += 1
            blocker = ""
            if missing_before and agg["deletion"].after_npz < agg["deletion"].promoted:
                blocker = "  (BLOCKED on deletion encode above)"
            print(
                f"  ⚠  addition       {s.promoted} promoted, "
                f"{missing_before} missing before.npz, "
                f"{missing_after} missing after.npz" + blocker
            )
            print("     next: link addition npz from encoded del.after.npz")
            next_cmds.append(
                "python -m scripts.cleaning.link_add_npz_from_del \\\n"
                f"    --v1-root {layout.root} \\\n"
                f"    --slat-root {slat_root_hint}"
            )

    # ── flux-branch (mod / scl / mat / clr / glb) ───────────────────────
    flux_total = sum(agg[t].promoted for t in
                     ("modification", "scale", "material", "color", "glb"))
    if flux_total == 0:
        print("  ·  flux-branch    no records yet (mod/scl/mat/clr/glb)")
    else:
        flux_after_ok = sum(agg[t].after_npz for t in
                            ("modification", "scale", "material", "color", "glb"))
        flux_before_ok = sum(agg[t].before_npz for t in
                             ("modification", "scale", "material", "color", "glb"))
        if flux_after_ok < flux_total or (
            ALL_NEED_PER_EDIT_BEFORE and flux_before_ok < flux_total
        ):
            nonzero_actions += 1
            print(
                f"  ⚠  flux-branch    {flux_total} promoted, "
                f"{flux_total - flux_after_ok} missing after.npz, "
                f"{flux_total - flux_before_ok} missing before.npz"
            )
            if flux_after_ok == flux_total and flux_before_ok < flux_total:
                print("     note: per-edit before.npz materialisation for "
                      "flux-branch is a pending promoter extension; "
                      "see docs/RUNBOOK_partverse_v1.md.")
            else:
                print("     next: rerun promote_to_v1 with promote_v1.yaml "
                      "(after gate_quality finishes upstream)")
        else:
            print(
                f"  OK flux-branch    all {flux_total} fully linked "
                f"(after & before)"
            )

    total_promoted = sum(
        a.promoted for a in agg.values()
    )
    if total_promoted == 0:
        nonzero_actions += 1
        print(
            "\nNo records found in v1 layout for this scope.\n"
            "  next: promote upstream pipeline run(s) into v1."
        )
        next_cmds.append(
            "# Track A: del + add (Gate-A only):\n"
            "  python -m scripts.cleaning.promote_to_v1 \\\n"
            "      --rules configs/cleaning/promote_v1_addel_textalign.yaml \\\n"
            "      --source-runs <outputs/partverse/...shard08...>\n"
            "# Track B: other edit types (waits for gate_quality to finish):\n"
            "  python -m scripts.cleaning.promote_to_v1 \\\n"
            "      --rules configs/cleaning/promote_v1.yaml \\\n"
            "      --source-runs <outputs/partverse/...shard08...>"
        )

    if next_cmds:
        print("\nNEXT COMMAND(S):")
        for c in next_cmds:
            print("  $ " + c)
    else:
        print("\nNEXT COMMAND(S): none — v1 looks consistent for the requested scope.")

    return nonzero_actions


# ─────────────────────────── main ────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--v1-root", type=Path, required=True)
    ap.add_argument("--shard", default=None,
                    help="Limit to one shard, e.g. '08'")
    ap.add_argument(
        "--slat-root", default="data/partverse/inputs/slat",
        help="Hint for the link_add_npz_from_del NEXT COMMAND.",
    )
    args = ap.parse_args(argv)

    if not args.v1_root.is_dir():
        print(f"v1-root does not exist: {args.v1_root}", file=sys.stderr)
        return 2

    layout = V1Layout(root=args.v1_root)
    print("=" * 60)
    print(f"v1 STATUS  v1_root={args.v1_root}  "
          f"shard={args.shard or '<all>'}")
    print("=" * 60 + "\n")

    stats = _scan(layout, only_shard=args.shard)
    _print_promoted(stats)
    n_del_pending, _ = _print_pending(layout)
    actions = _print_verdict(
        layout, stats, n_del_pending, slat_root_hint=args.slat_root,
    )
    return 0 if actions == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
