"""Shared utilities for the three ``pull_*`` H3D_v1 CLIs.

Keeps each CLI script ~50-line orchestration code by pulling out the
common parts: argparse skeleton, obj-id allowlist resolution, manifest
append, summary printing.
"""
from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from partcraft.cleaning.h3d_v1.layout import H3DLayout
from partcraft.cleaning.h3d_v1.manifest import append_jsonl
from partcraft.cleaning.h3d_v1.pipeline_io import (
    PipelineEdit,
    PipelinePaths,
    iter_edits,
    resolve_paths,
)
from partcraft.cleaning.h3d_v1.promoter import PromoteContext, PromoteResult


def add_common_args(ap: argparse.ArgumentParser) -> None:
    """Args shared by every pull_* CLI."""
    ap.add_argument("--pipeline-cfg", required=True, type=Path,
                    help="Path to the pipeline_v3 YAML for this shard.")
    ap.add_argument("--shard", required=True, type=str,
                    help='Two-digit shard string (e.g. "08").')
    ap.add_argument("--dataset-root", required=True, type=Path,
                    help="H3D_v1 dataset root, e.g. data/H3D_v1/.")
    ap.add_argument("--workers", type=int, default=8,
                    help="Thread pool size for the promote phase (default 8).")
    ap.add_argument("--limit", type=int, default=None,
                    help="Process at most N edits (after filter); for testing.")
    ap.add_argument("--obj-id", action="append", default=[],
                    help="Restrict to this obj_id (repeatable).")
    ap.add_argument("--obj-ids-file", type=Path, default=None,
                    help="One obj_id per line; combined with --obj-id.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Count + estimate only; write nothing to dataset.")
    ap.add_argument("--log-level", default="INFO",
                    choices=("DEBUG", "INFO", "WARNING", "ERROR"))


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s [%(levelname)s] %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def resolve_obj_filter(args: argparse.Namespace) -> set[str] | None:
    ids: set[str] = set(args.obj_id or [])
    if args.obj_ids_file:
        for line in args.obj_ids_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                ids.add(line)
    return ids or None


@dataclass
class PullStats:
    accepted: int = 0
    rejected: int = 0
    promoted: int = 0
    skipped: int = 0  # promote returned ok=False with reason
    failed: int = 0   # raised an exception
    reasons: dict[str, int] = field(default_factory=dict)

    def add_reject(self, reason: str) -> None:
        self.rejected += 1
        self.reasons[reason] = self.reasons.get(reason, 0) + 1

    def add_skip(self, reason: str) -> None:
        self.skipped += 1
        self.reasons[reason] = self.reasons.get(reason, 0) + 1

    def banner(self, label: str) -> str:
        line = f"=== {label} === accept={self.accepted} reject={self.rejected} " \
               f"promoted={self.promoted} skip={self.skipped} fail={self.failed}"
        if self.reasons:
            top = sorted(self.reasons.items(), key=lambda kv: -kv[1])[:5]
            line += "\n  top reasons: " + ", ".join(f"{r}×{n}" for r, n in top)
        return line


def collect_edits(
    cfg_path: Path,
    shard: str,
    types: Iterable[str],
    obj_filter: set[str] | None,
    *,
    accept_fn: Callable[[dict[str, Any], str], Any],
    load_status: Callable[[Path], dict[str, Any]],
) -> tuple[list[PipelineEdit], PullStats, PipelinePaths]:
    """Walk pipeline output, apply ``accept_fn``, return accepted edits + stats."""
    paths = resolve_paths(cfg_path)
    accepted: list[PipelineEdit] = []
    stats = PullStats()
    last_obj: str | None = None
    cached_status: dict[str, Any] = {}
    for edit in iter_edits(cfg_path, shard, types=types, obj_id_allowlist=obj_filter):
        if edit.obj_id != last_obj:
            cached_status = load_status(edit.obj_dir)
            last_obj = edit.obj_id
        decision = accept_fn(cached_status, edit.edit_id)
        if decision.ok:
            stats.accepted += 1
            accepted.append(edit)
        else:
            stats.add_reject(decision.reason or "unknown")
    return accepted, stats, paths


def build_promote_context(paths: PipelinePaths, *, ss_encoder=None) -> PromoteContext:
    return PromoteContext(
        pipeline_obj_root=paths.objects_root,
        slat_dir=paths.slat_dir,
        images_root=paths.images_root,
        ss_encoder=ss_encoder,
    )


def run_promote_pool(
    edits: list[PipelineEdit],
    layout: H3DLayout,
    ctx: PromoteContext,
    *,
    promote_fn: Callable[[PipelineEdit, H3DLayout], PromoteResult],
    workers: int,
    stats: PullStats,
) -> None:
    """Promote ``edits`` concurrently; append results to per-shard manifest."""
    log = logging.getLogger("h3d_v1.pull")
    log.info("promoting %d edits with %d workers", len(edits), workers)

    def _do(edit: PipelineEdit) -> tuple[PipelineEdit, PromoteResult | Exception]:
        try:
            return edit, promote_fn(edit, layout)
        except Exception as exc:  # noqa: BLE001 — we want the message
            return edit, exc

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = [pool.submit(_do, e) for e in edits]
        for fut in as_completed(futures):
            edit, outcome = fut.result()
            if isinstance(outcome, Exception):
                stats.failed += 1
                log.warning("[%s] failed: %s", edit.edit_id, outcome)
                continue
            if not outcome.ok:
                stats.add_skip(outcome.reason or "skip")
                continue
            stats.promoted += 1
            if outcome.manifest_record is not None:
                append_jsonl(layout.manifest_path(edit.edit_type, edit.shard),
                             outcome.manifest_record)


def print_summary(stats: PullStats, label: str) -> None:
    sys.stdout.write(stats.banner(label) + "\n")
    sys.stdout.flush()


__all__ = [
    "PullStats",
    "add_common_args",
    "build_promote_context",
    "collect_edits",
    "print_summary",
    "resolve_obj_filter",
    "run_promote_pool",
    "setup_logging",
]
