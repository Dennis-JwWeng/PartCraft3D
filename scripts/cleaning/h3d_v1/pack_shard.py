#!/usr/bin/env python3
"""Build a release tarball for one H3D_v1 shard.

Usage::

    python -m scripts.cleaning.h3d_v1.pack_shard \
        --dataset-root data/H3D_v1 \
        --shard 08 \
        --out releases/H3D_v1__shard08.tar \
        --drop-orig-views --drop-agg-manifest    # flags for HF upload

Includes (per spec §6.5, modulo ``--drop-*`` flags):

* ``_assets/<NN>/``               (object.npz + orig_views/ unless dropped)
* ``<edit_type>/<NN>/``           for every known edit_type
* ``manifests/<edit_type>/<NN>.jsonl``
* ``manifests/all.jsonl``         (unless dropped; unsafe for per-shard
                                    distribution because it aggregates
                                    across the entire local dataset)

Python's ``tarfile`` recognises and preserves hardlinks, so the
in-archive footprint stays close to the on-disk dedup'd footprint
when packed with default settings. A post-pack assertion lists
``tar -tvf`` to verify the count of hardlink entries (``h`` type).
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import tarfile
from pathlib import Path

from partcraft.cleaning.h3d_v1.layout import EDIT_TYPES_ALL, H3DLayout

LOG = logging.getLogger("h3d_v1.pack_shard")


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-root", required=True, type=Path)
    ap.add_argument("--shard", required=True, type=str,
                    help='Two-digit shard string (e.g. "08").')
    ap.add_argument("--out", required=True, type=Path,
                    help="Output tar path (created; parent dir auto-mkdir).")
    ap.add_argument("--compression", default="none",
                    choices=("none", "gz", "bz2", "xz"),
                    help='Tar compression. "none" preserves hardlinks best (recommended).')
    ap.add_argument(
        "--drop-orig-views",
        action="store_true",
        help="Exclude ``_assets/<NN>/<obj>/orig_views/`` from the archive. "
             "Use this for public HF distribution where only ``object.npz`` "
             "is needed as the NPZ hardlink anchor (saves ~200 MB per shard).",
    )
    ap.add_argument(
        "--drop-agg-manifest",
        action="store_true",
        help="Exclude ``manifests/all.jsonl`` from the archive. "
             "Recommended for per-shard distribution, since that file "
             "indexes every shard available on the packing host.",
    )
    ap.add_argument("--log-level", default="INFO",
                    choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return ap.parse_args()


def _iter_shard_paths(
    layout: H3DLayout, shard: str, drop_agg_manifest: bool
) -> list[Path]:
    """Return all files/dirs in spec §6.5 pack list, in deterministic order."""
    root = layout.root
    paths: list[Path] = []
    assets_shard = root / "_assets" / shard
    if assets_shard.is_dir():
        paths.append(assets_shard)
    for et in EDIT_TYPES_ALL:
        d = root / et / shard
        if d.is_dir():
            paths.append(d)
        m = layout.manifest_path(et, shard)
        if m.is_file():
            paths.append(m)
    if not drop_agg_manifest:
        agg = layout.aggregated_manifest()
        if agg.is_file():
            paths.append(agg)
    return paths


def _open_tar(out: Path, compression: str) -> tarfile.TarFile:
    mode = {"none": "w", "gz": "w:gz", "bz2": "w:bz2", "xz": "w:xz"}[compression]
    return tarfile.open(out, mode)


def _make_tar_filter(drop_orig_views: bool):
    """Return a ``tarfile.add`` filter that applies the drop-* flags.

    ``tarfile.add`` calls the filter on every ``TarInfo`` right before
    it would write the entry. Returning ``None`` drops the entry.
    """
    if not drop_orig_views:
        return None

    def _filter(info: tarfile.TarInfo):
        # Entries look like ``_assets/<NN>/<obj>/orig_views`` (dir) or
        # ``_assets/<NN>/<obj>/orig_views/<xx>.png`` (file). Match both
        # by checking for the ``/orig_views`` segment.
        parts = info.name.split("/")
        if "orig_views" in parts:
            return None
        return info

    return _filter


def _verify_hardlinks(out: Path) -> None:
    try:
        proc = subprocess.run(
            ["tar", "-tvf", str(out)], capture_output=True, text=True, check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        LOG.warning("could not run system tar for hardlink check: %s", exc)
        return
    n_hard = sum(1 for line in proc.stdout.splitlines() if line.startswith("h"))
    n_total = sum(1 for line in proc.stdout.splitlines())
    LOG.info("tar listing: %d entries, %d hardlinks (h-type)", n_total, n_hard)
    if n_hard == 0:
        LOG.warning("no hardlinks detected in archive — dedup not preserved")


def main() -> int:
    args = _parse_args()
    logging.basicConfig(level=args.log_level,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S")
    layout = H3DLayout(root=args.dataset_root)

    paths = _iter_shard_paths(layout, args.shard, args.drop_agg_manifest)
    if not paths:
        LOG.error("no shard data found under %s for shard=%s",
                  layout.root, args.shard)
        return 2

    args.out.parent.mkdir(parents=True, exist_ok=True)
    LOG.info(
        "packing %d top-level entries from %s into %s "
        "(compression=%s, drop_orig_views=%s, drop_agg_manifest=%s)",
        len(paths), layout.root, args.out, args.compression,
        args.drop_orig_views, args.drop_agg_manifest,
    )

    tar_filter = _make_tar_filter(drop_orig_views=args.drop_orig_views)

    n_files = 0
    with _open_tar(args.out, args.compression) as tar:
        for p in paths:
            arcname = p.relative_to(layout.root)
            tar.add(p, arcname=str(arcname), recursive=True, filter=tar_filter)
            if p.is_dir():
                for fp in p.rglob("*"):
                    if not fp.is_file():
                        continue
                    if args.drop_orig_views and "orig_views" in fp.parts:
                        continue
                    n_files += 1
            else:
                n_files += 1

    size_mb = args.out.stat().st_size / (1024 * 1024)
    LOG.info("packed: %d files, archive size %.1f MB", n_files, size_mb)
    _verify_hardlinks(args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
