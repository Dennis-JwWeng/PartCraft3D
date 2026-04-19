#!/usr/bin/env python3
"""Promote deletion edits from a pipeline_v3 shard into ``H3D_v1``.

Usage::

    python -m scripts.cleaning.h3d_v1.pull_deletion \
        --pipeline-cfg configs/pipeline_v3_shard08.yaml \
        --shard 08 \
        --dataset-root data/H3D_v1 \
        --device cuda:0 \
        --blender /usr/local/bin/blender \
        --workers 8

End-to-end per shard:

1. Walk ``<output_dir>/objects/<NN>/<obj_id>/edits_3d/del_*/``.
2. Filter via ``filter.accept_deletion`` (gate_a status == "pass").
3. **Encode** any accepted edit whose ``after.npz`` is missing — this
   is pipeline_v3's s6b step (Blender → DINOv2 → SLAT + SS encoder),
   inlined here because pipeline_v3 currently does not run s6b
   automatically. Sequentially per device; rerun the CLI on multiple
   GPUs in parallel by sharding obj_ids.
4. ``promote_deletion`` for each (now-ready) edit; append manifest.

``--skip-encode`` skips step 3 — useful when running with ``--dry-run``
or on a CPU-only machine to count what's pending.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

from partcraft.cleaning.h3d_v1.filter import accept_deletion
from partcraft.cleaning.h3d_v1.layout import H3DLayout
from partcraft.cleaning.h3d_v1.pipeline_io import PipelineEdit, load_edit_status
from partcraft.cleaning.h3d_v1.promoter import promote_deletion
from scripts.cleaning.h3d_v1._common import (
    add_common_args,
    build_promote_context,
    collect_edits,
    print_summary,
    resolve_obj_filter,
    run_promote_pool,
    setup_logging,
)

LOG = logging.getLogger("h3d_v1.pull_deletion")


def _add_encode_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--device", default="cuda:0",
                    help='torch device for s6b encoding (e.g. "cuda:0", "cuda:3").')
    ap.add_argument("--blender",
                    default=os.environ.get("BLENDER_PATH", "blender"),
                    help='Blender 3.5 binary path (defaults to $BLENDER_PATH or "blender").')
    ap.add_argument("--ckpt-root", type=Path,
                    default=Path(os.environ.get("PARTCRAFT_CKPT_ROOT",
                                                Path.cwd() / "checkpoints")),
                    help="Trellis checkpoints root for ss_encoder.")
    ap.add_argument("--num-views", type=int, default=40,
                    help="DINOv2 multi-view count for s6b (default 40, matches s6b).")
    ap.add_argument("--encode-work-dir", type=Path,
                    default=Path("outputs/h3d_v1_encode"),
                    help="Scratch dir for Blender renders during encode.")
    ap.add_argument("--skip-encode", action="store_true",
                    help="Don't run s6b for missing after.npz; skip those edits instead.")


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    _add_encode_args(ap)
    return ap.parse_args()


def _normalize_device_env(device: str) -> str:
    """Map ``cuda:N`` → ``cuda`` after pinning ``CUDA_VISIBLE_DEVICES=N``.

    The downstream loaders (``trellis``,
    ``encode_asset.encode_into_SLAT._get_slat_encoder``) check only for
    ``device == "cuda"``; passing ``cuda:0`` causes
    ``partcraft.io.npz_utils.load_ss_encoder`` to silently fall through
    to CPU and the encode step then fails with a torch type mismatch
    on the first conv. Pin the visible device and call everything
    below with the bare ``"cuda"`` token.
    """
    if device == "cuda" or not device.startswith("cuda:"):
        return device
    idx = device.split(":", 1)[1]
    existing = os.environ.get("CUDA_VISIBLE_DEVICES")
    if existing is None:
        os.environ["CUDA_VISIBLE_DEVICES"] = idx
        LOG.info("pinned CUDA_VISIBLE_DEVICES=%s (from --device %s)", idx, device)
    elif existing != idx:
        LOG.warning("--device %s but CUDA_VISIBLE_DEVICES=%s already set; "
                    "trusting existing env", device, existing)
    return "cuda"


def _maybe_load_encoder(ckpt_root: Path, device: str):
    """Lazy-load Trellis ss_encoder; deferred so CPU-only paths skip torch."""
    from scripts.tools.migrate_slat_to_npz import _load_ss_encoder  # noqa: PLC0415
    LOG.info("loading ss_encoder from ckpt_root=%s on %s", ckpt_root, device)
    return _load_ss_encoder(ckpt_root, device)


def _encode_after_npz(edit: PipelineEdit, *, ss_encoder, device: str,
                      work_dir: Path, num_views: int, blender_path: str) -> bool:
    """Run s6b inline for one deletion edit. Returns True on success."""
    import numpy as np  # noqa: PLC0415
    from scripts.tools.migrate_slat_to_npz import _render_and_full_encode  # noqa: PLC0415

    glb = edit.edit_dir / "after_new.glb"
    if not glb.is_file():
        LOG.warning("[%s] no after_new.glb; cannot encode", edit.edit_id)
        return False
    out_npz = edit.edit_dir / "after.npz"
    work_dir.mkdir(parents=True, exist_ok=True)
    name = f"{edit.shard}_{edit.obj_id}_{edit.edit_id}"
    try:
        payload = _render_and_full_encode(
            glb, name, work_dir, ss_encoder, device,
            num_views=num_views, blender_path=blender_path,
        )
    except Exception as exc:  # noqa: BLE001
        LOG.warning("[%s] encode failed: %s", edit.edit_id, exc)
        return False
    np.savez(out_npz, **payload)
    return True


def main() -> int:
    args = _parse_args()
    setup_logging(args.log_level)
    layout = H3DLayout(root=args.dataset_root)
    obj_filter = resolve_obj_filter(args)

    LOG.info("scanning shard=%s for deletion edits", args.shard)
    edits, stats, paths = collect_edits(
        args.pipeline_cfg, args.shard, ("deletion",), obj_filter,
        accept_fn=accept_deletion, load_status=load_edit_status,
    )
    if args.limit:
        edits = edits[: args.limit]
    LOG.info("after filter+limit: %d deletion edits to consider", len(edits))

    if args.dry_run:
        n_missing = sum(1 for e in edits if not (e.edit_dir / "after.npz").is_file())
        LOG.info("dry-run: %d edits already have after.npz, %d need s6b",
                  len(edits) - n_missing, n_missing)
        print_summary(stats, "pull_deletion (dry-run)")
        return 0

    needs_encode = [e for e in edits if not (e.edit_dir / "after.npz").is_file()]
    ready = [e for e in edits if (e.edit_dir / "after.npz").is_file()]
    LOG.info("ready=%d needs_encode=%d", len(ready), len(needs_encode))

    if needs_encode and not args.skip_encode:
        device = _normalize_device_env(args.device)
        ss_encoder = _maybe_load_encoder(args.ckpt_root, device)
        t0 = time.time()
        for i, edit in enumerate(needs_encode, 1):
            t = time.time()
            ok = _encode_after_npz(
                edit, ss_encoder=ss_encoder, device=device,
                work_dir=args.encode_work_dir,
                num_views=args.num_views, blender_path=args.blender,
            )
            dt = time.time() - t
            if ok:
                ready.append(edit)
                LOG.info("[encode %d/%d] %s ok (%.1fs)", i, len(needs_encode), edit.edit_id, dt)
            else:
                stats.add_skip("encode_failed")
                LOG.warning("[encode %d/%d] %s FAILED", i, len(needs_encode), edit.edit_id)
        LOG.info("encode phase: %d ok / %d in %.1fs", len(ready) - (len(edits) - len(needs_encode)),
                 len(needs_encode), time.time() - t0)
    elif needs_encode and args.skip_encode:
        LOG.warning("--skip-encode set; skipping %d edits with missing after.npz", len(needs_encode))
        for e in needs_encode:
            stats.add_skip("missing_after_npz_skip_encode")

    ctx = build_promote_context(paths)
    run_promote_pool(
        ready, layout, ctx,
        promote_fn=lambda e, l: promote_deletion(e, l, ctx=ctx),
        workers=args.workers, stats=stats,
    )
    print_summary(stats, "pull_deletion")
    return 0 if stats.failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
