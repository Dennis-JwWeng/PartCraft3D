#!/usr/bin/env python3
"""bench_color_fix.py - isolated re-run of s5_trellis + s6p_flux for
color edits on a sample of shard08 objects, to validate the
``resolve_2d_conditioning`` whitelist fix (Color added to the 2D-cond
whitelist in ``partcraft/pipeline_v3/trellis_utils.py``).

Inputs (read-only, mirrored via symlinks from the live shard root):
  - phase1/parsed.json
  - edits_2d/{eid}_{input,edited}.png    (cached FLUX results, re-used)

Outputs written to an isolated experiment root (default
``outputs/partverse/_experiments/color_fix_exp``).  The live shard root
is NEVER touched.

Only edits of type ``color`` are re-run - this is the bug under test.
For each re-run edit the script writes::

    <exp_root>/objects/<shard>/<obj>/edits_3d/clr_<obj>_<seq>/
        before.npz          (re-generated; deterministic from slat input)
        after.npz           (new: colour-conditioned via FLUX 2D)
        preview_{0..4}.png  (rendered via TRELLIS)

For side-by-side visual inspection the script also writes an HTML diff
report to ``<exp_root>/_report_color_fix.html`` (unless ``--no-report``).

Typical invocation (single GPU)::

    CUDA_VISIBLE_DEVICES=0 \\
    python scripts/tools/bench_color_fix.py \\
        --cfg configs/pipeline_v3_shard08.yaml \\
        --obj-id be1691a3b8484eab823c69e135299e2f \\
        --exp-root outputs/partverse/_experiments/color_fix_exp

Random N-sample across shard08::

    CUDA_VISIBLE_DEVICES=0 \\
    python scripts/tools/bench_color_fix.py \\
        --cfg configs/pipeline_v3_shard08.yaml \\
        --sample 5 --seed 2026

The script reuses the normal ``trellis_3d.run_for_object`` and
``preview_render.render_flux_previews_for_object`` entrypoints, with
the per-object ``iter_flux_specs`` / ``iter_all_specs`` calls
monkey-patched to only yield color specs so nothing else gets re-run.
"""
from __future__ import annotations

import argparse
import logging
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Iterator

import yaml

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from partcraft.pipeline_v3 import trellis_3d, preview_render  # noqa: E402
from partcraft.pipeline_v3.paths import PipelineRoot  # noqa: E402
from partcraft.pipeline_v3.specs import iter_flux_specs as _orig_iter_flux_specs  # noqa: E402
from partcraft.pipeline_v3.specs import iter_all_specs as _orig_iter_all_specs  # noqa: E402
from partcraft.pipeline_v3 import services_cfg as psvc  # noqa: E402


log = logging.getLogger("bench_color_fix")


# --- color-only spec iterators ---------------------------------------------

def _iter_color_flux_specs(ctx) -> Iterator:
    for s in _orig_iter_flux_specs(ctx):
        if s.edit_type == "color":
            yield s


def _iter_color_all_specs(ctx) -> Iterator:
    for s in _orig_iter_all_specs(ctx):
        if s.edit_type == "color":
            yield s


def _patch_color_only() -> None:
    """Swap the two spec iterators so only color edits get processed.

    ``trellis_3d.run_for_object`` iterates ``iter_flux_specs`` twice
    (pending queue + final write); ``preview_render.render_flux_previews_for_object``
    iterates ``iter_all_specs`` (filtered by FLUX_TYPES).  Patching both
    call sites restricts the re-run to colour edits without touching
    any live status.
    """
    trellis_3d.iter_flux_specs = _iter_color_flux_specs  # type: ignore[attr-defined]
    preview_render.iter_flux_specs = _iter_color_flux_specs  # type: ignore[attr-defined]
    preview_render.iter_all_specs = _iter_color_all_specs  # type: ignore[attr-defined]


# --- per-object mirror setup -----------------------------------------------

def _mirror_obj_inputs(live_obj_dir: Path, exp_obj_dir: Path) -> None:
    """Set up ``exp_obj_dir`` with symlinks to the live read-only inputs.

    Writable state files (``edit_status.json``, ``status.json``,
    ``qc.json``) are COPIED, not linked, so mutations during the re-run
    do not leak back into the live shard.

    ``edits_3d/`` is left for the re-run to create fresh.
    """
    exp_obj_dir.mkdir(parents=True, exist_ok=True)

    for sub in ("phase1", "edits_2d", "highlights"):
        src = live_obj_dir / sub
        if not src.is_dir():
            continue
        dst = exp_obj_dir / sub
        if dst.is_symlink() or dst.exists():
            if dst.is_symlink():
                dst.unlink()
            else:
                shutil.rmtree(dst)
        dst.symlink_to(src.resolve(), target_is_directory=True)

    for fname in ("edit_status.json", "status.json", "qc.json", "meta.json"):
        src = live_obj_dir / fname
        if not src.is_file():
            continue
        dst = exp_obj_dir / fname
        shutil.copy2(src, dst)


# --- obj discovery ---------------------------------------------------------

def _discover_color_objs(live_root: Path, shard: str) -> list[str]:
    shard_dir = live_root / "objects" / shard
    if not shard_dir.is_dir():
        return []
    out: list[str] = []
    for od in sorted(shard_dir.iterdir()):
        if not od.is_dir():
            continue
        e2d = od / "edits_2d"
        if not e2d.is_dir():
            continue
        for p in e2d.iterdir():
            if p.name.startswith("clr_") and p.name.endswith("_edited.png"):
                out.append(od.name)
                break
    return out


# --- HTML side-by-side report ----------------------------------------------

def _write_report(
    *,
    exp_root: Path,
    live_root: Path,
    shard: str,
    obj_ids: list[str],
    out_path: Path,
) -> None:
    import html as _html
    from urllib.parse import quote

    rows = []
    for obj in obj_ids:
        live_obj_3d = live_root / "objects" / shard / obj / "edits_3d"
        exp_obj_3d = exp_root / "objects" / shard / obj / "edits_3d"
        if not exp_obj_3d.is_dir():
            continue
        e2d_dir = live_root / "objects" / shard / obj / "edits_2d"
        for eid_dir in sorted(exp_obj_3d.iterdir()):
            eid = eid_dir.name
            if not eid.startswith("clr_"):
                continue
            live_previews = [live_obj_3d / eid / f"preview_{i}.png"
                             for i in range(5)]
            exp_previews = [eid_dir / f"preview_{i}.png" for i in range(5)]
            in2d = e2d_dir / f"{eid}_input.png"
            ed2d = e2d_dir / f"{eid}_edited.png"

            def _link(p: Path) -> str:
                rel = os.path.relpath(p, out_path.parent)
                return quote(rel)

            def _img(p: Path, w: int = 110) -> str:
                if not p.is_file():
                    return (f'<div class="miss" style="width:{w}px;'
                            f'height:{w}px">missing</div>')
                return (f'<img src="{_link(p)}" width="{w}"'
                        f' title="{_html.escape(str(p))}"/>')

            in_img = _img(in2d, 150)
            ed_img = _img(ed2d, 150)
            live_row = "".join(_img(p) for p in live_previews)
            exp_row = "".join(_img(p) for p in exp_previews)
            rows.append(
                f'<section class="card">'
                f'<h3>{_html.escape(obj)} &middot; {_html.escape(eid)}</h3>'
                f'<div class="twod"><div>input<br>{in_img}</div>'
                f'<div>edited (FLUX)<br>{ed_img}</div></div>'
                f'<div class="row"><div class="lbl">LIVE (before fix):</div>'
                f'<div class="frames">{live_row}</div></div>'
                f'<div class="row"><div class="lbl">EXP (with fix):</div>'
                f'<div class="frames">{exp_row}</div></div>'
                f'</section>'
            )

    css = (
        "body{font:13px/1.4 system-ui,sans-serif;background:#111;color:#ddd;"
        "padding:16px}"
        ".card{background:#1b1b1f;border:1px solid #333;border-radius:8px;"
        "padding:12px;margin-bottom:14px}"
        "h3{margin:0 0 6px;font-size:14px;color:#9cf}"
        ".twod{display:flex;gap:10px;margin:6px 0 10px;font-size:11px;"
        "color:#9c9}"
        ".row{display:flex;gap:10px;align-items:center;margin:4px 0}"
        ".lbl{width:160px;color:#fc9;font-weight:600}"
        ".frames{display:flex;gap:4px}"
        ".frames img{border:1px solid #333;border-radius:3px}"
        ".miss{display:flex;align-items:center;justify-content:center;"
        "color:#f66;background:#222;border:1px dashed #444;font-size:10px}"
    )
    out_path.write_text(
        "<!doctype html><meta charset=utf-8><title>color_fix bench</title>"
        f"<style>{css}</style>"
        f"<h1>Color-fix bench &mdash; {len(rows)} edits from {len(obj_ids)} "
        "objects</h1>"
        "<p>LIVE = current shard08 disk state (before fix); "
        "EXP = re-run with Color added to the resolve_2d_conditioning "
        "whitelist in trellis_utils.py.</p>"
        + "\n".join(rows)
    )
    log.info("[report] wrote %s (%d cards)", out_path, len(rows))


# --- main ------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True, type=Path,
                    help="pipeline_v3 shard yaml (for dataset/ckpt paths)")
    ap.add_argument("--live-root", type=Path, default=Path(
        "outputs/partverse/shard08/mode_e_text_align"))
    ap.add_argument("--exp-root", type=Path, default=Path(
        "outputs/partverse/_experiments/color_fix_exp"))
    ap.add_argument("--shard", default="08")
    ap.add_argument("--obj-id", action="append", default=[],
                    help="explicit obj id; repeatable")
    ap.add_argument("--sample", type=int, default=0,
                    help="if set, pick N random color-eligible objs from live")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--skip-render", action="store_true",
                    help="run s5 only; do NOT re-render previews (faster)")
    ap.add_argument("--no-report", action="store_true")
    ap.add_argument("--repaint-mode", default=None,
                    choices=["image", "text", "interleaved"],
                    help="override cfg services.image_edit.repaint_mode")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    cfg = yaml.safe_load(args.cfg.read_text())

    if args.repaint_mode:
        # Deep-patch cfg so psvc.trellis_image_edit_flat sees the override.
        svc = cfg.setdefault("services", {}).setdefault("image_edit", {})
        svc["repaint_mode"] = args.repaint_mode
        log.info("[cfg] repaint_mode overridden -> %s", args.repaint_mode)

    if args.obj_id:
        target_ids = list(args.obj_id)
    elif args.sample > 0:
        candidates = _discover_color_objs(args.live_root, args.shard)
        random.seed(args.seed)
        random.shuffle(candidates)
        target_ids = candidates[: args.sample]
        log.info("[sample] picked %d/%d color-eligible objs with seed=%d",
                 len(target_ids), len(candidates), args.seed)
    else:
        ap.error("must pass --obj-id or --sample N")

    log.info("[target] %d obj: %s",
             len(target_ids), ", ".join(target_ids[:5])
             + (" ..." if len(target_ids) > 5 else ""))

    args.exp_root.mkdir(parents=True, exist_ok=True)
    exp_proot = PipelineRoot(args.exp_root)
    exp_proot.ensure()

    live_proot = PipelineRoot(args.live_root)

    data_cfg = cfg.get("data") or {}
    images_root = Path(data_cfg["images_root"])
    mesh_root = Path(data_cfg["mesh_root"])

    exp_ctxs = []
    for obj_id in target_ids:
        live_obj = live_proot.object_dir(args.shard, obj_id)
        if not (live_obj / "phase1" / "parsed.json").is_file():
            log.warning("[skip] %s: no phase1/parsed.json in live", obj_id)
            continue
        exp_obj = exp_proot.object_dir(args.shard, obj_id)
        _mirror_obj_inputs(live_obj, exp_obj)
        ctx = exp_proot.context(
            args.shard, obj_id,
            mesh_npz=mesh_root / args.shard / f"{obj_id}.npz",
            image_npz=images_root / args.shard / f"{obj_id}.npz",
        )
        exp_ctxs.append(ctx)

    if not exp_ctxs:
        log.error("no valid objs to run")
        sys.exit(2)

    _patch_color_only()

    log.info("[s5] running TRELLIS 3D edits (color only) on %d obj",
             len(exp_ctxs))
    trellis_3d.run(
        exp_ctxs,
        cfg=cfg,
        images_root=images_root,
        mesh_root=mesh_root,
        shard=args.shard,
        seed=args.seed,
        debug=args.debug,
        force=True,
        logger=logging.getLogger("bench.s5"),
    )

    if not args.skip_render:
        log.info("[s6p] rendering 5-view previews (color only)")
        ckpt = psvc.image_edit_service(cfg).get(
            "trellis_text_ckpt", "checkpoints/TRELLIS-text-xlarge")
        preview_render.render_flux_previews_batch(
            exp_ctxs,
            ckpt=ckpt,
            force=True,
            logger=logging.getLogger("bench.s6p"),
        )

    if not args.no_report:
        report_path = args.exp_root / "_report_color_fix.html"
        _write_report(
            exp_root=args.exp_root,
            live_root=args.live_root,
            shard=args.shard,
            obj_ids=[c.obj_id for c in exp_ctxs],
            out_path=report_path,
        )


if __name__ == "__main__":
    main()
