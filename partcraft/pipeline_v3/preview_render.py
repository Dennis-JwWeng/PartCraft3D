"""Step s6p — canonical-view preview renders for all edit types (pre-VLM-gate).

Renders preview_{0..4}.png for every non-identity edit using the same
VIEW_INDICES cameras as the phase1 VLM overview.  These previews are
consumed by sq3 (VLM quality gate) without needing a full 40-view render,
and later by H3D_v1 ``pull_deletion``/``pull_addition`` as the source for
``after.png``.

Route by type:
  deletion   → Blender renders after_new.glb (after state = object minus parts)
  addition   → copies source_del's already-rendered preview_{k}.png (addition
               before-state = del after-state)
  mod/scl/mat/glb → TRELLIS decode+render from after.npz

Output per edit: ``edits_3d/<edit_id>/preview_{k}.png`` for k in target slots.
Step key: s6p_preview (split entrypoints: ``s6p_del`` / ``s6p_flux``).

Fast-path (``best_view_only=True`` in ``render_del_previews_{for_object,batch}``
or CLI flag ``--best-view-only``): renders/copies only the single canonical
slot picked by ``edit_status.json → gates.A.vlm.best_view`` per edit (fallback
:data:`DEFAULT_FRONT_VIEW_INDEX`, slot 4).  Used to backfill ``after.png`` on
shards where the pipeline's ``preview_del`` stage was skipped — the resulting
``preview_{k}.png`` matches exactly the slot that H3D_v1 promoter hardlinks as
``after.png`` (see ``partcraft.cleaning.h3d_v1.promoter._views_block``), so
``pull_deletion`` → ``pull_addition`` then proceeds without any upstream
pipeline rerun.  Runbook: ``docs/runbooks/h3d-v1-promote.md`` §2b.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

_ROOT = Path(__file__).resolve().parents[2]


def _encode_asset_script() -> str:
    """Return absolute path to encode_asset/blender_script/render.py."""
    p = _ROOT / "third_party" / "encode_asset" / "blender_script" / "render.py"
    if not p.is_file():
        raise FileNotFoundError(f"encode_asset render script not found: {p}")
    return str(p)


sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts" / "standalone"))
sys.path.insert(0, str(_ROOT / "third_party"))

from .paths import ObjectContext
from .specs import VIEW_INDICES
from .status import update_step, STATUS_OK, STATUS_FAIL
from .qc_io import is_gate_a_failed
from .edit_status_io import edit_needs_step, update_edit_stage, obj_needs_stage


@dataclass
class PreviewResult:
    obj_id: str
    n_ok: int = 0
    n_fail: int = 0
    n_skip: int = 0
    error: str | None = None


def _load_trellis_pipeline(ckpt: str, logger: logging.Logger):
    """Load TRELLIS pipeline onto GPU."""
    from trellis.pipelines import TrellisTextTo3DPipeline  # type: ignore
    logger.info("[s6p] loading TRELLIS %s", ckpt)
    pipe = TrellisTextTo3DPipeline.from_pretrained(ckpt)
    pipe.cuda()
    logger.info("[s6p] pipeline ready")
    return pipe


def _all_previews_exist(edit_dir: Path, n: int = 5) -> bool:
    """Return True if all preview_{0..n-1}.png files exist."""
    return all((edit_dir / f"preview_{i}.png").is_file() for i in range(n))


DEFAULT_FRONT_VIEW_INDEX: int = 4  # fallback slot when gate_a.vlm.best_view unavailable


def _best_view_slot_for_edit(
    ctx: ObjectContext,
    edit_id: str,
    *,
    default: int = DEFAULT_FRONT_VIEW_INDEX,
) -> int:
    """Return the canonical preview slot (0..N_VIEWS-1) for ``edit_id``.

    Source of truth: ``edit_status.json -> edits[edit_id].gates.A.vlm.best_view``
    (pipeline_v3 text_gen_gate_a).  For addition edits, Gate A is synthesised
    from the paired deletion, so we fall back to the paired ``del_*``'s
    best_view.  Final fallback is ``default`` (front view = 4).

    Keeps the same resolution order used by
    ``partcraft.cleaning.h3d_v1.promoter._views_block`` so single-view
    previews land at the exact slot H3D_v1 later hardlinks as ``after.png``.
    """
    from .edit_status_io import load_edit_status

    def _pick(eid: str) -> int | None:
        es = load_edit_status(ctx)
        e = (es.get("edits") or {}).get(eid) or {}
        ga = (e.get("gates") or {}).get("A")
        vlm = ga.get("vlm") if isinstance(ga, dict) else None
        bv = vlm.get("best_view") if isinstance(vlm, dict) else None
        if isinstance(bv, int) and 0 <= bv < len(VIEW_INDICES):
            return int(bv)
        return None

    k = _pick(edit_id)
    if k is None and edit_id.startswith("add_"):
        k = _pick("del_" + edit_id[4:])
    return k if k is not None else int(default)


def _slot_previews_exist(edit_dir: Path, slots: list[int]) -> bool:
    """Return True iff every ``preview_{k}.png`` for k in ``slots`` exists."""
    return bool(slots) and all((edit_dir / f"preview_{k}.png").is_file() for k in slots)


def _write_preview_images(edit_dir: Path, imgs: list[np.ndarray]) -> None:
    """Save list of BGR images as preview_{0..}.png.

    Uses cv2.imwrite (not PIL) to preserve BGR channel order correctly.
    run_blender() returns BGR; PIL.Image.fromarray() would treat it as RGB
    and silently swap R/B channels in the saved file.
    """
    import cv2 as _cv2
    for i, img in enumerate(imgs):
        _cv2.imwrite(str(edit_dir / f"preview_{i}.png"), img)


def _write_preview_images_by_slot(edit_dir: Path, imgs_by_slot: dict[int, np.ndarray]) -> None:
    """Save BGR images keyed by canonical slot as preview_{slot}.png."""
    import cv2 as _cv2
    for slot, img in imgs_by_slot.items():
        _cv2.imwrite(str(edit_dir / f"preview_{int(slot)}.png"), img)


def _render_ply_views(
    ply_path: Path,
    frames: list[dict],
    blender: str,
    resolution: int,
    samples: int = 32,
) -> list[np.ndarray]:
    """Render a single PLY file at the given camera frames using Blender.

    Uses a temporary directory with a copy of the PLY as part_0.ply to satisfy
    run_blender's expected parts_dir layout (part_*.ply convention).

    ``samples`` controls Cycles sample count.  Default is 32 (GPU, denoising ON)
    which matches the minimum quality needed for VLM judgment in sq3.
    The dataset prerender uses 128 samples; 32+denoise gives acceptable sharpness
    (~15% below original) without excessive render time (~45s vs 14s per edit).
    """
    from partcraft.render.overview import run_blender as _run_blender
    with tempfile.TemporaryDirectory(prefix="pcv2_s6p_ply_") as tmp:
        tmp_path = Path(tmp)
        # run_blender expects part_*.ply files in parts_dir
        shutil.copy2(ply_path, tmp_path / "part_0.ply")
        imgs = _run_blender(
            tmp_path, blender, resolution,
            [[128, 128, 128]],   # palette unused in vertex-color mode
            frames,
            use_vertex_colors=True,
            samples=samples,
        )
    return imgs


def _read_camera_views_from_npz(
    image_npz: Path,
    *,
    slots: list[int] | None = None,
) -> list[dict]:
    """Extract yaw/pitch/radius/fov camera params from image NPZ.

    By default returns one entry per ``VIEW_INDICES`` slot (canonical 5 views).
    If ``slots`` is given, only those 0..N_VIEWS-1 indices are returned, in
    the caller-supplied order.  Used by the best-view-only preview path where
    we only need a single canonical camera per edit.
    """
    import math
    npz = np.load(str(image_npz), allow_pickle=True)
    frames = json.loads(bytes(npz["transforms.json"]))["frames"]
    if slots is None:
        src_slots = list(range(len(VIEW_INDICES)))
    else:
        src_slots = [int(s) for s in slots]
    views = []
    for s in src_slots:
        if not (0 <= s < len(VIEW_INDICES)):
            continue
        vi = VIEW_INDICES[s]
        if vi >= len(frames):
            continue
        frame = frames[vi]
        m = frame["transform_matrix"]
        # Camera position from c2w matrix (last column, first 3 rows)
        cx, cy, cz = m[0][3], m[1][3], m[2][3]
        r = math.sqrt(cx ** 2 + cy ** 2 + cz ** 2)
        if r < 1e-6:
            continue
        views.append({
            "yaw":    math.atan2(cy, cx),   # atan2(y, x): encode_asset x=r·cos(yaw)·cos(p), y=r·sin(yaw)·cos(p)
            "pitch":  math.asin(max(-1.0, min(1.0, cz / r))),
            "radius": r,
            "fov":    frame.get("camera_angle_x", math.radians(40)),
        })
    return views


def _read_scene_normalization(image_npz: Path) -> tuple[float, list[float]] | tuple[None, None]:
    """Read prerender normalization scale+offset from image_npz transforms.json.

    The encode_asset render script saves scale/offset from normalize_scene() into
    transforms.json so we can replay the *same* normalization on partial meshes
    (e.g. after deletion) and keep the object at the identical apparent size.
    Returns (scale, [ox, oy, oz]) or (None, None) if not stored.
    """
    try:
        npz = np.load(str(image_npz), allow_pickle=True)
        meta = json.loads(bytes(npz["transforms.json"]))
        scale = meta.get("scale")
        offset = meta.get("offset")   # [x, y, z]
        if scale is not None and offset is not None and len(offset) == 3:
            return float(scale), [float(v) for v in offset]
    except Exception:
        pass
    return None, None


def _render_glb_views(
    glb_path: Path,
    image_npz: Path,
    encode_script: str,
    blender: str,
    resolution: int,
    *,
    view_slots: list[int] | None = None,
    force_cpu: bool = False,
    cuda_device: str | None = None,
) -> dict[int, np.ndarray]:
    """Render GLB at a subset of VIEW_INDICES cameras.

    ``view_slots`` is a list of canonical slot indices (0..N_VIEWS-1).  None
    means "all N_VIEWS canonical slots" (legacy behaviour).  Returns a mapping
    ``{slot: BGR_image}`` so callers can write directly to ``preview_{slot}.png``.

    Uses the *same* normalization (scale + offset) that was applied when the
    original object was pre-rendered, so deleted-part renders appear at the
    correct apparent size instead of being zoomed-in to the remaining bbox.
    """
    import subprocess
    import cv2 as _cv2

    if view_slots is None:
        slots = list(range(len(VIEW_INDICES)))
    else:
        slots = [int(s) for s in view_slots if 0 <= int(s) < len(VIEW_INDICES)]
        if not slots:
            raise RuntimeError(f"no valid view_slots in {view_slots!r}")

    views = _read_camera_views_from_npz(image_npz, slots=slots)
    if len(views) != len(slots):
        raise RuntimeError(
            f"Expected {len(slots)} camera views, got {len(views)}"
        )

    norm_scale, norm_offset = _read_scene_normalization(image_npz)

    with tempfile.TemporaryDirectory(prefix="pcv2_s6p_glb_") as tmp:
        tmp_path = Path(tmp)
        cmd = [
            blender, "-b", "-P", encode_script, "--",
            "--object",        str(glb_path),
            "--output_folder", str(tmp_path),
            "--views",         json.dumps(views),
            "--resolution",    str(resolution),
        ]
        if norm_scale is not None and norm_offset is not None:
            cmd += ["--normalize_scale", str(norm_scale),
                    "--normalize_offset", str(norm_offset[0]), str(norm_offset[1]), str(norm_offset[2])]
        env = dict(os.environ)
        if force_cpu:
            env["CUDA_VISIBLE_DEVICES"] = ""  # hide all GPUs → Blender falls back to CPU
        elif cuda_device is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(cuda_device)  # pin to specific GPU
        result = subprocess.run(
            cmd,
            capture_output=True, text=True, timeout=600,
            env=env,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"encode_asset Blender failed:\n{result.stderr[-400:]}"
            )

        out: dict[int, np.ndarray] = {}
        for i, slot in enumerate(slots):
            png = tmp_path / f"{i:03d}.png"
            if not png.is_file():
                raise RuntimeError(f"Missing render output: {png}")
            arr = np.array(Image.open(str(png)).convert("RGBA"), dtype=np.float32) / 255.0
            r_ch, g_ch, b_ch, a = arr[..., 0], arr[..., 1], arr[..., 2], arr[..., 3]
            rgb = np.stack([
                (r_ch * a + (1 - a)) * 255,
                (g_ch * a + (1 - a)) * 255,
                (b_ch * a + (1 - a)) * 255,
            ], axis=-1).clip(0, 255).astype(np.uint8)
            out[slot] = _cv2.cvtColor(rgb, _cv2.COLOR_RGB2BGR)
        return out


def _render_glb_five_views(
    glb_path: Path,
    image_npz: Path,
    encode_script: str,
    blender: str,
    resolution: int,
    *,
    force_cpu: bool = False,
    cuda_device: str | None = None,
) -> list[np.ndarray]:
    """Legacy 5-view wrapper returning a list aligned with VIEW_INDICES."""
    out = _render_glb_views(
        glb_path, image_npz, encode_script, blender, resolution,
        view_slots=list(range(len(VIEW_INDICES))),
        force_cpu=force_cpu, cuda_device=cuda_device,
    )
    return [out[k] for k in range(len(VIEW_INDICES))]


def _render_trellis_five_views(
    npz_path: Path,
    pipeline,
    frames: list[dict],
    resolution: int,
) -> list[np.ndarray]:
    """Render 5 views from a SLAT npz using TRELLIS pipeline.

    render_one_view() returns RGB; convert to BGR so _write_preview_images()
    (which calls cv2.imwrite) writes the correct channel order.
    """
    import cv2 as _cv2
    from render_phase1v2_3d_results import render_one_view as _render_one_view, load_slat as _load_slat  # type: ignore
    slat = _load_slat(npz_path)
    imgs = []
    for frame in frames:
        img = _render_one_view(pipeline, slat, frame, resolution)
        imgs.append(_cv2.cvtColor(img, _cv2.COLOR_RGB2BGR))
    return imgs


def _iter_addition_edits(ctx: ObjectContext):
    """Yield (edit_id, meta_dict) for addition edits discovered from meta.json on disk."""
    if not ctx.edits_3d_dir.is_dir():
        return
    for add_dir in sorted(ctx.edits_3d_dir.iterdir()):
        if not add_dir.is_dir():
            continue
        meta_path = add_dir / "meta.json"
        if not meta_path.is_file():
            continue
        try:
            meta = json.loads(meta_path.read_text())
        except Exception:
            continue
        if meta.get("edit_type") == "addition":
            yield add_dir.name, meta


def render_trellis_previews_for_object(
    ctx: ObjectContext,
    *,
    pipeline,          # TRELLIS pipeline (None if no SLAT edits present)
    blender: str,
    resolution: int = 518,
    force: bool = False,
    logger: logging.Logger | None = None,
) -> PreviewResult:
    """Render preview_{0..4}.png for every non-identity edit in this object."""
    log = logger or logging.getLogger("pipeline_v3.s6p")
    res = PreviewResult(obj_id=ctx.obj_id)

    if ctx.image_npz is None or not ctx.image_npz.is_file():
        update_step(ctx, "s6p_preview", status=STATUS_FAIL,
                    error="missing_image_npz")
        res.error = "missing_image_npz"
        return res

    from partcraft.render.overview import load_views_from_npz
    orig_imgs, frames = load_views_from_npz(ctx.image_npz, VIEW_INDICES)  # orig_imgs: BGR, reused for add previews

    if not ctx.edits_3d_dir.is_dir():
        update_step(ctx, "s6p_preview", status=STATUS_OK, n=0,
                    reason="no_edits_3d")
        return res

    t0 = time.time()

    # --- deletion edits ---
    from .specs import iter_deletion_specs
    for spec in iter_deletion_specs(ctx):
        if prereq_map is not None:
            if not edit_needs_step(ctx, spec.edit_id, "s6p", prereq_map, force=force):
                res.n_skip += 1
                continue
        else:
            if is_gate_a_failed(ctx, spec.edit_id):
                res.n_skip += 1
                continue
            edit_dir_chk = ctx.edit_3d_dir(spec.edit_id)
            if _all_previews_exist(edit_dir_chk) and not force:
                res.n_skip += 1
                continue
        edit_dir = ctx.edit_3d_dir(spec.edit_id)
        a_ply = edit_dir / "after.ply"
        after_glb = edit_dir / "after_new.glb"
        if not a_ply.is_file() and not after_glb.is_file():
            log.warning("[s6p] del %s: both after.ply and after_new.glb missing", spec.edit_id)
            res.n_fail += 1
            continue
        try:
            if after_glb.is_file():
                imgs = _render_glb_five_views(
                    after_glb, ctx.image_npz, _encode_asset_script(),
                    blender, resolution,
                )
            else:
                log.debug("[s6p] del %s: no after_new.glb, PLY fallback", spec.edit_id)
                imgs = _render_ply_views(a_ply, frames, blender, resolution, samples=32)
            _write_preview_images(edit_dir, imgs)
            res.n_ok += 1
        except Exception as e:
            log.warning("[s6p] del %s: %s", spec.edit_id, e)
            res.n_fail += 1

    # --- addition edits (use source_del's before.ply as after state) ---
    for add_id, meta in _iter_addition_edits(ctx):
        if prereq_map is not None:
            if not edit_needs_step(ctx, add_id, "s6p", prereq_map, force=force):
                res.n_skip += 1
                continue
        else:
            if is_gate_a_failed(ctx, add_id):
                res.n_skip += 1
                continue
            add_dir_chk = ctx.edit_3d_dir(add_id)
            if _all_previews_exist(add_dir_chk) and not force:
                res.n_skip += 1
                continue
        add_dir = ctx.edit_3d_dir(add_id)
        source_del_id = meta.get("source_del_id")
        if not source_del_id:
            log.warning("[s6p] add %s: no source_del_id in meta", add_id)
            res.n_fail += 1
            continue
        # add preview = "before-add" state = del's already-rendered preview images.
        # The del edit already rendered its after mesh (object with part missing) as
        # preview_*.png; we copy those directly — no Blender call needed.
        del_dir = ctx.edit_3d_dir(source_del_id)
        del_previews = sorted(del_dir.glob("preview_*.png"))
        if not del_previews:
            log.warning("[s6p] add %s: del %s has no preview_*.png yet", add_id, source_del_id)
            res.n_fail += 1
            continue
        try:
            for src in del_previews:
                shutil.copy2(src, add_dir / src.name)
            res.n_ok += 1
        except Exception as e:
            log.warning("[s6p] add %s: %s", add_id, e)
            res.n_fail += 1

    # --- SLAT-based edits (mod, scl, mat, glb) ---
    from .specs import iter_all_specs
    PLY_TYPES = {"deletion", "addition", "identity"}
    for spec in iter_all_specs(ctx):
        if spec.edit_type in PLY_TYPES:
            continue
        if prereq_map is not None:
            if not edit_needs_step(ctx, spec.edit_id, "s6p", prereq_map, force=force):
                res.n_skip += 1
                continue
        else:
            if is_gate_a_failed(ctx, spec.edit_id):
                res.n_skip += 1
                continue
            edit_dir_chk = ctx.edit_3d_dir(spec.edit_id)
            if _all_previews_exist(edit_dir_chk) and not force:
                res.n_skip += 1
                continue
        edit_dir = ctx.edit_3d_dir(spec.edit_id)
        a_npz = edit_dir / "after.npz"
        if not a_npz.is_file():
            log.warning("[s6p] %s %s: after.npz missing", spec.edit_type, spec.edit_id)
            res.n_fail += 1
            continue
        if pipeline is None:
            log.error("[s6p] TRELLIS pipeline not loaded but needed for %s", spec.edit_id)
            res.n_fail += 1
            continue
        try:
            imgs = _render_trellis_five_views(a_npz, pipeline, frames, resolution)
            _write_preview_images(edit_dir, imgs)
            res.n_ok += 1
        except Exception as e:
            log.warning("[s6p] %s %s: %s", spec.edit_type, spec.edit_id, e)
            res.n_fail += 1

    update_step(
        ctx, "s6p_preview",
        status=STATUS_OK if res.n_fail == 0 else STATUS_FAIL,
        n_ok=res.n_ok, n_fail=res.n_fail, n_skip=res.n_skip,
        wall_s=round(time.time() - t0, 2),
    )
    return res


def _has_flux_or_trellis_edits(ctxs: list[ObjectContext]) -> bool:
    """Check whether any object has SLAT-based (non-PLY) edits needing TRELLIS."""
    from .specs import iter_all_specs
    PLY_TYPES = {"deletion", "addition", "identity"}
    for ctx in ctxs:
        for spec in iter_all_specs(ctx):
            if spec.edit_type not in PLY_TYPES:
                a_npz = ctx.edit_3d_dir(spec.edit_id) / "after.npz"
                if a_npz.is_file():
                    return True
    return False


def render_trellis_previews_batch(
    ctxs: Iterable[ObjectContext],
    *,
    ckpt: str = "checkpoints/TRELLIS-text-xlarge",
    blender: str = "blender",
    resolution: int = 518,
    force: bool = False,
    logger: logging.Logger | None = None,
) -> list[PreviewResult]:
    """Batch entry point. Lazily loads TRELLIS only if SLAT edits are present."""
    log = logger or logging.getLogger("pipeline_v3.s6p")
    log.info("[s6p] CUDA_VISIBLE_DEVICES=%s",
             os.environ.get("CUDA_VISIBLE_DEVICES"))

    ctx_list = list(ctxs)
    # Filter using per-edit status (authoritative)
    from .specs import iter_flux_specs, iter_deletion_specs
    def _s6p_obj_pending(c):
        flux_ids = [sp.edit_id for sp in iter_flux_specs(c)]
        del_ids  = [sp.edit_id for sp in iter_deletion_specs(c)]
        add_ids  = [aid for aid, _ in _iter_addition_edits(c)]
        all_ids  = flux_ids + del_ids + add_ids
        return not all_ids or obj_needs_stage(c, all_ids, "s6p", prereq_map or {}, force=force)
    pending = [c for c in ctx_list if force or _s6p_obj_pending(c)]
    pending_set = set(pending)
    done = [c for c in ctx_list if c not in pending_set]

    pipeline = None
    if _has_flux_or_trellis_edits(pending):
        pipeline = _load_trellis_pipeline(ckpt, log)

    results: list[PreviewResult] = [PreviewResult(c.obj_id) for c in done]
    for ctx in pending:
        results.append(render_trellis_previews_for_object(
            ctx, pipeline=pipeline, blender=blender,
            resolution=resolution, force=force, logger=log,
        ))
    return results


__all__ = [
    "PreviewResult",
    "render_trellis_previews_for_object", "render_trellis_previews_batch",
    "render_del_previews_for_object", "render_del_previews_batch",
    "render_flux_previews_for_object", "render_flux_previews_batch",
]


# ─────────────────── split entry points (s6p_del / s6p_flux) ────────────────

def render_del_previews_for_object(
    ctx: ObjectContext,
    *,
    blender: str,
    resolution: int = 518,
    prereq_map: dict[str, str | None] | None = None,
    force: bool = False,
    cuda_device: str | None = None,
    progress_bar=None,
    best_view_only: bool = False,
    logger: logging.Logger | None = None,
) -> PreviewResult:
    """Render preview_{k}.png for deletion and addition edits only.

    Step key: s6p_del.  CPU/Blender only — no TRELLIS pipeline needed.
    Deletion: renders after_new.glb (or after.ply fallback) via Blender.
    Addition: copies source deletion's preview_*.png (no Blender call).

    ``best_view_only=True`` renders / copies only the single canonical slot
    picked by gate_a.vlm.best_view (fallback DEFAULT_FRONT_VIEW_INDEX) per
    edit.  This cuts Blender cost ~5x and is enough for H3D_v1 promotion,
    which only hardlinks ``preview_{best_view_index}.png``.
    """
    log = logger or logging.getLogger("pipeline_v3.s6p_del")
    res = PreviewResult(obj_id=ctx.obj_id)

    # Per-edit gate check: skip object only if no del/add edits need s6p
    from .specs import iter_deletion_specs as _iter_del_specs
    _del_add_ids = (
        [sp.edit_id for sp in _iter_del_specs(ctx)]
        + [aid for aid, _ in _iter_addition_edits(ctx)]
    )
    if _del_add_ids and not force and not obj_needs_stage(
        ctx, _del_add_ids, "s6p", prereq_map or {}, force=force
    ):
        return res

    if ctx.image_npz is None or not ctx.image_npz.is_file():
        update_step(ctx, "s6p_del", status=STATUS_FAIL, error="missing_image_npz")
        res.error = "missing_image_npz"
        return res

    from partcraft.render.overview import load_views_from_npz
    _, frames = load_views_from_npz(ctx.image_npz, VIEW_INDICES)

    if not ctx.edits_3d_dir.is_dir():
        update_step(ctx, "s6p_del", status=STATUS_OK, n=0, reason="no_edits_3d")
        return res

    t0 = time.time()

    # --- deletion edits ---
    from .specs import iter_deletion_specs
    for spec in iter_deletion_specs(ctx):
        if prereq_map is not None:
            if not edit_needs_step(ctx, spec.edit_id, "s6p", prereq_map, force=force):
                res.n_skip += 1
                continue
        else:
            if is_gate_a_failed(ctx, spec.edit_id):
                res.n_skip += 1
                continue
            edit_dir_chk = ctx.edit_3d_dir(spec.edit_id)
            if _all_previews_exist(edit_dir_chk) and not force:
                res.n_skip += 1
                continue
        edit_dir = ctx.edit_3d_dir(spec.edit_id)
        a_ply = edit_dir / "after.ply"
        after_glb = edit_dir / "after_new.glb"
        if not a_ply.is_file() and not after_glb.is_file():
            log.warning("[s6p_del] del %s: both after.ply and after_new.glb missing", spec.edit_id)
            res.n_fail += 1
            continue
        # Resolve target slots: single best-view in fast mode, else all 5.
        if best_view_only:
            k = _best_view_slot_for_edit(ctx, spec.edit_id)
            target_slots = [k]
        else:
            target_slots = list(range(len(VIEW_INDICES)))
        # Re-check skip with slot-precise existence for fast mode.
        if (
            best_view_only
            and prereq_map is None
            and not force
            and _slot_previews_exist(edit_dir, target_slots)
        ):
            res.n_skip += 1
            if progress_bar is not None:
                progress_bar.update(1)
            continue
        try:
            if after_glb.is_file():
                imgs_by_slot = _render_glb_views(
                    after_glb, ctx.image_npz, _encode_asset_script(),
                    blender, resolution,
                    view_slots=target_slots,
                    cuda_device=cuda_device,
                )
                _write_preview_images_by_slot(edit_dir, imgs_by_slot)
            else:
                # PLY fallback renders the exact camera frames we ask for.
                sub_frames = [frames[k] for k in target_slots]
                imgs = _render_ply_views(a_ply, sub_frames, blender, resolution, samples=32)
                _write_preview_images_by_slot(
                    edit_dir, {k: img for k, img in zip(target_slots, imgs)}
                )
            res.n_ok += 1
            update_edit_stage(ctx, spec.edit_id, spec.edit_type, "s6p", status="done")
        except Exception as e:
            log.warning("[s6p_del] del %s: %s", spec.edit_id, e)
            res.n_fail += 1
            update_edit_stage(ctx, spec.edit_id, spec.edit_type, "s6p",
                              status="error", reason=str(e)[:200])
        finally:
            if progress_bar is not None:
                progress_bar.update(1)

    # --- addition edits (copy source del's preview_*.png) ---
    for add_id, meta in _iter_addition_edits(ctx):
        if prereq_map is not None:
            if not edit_needs_step(ctx, add_id, "s6p", prereq_map, force=force):
                res.n_skip += 1
                continue
        else:
            if is_gate_a_failed(ctx, add_id):
                res.n_skip += 1
                continue
            add_dir_chk = ctx.edit_3d_dir(add_id)
            if _all_previews_exist(add_dir_chk) and not force:
                res.n_skip += 1
                continue
        add_dir = ctx.edit_3d_dir(add_id)
        source_del_id = meta.get("source_del_id")
        if not source_del_id:
            log.warning("[s6p_del] add %s: no source_del_id in meta", add_id)
            res.n_fail += 1
            continue
        del_dir = ctx.edit_3d_dir(source_del_id)
        if best_view_only:
            k_add = _best_view_slot_for_edit(ctx, add_id)
            src_png = del_dir / f"preview_{k_add}.png"
            if not src_png.is_file():
                log.warning(
                    "[s6p_del] add %s: del %s missing preview_%d.png (best-view-only)",
                    add_id, source_del_id, k_add,
                )
                res.n_fail += 1
                continue
            del_previews = [src_png]
        else:
            del_previews = sorted(del_dir.glob("preview_*.png"))
            if not del_previews:
                log.warning("[s6p_del] add %s: del %s has no preview_*.png yet", add_id, source_del_id)
                res.n_fail += 1
                continue
        try:
            for src in del_previews:
                shutil.copy2(src, add_dir / src.name)
            res.n_ok += 1
            update_edit_stage(ctx, add_id, "addition", "s6p", status="done")
        except Exception as e:
            log.warning("[s6p_del] add %s: %s", add_id, e)
            res.n_fail += 1
            update_edit_stage(ctx, add_id, "addition", "s6p",
                              status="error", reason=str(e)[:200])
        finally:
            if progress_bar is not None:
                progress_bar.update(1)

    update_step(
        ctx, "s6p_del",
        status=STATUS_OK if res.n_fail == 0 else STATUS_FAIL,
        n_ok=res.n_ok, n_fail=res.n_fail, n_skip=res.n_skip,
        wall_s=round(time.time() - t0, 2),
    )
    return res


def render_flux_previews_for_object(
    ctx: ObjectContext,
    *,
    pipeline,
    resolution: int = 518,
    prereq_map: dict[str, str | None] | None = None,
    force: bool = False,
    logger: logging.Logger | None = None,
) -> PreviewResult:
    """Render preview_{0..4}.png for TRELLIS-based edits (mod/scl/mat/glb).

    Step key: s6p_flux.  GPU required (TRELLIS decode).
    """
    log = logger or logging.getLogger("pipeline_v3.s6p_flux")
    res = PreviewResult(obj_id=ctx.obj_id)

    # Per-edit gate check: skip object only if no flux edits need s6p
    from .specs import iter_flux_specs as _iter_flux_specs_inner
    _flux_ids = [sp.edit_id for sp in _iter_flux_specs_inner(ctx)]
    if _flux_ids and not force and not obj_needs_stage(
        ctx, _flux_ids, "s6p", prereq_map or {}, force=force
    ):
        return res

    if ctx.image_npz is None or not ctx.image_npz.is_file():
        update_step(ctx, "s6p_flux", status=STATUS_FAIL, error="missing_image_npz")
        res.error = "missing_image_npz"
        return res

    from partcraft.render.overview import load_views_from_npz
    _, frames = load_views_from_npz(ctx.image_npz, VIEW_INDICES)

    if not ctx.edits_3d_dir.is_dir():
        update_step(ctx, "s6p_flux", status=STATUS_OK, n=0, reason="no_edits_3d")
        return res

    t0 = time.time()

    from .specs import iter_all_specs
    from partcraft.edit_types import FLUX_TYPES
    for spec in iter_all_specs(ctx):
        if spec.edit_type not in FLUX_TYPES:
            continue
        if prereq_map is not None:
            if not edit_needs_step(ctx, spec.edit_id, "s6p", prereq_map, force=force):
                res.n_skip += 1
                continue
        else:
            if is_gate_a_failed(ctx, spec.edit_id):
                res.n_skip += 1
                continue
            edit_dir_chk = ctx.edit_3d_dir(spec.edit_id)
            if _all_previews_exist(edit_dir_chk) and not force:
                res.n_skip += 1
                continue
        edit_dir = ctx.edit_3d_dir(spec.edit_id)
        a_npz = edit_dir / "after.npz"
        if not a_npz.is_file():
            log.warning("[s6p_flux] %s %s: after.npz missing", spec.edit_type, spec.edit_id)
            res.n_fail += 1
            continue
        if pipeline is None:
            log.error("[s6p_flux] TRELLIS pipeline not loaded but needed for %s", spec.edit_id)
            res.n_fail += 1
            continue
        try:
            imgs = _render_trellis_five_views(a_npz, pipeline, frames, resolution)
            _write_preview_images(edit_dir, imgs)
            res.n_ok += 1
            update_edit_stage(ctx, spec.edit_id, spec.edit_type, "s6p", status="done")
        except Exception as e:
            log.warning("[s6p_flux] %s %s: %s", spec.edit_type, spec.edit_id, e)
            res.n_fail += 1
            update_edit_stage(ctx, spec.edit_id, spec.edit_type, "s6p",
                              status="error", reason=str(e)[:200])

    update_step(
        ctx, "s6p_flux",
        status=STATUS_OK if res.n_fail == 0 else STATUS_FAIL,
        n_ok=res.n_ok, n_fail=res.n_fail, n_skip=res.n_skip,
        wall_s=round(time.time() - t0, 2),
    )
    return res


def render_del_previews_batch(
    ctxs: Iterable[ObjectContext],
    *,
    blender: str = "blender",
    resolution: int = 518,
    prereq_map: dict[str, str | None] | None = None,
    force: bool = False,
    n_workers: int = 8,
    gpus: list[int] | None = None,
    best_view_only: bool = False,
    logger: logging.Logger | None = None,
) -> list[PreviewResult]:
    """Batch entry for s6p_del (deletion + addition preview, GPU/Blender).

    ``best_view_only=True`` renders only one canonical view per edit (gate_a
    ``best_view``, fallback front = slot 4).  This is the fast path used to
    backfill ``after.png`` on shards where the pipeline preview stage was
    skipped.

    GPU binding strategy (when ``gpus`` is provided):
    - n_workers is overridden to len(gpus): one worker thread per GPU.
    - Each thread is initialised with exactly one GPU index via a thread-local,
      so CUDA_VISIBLE_DEVICES is set once per thread and never shared.
    - This guarantees at most one Blender process per GPU at any moment.
    """
    import concurrent.futures
    import itertools
    import threading

    log = logger or logging.getLogger("pipeline_v3.s6p_del")

    # --- GPU affinity setup ---
    _thread_local = threading.local()
    if gpus:
        effective_workers = len(gpus)
        _gpu_counter = itertools.count()

        def _thread_init():
            # Called once when the thread is first created.
            _thread_local.cuda_device = str(gpus[next(_gpu_counter) % len(gpus)])

        pool_kwargs: dict = {"max_workers": effective_workers, "initializer": _thread_init}
    else:
        effective_workers = n_workers
        pool_kwargs = {"max_workers": effective_workers}

    log.info("[s6p_del] n_workers=%d  gpus=%s  best_view_only=%s",
             effective_workers, gpus, best_view_only)

    ctx_list = list(ctxs)
    from .specs import iter_deletion_specs as _iter_del_specs_outer
    def _del_obj_pending(c):
        ids = ([sp.edit_id for sp in _iter_del_specs_outer(c)]
               + [aid for aid, _ in _iter_addition_edits(c)])
        return not ids or obj_needs_stage(c, ids, "s6p", prereq_map or {}, force=force)
    pending = [c for c in ctx_list if force or _del_obj_pending(c)]
    done_results = [PreviewResult(c.obj_id) for c in ctx_list if c not in set(pending)]

    # --- pre-scan: count total pending edits for accurate progress bar ---
    def _count_pending_edits(ctx: ObjectContext) -> int:
        if not ctx.edits_3d_dir.is_dir():
            return 0
        from .specs import iter_deletion_specs
        count = 0
        for spec in iter_deletion_specs(ctx):
            if is_gate_a_failed(ctx, spec.edit_id):
                continue
            edit_dir = ctx.edit_3d_dir(spec.edit_id)
            if not force:
                if best_view_only:
                    k = _best_view_slot_for_edit(ctx, spec.edit_id)
                    if _slot_previews_exist(edit_dir, [k]):
                        continue
                elif _all_previews_exist(edit_dir):
                    continue
            count += 1
        for add_id, _ in _iter_addition_edits(ctx):
            if is_gate_a_failed(ctx, add_id):
                continue
            add_dir = ctx.edit_3d_dir(add_id)
            if not force:
                if best_view_only:
                    k = _best_view_slot_for_edit(ctx, add_id)
                    if _slot_previews_exist(add_dir, [k]):
                        continue
                elif _all_previews_exist(add_dir):
                    continue
            count += 1
        return count

    total_edits = sum(_count_pending_edits(c) for c in pending)
    log.info("[s6p_del] %d objects  %d edits pending", len(pending), total_edits)

    # --- shared per-edit progress bar (thread-safe tqdm updates) ---
    try:
        from tqdm import tqdm as _tqdm
        _bar = _tqdm(total=total_edits, desc="s6p_del", unit="edit",
                     dynamic_ncols=True, leave=True)
    except ImportError:
        _bar = None

    _n_ok = _n_fail = _n_skip = 0
    _bar_lock = threading.Lock()

    def _do(ctx):
        nonlocal _n_ok, _n_fail, _n_skip
        res = render_del_previews_for_object(
            ctx, blender=blender, resolution=resolution,
            cuda_device=getattr(_thread_local, "cuda_device", None),
            progress_bar=_bar,
            prereq_map=prereq_map,
            force=force,
            best_view_only=best_view_only,
            logger=log,
        )
        with _bar_lock:
            _n_ok   += res.n_ok
            _n_fail += res.n_fail
            _n_skip += res.n_skip
            if _bar is not None:
                _bar.set_postfix(ok=_n_ok, fail=_n_fail, skip=_n_skip,
                                 gpu=getattr(_thread_local, "cuda_device", "?"))
        return res

    results: list[PreviewResult] = list(done_results)

    with concurrent.futures.ThreadPoolExecutor(**pool_kwargs) as pool:
        futures = {pool.submit(_do, ctx): ctx for ctx in pending}
        for fut in concurrent.futures.as_completed(futures):
            results.append(fut.result())

    if _bar is not None:
        _bar.close()

    log.info("[s6p_del] done: ok=%d fail=%d skip=%d", _n_ok, _n_fail, _n_skip)
    return results


def render_flux_previews_batch(
    ctxs: Iterable[ObjectContext],
    *,
    ckpt: str = "checkpoints/TRELLIS-text-xlarge",
    resolution: int = 518,
    prereq_map: dict[str, str | None] | None = None,
    force: bool = False,
    logger: logging.Logger | None = None,
) -> list[PreviewResult]:
    """Batch entry for s6p_flux (mod/scl/mat/glb preview, GPU/TRELLIS)."""
    log = logger or logging.getLogger("pipeline_v3.s6p_flux")
    log.info("[s6p_flux] CUDA_VISIBLE_DEVICES=%s", os.environ.get("CUDA_VISIBLE_DEVICES"))
    ctx_list = list(ctxs)
    from .specs import iter_flux_specs as _iter_flux_outer
    def _flux_obj_pending(c):
        ids = [sp.edit_id for sp in _iter_flux_outer(c)]
        return not ids or obj_needs_stage(c, ids, "s6p", prereq_map or {}, force=force)
    pending = [c for c in ctx_list if force or _flux_obj_pending(c)]
    pipeline = None
    if _has_flux_or_trellis_edits(pending):
        pipeline = _load_trellis_pipeline(ckpt, log)
    results: list[PreviewResult] = [PreviewResult(c.obj_id) for c in ctx_list if c not in set(pending)]
    for ctx in pending:
        results.append(render_flux_previews_for_object(
            ctx, pipeline=pipeline, resolution=resolution,
            prereq_map=prereq_map, force=force, logger=log,
        ))
    return results
