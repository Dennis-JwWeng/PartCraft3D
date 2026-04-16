#!/usr/bin/env python3
"""Visualize SLAT voxels + part mask for Trellis edits.

For each edit, reconstructs the 64³ mask via refiner.build_part_mask()
and renders 3-projection scatter plots (XY / XZ / YZ):
  gray  = SLAT voxels not in mask (preserved)
  red   = SLAT voxels inside mask (edit region)
  blue  = mask voxels with no SLAT (empty edit space)

Usage:
    python scripts/tools/visualize_trellis_mask.py \
        --obj-id bde1b486ee284e4d94f54bdbb3b3d6d7 \
        --shard 08 \
        --config configs/pipeline_v3_shard08_bench100.yaml \
        [--edit-ids mod_..._000 mod_..._001] \
        [--out /tmp/mask_viz.png]
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import yaml

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))


# ─── tiny SLAT stub ───────────────────────────────────────────────────────────

def _load_slat_stub(before_npz: Path):
    import torch
    class _S: pass
    d = np.load(str(before_npz))
    s = _S()
    # before.npz stores slat_coords as [N, 4] — batch col already included
    s.coords = torch.from_numpy(d["slat_coords"])
    s.feats  = torch.from_numpy(d["slat_feats"])
    return s


# ─── build mask via refiner (CPU-only, no GPU models loaded) ─────────────────

def _stub_refiner(cfg):
    import torch
    from partcraft.trellis.refiner import TrellisRefiner
    r = TrellisRefiner.__new__(TrellisRefiner)
    r.device      = torch.device("cpu")
    r.ckpt_root   = Path(cfg.get("ckpt_root", "checkpoints"))
    r.slat_dir    = Path(cfg["data"].get("slat_dir", ""))
    r.img_enc_dir = None
    r.debug       = False
    r.pipeline = r.image_enc = None
    return r


def _get_edit_part_ids(spec):
    et = spec.edit_type.capitalize()
    if et == "Global":    return []
    if et == "Material":  return [spec.selected_part_ids[0]] if spec.selected_part_ids else []
    return list(spec.selected_part_ids) if spec.selected_part_ids else []


def build_mask(refiner, ctx, spec, cfg):
    from partcraft.io.partcraft_loader import PartCraftDataset
    before_npz = ctx.edit_3d_dir(spec.edit_id) / "before.npz"
    if not before_npz.is_file():
        return None, None, None
    slat    = _load_slat_stub(before_npz)
    dataset = PartCraftDataset(
        render_dir=cfg["data"]["images_root"],
        mesh_dir=cfg["data"]["mesh_root"],
    )
    obj_rec = dataset.load_object(ctx.shard, ctx.obj_id)
    mask, eff = refiner.build_part_mask(
        ctx.obj_id, obj_rec, _get_edit_part_ids(spec),
        slat, spec.edit_type.capitalize())
    return mask, slat, eff


# ─── plotting ─────────────────────────────────────────────────────────────────

def _plot_row(axes_row, mask, slat, spec, effective_type):
    sc      = slat.coords[:, 1:].numpy()          # [N,3] in [0,63]
    mask_np = mask.numpy()

    in_mask   = mask_np[sc[:,0], sc[:,1], sc[:,2]].astype(bool)
    edit_sc   = sc[in_mask]
    pres_sc   = sc[~in_mask]
    n_total   = len(sc)
    pct       = 100 * len(edit_sc) / n_total if n_total else 0

    proj = [("X","Y",0,1), ("X","Z",0,2), ("Y","Z",1,2)]
    for ax, (xl, yl, xi, yi) in zip(axes_row, proj):
        def _p(arr, xi=xi, yi=yi):
            return (arr[:,xi], arr[:,yi]) if len(arr) else ([], [])
        # draw preserved first (background), then edit on top
        if len(pres_sc):  ax.scatter(*_p(pres_sc),  s=4,  c="#cccccc", alpha=0.40, lw=0)
        if len(edit_sc):  ax.scatter(*_p(edit_sc),  s=12, c="#dd2222", alpha=0.90, lw=0)
        ax.set_xlim(-1,64); ax.set_ylim(-1,64)
        ax.set_xlabel(xl, fontsize=8); ax.set_ylabel(yl, fontsize=8)
        ax.set_aspect("equal"); ax.grid(True, alpha=0.15)
        ax.tick_params(labelsize=6)

    # row title on left axis
    promote_note = f"  ⇒ promoted to {effective_type}" if effective_type.lower() != spec.edit_type.lower() else ""
    t = (f"{spec.edit_id}   parts={spec.selected_part_ids}{promote_note}\n"
         f"edit SLAT: {len(edit_sc)}/{n_total} ({pct:.1f}%)   "
         f"preserved SLAT: {len(pres_sc)}")
    axes_row[0].set_title(t, fontsize=6.5, loc="left")


# ─── main ─────────────────────────────────────────────────────────────────────

def run(obj_id, shard, cfg, edit_ids, out_path):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    from partcraft.pipeline_v3.paths import PipelineRoot, DatasetRoots
    from partcraft.pipeline_v3.specs import iter_flux_specs

    root  = PipelineRoot(Path(cfg["data"]["output_dir"]))
    roots = DatasetRoots.from_pipeline_cfg(cfg)
    mesh_npz, image_npz = roots.input_npz_paths(shard, obj_id)
    ctx = root.context(shard, obj_id, mesh_npz=mesh_npz, image_npz=image_npz)

    refiner = _stub_refiner(cfg)

    specs = [s for s in iter_flux_specs(ctx)
             if (edit_ids is None or s.edit_id in set(edit_ids))
             and (ctx.edit_3d_dir(s.edit_id) / "before.npz").is_file()]
    if not specs:
        print("[WARN] no specs with before.npz — run trellis_3d first"); return

    print(f"Visualizing {len(specs)} edits")
    nrows, ncols = len(specs), 3
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols*4.5, nrows*3.6),
                             squeeze=False)

    for i, spec in enumerate(specs):
        mask, slat, eff = build_mask(refiner, ctx, spec, cfg)
        if mask is None:
            for ax in axes[i]: ax.set_title("no before.npz"); continue
        _plot_row(axes[i], mask, slat, spec, eff)

    legend = [
        mpatches.Patch(color="#cccccc", label="SLAT voxels — preserved"),
        mpatches.Patch(color="#dd2222", label="SLAT voxels — edit region"),
    ]
    fig.legend(handles=legend, loc="upper right", fontsize=8)
    for ax in axes[0]: ax.set_title(ax.get_title() or "", fontsize=7)
    fig.suptitle(f"Trellis mask — {obj_id}  shard={shard}", fontsize=9)
    plt.tight_layout(rect=[0,0,1,0.97])
    plt.savefig(str(out_path), dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obj-id",   required=True)
    ap.add_argument("--shard",    default="08")
    ap.add_argument("--config",   default="configs/pipeline_v3_shard08_bench100.yaml")
    ap.add_argument("--edit-ids", nargs="*", default=None)
    ap.add_argument("--out",      default=None)
    args = ap.parse_args()
    cfg  = yaml.safe_load(Path(args.config).read_text())
    out  = Path(args.out) if args.out else Path(f"/tmp/{args.obj_id}_mask_viz.png")
    run(args.obj_id, args.shard.zfill(2), cfg, args.edit_ids, out)

if __name__ == "__main__":
    main()
