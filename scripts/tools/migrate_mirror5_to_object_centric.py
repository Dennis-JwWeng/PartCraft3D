#!/usr/bin/env python3
"""One-shot: reorganize the flat phase1_v2_mirror5 outputs into the new
object-centric layout under outputs/partverse/pipeline_v2_mirror5/.

Source layout (flat, legacy):
    outputs/_debug/phase1_v2_mirror5/
        <obj>.parsed.json / .overview.png / .raw.txt
        _hl_edits/<obj>__e{idx:02d}.png
        2d_edits/{type}_{obj}_{seq:03d}_{input,edited}.png
        _3d_renders/{type}_{obj}_{seq:03d}_{before,after}.png
        mesh_pairs_mirror5/{type}_{obj}_{seq:03d}/{before,after}.npz

Target layout (object-centric):
    outputs/partverse/pipeline_v2_mirror5/
        _global/manifest.jsonl
        objects/<shard>/<obj_id>/
            meta.json
            phase1/{overview.png, parsed.json, raw.txt}
            highlights/e{idx:02d}.png
            edits_2d/{edit_id}_{input,edited}.png
            edits_3d/{edit_id}/{before,after}.{npz,png}
            status.json

Notes
- Files are HARDLINKED (no copy, no symlink) so the new tree is portable
  but takes ~0 extra disk.
- edit_id numbering matches parsed_to_edit_specs.py: shared seq across
  mod/scl/mat/glb; deletion uses its own per-obj seq (`del_<obj>_<NNN>`).
- addition is NOT materialized here (will be done by a future backfill
  step that hardlinks deletion before/after swapped).
"""
from __future__ import annotations
import argparse
import json
import os
import shutil
from pathlib import Path

PREFIX_FLUX = {"modification": "mod", "scale": "scl",
               "material": "mat", "global": "glb"}
PREFIX_DEL = "del"


def hardlink(src: Path, dst: Path):
    if not src.is_file():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path,
                    default=Path("outputs/_debug/phase1_v2_mirror5"))
    ap.add_argument("--dst", type=Path,
                    default=Path("outputs/partverse/pipeline_v2_mirror5"))
    ap.add_argument("--shard", default="01")
    ap.add_argument("--tag", default="mirror5")
    args = ap.parse_args()

    src = args.src.resolve()
    dst = args.dst.resolve()
    pairs = src / f"mesh_pairs_{args.tag}"
    flux_dir = src / "2d_edits"
    rd3 = src / "_3d_renders"
    hl_dir = src / "_hl_edits"

    obj_root = dst / "objects" / args.shard
    glob_dir = dst / "_global"
    obj_root.mkdir(parents=True, exist_ok=True)
    glob_dir.mkdir(parents=True, exist_ok=True)
    manifest = open(glob_dir / "manifest.jsonl", "w")

    parsed_files = sorted(src.glob("*.parsed.json"))
    print(f"[migrate] {len(parsed_files)} objects: {src} -> {dst}")

    n_obj = n_edits = n_flux = n_3d = n_hl = n_del = 0

    for pf in parsed_files:
        obj_id = pf.stem.replace(".parsed", "")
        j = json.loads(pf.read_text())
        edits = (j.get("parsed") or {}).get("edits") or []

        odir = obj_root / obj_id
        (odir / "phase1").mkdir(parents=True, exist_ok=True)
        (odir / "highlights").mkdir(parents=True, exist_ok=True)
        (odir / "edits_2d").mkdir(parents=True, exist_ok=True)
        (odir / "edits_3d").mkdir(parents=True, exist_ok=True)

        # phase1
        hardlink(pf, odir / "phase1" / "parsed.json")
        hardlink(src / f"{obj_id}.overview.png",
                 odir / "phase1" / "overview.png")
        hardlink(src / f"{obj_id}.raw.txt",
                 odir / "phase1" / "raw.txt")

        flux_seq = 0
        del_seq = 0
        rec_edits = []
        status = {"phase1": "ok", "highlights": 0,
                  "edits_2d": 0, "edits_3d": 0, "deletions": 0}

        for idx, e in enumerate(edits):
            n_edits += 1
            et = e.get("edit_type", "?")

            # highlight (per edit index, regardless of type)
            hl_src = hl_dir / f"{obj_id}__e{idx:02d}.png"
            if hardlink(hl_src, odir / "highlights" / f"e{idx:02d}.png"):
                status["highlights"] += 1
                n_hl += 1

            edit_id = None
            if et in PREFIX_FLUX:
                edit_id = f"{PREFIX_FLUX[et]}_{obj_id}_{flux_seq:03d}"
                flux_seq += 1
                # 2D
                for kind in ("input", "edited"):
                    if hardlink(flux_dir / f"{edit_id}_{kind}.png",
                                odir / "edits_2d" / f"{edit_id}_{kind}.png"):
                        if kind == "edited":
                            status["edits_2d"] += 1
                            n_flux += 1
                # 3D pair (npz + rerender)
                pair = pairs / edit_id
                e3 = odir / "edits_3d" / edit_id
                got_3d = False
                for which in ("before", "after"):
                    if hardlink(pair / f"{which}.npz", e3 / f"{which}.npz"):
                        got_3d = True
                    hardlink(rd3 / f"{edit_id}_{which}.png",
                             e3 / f"{which}.png")
                if got_3d:
                    status["edits_3d"] += 1
                    n_3d += 1
            elif et == "deletion":
                edit_id = f"{PREFIX_DEL}_{obj_id}_{del_seq:03d}"
                del_seq += 1
                # mesh-delete artifacts not produced by this run; placeholder dir
                (odir / "edits_3d" / edit_id).mkdir(parents=True, exist_ok=True)
                status["deletions"] += 1
                n_del += 1

            rec_edits.append({
                "idx": idx,
                "edit_id": edit_id,
                "edit_type": et,
                "view_index": e.get("view_index"),
                "selected_part_ids": e.get("selected_part_ids"),
                "prompt": e.get("prompt"),
                "target_part_desc": e.get("target_part_desc"),
            })

        meta = {
            "obj_id": obj_id,
            "shard": args.shard,
            "n_edits": len(edits),
            "edits": rec_edits,
            "object": (j.get("parsed") or {}).get("object"),
        }
        (odir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
        (odir / "status.json").write_text(json.dumps(status, indent=2))
        manifest.write(json.dumps({
            "obj_id": obj_id, "shard": args.shard,
            "n_edits": len(edits), **status,
        }) + "\n")
        n_obj += 1
        print(f"  [{n_obj}/{len(parsed_files)}] {obj_id} "
              f"edits={len(edits)} flux={status['edits_2d']} "
              f"3d={status['edits_3d']} del={status['deletions']}")

    manifest.close()
    print(f"\n[done] objs={n_obj} edits={n_edits} hl={n_hl} "
          f"flux={n_flux} 3d_pairs={n_3d} del_placeholder={n_del}")
    print(f"       -> {dst}")


if __name__ == "__main__":
    main()
