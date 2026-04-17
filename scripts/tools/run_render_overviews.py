"""Render overview.png for a list of objects using the pipeline's render_overview_png.

Usage:
    python -m scripts.tools.run_render_overviews \
        --config configs/pipeline_v3_shard09.yaml \
        --ids    configs/shard09_all_obj_ids.txt \
        --shard  09 \
        --out    /mnt/cfs/vffey4/omni3d/partverse/outputs/partverse/pipeline_v3_shard09

(Or invoke as a script from the repo root with PYTHONPATH=. .)
"""

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import yaml

# Make ``partcraft`` importable when this file is run as a script from the
# repo root (``python scripts/tools/run_render_overviews.py``).  No hard-
# coded /mnt path — works on any host.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _render_one(args: tuple) -> tuple[str, str]:
    """Worker: renders one overview and returns (obj_id, status)."""
    mesh_npz, image_npz, blender, out_png, force = args
    # Re-add repo root in the worker process (ProcessPoolExecutor spawns
    # fresh interpreters that don't inherit the parent's sys.path mods).
    import sys as _s, os as _os
    _root = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
    if _root not in _s.path:
        _s.path.insert(0, _root)
    from partcraft.pipeline_v3.vlm_core import render_overview_png  # noqa: E402
    out_p = Path(out_png)
    if not force and out_p.is_file() and out_p.stat().st_size > 1000:
        return Path(mesh_npz).stem, "skip"
    out_p.parent.mkdir(parents=True, exist_ok=True)
    png = render_overview_png(Path(mesh_npz), Path(image_npz), blender)
    tmp = out_p.with_suffix(".png.tmp")
    tmp.write_bytes(png)
    tmp.replace(out_p)
    return Path(mesh_npz).stem, "ok"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Pipeline YAML config")
    ap.add_argument("--ids",    required=True, help="Text file with one obj_id per line (# comments ok)")
    ap.add_argument("--shard",  default="08",  help="Shard id (default: 08)")
    ap.add_argument("--out",    required=True, help="Output root dir (overrides config output_dir)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--force",  action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    blender: str = cfg["blender"]
    data = cfg["data"]
    mesh_root   = Path(data["mesh_root"])
    images_root = Path(data["images_root"])
    shard = args.shard.zfill(2)

    obj_ids = [
        ln.split("#")[0].strip()
        for ln in Path(args.ids).read_text().splitlines()
        if ln.split("#")[0].strip()
    ]
    print(f"Objects : {len(obj_ids)}")
    print(f"Output  : {args.out}")
    print(f"Blender : {blender}")

    out_root = Path(args.out)
    tasks = []
    for oid in obj_ids:
        mesh_npz  = mesh_root   / shard / f"{oid}.npz"
        image_npz = images_root / shard / f"{oid}.npz"
        out_png   = out_root / "objects" / shard / oid / "phase1" / "overview.png"
        if not mesh_npz.is_file():
            print(f"  WARN  {oid}: mesh not found at {mesh_npz}")
            continue
        if not image_npz.is_file():
            print(f"  WARN  {oid}: images not found at {image_npz}")
            continue
        tasks.append((str(mesh_npz), str(image_npz), blender, str(out_png), args.force))

    ok = skip = err = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(_render_one, t): t for t in tasks}
        for fut in as_completed(futs):
            try:
                oid, status = fut.result()
                if status == "ok":
                    ok += 1
                    print(f"  [OK]   {oid}")
                else:
                    skip += 1
                    print(f"  [SKIP] {oid}")
            except Exception as e:
                err += 1
                t = futs[fut]
                print(f"  [ERR]  {Path(t[0]).stem}: {e}")

    print(f"\nDone: ok={ok} skip={skip} err={err}")
    if err:
        sys.exit(1)


if __name__ == "__main__":
    main()
