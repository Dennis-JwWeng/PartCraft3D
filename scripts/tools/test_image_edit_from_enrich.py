#!/usr/bin/env python3
"""Drive local image-edit HTTP server using prompts from enrich JSON.

Reads ``enrich_*_raw.json`` (``part_groups`` / ``global_edits``), loads the
object NPZ, picks a view, builds prompts like :func:`run_2d_edit.call_local_edit`.

Example:
  PARTCRAFT_DATA_ROOT=data/partverse \\
    python scripts/tools/test_image_edit_from_enrich.py \\
    --enrich outputs/vlm_compare_shard05_5dc4/enrich_api_raw.json \\
    --obj-id 5dc4ca7d607c495bb82eca3d0153cc2c \\
    --shard 05 \\
    --base-url http://localhost:8001
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from partcraft.utils.config import load_config
from scripts.pipeline_common import create_dataset
from scripts.run_2d_edit import call_local_edit, check_edit_server, prepare_input_image


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/partverse_local.yaml")
    ap.add_argument("--enrich", required=True, type=Path, help="enrich_*_raw.json")
    ap.add_argument("--obj-id", required=True)
    ap.add_argument("--shard", required=True)
    ap.add_argument("--base-url", default="http://localhost:8001")
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()

    if not check_edit_server(args.base_url):
        print(f"Image edit server not reachable at {args.base_url}/health", file=sys.stderr)
        return 1

    data = json.loads(args.enrich.read_text(encoding="utf-8"))
    cfg = load_config(_PROJECT_ROOT / args.config)
    ds = create_dataset(cfg)
    obj = ds.load_object(args.shard, args.obj_id)

    out_dir = args.out_dir or (_PROJECT_ROOT / "outputs" / "image_edit_from_enrich")
    out_dir.mkdir(parents=True, exist_ok=True)

    ortho = data.get("orthogonal_views") or [32, 33, 34, 35]

    # (name, edit_type, prompt, after, old_label, before_desc, part_ids, view_override)
    jobs: list[tuple[str, str, str, str, str, str, list[int], int | None]] = []

    def _group_view(g: dict) -> int | None:
        bvi = g.get("best_view_idx")
        if isinstance(bvi, int) and 0 <= bvi < len(ortho):
            return ortho[bvi]
        return None

    for gi, g in enumerate(data.get("part_groups") or []):
        gname = g.get("group_name", f"group_{gi}")
        pids = g.get("part_ids") or []
        gv = _group_view(g)
        for si, sw in enumerate(g.get("swaps") or []):
            jobs.append((
                f"swap_{gname}_{si}",
                "modification",
                sw.get("prompt", ""),
                sw.get("after_desc", ""),
                gname.replace("_", " "),
                sw.get("before_desc", ""),
                pids,
                gv,
            ))
        for mi, mat in enumerate(g.get("materials") or []):
            jobs.append((
                f"material_{gname}_{mi}",
                "modification",
                mat.get("prompt", ""),
                mat.get("after_desc", ""),
                gname.replace("_", " "),
                "",
                pids,
                gv,
            ))
        del_ = g.get("deletion")
        if del_:
            jobs.append((
                f"deletion_{gname}",
                "deletion",
                del_.get("prompt", ""),
                del_.get("after_desc", ""),
                gname.replace("_", " "),
                "",
                pids,
                gv,
            ))

    for gi, ge in enumerate(data.get("global_edits") or []):
        bvi = ge.get("best_view_idx")
        if isinstance(bvi, int) and 0 <= bvi < len(ortho):
            g_view = ortho[bvi]
        else:
            g_view = ortho[0]
        jobs.append((
            f"global_{gi}",
            "global",
            ge.get("prompt", ""),
            ge.get("after_desc", ""),
            "",
            "",
            [],
            g_view,
        ))

    if not jobs:
        print("No swaps/materials/deletions/global_edits in enrich JSON", file=sys.stderr)
        obj.close()
        return 1

    # Run first swap + first global only (full grid can be huge / slow)
    priority = []
    for j in jobs:
        if j[0].startswith("swap_"):
            priority.append(j)
            break
    for j in jobs:
        if j[0].startswith("global_"):
            priority.append(j)
            break
    if not priority:
        priority = jobs[:2]

    for name, et, prompt, after, old_lbl, before_d, pids, view_ov in priority:
        if not prompt.strip():
            continue
        if view_ov is not None:
            view_id = view_ov
        else:
            view_id = ortho[0]
        png_bytes, pil_in = prepare_input_image(obj, view_id)
        pil_in.save(out_dir / f"{args.obj_id}_{name}_input.png")
        print(f"[{name}] view={view_id} type={et} …")
        out = call_local_edit(
            args.base_url,
            png_bytes,
            prompt,
            after,
            old_part_label=old_lbl,
            before_part_desc=before_d,
            edit_type=et,
        )
        if out is None:
            print(f"  FAILED: {name}", file=sys.stderr)
            continue
        out_path = out_dir / f"{args.obj_id}_{name}_edited.png"
        out.save(out_path)
        print(f"  saved {out_path}")

    obj.close()
    print(f"Done. Inputs/edits under {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
