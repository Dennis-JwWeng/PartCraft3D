#!/usr/bin/env python3
"""bench_v3_color_scale.py — qualitative bench for color / scale edits.

Pulls samples from a finished pipeline_v3 shard root and renders a
self-contained HTML report so we can eyeball where the FLUX 2D edit
and the downstream Trellis 3D reconstruction go wrong for the two
problematic edit families (color, scale).

Each sampled edit row shows:

    [INPUT 2D] [EDITED 2D] [PREVIEW × 5]

plus the parsed prompts (original_prompt / improved_prompt /
new_parts_desc / target_part_desc) and any Gate-A / Gate-E judge
reasoning that was written by the pipeline.

Buckets the samples by gate_e outcome (pass / fail / missing) so we
can A/B "pipeline thought it was OK" vs "pipeline flagged" cases at a
glance — without ever judging quality ourselves.

Usage
-----
  python scripts/tools/bench_v3_color_scale.py \\
    --root outputs/partverse/shard07/mode_e_text_align \\
    --shard 07 \\
    --types color,scale \\
    --per-bucket 12 \\
    --out reports/bench_v3_shard07_color_scale.html
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import random
import sys
from collections import defaultdict
from html import escape
from pathlib import Path

from PIL import Image


EID_PFX = {"deletion": "del", "modification": "mod", "scale": "scl",
           "material": "mat", "color": "clr", "global": "glb",
           "addition": "add"}
PFX_TO_TYPE = {v: k for k, v in EID_PFX.items()}
TYPE_CHIP = {
    "scale": "#d49c1a",
    "color": "#b83db8",
    "modification": "#3778d4",
    "material": "#27ae60",
    "global": "#16a085",
}


# ─────────────────── image utils ────────────────────────────────────

def _flatten(im: Image.Image) -> Image.Image:
    if im.mode == "RGBA":
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bg.paste(im, mask=im.split()[-1])
        return bg
    return im.convert("RGB") if im.mode != "RGB" else im


def _resize_h(im: Image.Image, h: int) -> Image.Image:
    if im.height == h:
        return im
    r = h / im.height
    return im.resize((max(1, int(im.width * r)), h), Image.LANCZOS)


def _b64(im: Image.Image, quality: int = 78) -> str:
    im = _flatten(im)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=quality, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _open(p: Path):
    if not p.is_file():
        return None
    try:
        return Image.open(p)
    except Exception:
        return None


def _strip(images, h: int = 180, gap: int = 3) -> str | None:
    """Render N images into a horizontal strip, equal height."""
    norms = []
    any_real = False
    for im in images:
        if im is None:
            norms.append(Image.new("RGB", (h, h), (240, 240, 240)))
        else:
            any_real = True
            norms.append(_resize_h(_flatten(im), h))
    if not any_real:
        return None
    total = sum(im.width for im in norms) + gap * (len(norms) - 1)
    out = Image.new("RGB", (total, h), (255, 255, 255))
    x = 0
    for im in norms:
        out.paste(im, (x, 0))
        x += im.width + gap
    return _b64(out)


# ─────────────────── status / prompt loaders ────────────────────────

def _load_json(p: Path) -> dict:
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def _build_prompt_index(parsed_path: Path) -> dict[str, dict]:
    """Map edit_id -> raw spec dict from phase1/parsed.json.

    Mirrors partcraft.pipeline_v3.specs.iter_all_specs: deletion has its
    own 0-based counter; FLUX-types share `flux_seq` across the whole
    parsed.edits list. Addition is backfilled later (not in parsed.json).
    """
    pj = _load_json(parsed_path)
    edits = pj.get("parsed", {}).get("edits", [])
    obj_id = pj.get("obj_id") or parsed_path.parent.parent.name
    out: dict[str, dict] = {}
    flux_seq, del_seq = 0, 0
    for e in edits:
        et = e.get("edit_type", "?")
        if et == "deletion":
            eid = f"del_{obj_id}_{del_seq:03d}"
            del_seq += 1
        elif et in {"modification", "scale", "material", "color", "global"}:
            eid = f"{EID_PFX[et]}_{obj_id}_{flux_seq:03d}"
            flux_seq += 1
        else:  # addition / unknown — skip
            continue
        out[eid] = e
    return out


# ─────────────────── candidate gathering ────────────────────────────

def _bucket_for(stages: dict) -> str:
    ge = (stages or {}).get("gate_e") or {}
    s = ge.get("status")
    if s == "pass":
        return "pass"
    if s == "fail":
        return "fail"
    return "missing"


def _gather(root: Path, shard: str, types: set[str]) -> dict:
    obj_root = root / "objects" / shard
    if not obj_root.is_dir():
        sys.exit(f"[bench] missing obj_root: {obj_root}")
    by_type_bucket: dict[tuple[str, str], list[dict]] = defaultdict(list)
    n_obj = 0
    for od in sorted(obj_root.iterdir()):
        if not od.is_dir():
            continue
        sf = od / "edit_status.json"
        st = _load_json(sf)
        edits = (st or {}).get("edits") or {}
        if not edits:
            continue
        n_obj += 1
        prompt_idx = _build_prompt_index(od / "phase1" / "parsed.json")
        for eid, einfo in edits.items():
            et = einfo.get("edit_type", "?")
            if et not in types:
                continue
            stages = einfo.get("stages") or {}
            if (stages.get("s4") or {}).get("status") != "done":
                continue  # FLUX never produced
            bucket = _bucket_for(stages)
            row = {
                "obj_id": od.name,
                "shard": shard,
                "edit_id": eid,
                "edit_type": et,
                "bucket": bucket,
                "stages": stages,
                "gates": einfo.get("gates") or {},
                "fail_gate": einfo.get("fail_gate"),
                "fail_reason": einfo.get("fail_reason"),
                "obj_dir": str(od),
                "spec": prompt_idx.get(eid) or {},
            }
            by_type_bucket[(et, bucket)].append(row)
    print(f"[bench] scanned {n_obj} objects; collected:")
    for (et, b), lst in sorted(by_type_bucket.items()):
        print(f"  {et:<8s}  {b:<8s}  {len(lst)}")
    return by_type_bucket


# ─────────────────── HTML rendering ─────────────────────────────────

def _refined_meta(obj_dir: Path, eid: str) -> dict:
    return _load_json(Path(obj_dir) / "edits_3d" / eid / "refined_prompt.json")


def _row_html(row: dict, *, root_b: Path | None = None) -> str:
    obj_dir = Path(row["obj_dir"])
    eid = row["edit_id"]
    et = row["edit_type"]
    chip_color = TYPE_CHIP.get(et, "#888")
    shard = row["shard"]
    obj_id = row["obj_id"]

    # 2D pair (A = baseline)
    in_im  = _open(obj_dir / "edits_2d" / f"{eid}_input.png")
    out_im = _open(obj_dir / "edits_2d" / f"{eid}_edited.png")

    if root_b is None:
        twoD_a = _strip([in_im, out_im], h=260)
        twoD_b = None
    else:
        b_obj = root_b / "objects" / shard / obj_id
        b_in  = _open(b_obj / "edits_2d" / f"{eid}_input.png")  # may be missing
        b_out = _open(b_obj / "edits_2d" / f"{eid}_edited.png")
        # Use A's input when B's input is missing (rerun may skip writing it).
        if b_in is None:
            b_in = in_im
        twoD_a = _strip([in_im, out_im], h=260)
        twoD_b = _strip([b_in, b_out], h=260)

    # 3D previews
    previews_a = [_open(obj_dir / "edits_3d" / eid / f"preview_{i}.png")
                  for i in range(5)]
    threeD_a = _strip(previews_a, h=180)
    threeD_b = None
    if root_b is not None:
        b_obj = root_b / "objects" / shard / obj_id
        previews_b = [_open(b_obj / "edits_3d" / eid / f"preview_{i}.png")
                      for i in range(5)]
        threeD_b = _strip(previews_b, h=180)

    # text bits
    spec = row["spec"] or {}
    refined = _refined_meta(obj_dir, eid)
    a_vlm = ((row["gates"].get("A") or {}).get("vlm") or {})
    e_vlm = ((row["gates"].get("E") or {}).get("vlm") or {})

    def _kv(k, v):
        if v in (None, "", []):
            return ""
        return (f'<div class="kv"><span class="k">{escape(k)}</span>'
                f'<span class="v">{escape(str(v))}</span></div>')

    text_html = (
        _kv("original_prompt", spec.get("edit_prompt") or spec.get("prompt"))
        + _kv("improved_prompt", refined.get("improved_prompt"))
        + _kv("target_part_desc", spec.get("target_part_desc")
              or spec.get("before_part_desc"))
        + _kv("new_parts_desc", spec.get("new_parts_desc")
              or refined.get("improved_after_desc"))
        + _kv("part_labels", spec.get("part_labels"))
        + _kv("gateA reason", a_vlm.get("reason"))
        + _kv("gateA score", a_vlm.get("score"))
        + _kv("gateE reason", e_vlm.get("reason"))
        + _kv("gateE score", e_vlm.get("score"))
        + _kv("refiner judge_pass", refined.get("judge_pass"))
        + _kv("refiner prompt_quality", refined.get("prompt_quality"))
        + _kv("fail_gate", row.get("fail_gate"))
    )

    bucket_chip = {
        "pass":    ("#27ae60", "GATE-E PASS"),
        "fail":    ("#c0392b", "GATE-E FAIL"),
        "missing": ("#7f8c8d", "GATE-E NONE"),
    }[row["bucket"]]

    return (
        f'<section class="card">'
        f'<header>'
        f'<span class="chip" style="background:{chip_color}">{et}</span>'
        f'<span class="chip" style="background:{bucket_chip[0]}">{bucket_chip[1]}</span>'
        f'<code class="eid">{escape(eid)}</code>'
        f'<code class="oid">{escape(row["obj_id"])}</code>'
        f'</header>'
        + (f'<div class="row twod"><div class="lbl">A · baseline</div>'
           f'<img src="{twoD_a}"/></div>' if twoD_a else "")
        + (f'<div class="row twod"><div class="lbl">B · variant</div>'
           f'<img src="{twoD_b}"/></div>' if twoD_b else "")
        + (f'<div class="row threed"><div class="lbl">A · 3D preview</div>'
           f'<img src="{threeD_a}"/></div>' if threeD_a else "")
        + (f'<div class="row threed"><div class="lbl">B · 3D preview</div>'
           f'<img src="{threeD_b}"/></div>' if threeD_b else "")
        + f'<div class="meta">{text_html}</div>'
        f'</section>'
    )


def _section_html(et: str, bucket: str, rows: list[dict],
                  *, root_b: Path | None = None) -> str:
    body = "\n".join(_row_html(r, root_b=root_b) for r in rows)
    return (
        f'<h2 id="{et}_{bucket}">{et} · gate-E {bucket} '
        f'<small>(showing {len(rows)})</small></h2>{body}'
    )


CSS = """
:root { color-scheme: light; }
body { font: 13px/1.4 -apple-system,Segoe UI,sans-serif;
       background:#f6f7f9; margin:0; padding:24px; }
h1 { margin:0 0 6px; font-size:20px; }
h2 { margin:32px 0 12px; padding:6px 10px; background:#222; color:#fff;
     border-radius:6px; font-size:15px; }
h2 small { font-weight:normal; opacity:.7; }
nav { background:#fff; padding:10px 14px; border-radius:8px;
      margin-bottom:18px; box-shadow:0 1px 3px rgba(0,0,0,.06); }
nav a { margin-right:14px; color:#3778d4; text-decoration:none; font-weight:500; }
.card { background:#fff; border-radius:10px; padding:14px;
        margin:10px 0; box-shadow:0 1px 4px rgba(0,0,0,.06); }
.card header { display:flex; gap:8px; align-items:center;
               flex-wrap:wrap; margin-bottom:10px; }
.chip { color:#fff; padding:2px 9px; border-radius:99px;
        font-size:11px; font-weight:600; letter-spacing:.4px; }
.eid, .oid { font-family:ui-monospace,Menlo,monospace; font-size:11px;
             background:#f0f1f3; padding:2px 7px; border-radius:4px; }
.row { margin:6px 0; }
.row img { display:block; max-width:100%; border-radius:6px;
           border:1px solid #e3e5e8; }
.row .lbl { font-size:10px; color:#666; letter-spacing:.6px;
           text-transform:uppercase; margin:6px 0 2px; font-weight:600; }
.meta { display:grid; grid-template-columns: minmax(150px,180px) 1fr;
        gap:4px 14px; margin-top:10px; padding-top:10px;
        border-top:1px dashed #d8dadd; }
.kv { display:contents; }
.k { color:#555; font-weight:600; font-size:11px;
     text-transform:uppercase; letter-spacing:.5px; padding-top:2px; }
.v { color:#222; word-break:break-word; }
"""


def render_html(buckets: dict, *, root: Path, shard: str,
                per_bucket: int, types: list[str], seed: int,
                root_b: Path | None = None) -> str:
    rng = random.Random(seed)
    sections = []
    nav_links = []
    counts = []
    for et in types:
        for b in ("fail", "pass", "missing"):
            pool = buckets.get((et, b), [])
            counts.append((et, b, len(pool)))
            if not pool:
                continue
            sample = pool if len(pool) <= per_bucket \
                     else rng.sample(pool, per_bucket)
            sample.sort(key=lambda r: r["edit_id"])
            sections.append(_section_html(et, b, sample, root_b=root_b))
            nav_links.append(
                f'<a href="#{et}_{b}">{et}/{b} ({len(sample)}/{len(pool)})</a>'
            )
    nav_html = "<nav>" + "".join(nav_links) + "</nav>" if nav_links else ""
    summary = " · ".join(f"{et}/{b}={n}" for et, b, n in counts)
    head = (
        f'<h1>pipeline_v3 color/scale bench</h1>'
        f'<div style="color:#555; margin-bottom:12px;">'
        f'<code>{escape(str(root))}</code> shard <b>{shard}</b><br>'
        f'pool counts: {escape(summary)}<br>'
        f'showing per bucket: <b>{per_bucket}</b> · seed={seed}'
        f'</div>'
    )
    return (
        f'<!doctype html><html><head><meta charset="utf-8">'
        f'<title>v3 bench {shard} color/scale</title>'
        f'<style>{CSS}</style></head><body>'
        f'{head}{nav_html}{"".join(sections)}'
        f'</body></html>'
    )


# ─────────────────── CLI ────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, type=Path,
                    help="shard root, e.g. outputs/partverse/shard07/mode_e_text_align")
    ap.add_argument("--shard", required=True,
                    help="2-digit shard string, e.g. 07")
    ap.add_argument("--types", default="color,scale",
                    help="comma list, default 'color,scale'")
    ap.add_argument("--per-bucket", type=int, default=12,
                    help="samples per (type × gate-E bucket)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--root-b", type=Path, default=None,
                    help="optional second root (A/B variant). "
                         "When set, each row also shows the B 2D edit.")
    ap.add_argument("--out", required=True, type=Path,
                    help="output HTML path")
    args = ap.parse_args()

    types = [t.strip() for t in args.types.split(",") if t.strip()]
    bad = [t for t in types if t not in EID_PFX]
    if bad:
        sys.exit(f"[bench] unknown types: {bad}")

    buckets = _gather(args.root, args.shard, set(types))
    html = render_html(buckets, root=args.root, shard=args.shard,
                       per_bucket=args.per_bucket, types=types,
                       seed=args.seed, root_b=args.root_b)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html, encoding="utf-8")
    print(f"[bench] wrote {args.out}  ({args.out.stat().st_size/1024:.1f} KB)")


if __name__ == "__main__":
    main()
