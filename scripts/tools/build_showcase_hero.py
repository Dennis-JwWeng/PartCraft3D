#!/usr/bin/env python3
"""build_showcase_hero.py
==========================

Render the FINAL high-quality showcase HTML from a hand-picked JSON list
of "<obj_id>/<edit_id>" strings (produced by clicking through
build_showcase_candidates.py output).

Per card we render:
  * type chip + obj/edit_id + score
  * prompt + target part description
  * BEFORE row (5 views, full resolution)
  * AFTER  row (5 views, full resolution)

Sources mirror the candidate report (same plumbing) but every image is
preserved at native resolution and re-encoded at JPEG q=92.

Usage
-----
    python scripts/tools/build_showcase_hero.py \
        --root outputs/partverse/shard08/mode_e_text_align \
        --shard 08 \
        --images-root data/partverse/inputs/images \
        --picks reports/shard08_picks.json \
        --out reports/shard08_showcase_hero.html
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import sys
from collections import defaultdict
from html import escape
from pathlib import Path

import numpy as np
from PIL import Image

VIEW_INDICES = [89, 90, 91, 100, 8]
EID_PFX = {"deletion": "del", "modification": "mod", "scale": "scl",
           "material": "mat", "color": "clr", "global": "glb",
           "addition": "add"}
PFX_TO_TYPE = {v: k for k, v in EID_PFX.items()}
FLUX_TYPES_SET = {"modification", "scale", "material", "color", "global"}
TYPE_CHIP = {
    "deletion":     "#c0392b",
    "modification": "#3778d4",
    "scale":        "#d49c1a",
    "material":     "#27ae60",
    "color":        "#b83db8",
    "global":       "#16a085",
    "addition":     "#2ecc71",
}
TYPE_LABEL = {
    "deletion": "DELETION", "modification": "MODIFICATION",
    "scale": "SCALE", "material": "MATERIAL", "color": "COLOR",
    "global": "GLOBAL STYLE", "addition": "ADDITION",
}


def _flatten_to_white(im: Image.Image) -> Image.Image:
    if im.mode == "RGBA":
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bg.paste(im, mask=im.split()[-1])
        return bg
    if im.mode != "RGB":
        return im.convert("RGB")
    return im


def _img_to_jpeg_b64(im: Image.Image, *, quality: int = 92) -> str:
    im = _flatten_to_white(im)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=quality, optimize=True,
            subsampling=0)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _png_bytes_from_npz(npz_path: Path, view_idx: int):
    if not npz_path.is_file():
        return None
    try:
        d = np.load(npz_path)
    except Exception:
        return None
    key = f"{view_idx:03d}.png"
    if key not in d.files:
        return None
    return bytes(d[key].tobytes())


def _open_png_bytes(b):
    if not b:
        return None
    try:
        return Image.open(io.BytesIO(b))
    except Exception:
        return None


def _open_path(p: Path):
    if not p.is_file():
        return None
    try:
        return Image.open(p)
    except Exception:
        return None


def _build_view_strip_hd(images, *, h: int = 360, gap: int = 4,
                         quality: int = 92) -> str:
    """Horizontal strip of 5 views at native height `h`, JPEG q=92."""
    if not images:
        return ""
    norm = []
    for im in images:
        if im is None:
            im = Image.new("RGB", (h, h), (255, 255, 255))
        else:
            im = _flatten_to_white(im)
        if im.height != h:
            ratio = h / im.height
            im = im.resize((int(round(im.width * ratio)), h), Image.LANCZOS)
        norm.append(im)
    total_w = sum(im.width for im in norm) + gap * (len(norm) - 1)
    out = Image.new("RGB", (total_w, h), (255, 255, 255))
    x = 0
    for im in norm:
        out.paste(im, (x, 0))
        x += im.width + gap
    return _img_to_jpeg_b64(out, quality=quality)


def _eid_type(eid: str) -> str:
    pfx = eid.split("_", 1)[0]
    return PFX_TO_TYPE.get(pfx, "?")


def _load_meta_for_pick(root: Path, shard: str, oid: str, eid: str) -> dict:
    """Pull prompt, target_part_desc, score, source_del_id (for add) for
    one pick. Mirrors build_showcase_candidates._gather_candidates."""
    sf = root / "objects" / shard / oid / "edit_status.json"
    parsed_path = root / "objects" / shard / oid / "phase1" / "parsed.json"
    meta = {"prompt": "", "target_part_desc": "", "score": None,
            "source_del_id": "", "ge_reason": ""}
    et = _eid_type(eid)

    if sf.is_file():
        try:
            stat = json.loads(sf.read_text())
            ed = (stat.get("edits") or {}).get(eid) or {}
            ge_vlm = ((ed.get("gates") or {}).get("E") or {}).get("vlm") or {}
            meta["score"] = ge_vlm.get("score")
            meta["ge_reason"] = ge_vlm.get("reason") or ""
        except Exception:
            pass

    if et == "addition":
        mp = root / "objects" / shard / oid / "edits_3d" / eid / "meta.json"
        if mp.is_file():
            try:
                m = json.loads(mp.read_text())
                meta.update({
                    "prompt": m.get("prompt") or "",
                    "target_part_desc": m.get("target_part_desc") or "",
                    "source_del_id": m.get("source_del_id") or "",
                })
            except Exception:
                pass
    elif parsed_path.is_file():
        try:
            pj = json.loads(parsed_path.read_text())
            edits_list = pj.get("parsed", {}).get("edits", [])
            flux_seq = 0
            del_seq = 0
            for ed in edits_list:
                t = ed.get("edit_type")
                pfx = EID_PFX.get(t)
                if not pfx:
                    continue
                if t == "deletion":
                    seq = del_seq; del_seq += 1
                elif t in FLUX_TYPES_SET:
                    seq = flux_seq; flux_seq += 1
                else:
                    continue
                cand_eid = f"{pfx}_{oid}_{seq:03d}"
                if cand_eid == eid:
                    meta.update({
                        "prompt": ed.get("prompt") or "",
                        "target_part_desc": ed.get("target_part_desc") or "",
                    })
                    break
        except Exception:
            pass

    return meta


def _build_pair_for_pick(root: Path, shard: str, images_root: Path,
                         oid: str, eid: str, meta: dict, *,
                         view_h: int = 360):
    """Returns (before_uri, after_uri).

    Sources per edit_type (white-bg, alpha-composited, JPEG q=92):
      deletion / FLUX
        BEFORE = NPZ frames at VIEW_INDICES
        AFTER  = edits_3d/<eid>/preview_{0..4}.png
      addition (= inverse of source deletion; show del's pair reversed)
        BEFORE = edits_3d/<source_del_id>/preview_{0..4}.png  (no part)
        AFTER  = NPZ frames at VIEW_INDICES                   (with part)
    """
    et = _eid_type(eid)
    edits_3d_root = root / "objects" / shard / oid / "edits_3d"
    npz_path = images_root / shard / f"{oid}.npz"

    if et == "addition":
        src = meta.get("source_del_id") or ""
        src_dir = edits_3d_root / src if src else None
        if src_dir and src_dir.is_dir():
            before_imgs = [_open_path(src_dir / f"preview_{i}.png")
                            for i in range(5)]
        else:
            before_imgs = [None] * 5
        after_imgs = [_open_png_bytes(_png_bytes_from_npz(npz_path, v))
                       for v in VIEW_INDICES]
    else:
        before_imgs = [_open_png_bytes(_png_bytes_from_npz(npz_path, v))
                        for v in VIEW_INDICES]
        edit_dir = edits_3d_root / eid
        after_imgs = [_open_path(edit_dir / f"preview_{i}.png")
                       for i in range(5)]

    return (_build_view_strip_hd(before_imgs, h=view_h),
            _build_view_strip_hd(after_imgs, h=view_h))


CSS = """
:root{--bg:#0e0e10;--s1:#16161a;--s2:#1c1c22;--bd:#2a2a32;--tx:#e6e6ea;
      --sub:#7e7e88;--ac:#4a9eff;--gold:#f1c40f;}
*{box-sizing:border-box;margin:0;padding:0;}
body{background:var(--bg);color:var(--tx);
     font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
     padding:28px 32px;line-height:1.45;}
h1{font-size:1.7rem;color:var(--ac);margin-bottom:6px;font-weight:700;
   letter-spacing:.01em;}
.sub{font-size:.85rem;color:var(--sub);margin-bottom:24px;}
.card{background:var(--s1);border:1px solid var(--bd);border-radius:10px;
      padding:18px 22px;margin-bottom:22px;}
.card .hdr{display:flex;align-items:center;gap:12px;margin-bottom:8px;
           flex-wrap:wrap;}
.chip{display:inline-block;padding:4px 12px;border-radius:13px;
      font-size:.72rem;text-transform:uppercase;letter-spacing:.06em;
      font-weight:700;color:#fff;}
.eid{font-family:'JetBrains Mono','SF Mono',monospace;font-size:.78rem;
     color:var(--tx);}
.eid .obj{color:var(--sub);}
.score{font-family:'JetBrains Mono',monospace;font-size:.74rem;
       color:var(--gold);margin-left:auto;}
.idx{font-family:'JetBrains Mono',monospace;font-size:.68rem;
     color:var(--sub);}
.prompt{font-size:1.0rem;color:var(--tx);font-style:italic;
        line-height:1.55;margin:8px 0 6px 0;max-width:1400px;}
.prompt .quote{color:var(--ac);font-style:normal;}
.target{font-size:.78rem;color:var(--sub);font-style:normal;
        margin-bottom:14px;}
.row{display:flex;align-items:center;gap:14px;margin:6px 0;}
.row .lbl{font-size:.7rem;color:var(--sub);text-transform:uppercase;
          letter-spacing:.08em;font-weight:700;width:64px;
          font-family:'JetBrains Mono',monospace;}
.row.before .lbl{color:#aaa;}
.row.after  .lbl{color:var(--gold);}
.row img{display:block;border-radius:6px;border:1px solid var(--bd);
         background:#fff;max-width:100%;height:auto;}
.toolbar{position:sticky;top:0;background:var(--bg);z-index:30;
         padding:10px 0;border-bottom:1px solid var(--bd);
         margin-bottom:18px;display:flex;gap:8px;flex-wrap:wrap;
         align-items:center;}
.btn{background:var(--s2);color:var(--tx);border:1px solid var(--bd);
     border-radius:5px;padding:5px 11px;font-size:.78rem;cursor:pointer;
     font-family:inherit;}
.btn.active{background:var(--ac);border-color:var(--ac);color:#fff;}
"""

JS = r"""
const filters = {type: 'all'};
function applyFilters(){
  document.querySelectorAll('.card').forEach(c => {
    c.style.display = (filters.type === 'all'
      || c.dataset.type === filters.type) ? '' : 'none';
  });
}
document.addEventListener('DOMContentLoaded', () => {
  document.querySelectorAll('.btn[data-type]').forEach(b => {
    b.addEventListener('click', () => {
      document.querySelectorAll('.btn[data-type]').forEach(x => x.classList.remove('active'));
      b.classList.add('active');
      filters.type = b.dataset.type;
      applyFilters();
    });
  });
});
"""


def _render_card(idx: int, oid: str, eid: str, et: str,
                 meta: dict, before_uri: str, after_uri: str) -> str:
    chip_color = TYPE_CHIP.get(et, "#666")
    label = TYPE_LABEL.get(et, et.upper())
    prompt = escape(meta.get("prompt") or "(no prompt)")
    tgt = escape(meta.get("target_part_desc") or "")
    score = meta.get("score")
    if et in ("addition", "deletion"):
        score_html = '<span class="score">auto-pass</span>'
    elif score is not None:
        score_html = f'<span class="score">VLM score = {score:.2f}</span>'
    else:
        score_html = ""
    target_html = f'<div class="target">target part: {tgt}</div>' if tgt else ""
    return f"""
<div class="card" data-type="{et}">
  <div class="hdr">
    <span class="idx">#{idx:02d}</span>
    <span class="chip" style="background:{chip_color};">{label}</span>
    <span class="eid"><span class="obj">{escape(oid)}</span> / {escape(eid)}</span>
    {score_html}
  </div>
  <div class="prompt"><span class="quote">&ldquo;</span>{prompt}<span class="quote">&rdquo;</span></div>
  {target_html}
  <div class="row before">
    <span class="lbl">BEFORE</span>
    <img src="{before_uri}" alt="before">
  </div>
  <div class="row after">
    <span class="lbl">AFTER</span>
    <img src="{after_uri}" alt="after">
  </div>
</div>
""".strip()


def render(picks: list, root: Path, shard: str, images_root: Path,
           out_path: Path, view_h: int = 360) -> None:
    type_counts = defaultdict(int)
    rendered_cards = []

    for i, pick in enumerate(picks, 1):
        if "/" not in pick:
            print(f"  [skip] bad pick (no /): {pick}", file=sys.stderr)
            continue
        oid, eid = pick.split("/", 1)
        et = _eid_type(eid)
        meta = _load_meta_for_pick(root, shard, oid, eid)
        before_uri, after_uri = _build_pair_for_pick(
            root, shard, images_root, oid, eid, meta, view_h=view_h,
        )
        rendered_cards.append(_render_card(i, oid, eid, et, meta,
                                            before_uri, after_uri))
        type_counts[et] += 1
        print(f"  [{i:02d}/{len(picks)}] rendered {et:12s} {oid[:8]}... {eid}",
              file=sys.stderr)

    parts = []
    parts.append("<!doctype html><html><head><meta charset='utf-8'>"
                 "<title>H3D Dataset Overview</title>")
    parts.append(f"<style>{CSS}</style></head><body>")
    parts.append("<h1>H3D Dataset Overview</h1>")
    type_summary = ", ".join(
        f"{TYPE_LABEL[t].lower()}={type_counts[t]}"
        for t in TYPE_LABEL
        if type_counts[t] > 0
    )
    parts.append(
        f"<div class='sub'>{len(rendered_cards)} hand-picked edits "
        f"({type_summary}) &middot; view height {view_h}px &middot; "
        f"JPEG quality 92</div>"
    )
    parts.append("<div class='toolbar'>")
    parts.append("<span style='font-size:.7rem;color:var(--sub);"
                 "margin-right:6px;text-transform:uppercase;'>filter:</span>")
    parts.append("<button class='btn active' data-type='all'>"
                  f"all ({len(rendered_cards)})</button>")
    for t, n in type_counts.items():
        if n > 0:
            chip_color = TYPE_CHIP.get(t)
            parts.append(
                f"<button class='btn' data-type='{t}' "
                f"style='border-left:3px solid {chip_color};'>{t} ({n})</button>"
            )
    parts.append("</div>")
    parts.extend(rendered_cards)
    parts.append(f"<script>{JS}</script></body></html>")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(parts))
    print(f"\n[render] wrote {out_path}  "
          f"({out_path.stat().st_size/1024/1024:.1f} MB)", file=sys.stderr)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--shard", default="08")
    ap.add_argument("--images-root", required=True, type=Path)
    ap.add_argument("--picks", required=True, type=Path,
                    help="JSON file: list of '<obj>/<edit_id>' strings")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--view-h", type=int, default=360,
                    help="Per-view image height in px (default 360)")
    a = ap.parse_args()

    picks = json.loads(a.picks.read_text())
    if not isinstance(picks, list):
        print("ERROR: --picks must be a JSON array of strings", file=sys.stderr)
        return 2
    print(f"[picks] {len(picks)} ids loaded from {a.picks}", file=sys.stderr)
    render(picks, a.root, a.shard, a.images_root, a.out, view_h=a.view_h)
    return 0


if __name__ == "__main__":
    sys.exit(main())
