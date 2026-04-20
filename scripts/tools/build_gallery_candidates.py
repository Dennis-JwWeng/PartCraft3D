#!/usr/bin/env python3
"""build_gallery_candidates.py - multi-shard candidate-pool HTML for the
H3D showcase gallery. Sits next to build_showcase_candidates.py and shares
the same data conventions, but:

  * --shards accepts multiple shards (e.g. 07 08); pool is unioned.
  * card id is "<shard>/<obj>/<eid>" so picks JSON disambiguates shard.
  * JS counter target is configurable via --target.
  * Prints a compact GLB-availability badge per card so the user knows
    which picks already have a *.glb on disk vs need TRELLIS decode:
      - del  -> after_new.glb exists  (BEFORE = source full.glb)
      - add  -> derives from source del's after_new.glb (BEFORE) + full.glb (AFTER)
      - mod/mat/clr/glb/scl -> only slat tokens (after.npz); needs decode
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
from typing import Iterable

import numpy as np
from PIL import Image

VIEW_INDICES = [89, 90, 91, 100, 8]
EDIT_TYPES = ("deletion", "modification", "scale", "material",
              "color", "global", "addition")
FLUX_TYPES_SET = {"modification", "scale", "material", "color", "global"}
EID_PFX = {"deletion": "del", "modification": "mod", "scale": "scl",
           "material": "mat", "color": "clr", "global": "glb",
           "addition": "add"}
TYPE_CHIP = {
    "deletion":     "#c0392b",
    "modification": "#3778d4",
    "scale":        "#d49c1a",
    "material":     "#27ae60",
    "color":        "#b83db8",
    "global":       "#16a085",
    "addition":     "#2ecc71",
}
DEFAULT_QUOTAS = {
    "deletion": 60, "modification": 70, "global": 60, "material": 50,
    "scale": 30, "color": 30, "addition": 50,
}


# ----------------------------- I/O helpers ----------------------------------

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


def _flatten_to_white(im: Image.Image) -> Image.Image:
    if im.mode == "RGBA":
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bg.paste(im, mask=im.split()[-1])
        return bg
    if im.mode != "RGB":
        return im.convert("RGB")
    return im


def _img_to_jpeg_b64(im: Image.Image, *, quality: int = 78) -> str:
    im = _flatten_to_white(im)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=quality, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _build_view_strip(images, *, h: int = 140, gap: int = 3) -> str:
    if not images:
        return ""
    norm = []
    for im in images:
        if im is None:
            im = Image.new("RGB", (h, h), (255, 255, 255))
        else:
            im = _flatten_to_white(im)
        ratio = h / im.height
        norm.append(im.resize((int(im.width * ratio), h), Image.LANCZOS))
    total_w = sum(im.width for im in norm) + gap * (len(norm) - 1)
    out = Image.new("RGB", (total_w, h), (255, 255, 255))
    x = 0
    for im in norm:
        out.paste(im, (x, 0))
        x += im.width + gap
    return _img_to_jpeg_b64(out)


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


# ----------------------------- gather ---------------------------------------

def _gather_one_shard(root: Path, shard: str) -> dict:
    """Return {edit_type: [candidate_dict,...]} for one shard."""
    obj_root = root / "objects" / shard
    out = defaultdict(list)
    if not obj_root.is_dir():
        print(f"[gather] WARN missing {obj_root}", file=sys.stderr)
        return out
    objs = sorted(p.name for p in obj_root.iterdir() if p.is_dir())
    n_obj = 0
    for oid in objs:
        sf = obj_root / oid / "edit_status.json"
        if not sf.is_file():
            continue
        try:
            stat = json.loads(sf.read_text())
        except Exception:
            continue
        parsed_path = obj_root / oid / "phase1" / "parsed.json"
        prompt_by_id = {}
        if parsed_path.is_file():
            try:
                pj = json.loads(parsed_path.read_text())
                edits_list = pj.get("parsed", {}).get("edits", [])
                flux_seq = 0
                del_seq = 0
                for ed in edits_list:
                    et = ed.get("edit_type")
                    pfx = EID_PFX.get(et)
                    if not pfx:
                        continue
                    if et == "deletion":
                        seq = del_seq; del_seq += 1
                    elif et in FLUX_TYPES_SET:
                        seq = flux_seq; flux_seq += 1
                    else:
                        continue
                    eid = f"{pfx}_{oid}_{seq:03d}"
                    prompt_by_id[eid] = ed
            except Exception:
                pass
        # add prompts from meta.json
        add_dir_root = obj_root / oid / "edits_3d"
        if add_dir_root.is_dir():
            for sub in add_dir_root.iterdir():
                if sub.is_dir() and sub.name.startswith("add_"):
                    mp = sub / "meta.json"
                    if mp.is_file():
                        try:
                            m = json.loads(mp.read_text())
                            prompt_by_id[sub.name] = {
                                "prompt": m.get("prompt") or "",
                                "target_part_desc": m.get("target_part_desc") or "",
                                "selected_part_ids": m.get("selected_part_ids") or [],
                                "edit_type": "addition",
                                "source_del_id": m.get("source_del_id") or "",
                            }
                        except Exception:
                            pass

        for eid, ed in stat.get("edits", {}).items():
            if not ed.get("final_pass"):
                continue
            et = ed.get("edit_type")
            if et not in EDIT_TYPES:
                continue
            ga = (ed.get("gates") or {}).get("A") or {}
            ge = (ed.get("gates") or {}).get("E") or {}
            ga_vlm = ga.get("vlm") or {}
            ge_vlm = ge.get("vlm") or {}
            score = ge_vlm.get("score") if ge_vlm else None
            pixel_sum = sum(ga_vlm.get("pixel_counts") or []) or 0
            parsed_ed = prompt_by_id.get(eid, {})
            # GLB availability:
            edit_dir = add_dir_root / eid
            glb_present = (edit_dir / "after_new.glb").is_file()
            if et == "addition":
                src = parsed_ed.get("source_del_id") or ""
                if src:
                    glb_present = (add_dir_root / src / "after_new.glb").is_file()
            out[et].append({
                "shard": shard,
                "obj": oid,
                "eid": eid,
                "edit_type": et,
                "score": score if score is not None else 0.0,
                "pixel_sum": int(pixel_sum),
                "best_view": ga_vlm.get("best_view"),
                "prompt": parsed_ed.get("prompt") or "",
                "target_part_desc": parsed_ed.get("target_part_desc") or "",
                "selected_part_ids": parsed_ed.get("selected_part_ids") or [],
                "source_del_id": parsed_ed.get("source_del_id") or "",
                "ge_reason": ge_vlm.get("reason") or "",
                "ga_reason": ga_vlm.get("reason") or "",
                "glb_present": bool(glb_present),
            })
        n_obj += 1
    print(f"[gather:{shard}] scanned {n_obj} objs", file=sys.stderr)
    return out


def _gather_pools(roots_per_shard: dict) -> dict:
    pool = defaultdict(list)
    for shard, root in roots_per_shard.items():
        sub = _gather_one_shard(root, shard)
        for et, cands in sub.items():
            pool[et].extend(cands)
    return pool


def _rank_and_select(pool, quotas, seed=0):
    rng = random.Random(seed)
    selected = []
    for et, n in quotas.items():
        cands = list(pool.get(et, []))
        if not cands:
            continue
        if et in ("deletion", "addition"):
            # high pixel coverage first, jitter ties; prefer present GLB
            cands.sort(key=lambda c: (
                0 if c.get("glb_present") else 1,
                -c["pixel_sum"],
                rng.random(),
            ))
        else:
            cands.sort(key=lambda c: (-c["score"], -c["pixel_sum"], rng.random()))
        selected.extend(cands[:n])
    rng.shuffle(selected)  # interleave types so the page isn't grouped
    return selected


# ----------------------------- HTML / JS ------------------------------------

CSS = """
:root{--bg:#0e0e10;--s1:#16161a;--s2:#1c1c22;--bd:#2a2a32;--tx:#e6e6ea;
      --sub:#7e7e88;--ac:#4a9eff;--gold:#f1c40f;}
*{box-sizing:border-box;margin:0;padding:0;}
body{background:var(--bg);color:var(--tx);font-family:-apple-system,
     BlinkMacSystemFont,'Segoe UI',sans-serif;padding:18px 22px;line-height:1.4;}
h1{font-size:1.35rem;color:var(--ac);margin-bottom:4px;font-weight:600;}
.sub{font-size:.78rem;color:var(--sub);margin-bottom:16px;}
.toolbar{position:sticky;top:0;background:var(--bg);z-index:30;
         padding:8px 0;border-bottom:1px solid var(--bd);margin-bottom:14px;
         display:flex;gap:8px;flex-wrap:wrap;align-items:center;}
.btn{background:var(--s2);color:var(--tx);border:1px solid var(--bd);
     border-radius:5px;padding:5px 11px;font-size:.78rem;cursor:pointer;
     transition:all .15s;font-family:inherit;}
.btn:hover{border-color:var(--ac);}
.btn.active{background:var(--ac);border-color:var(--ac);color:#fff;}
.btn.gold.active{background:var(--gold);border-color:var(--gold);color:#000;}
.counter{margin-left:auto;font-family:'JetBrains Mono',monospace;
         font-size:.78rem;color:var(--sub);}
.counter b{color:var(--gold);font-size:1rem;}
.grid{display:grid;grid-template-columns:1fr;gap:10px;}
.card{background:var(--s1);border:1px solid var(--bd);border-radius:7px;
      padding:10px 12px;display:grid;
      grid-template-columns:auto 1fr;column-gap:14px;row-gap:6px;
      align-items:center;transition:border-color .15s;}
.card.picked{border-color:var(--gold);box-shadow:0 0 0 1px var(--gold);}
.pick{grid-row:1/4;font-size:1.7rem;cursor:pointer;color:var(--sub);
      user-select:none;width:38px;height:38px;display:flex;
      align-items:center;justify-content:center;border-radius:6px;
      border:1px solid var(--bd);background:var(--s2);}
.pick:hover{color:var(--gold);}
.card.picked .pick{color:var(--gold);background:#2a2410;}
.meta{display:flex;align-items:center;gap:8px;flex-wrap:wrap;
      font-size:.78rem;color:var(--sub);font-family:'JetBrains Mono',monospace;}
.chip{display:inline-block;padding:2px 8px;border-radius:11px;
      font-size:.68rem;text-transform:uppercase;letter-spacing:.04em;
      font-weight:600;color:#fff;}
.chip.shard{background:#3b3b46;color:#cfcfdc;}
.chip.glb-yes{background:#1c5b2e;}
.chip.glb-no{background:#5b1c1c;}
.eid{color:var(--tx);}
.score{color:var(--gold);}
.prompt{font-size:.84rem;color:var(--tx);font-style:italic;
        line-height:1.45;padding-top:2px;}
.prompt .target{color:var(--sub);font-style:normal;font-size:.74rem;
                margin-left:6px;}
.strips{display:flex;gap:14px;align-items:flex-start;flex-wrap:wrap;}
.strip{display:flex;flex-direction:column;gap:3px;}
.strip .lbl{font-size:.62rem;color:var(--sub);text-transform:uppercase;
            letter-spacing:.06em;}
.strip img{display:block;border-radius:4px;border:1px solid var(--bd);
           background:#000;max-width:100%;}
.copy-pad{position:fixed;bottom:18px;right:18px;background:var(--s1);
          border:1px solid var(--bd);border-radius:8px;padding:10px 12px;
          font-family:'JetBrains Mono',monospace;font-size:.72rem;
          color:var(--tx);max-width:380px;max-height:280px;overflow:auto;
          box-shadow:0 6px 24px rgba(0,0,0,.4);z-index:40;display:none;}
.copy-pad.show{display:block;}
.copy-pad .copy-btn{display:inline-block;background:var(--ac);color:#fff;
                    border:none;border-radius:4px;padding:4px 10px;
                    font-size:.72rem;cursor:pointer;margin-bottom:6px;
                    font-family:inherit;}
"""

JS_TMPL = r"""
const TARGET = __TARGET__;
const picked = new Set();
function updateCount(){
  const c = document.getElementById('pick-count');
  c.innerHTML = '<b>' + picked.size + '</b> / ' + TARGET + ' picked';
  const pad = document.getElementById('copy-pad');
  if (picked.size > 0){ pad.classList.add('show'); }
  else { pad.classList.remove('show'); }
  const out = Array.from(picked).map(id => '  ' + JSON.stringify(id)).join(',\n');
  document.getElementById('copy-text').textContent = '[\n' + out + '\n]';
}
function togglePick(card){
  const id = card.dataset.id;
  if (picked.has(id)){ picked.delete(id); card.classList.remove('picked'); }
  else { picked.add(id); card.classList.add('picked'); }
  updateCount();
}
function copyPicks(){
  navigator.clipboard.writeText(document.getElementById('copy-text').textContent);
  const b = document.getElementById('copy-btn');
  const old = b.textContent; b.textContent = 'copied!';
  setTimeout(() => { b.textContent = old; }, 1200);
}
const filters = {type: 'all', mode: 'all', shard: 'all', glb: 'all'};
function applyFilters(){
  document.querySelectorAll('.card').forEach(c => {
    const ok_type = filters.type === 'all' || c.dataset.type === filters.type;
    const ok_mode = filters.mode === 'all'
      || (filters.mode === 'picked' && picked.has(c.dataset.id));
    const ok_shard = filters.shard === 'all' || c.dataset.shard === filters.shard;
    const ok_glb = filters.glb === 'all' || c.dataset.glb === filters.glb;
    c.style.display = (ok_type && ok_mode && ok_shard && ok_glb) ? '' : 'none';
  });
}
document.addEventListener('DOMContentLoaded', () => {
  ['type','mode','shard','glb'].forEach(group => {
    document.querySelectorAll('.btn[data-' + group + ']').forEach(b => {
      b.addEventListener('click', () => {
        document.querySelectorAll('.btn[data-' + group + ']').forEach(x => x.classList.remove('active'));
        b.classList.add('active'); filters[group] = b.dataset[group]; applyFilters();
      });
    });
  });
  updateCount();
});
"""


def _render_card(c, before_uri, after_uri):
    et = c["edit_type"]
    chip_color = TYPE_CHIP.get(et, "#666")
    pid = f"{c['shard']}/{c['obj']}/{c['eid']}"
    score_str = f"score={c['score']:.2f}"
    if et in ("deletion", "addition"):
        score_str = f"auto-pass | pix={c['pixel_sum']}"
    prompt = escape(c.get("prompt") or "(no prompt)")
    tgt = escape(c.get("target_part_desc") or "")
    glb_present = bool(c.get("glb_present"))
    glb_chip = ("<span class='chip glb-yes'>glb&nbsp;ready</span>"
                if glb_present
                else "<span class='chip glb-no'>needs&nbsp;decode</span>")
    glb_attr = "yes" if glb_present else "no"
    return f"""
<div class="card" data-id="{escape(pid)}" data-type="{et}" data-shard="{c['shard']}" data-glb="{glb_attr}">
  <div class="pick" onclick="togglePick(this.parentElement)">&#9733;</div>
  <div class="meta">
    <span class="chip" style="background:{chip_color};">{et}</span>
    <span class="chip shard">shard&nbsp;{c['shard']}</span>
    {glb_chip}
    <span class="eid">{escape(pid)}</span>
    <span class="score">{score_str}</span>
  </div>
  <div class="strips">
    <div class="strip">
      <span class="lbl">BEFORE</span>
      <img src="{before_uri}" alt="before">
    </div>
    <div class="strip">
      <span class="lbl">AFTER</span>
      <img src="{after_uri}" alt="after">
    </div>
  </div>
  <div class="prompt">"{prompt}"<span class="target">- {tgt}</span></div>
</div>
""".strip()


def _build_card_assets(c, root_per_shard, images_root: Path, view_h=140):
    obj = c["obj"]; eid = c["eid"]; et = c.get("edit_type"); shard = c["shard"]
    root = root_per_shard[shard]
    edits_3d_root = root / "objects" / shard / obj / "edits_3d"
    edit_dir = edits_3d_root / eid
    after_imgs = [_open_path(edit_dir / f"preview_{i}.png") for i in range(5)]

    if et == "addition":
        src = c.get("source_del_id") or ""
        src_dir = edits_3d_root / src if src else None
        if src_dir and src_dir.is_dir():
            before_imgs = [_open_path(src_dir / f"preview_{i}.png") for i in range(5)]
        else:
            before_imgs = [None] * 5
        npz_path = images_root / shard / f"{obj}.npz"
        after_imgs = [_open_png_bytes(_png_bytes_from_npz(npz_path, v)) for v in VIEW_INDICES]
    else:
        npz_path = images_root / shard / f"{obj}.npz"
        before_imgs = [_open_png_bytes(_png_bytes_from_npz(npz_path, v)) for v in VIEW_INDICES]

    return (_build_view_strip(before_imgs, h=view_h),
            _build_view_strip(after_imgs, h=view_h))


def render(selected, root_per_shard, images_root, out_path, *, target: int):
    parts = []
    parts.append("<!doctype html><html><head><meta charset='utf-8'>"
                 "<title>H3D gallery candidates</title>")
    parts.append(f"<style>{CSS}</style></head><body>")
    parts.append("<h1>H3D gallery &mdash; multi-shard candidate pool</h1>")
    shards = sorted({c["shard"] for c in selected})
    type_summary = ", ".join(
        f"{et}={sum(1 for c in selected if c['edit_type']==et)}"
        for et in EDIT_TYPES
        if any(c['edit_type']==et for c in selected)
    )
    glb_ready = sum(1 for c in selected if c.get("glb_present"))
    parts.append(
        f"<div class='sub'>{len(selected)} candidates &middot; shards "
        f"{','.join(shards)} &middot; {type_summary} &middot; "
        f"glb-ready {glb_ready}/{len(selected)} &middot; click "
        f"&#9733; to pick &middot; target = {target} final picks &middot; "
        f"output JSON in bottom-right pad</div>"
    )
    parts.append("<div class='toolbar'>")
    parts.append("<span style='font-size:.7rem;color:var(--sub);"
                 "margin-right:6px;text-transform:uppercase;'>type:</span>")
    parts.append("<button class='btn active' data-type='all'>all</button>")
    for et in EDIT_TYPES:
        n = sum(1 for c in selected if c["edit_type"] == et)
        if n == 0:
            continue
        chip_color = TYPE_CHIP.get(et)
        parts.append(
            f"<button class='btn' data-type='{et}' "
            f"style='border-left:3px solid {chip_color};'>{et} ({n})</button>"
        )
    parts.append("<span style='font-size:.7rem;color:var(--sub);"
                 "margin-left:14px;margin-right:6px;text-transform:uppercase;'>shard:</span>")
    parts.append("<button class='btn active' data-shard='all'>all</button>")
    for sh in shards:
        n = sum(1 for c in selected if c["shard"] == sh)
        parts.append(f"<button class='btn' data-shard='{sh}'>{sh} ({n})</button>")
    parts.append("<span style='font-size:.7rem;color:var(--sub);"
                 "margin-left:14px;margin-right:6px;text-transform:uppercase;'>glb:</span>")
    parts.append("<button class='btn active' data-glb='all'>all</button>")
    parts.append(f"<button class='btn' data-glb='yes'>ready ({glb_ready})</button>")
    parts.append(f"<button class='btn' data-glb='no'>needs decode ({len(selected)-glb_ready})</button>")
    parts.append("<span style='font-size:.7rem;color:var(--sub);"
                 "margin-left:14px;margin-right:6px;text-transform:uppercase;'>show:</span>")
    parts.append("<button class='btn active' data-mode='all'>all</button>")
    parts.append("<button class='btn gold' data-mode='picked'>picked</button>")
    parts.append(f"<span class='counter' id='pick-count'><b>0</b> / {target} picked</span>")
    parts.append("</div>")
    parts.append("<div class='grid'>")
    for i, c in enumerate(selected, 1):
        before, after = _build_card_assets(c, root_per_shard, images_root)
        parts.append(_render_card(c, before, after))
        if i % 25 == 0:
            print(f"  rendered {i}/{len(selected)}", file=sys.stderr)
    parts.append("</div>")
    parts.append("<div class='copy-pad' id='copy-pad'>")
    parts.append("<button class='copy-btn' id='copy-btn' onclick='copyPicks()'>copy JSON</button>")
    parts.append("<pre id='copy-text'></pre></div>")
    js = JS_TMPL.replace("__TARGET__", str(target))
    parts.append(f"<script>{js}</script></body></html>")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(parts))
    print(f"[render] wrote {out_path}  ({out_path.stat().st_size/1024:.0f} KB)",
          file=sys.stderr)


def _parse_quota_overrides(items: Iterable[str]) -> dict:
    out = {}
    for kv in items:
        if "=" not in kv:
            raise ValueError(f"--quota expects TYPE=N (got {kv!r})")
        k, v = kv.split("=", 1)
        if k not in EDIT_TYPES:
            raise ValueError(f"unknown edit_type: {k}")
        out[k] = int(v)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shards", nargs="+", required=True,
                    help="e.g. 07 08 (must match objects/<shard>/ folder names)")
    ap.add_argument("--root", action="append", required=True,
                    help="Repeat per shard, parallel to --shards: --root outputs/partverse/shard07/mode_e_text_align ...")
    ap.add_argument("--images-root", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--quota", nargs="*", default=[])
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--target", type=int, default=50,
                    help="JS counter target (final picks goal)")
    a = ap.parse_args()

    if len(a.shards) != len(a.root):
        ap.error(f"--shards ({len(a.shards)}) and --root ({len(a.root)}) length mismatch")
    root_per_shard = {sh: Path(r) for sh, r in zip(a.shards, a.root)}

    quotas = dict(DEFAULT_QUOTAS)
    quotas.update(_parse_quota_overrides(a.quota))
    print(f"[quotas] {quotas}  (total={sum(quotas.values())})", file=sys.stderr)
    print(f"[shards] {root_per_shard}", file=sys.stderr)
    print(f"[seed]   {a.seed}  [target] {a.target}", file=sys.stderr)

    pool = _gather_pools(root_per_shard)
    for et in EDIT_TYPES:
        n = len(pool.get(et, []))
        n_glb = sum(1 for c in pool.get(et, []) if c.get("glb_present"))
        print(f"  pool[{et:12s}] = {n:>5d} final-pass  ({n_glb} glb-ready)",
              file=sys.stderr)
    selected = _rank_and_select(pool, quotas, seed=a.seed)
    print(f"[selected] {len(selected)} candidates", file=sys.stderr)
    render(selected, root_per_shard, a.images_root, a.out, target=a.target)
    return 0


if __name__ == "__main__":
    sys.exit(main())
