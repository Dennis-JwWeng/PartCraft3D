#!/usr/bin/env python3
"""build_showcase_candidates.py - candidate-pool HTML for shard08 mode_e showcase."""
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
    "deletion": 20, "modification": 25, "global": 25, "material": 20,
    "scale": 12, "color": 10, "addition": 8,
}


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
    """RGBA -> RGB by alpha-compositing onto WHITE.

    NPZ source frames are RGBA with transparent backgrounds; gate_a /
    gate_e and preview_*.png all use WHITE backgrounds, so we mirror
    that here for visual parity in the report.
    """
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


def _gather_candidates(root: Path, shard: str) -> dict:
    obj_root = root / "objects" / shard
    out = defaultdict(list)
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
        # Mirrors partcraft.pipeline_v3.specs.iter_all_specs: deletion gets
        # its own 0-based counter; FLUX (mod/scl/mat/clr/glb) share a single
        # `flux_seq` counter that increments across types in parsed.json
        # order. Addition is NOT in parsed.json (backfilled from deletion in
        # s7), so we read its prompt from edits_3d/<add_eid>/meta.json below.
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
        # Backfill addition prompts from edits_3d/<add_eid>/meta.json.
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
            out[et].append({
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
            })
        n_obj += 1
    print(f"[gather] scanned {n_obj} objs", file=sys.stderr)
    return out


def _rank_and_select(pool, quotas, seed=0):
    rng = random.Random(seed)
    selected = []
    for et, n in quotas.items():
        cands = list(pool.get(et, []))
        if not cands:
            continue
        if et in ("deletion", "addition"):
            cands.sort(key=lambda c: (-c["pixel_sum"], rng.random()))
        else:
            cands.sort(key=lambda c: (-c["score"], -c["pixel_sum"], c["eid"]))
        selected.extend(cands[:n])
    return selected


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

JS = r"""
const picked = new Set();
function updateCount(){
  const c = document.getElementById('pick-count');
  c.innerHTML = '<b>' + picked.size + '</b> / 20 picked';
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
const filters = {type: 'all', mode: 'all'};
function applyFilters(){
  document.querySelectorAll('.card').forEach(c => {
    const ok_type = filters.type === 'all' || c.dataset.type === filters.type;
    const ok_mode = filters.mode === 'all'
      || (filters.mode === 'picked' && picked.has(c.dataset.id));
    c.style.display = (ok_type && ok_mode) ? '' : 'none';
  });
}
document.addEventListener('DOMContentLoaded', () => {
  document.querySelectorAll('.btn[data-type]').forEach(b => {
    b.addEventListener('click', () => {
      document.querySelectorAll('.btn[data-type]').forEach(x => x.classList.remove('active'));
      b.classList.add('active'); filters.type = b.dataset.type; applyFilters();
    });
  });
  document.querySelectorAll('.btn[data-mode]').forEach(b => {
    b.addEventListener('click', () => {
      document.querySelectorAll('.btn[data-mode]').forEach(x => x.classList.remove('active'));
      b.classList.add('active'); filters.mode = b.dataset.mode; applyFilters();
    });
  });
  updateCount();
});
"""


def _render_card(c, before_uri, after_uri):
    et = c["edit_type"]
    chip_color = TYPE_CHIP.get(et, "#666")
    pid = f"{c['obj']}/{c['eid']}"
    score_str = f"score={c['score']:.2f}"
    if et in ("deletion", "addition"):
        score_str = f"auto-pass | pix={c['pixel_sum']}"
    prompt = escape(c.get("prompt") or "(no prompt)")
    tgt = escape(c.get("target_part_desc") or "")
    return f"""
<div class="card" data-id="{escape(pid)}" data-type="{et}">
  <div class="pick" onclick="togglePick(this.parentElement)">&#9733;</div>
  <div class="meta">
    <span class="chip" style="background:{chip_color};">{et}</span>
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


def _build_card_assets(c, root: Path, shard: str, images_root: Path, view_h=140):
    """Return (before_uri, after_uri) data URIs for one candidate.

    Source per edit_type:
      deletion / FLUX (mod/scl/mat/clr/glb)
        BEFORE = NPZ frames at VIEW_INDICES (raw multi-view RGB)
        AFTER  = edits_3d/<eid>/preview_{0..4}.png
      addition (inverse of deletion)
        BEFORE = edits_3d/<source_del_id>/preview_{0..4}.png  (post-deletion)
        AFTER  = edits_3d/<eid>/preview_{0..4}.png            (part added back)
    """
    obj = c["obj"]; eid = c["eid"]; et = c.get("edit_type")
    edits_3d_root = root / "objects" / shard / obj / "edits_3d"
    edit_dir = edits_3d_root / eid
    after_imgs = [_open_path(edit_dir / f"preview_{i}.png") for i in range(5)]

    if et == "addition":
        # addition is the inverse of source deletion -> mirror del's pair:
        #   BEFORE = source del preview (no part), AFTER = original NPZ (part present)
        # We override `after_imgs` here because edit_dir/preview is the
        # Trellis re-reconstruction which can carry artifacts; the original
        # NPZ shows the ground-truth "part-present" target instead.
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


def render(selected, root, shard, images_root, out_path):
    parts = []
    parts.append("<!doctype html><html><head><meta charset='utf-8'>"
                 "<title>shard08 showcase candidates</title>")
    parts.append(f"<style>{CSS}</style></head><body>")
    parts.append("<h1>shard08 mode_e &mdash; showcase candidate pool</h1>")
    parts.append(
        f"<div class='sub'>{len(selected)} candidates &middot; click "
        f"&#9733; to pick &middot; target = 20 final picks &middot; "
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
                 "margin-left:14px;margin-right:6px;text-transform:uppercase;'>show:</span>")
    parts.append("<button class='btn active' data-mode='all'>all</button>")
    parts.append("<button class='btn gold' data-mode='picked'>picked</button>")
    parts.append("<span class='counter' id='pick-count'><b>0</b> / 20 picked</span>")
    parts.append("</div>")
    parts.append("<div class='grid'>")
    for i, c in enumerate(selected, 1):
        before, after = _build_card_assets(c, root, shard, images_root)
        parts.append(_render_card(c, before, after))
        if i % 25 == 0:
            print(f"  rendered {i}/{len(selected)}", file=sys.stderr)
    parts.append("</div>")
    parts.append("<div class='copy-pad' id='copy-pad'>")
    parts.append("<button class='copy-btn' id='copy-btn' onclick='copyPicks()'>copy JSON</button>")
    parts.append("<pre id='copy-text'></pre></div>")
    parts.append(f"<script>{JS}</script></body></html>")
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
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--shard", default="08")
    ap.add_argument("--images-root", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--quota", nargs="*", default=[])
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    quotas = dict(DEFAULT_QUOTAS)
    quotas.update(_parse_quota_overrides(a.quota))
    print(f"[quotas] {quotas}  (total={sum(quotas.values())})", file=sys.stderr)

    pool = _gather_candidates(a.root, a.shard)
    for et in EDIT_TYPES:
        print(f"  pool[{et:12s}] = {len(pool.get(et, []))} final-pass edits",
              file=sys.stderr)
    selected = _rank_and_select(pool, quotas, seed=a.seed)
    print(f"[selected] {len(selected)} candidates", file=sys.stderr)
    render(selected, a.root, a.shard, a.images_root, a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
