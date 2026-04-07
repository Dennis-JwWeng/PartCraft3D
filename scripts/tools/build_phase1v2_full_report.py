#!/usr/bin/env python3
"""End-to-end phase1_v2 pipeline report.

For every edit shows up to 5 columns:
    ORIGINAL · HIGHLIGHT · FLUX 2D EDIT · 3D BEFORE · 3D AFTER
plus the prompt, rationale, target_part_desc, edit_type, view, pids.

Sources (all under --in-dir):
    *.parsed.json                  edits + per-edit metadata
    _hl_edits/{obj}__e{idx:02d}.png  highlight (rendered on demand if missing)
    2d_edits/{edit_id}_edited.png  flux 2D output (mod/scl/mat/glb only)
    _3d_renders/{edit_id}_{before,after}.png   3D-edit gaussian rerender
"""
from __future__ import annotations
import argparse
import base64
import html
import json
import sys
from collections import Counter
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from render_part_overview import VIEW_INDICES, load_views_from_npz  # noqa: E402
from render_part_highlight import render_highlight  # noqa: E402


@lru_cache(maxsize=64)
def _orig_views(npz_str: str) -> tuple:
    imgs, _ = load_views_from_npz(Path(npz_str), VIEW_INDICES)
    return tuple(imgs)


def _embed(img, max_w: int) -> str:
    if img is None:
        return ""
    h, w = img.shape[:2]
    if w > max_w:
        s = max_w / w
        img = cv2.resize(img, (max_w, int(h * s)),
                         interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 82])
    if not ok:
        return ""
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def _placeholder(W: int, H: int, text: str) -> np.ndarray:
    img = np.full((H, W, 3), 245, dtype=np.uint8)
    cv2.putText(img, text, (W // 12, H // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (140, 140, 140), 1, cv2.LINE_AA)
    return img


def _label(text: str, W: int) -> np.ndarray:
    bar = np.full((26, W, 3), 232, dtype=np.uint8)
    cv2.putText(bar, text, (10, 19), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (35, 35, 35), 1, cv2.LINE_AA)
    return bar


def stitch_row(panels: list[tuple[str, np.ndarray | None]]) -> np.ndarray:
    """panels = [(label, img|None), ...] all panels resized to common H/W."""
    # use first non-None as reference
    ref = next((p for _, p in panels if p is not None), None)
    if ref is None:
        return np.zeros((10, 10, 3), dtype=np.uint8)
    H, W = ref.shape[:2]
    cells = []
    for lab, img in panels:
        if img is None:
            img = _placeholder(W, H, "(none)")
        elif img.shape[:2] != (H, W):
            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        cells.append(np.vstack([_label(lab, W), img]))
    sep = np.full((cells[0].shape[0], 6, 3), 180, dtype=np.uint8)
    out = cells[0]
    for c in cells[1:]:
        out = np.hstack([out, sep, c])
    return out


TYPE_COLORS = {
    "deletion": "#f85149",
    "modification": "#1f6feb",
    "scale": "#a371f7",
    "material": "#d29922",
    "global": "#3fb950",
}
PREFIX = {"modification": "mod", "scale": "scl",
          "material": "mat", "global": "glb"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True, type=Path)
    ap.add_argument("--shard", default="01")
    ap.add_argument("--tag", default="mirror5")
    ap.add_argument("--mesh-root", default="data/partverse/mesh", type=Path)
    ap.add_argument("--images-root", default="data/partverse/images", type=Path)
    ap.add_argument("--blender",
                    default="/Node11_nvme/artgen/lac/.tools/blender-4.2.0-linux-x64/blender")
    ap.add_argument("--out", default=None, type=Path)
    ap.add_argument("--max-width", type=int, default=1100)
    args = ap.parse_args()

    files = sorted(args.in_dir.glob("*.parsed.json"))
    if not files:
        raise SystemExit(f"no parsed.json in {args.in_dir}")
    out = args.out or (args.in_dir / "report_full.html")
    hl_cache = args.in_dir / "_hl_edits"
    hl_cache.mkdir(parents=True, exist_ok=True)
    flux_dir = args.in_dir / "2d_edits"
    rd3 = args.in_dir / "_3d_renders"

    type_counts = Counter()
    objs_meta = []
    for f in files:
        j = json.loads(f.read_text())
        edits = (j.get("parsed") or {}).get("edits") or []
        objs_meta.append((f, j, edits))
        for e in edits:
            type_counts[e.get("edit_type", "?")] += 1
    n_total = sum(type_counts.values())

    style = """
body { font-family: -apple-system, sans-serif; margin:0; background:#0d1117; color:#c9d1d9; }
header { padding:14px 22px; background:#161b22; border-bottom:1px solid #30363d;
         position:sticky; top:0; z-index:10; }
h1 { margin:0 0 4px; font-size:17px; }
.subtle { color:#8b949e; font-size:13px; }
.summary { display:flex; gap:10px; flex-wrap:wrap; margin-top:8px; }
.pill { padding:3px 10px; border-radius:12px; font-size:12px; font-weight:600; color:#0d1117; }
.controls { margin-top:10px; display:flex; gap:6px; flex-wrap:wrap; }
.controls button { background:#21262d; color:#c9d1d9; border:1px solid #30363d;
                   padding:3px 11px; border-radius:6px; cursor:pointer; font-size:12px; }
.controls button.active { background:#1f6feb; border-color:#1f6feb; color:white; }
.obj { padding:22px; border-bottom:1px solid #30363d; }
.objHeader { display:flex; align-items:baseline; gap:14px; margin-bottom:8px; flex-wrap:wrap; }
.objHeader h2 { margin:0; font-size:15px; font-family:monospace; }
.objDesc { color:#8b949e; font-size:12px; margin:6px 0 12px; max-width:1200px; line-height:1.4; }
.objDesc b { color:#c9d1d9; }
.partsList { font-size:11px; color:#8b949e; margin:4px 0 12px; font-family:monospace; }
.grid { display:grid; grid-template-columns:1fr; gap:14px; }
.card { background:#161b22; border:1px solid #30363d; border-radius:8px; overflow:hidden; }
.card img { width:100%; display:block; background:white; }
.card .meta { padding:10px 14px; font-size:12px; }
.card .row { display:flex; justify-content:space-between; align-items:center; margin-bottom:6px; }
.card .prompt { color:#c9d1d9; font-weight:600; font-size:13px; margin:2px 0 4px; }
.card .desc { color:#8b949e; font-size:11px; margin:2px 0 5px; }
.card .new { color:#3fb950; font-size:11px; margin:2px 0; }
.card .params { color:#a371f7; font-size:11px; margin:2px 0; font-family:monospace; }
.card .rationale { color:#6e7681; font-size:11px; margin-top:5px; font-style:italic; }
.typeBadge { padding:2px 7px; border-radius:4px; font-size:10px; font-weight:700;
             color:#0d1117; text-transform:uppercase; }
.hidden { display:none; }
"""
    pills = "".join(
        f'<span class="pill" style="background:{TYPE_COLORS[t]};">{t}: {type_counts[t]}</span>'
        for t in ("deletion", "modification", "scale", "material", "global")
        if type_counts[t]
    )
    parts_out = [f"""<!doctype html><html><head><meta charset="utf-8">
<title>Phase 1 v2 — full pipeline report</title><style>{style}</style></head><body>
<header>
  <h1>Phase 1 v2 — full pipeline (orig · highlight · flux 2D · 3D before · 3D after)</h1>
  <div class="subtle">{len(files)} objects · <b>{n_total}</b> edits ·
       source: <code>{html.escape(str(args.in_dir))}</code></div>
  <div class="summary">{pills}</div>
  <div class="controls" id="filters">
    <button class="active" data-type="all">all</button>
    <button data-type="deletion">deletion</button>
    <button data-type="modification">modification</button>
    <button data-type="scale">scale</button>
    <button data-type="material">material</button>
    <button data-type="global">global</button>
  </div>
</header>"""]

    for fi, (f, j, edits) in enumerate(objs_meta):
        obj_id = j["obj_id"]
        obj = (j.get("parsed") or {}).get("object") or {}
        full_desc = obj.get("full_desc", "")
        canonical_front = obj.get("canonical_front")
        frontal_vi = obj.get("frontal_view_index")
        parts = obj.get("parts", [])
        mesh = args.mesh_root / args.shard / f"{obj_id}.npz"
        img_npz = args.images_root / args.shard / f"{obj_id}.npz"
        if not (mesh.is_file() and img_npz.is_file()):
            print(f"[skip] {obj_id}: missing npz")
            continue
        try:
            origs = _orig_views(str(img_npz))
        except Exception as e:
            print(f"[skip] {obj_id}: {e}")
            continue

        print(f"[{fi+1}/{len(objs_meta)}] {obj_id}  ({len(edits)} edits)")
        type_count_obj = Counter(e.get("edit_type", "?") for e in edits)
        parts_out.append(f'<section class="obj" id="{obj_id}">')
        parts_out.append('<div class="objHeader">')
        parts_out.append(f'<h2>{obj_id}</h2>')
        for t in ("deletion", "modification", "scale", "material", "global"):
            n = type_count_obj.get(t, 0)
            if n:
                parts_out.append(
                    f'<span class="typeBadge" style="background:{TYPE_COLORS[t]};">{t} {n}</span>')
        parts_out.append('</div>')
        if full_desc:
            parts_out.append(f'<div class="objDesc"><b>desc:</b> {html.escape(full_desc)}</div>')
        if canonical_front:
            parts_out.append(
                f'<div class="objDesc"><b>canonical_front:</b> {html.escape(canonical_front)} '
                f'· <b>frontal_view_index:</b> {frontal_vi}</div>')
        parts_lines = ", ".join(
            f"#{p.get('part_id','?')}={html.escape(str(p.get('name','?')))}"
            for p in parts)
        parts_out.append(f'<div class="partsList">{parts_lines}</div>')

        flux_seq = 0   # SHARED across mod/scl/mat/glb (matches parsed_to_edit_specs)
        parts_out.append('<div class="grid">')
        for idx, e in enumerate(edits):
            et = e.get("edit_type", "?")
            vi = int(e.get("view_index", 0))
            pids = list(e.get("selected_part_ids") or [])

            # 1. ORIGINAL
            orig = origs[vi] if 0 <= vi < len(origs) else None

            # 2. HIGHLIGHT (cached)
            cache_path = hl_cache / f"{obj_id}__e{idx:02d}.png"
            if cache_path.is_file():
                hl = cv2.imread(str(cache_path), cv2.IMREAD_COLOR)
            else:
                try:
                    if pids:
                        _o, hl = render_highlight(mesh, img_npz, vi, pids, args.blender)
                    else:
                        hl = orig
                    if hl is not None:
                        cv2.imwrite(str(cache_path), hl)
                except Exception as ex:
                    print(f"  [hl fail] e{idx}: {ex}")
                    hl = None

            # 3-5. flux + 3d before/after — only for non-deletion w/ flux seq
            flux_2d = b3d = a3d = None
            if et in PREFIX:
                edit_id = f"{PREFIX[et]}_{obj_id}_{flux_seq:03d}"
                flux_seq += 1
                fp = flux_dir / f"{edit_id}_edited.png"
                if fp.is_file():
                    flux_2d = cv2.imread(str(fp), cv2.IMREAD_COLOR)
                bp = rd3 / f"{edit_id}_before.png"
                ap2 = rd3 / f"{edit_id}_after.png"
                if bp.is_file():
                    b3d = cv2.imread(str(bp), cv2.IMREAD_COLOR)
                if ap2.is_file():
                    a3d = cv2.imread(str(ap2), cv2.IMREAD_COLOR)

            row = stitch_row([
                ("ORIGINAL", orig),
                ("HIGHLIGHT", hl),
                ("FLUX 2D EDIT", flux_2d),
                ("3D BEFORE", b3d),
                ("3D AFTER", a3d),
            ])
            img_uri = _embed(row, args.max_width)

            params = e.get("edit_params") or {}
            param_str = ", ".join(f"{k}={v}" for k, v in params.items())
            new_desc = e.get("new_parts_desc") or ""
            new_html = (f'<div class="new">→ {html.escape(new_desc[:240])}</div>'
                        if new_desc else "")
            parts_out.append(f'''
<div class="card" data-type="{et}">
  <img src="{img_uri}">
  <div class="meta">
    <div class="row">
      <span style="font-family:monospace;color:#8b949e;font-size:11px;">
        #{idx} · v{vi} · pids={pids}
      </span>
      <span class="typeBadge" style="background:{TYPE_COLORS.get(et,"#888")};">{et}</span>
    </div>
    <div class="prompt">{html.escape(e.get("prompt",""))}</div>
    <div class="desc">target: {html.escape(e.get("target_part_desc",""))}</div>
    {f'<div class="params">{html.escape(param_str)}</div>' if param_str else ''}
    {new_html}
    <div class="rationale">{html.escape(e.get("rationale",""))}</div>
  </div>
</div>''')
        parts_out.append('</div></section>')

    parts_out.append("""
<script>
const buttons = document.querySelectorAll('#filters button');
buttons.forEach(b => b.addEventListener('click', () => {
  buttons.forEach(x => x.classList.remove('active'));
  b.classList.add('active');
  const t = b.dataset.type;
  document.querySelectorAll('.card').forEach(c => {
    c.classList.toggle('hidden', t !== 'all' && c.dataset.type !== t);
  });
  document.querySelectorAll('.obj').forEach(o => {
    const v = o.querySelectorAll('.card:not(.hidden)').length;
    o.style.display = v ? '' : 'none';
  });
}));
</script>
</body></html>""")

    out.write_text("".join(parts_out), encoding="utf-8")
    print(f"\n[OK] {out}  ({n_total} edits, {len(files)} objects)")


if __name__ == "__main__":
    main()
