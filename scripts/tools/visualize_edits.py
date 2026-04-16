#!/usr/bin/env python3
"""
visualize_edits.py
──────────────────
Generates a self-contained HTML file (base64-embedded images, no server)
showing, for every edit in a parsed.json:
  • the 5-view overview with ONLY the selected parts highlighted (others greyed)
  • the edit type badge + instruction text

Usage
-----
  # single object
  python scripts/tools/visualize_edits.py \
      --parsed  <path/to/parsed.json> \
      --output  <path/to/out.html>

  # batch — all parsed.json under a run directory
  python scripts/tools/visualize_edits.py \
      --run-dir <path/to/mode_x_dir> \
      --output  <path/to/out.html>   # merged into one file
"""
from __future__ import annotations
import argparse, base64, io, json, os, sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ── palette (mirrors partcraft/render/overview.py) ──────────────────────────
_PALETTE: list[list[int]] = [
    [220,  30,  30],  # 0  red
    [255, 140,   0],  # 1  orange
    [255, 220,   0],  # 2  yellow
    [130, 220,  30],  # 3  lime
    [ 30, 160,  50],  # 4  green
    [  0, 150, 150],  # 5  teal
    [ 60, 220, 230],  # 6  cyan
    [ 30,  90, 240],  # 7  blue
    [ 20,  30, 130],  # 8  navy
    [140,  40, 200],  # 9  purple
    [230,  40, 200],  # 10 magenta
    [255, 150, 200],  # 11 pink
    [130,  70,  30],  # 12 brown
    [220, 180, 130],  # 13 tan
    [ 30,  30,  30],  # 14 black
    [130, 130, 130],  # 15 gray
]
_PALETTE_NAMES = [
    "red","orange","yellow","lime","green","teal","cyan","blue",
    "navy","purple","magenta","pink","brown","tan","black","gray",
]
_PAL_ARR = np.array(_PALETTE, dtype=np.float32)   # 16 × 3

_TYPE_CSS: dict[str, str] = {
    "deletion":     "#d43f3f",
    "modification": "#3778d4",
    "scale":        "#d49c1a",
    "material":     "#3daa50",
    "color":        "#b83db8",
    "global":       "#23aaaa",
}
_TYPE_DEFAULT_CSS = "#777"


# ── fonts ────────────────────────────────────────────────────────────────────
def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    for p in [
        f"/usr/share/fonts/truetype/dejavu/DejaVuSans{'-Bold' if bold else ''}.ttf",
        f"/usr/share/fonts/dejavu/DejaVuSans{'-Bold' if bold else ''}.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


# ── highlighted image generation ─────────────────────────────────────────────
def _make_highlight_png_b64(
    overview: np.ndarray,        # H × W × 3 uint8
    selected_part_ids: list[int],
    pal_tol: float = 60.0,
    red: tuple  = (220, 45, 45),   # selected parts
    grey: tuple = (65, 65, 65),    # unselected parts
    scale: float = 0.5,
) -> str:
    """
    Return a base64-encoded JPEG of the 5-view grid where selected parts
    are painted red and all other palette pixels are greyed out.
    """
    H, W, _ = overview.shape
    half = H // 2
    top  = overview[:half].copy()
    bot  = overview[half:].copy()

    # nearest-palette classification
    selected_slots = set(pid % len(_PALETTE) for pid in selected_part_ids)
    flat  = bot.reshape(-1, 3).astype(np.float32)
    dists = np.sum((flat[:, None] - _PAL_ARR[None]) ** 2, axis=2)  # N×16
    near  = dists.argmin(axis=1)
    ndist = dists[np.arange(len(flat)), near]

    is_pal = ndist < pal_tol ** 2
    is_sel = np.array([s in selected_slots for s in near], dtype=bool)

    # selected → red, unselected palette pixels → grey
    flat[is_pal &  is_sel] = red
    flat[is_pal & ~is_sel] = grey
    bot_mod = flat.reshape(half, W, 3).astype(np.uint8)

    combined = np.concatenate([top, bot_mod], axis=0)
    img = Image.fromarray(combined)
    if scale != 1.0:
        nw, nh = int(W * scale), int(H * scale)
        img = img.resize((nw, nh), Image.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=82)
    return base64.b64encode(buf.getvalue()).decode()


def _img_b64(path: Path, scale: float = 0.5) -> str:
    """Return base64-encoded JPEG of overview.png (full, unmodified)."""
    img = Image.open(path).convert("RGB")
    if scale != 1.0:
        img = img.resize((int(img.width * scale), int(img.height * scale)),
                         Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=82)
    return base64.b64encode(buf.getvalue()).decode()


# ── palette chip HTML helper ──────────────────────────────────────────────────
def _chips_html(pids: list[int], names: dict[int,str] | None = None) -> str:
    chips = []
    for pid in pids:
        slot = pid % len(_PALETTE)
        r, g, b = _PALETTE[slot]
        lum = 0.299*r + 0.587*g + 0.114*b
        tc  = "#000" if lum > 150 else "#fff"
        label = f"p{pid}"
        if names and pid in names:
            label = f"p{pid} {names[pid]}"
        chips.append(
            f'<span class="chip" style="background:rgb({r},{g},{b});color:{tc};">'
            f'{label}</span>'
        )
    return "".join(chips)


def _parts_legend_html(parts: list[dict]) -> str:
    """Compact colour-coded table of all parts for this object."""
    rows = []
    for p in parts:
        pid  = p["part_id"]
        name = p.get("name", "?")
        slot = pid % len(_PALETTE)
        r, g, b = _PALETTE[slot]
        lum  = 0.299*r + 0.587*g + 0.114*b
        tc   = "#000" if lum > 150 else "#fff"
        rows.append(
            f'<div class="part-row">'
            f'<span class="chip" style="background:rgb({r},{g},{b});color:{tc};">p{pid}</span>'
            f'<span class="part-name">{name}</span>'
            f'</div>'
        )
    return '<div class="parts-legend">' + "".join(rows) + '</div>'


# ── per-object HTML block ─────────────────────────────────────────────────────
def _object_html(
    obj_id: str,
    mode: str,
    overview: np.ndarray,
    overview_b64: str,
    edits: list[dict],
    parts: list[dict],
    scale: float = 0.5,
) -> str:
    ov_w  = int(overview.shape[1] * scale)
    names = {p["part_id"]: p.get("name", "?") for p in parts}

    edits_html_parts = []
    for idx, edit in enumerate(edits):
        etype  = edit.get("edit_type", "?")
        pids   = edit.get("selected_part_ids", [])
        prompt = edit.get("prompt", edit.get("edit_desc", ""))
        badge_color = _TYPE_CSS.get(etype.lower(), _TYPE_DEFAULT_CSS)
        hi_b64 = _make_highlight_png_b64(overview, pids, scale=scale)

        edits_html_parts.append(f"""
      <div class="edit-card">
        <img src="data:image/jpeg;base64,{hi_b64}" width="{ov_w}" alt="edit {idx}">
        <div class="edit-meta">
          <span class="badge" style="background:{badge_color};">{etype.upper()}</span>
          {_chips_html(pids, names)}
          <span class="prompt">{prompt}</span>
        </div>
      </div>""")

    edits_block = "\n".join(edits_html_parts)
    legend = _parts_legend_html(parts)
    return f"""
  <section class="obj-section">
    <div class="obj-header">
      <span class="obj-id">{obj_id}</span>
      <span class="obj-mode">[{mode}]</span>
      <span class="obj-count">{len(edits)} edits · {len(parts)} parts</span>
    </div>
    <div class="overview-row">
      <div class="ov-label">Full overview</div>
      <img src="data:image/jpeg;base64,{overview_b64}" width="{ov_w}" alt="overview">
      {legend}
    </div>
    <div class="edits-grid">
{edits_block}
    </div>
  </section>"""


# ── HTML shell ────────────────────────────────────────────────────────────────
_HTML_HEAD = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Edit Highlight Viewer</title>
<style>
  :root {
    --bg:      #111;
    --surface: #1a1a1a;
    --border:  #2a2a2a;
    --text:    #ddd;
    --sub:     #888;
    --accent:  #4a9eff;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: var(--bg);
    color: var(--text);
    font-family: 'Courier New', monospace;
    padding: 24px;
  }
  h1 { font-size: 1.3rem; color: var(--accent); margin-bottom: 20px; }

  .obj-section {
    border: 1px solid var(--border);
    border-radius: 8px;
    margin-bottom: 40px;
    padding: 16px;
    background: var(--surface);
  }
  .obj-header {
    display: flex; align-items: baseline; gap: 12px;
    margin-bottom: 14px; padding-bottom: 10px;
    border-bottom: 1px solid var(--border);
  }
  .obj-id   { font-size: 1rem; font-weight: bold; color: var(--accent); }
  .obj-mode { font-size: 0.85rem; color: var(--sub); }
  .obj-count{ font-size: 0.85rem; color: #aaa; margin-left: auto; }

  .overview-row {
    display: flex; align-items: flex-start; gap: 10px;
    margin-bottom: 20px;
  }
  .ov-label {
    writing-mode: vertical-rl; text-orientation: mixed;
    font-size: 0.7rem; color: var(--sub); padding-top: 8px;
  }
  .overview-row img { border-radius: 4px; }

  .edits-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(520px, 1fr));
    gap: 14px;
  }
  .edit-card {
    background: #141414;
    border: 1px solid var(--border);
    border-radius: 6px;
    overflow: hidden;
  }
  .edit-card img { display: block; width: 100%; height: auto; }
  .edit-meta {
    display: flex; flex-wrap: wrap; align-items: center;
    gap: 6px;
    padding: 8px 10px;
    background: #181818;
  }
  .badge {
    font-size: 0.7rem; font-weight: bold; letter-spacing: .05em;
    padding: 2px 8px; border-radius: 3px; color: #fff;
    white-space: nowrap;
  }
  .chip {
    font-size: 0.7rem; font-weight: bold;
    padding: 1px 5px; border-radius: 3px;
    white-space: nowrap;
  }
  .prompt {
    font-size: 0.82rem; color: #ccc; flex: 1; min-width: 0;
    word-break: break-word;
  }
  .parts-legend {
    display: flex; flex-direction: column; gap: 3px;
    margin-left: 14px; align-self: flex-start;
    background: #141414; border: 1px solid var(--border);
    border-radius: 5px; padding: 8px 10px;
    min-width: 140px; max-height: 480px; overflow-y: auto;
  }
  .part-row {
    display: flex; align-items: center; gap: 6px;
  }
  .part-name {
    font-size: 0.75rem; color: #bbb; white-space: nowrap;
  }
</style>
</head>
<body>
<h1>Edit Highlight Viewer</h1>
"""
_HTML_TAIL = "</body></html>\n"


# ── process one object ────────────────────────────────────────────────────────
def collect_object(
    parsed_path: Path,
    overview_path: Path,
    scale: float = 0.5,
) -> str | None:
    raw   = json.loads(parsed_path.read_text())
    edits = raw.get("parsed", raw).get("edits", [])
    if not edits:
        print(f"  [skip] no edits in {parsed_path}")
        return None

    overview = np.array(Image.open(overview_path).convert("RGB"))
    ov_b64   = _img_b64(overview_path, scale=scale)
    obj_id   = raw.get("obj_id", parsed_path.parent.parent.name)
    mode     = raw.get("mode", "")

    print(f"  {obj_id}  [{mode}]  {len(edits)} edits …", end=" ", flush=True)
    parts = raw.get("parsed", raw).get("object", {}).get("parts", [])
    block = _object_html(obj_id, mode, overview, ov_b64, edits, parts, scale=scale)
    print("done")
    return block


def _find_overview(parsed_path: Path) -> Path | None:
    p = parsed_path.parent / "overview.png"
    return p if p.exists() else None


# ── CLI ───────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parsed",   type=Path, help="Single parsed.json")
    ap.add_argument("--overview", type=Path, help="Overview PNG (auto if omitted)")
    ap.add_argument("--run-dir",  type=Path,
                    help="Batch: process all parsed.json under this dir")
    ap.add_argument("--output",   type=Path, required=True,
                    help="Output .html path")
    ap.add_argument("--scale",    type=float, default=0.5,
                    help="Image scale factor for embedding (default: 0.5)")
    args = ap.parse_args()

    blocks: list[str] = []

    if args.run_dir:
        run_dir = args.run_dir.resolve()
        pfiles  = sorted(run_dir.rglob("parsed.json"))
        if not pfiles:
            print(f"No parsed.json found under {run_dir}"); sys.exit(1)
        print(f"Processing {len(pfiles)} objects under {run_dir}")
        for pf in pfiles:
            ov = _find_overview(pf)
            if ov is None:
                print(f"  [skip] no overview.png beside {pf}"); continue
            b = collect_object(pf, ov, scale=args.scale)
            if b:
                blocks.append(b)
    elif args.parsed:
        pf = args.parsed.resolve()
        ov = args.overview or _find_overview(pf)
        if ov is None:
            print("Cannot find overview.png; pass --overview."); sys.exit(1)
        b = collect_object(pf, ov, scale=args.scale)
        if b:
            blocks.append(b)
    else:
        ap.print_help(); sys.exit(1)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(_HTML_HEAD + "\n".join(blocks) + _HTML_TAIL, encoding="utf-8")
    print(f"\nSaved → {args.output}  ({len(blocks)} objects)")


if __name__ == "__main__":
    main()
