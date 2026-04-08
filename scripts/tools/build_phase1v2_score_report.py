#!/usr/bin/env python3
"""Build a single-file HTML report visualizing phase1_v2 + phase 1.5 scoring.

Reads:
  <in-dir>/*.scored.json   per-object scoring
  <in-dir>/_hl/*.png       per-edit highlight images

Outputs:
  <in-dir>/report.html     standalone HTML using relative image paths

Usage:
    python scripts/tools/build_phase1v2_score_report.py \
        --in-dir outputs/_debug/phase1_v2_5view_morequota
"""
from __future__ import annotations
import argparse
import base64
import html
import io
import json
import sys
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from render_part_overview import VIEW_INDICES, load_views_from_npz  # noqa: E402
from render_part_highlight import stitch_pair  # noqa: E402

@lru_cache(maxsize=64)
def _orig_views(img_npz_str: str) -> tuple:
    """Cache (per obj) the 5 original photos as a tuple of BGR ndarrays."""
    imgs, _ = load_views_from_npz(Path(img_npz_str), VIEW_INDICES)
    return tuple(imgs)


def _embed_image(img, max_width: int) -> str:
    """Encode an in-memory BGR ndarray as a base64 jpeg data URI."""
    import cv2
    if img is None:
        return ""
    h, w = img.shape[:2]
    if w > max_width:
        scale = max_width / w
        img = cv2.resize(img, (max_width, int(h * scale)),
                         interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ok:
        return ""
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def _embed_png(path: Path, max_width: int) -> str:
    """Return a data: URI for the image, optionally downscaled with cv2."""
    if not path.is_file():
        return ""
    try:
        import cv2
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            return ""
        h, w = img.shape[:2]
        if w > max_width:
            scale = max_width / w
            img = cv2.resize(img, (max_width, int(h * scale)),
                             interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", img,
                               [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            return ""
        b64 = base64.b64encode(buf.tobytes()).decode()
        return f"data:image/jpeg;base64,{b64}"
    except Exception:
        # fallback: raw bytes, no resize
        b64 = base64.b64encode(path.read_bytes()).decode()
        return f"data:image/png;base64,{b64}"


TIER_COLORS = {
    "high":   "#3fb950",
    "medium": "#d29922",
    "low":    "#db6d28",
    "reject": "#f85149",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True, type=Path)
    ap.add_argument("--out", default=None, type=Path,
                    help="default: <in-dir>/report.html")
    ap.add_argument("--embed", action="store_true",
                    help="inline all highlight images as base64 (single-file, "
                         "no relative paths needed)")
    ap.add_argument("--max-width", type=int, default=720,
                    help="downscale embedded images to this max width to "
                         "shrink the HTML file (only with --embed)")
    ap.add_argument("--shard", default="01")
    ap.add_argument("--images-root", default="data/partverse/images", type=Path)
    args = ap.parse_args()

    files = sorted(args.in_dir.glob("*.scored.json"))
    if not files:
        raise SystemExit(f"no *.scored.json in {args.in_dir}")

    out = args.out or (args.in_dir / "report.html")

    # ── aggregate ──
    objs = []
    g_tier = Counter()
    g_by_type: dict[str, Counter] = defaultdict(Counter)
    n_total_edits = 0
    for f in files:
        j = json.loads(f.read_text())
        sc = j.get("scoring") or {}
        edits = (j.get("parsed") or {}).get("edits") or []
        n_total_edits += sum(1 for e in edits if e.get("score"))
        for e in edits:
            s = e.get("score")
            if not s:
                continue
            tier = s.get("tier", "low")
            g_tier[tier] += 1
            g_by_type[e.get("edit_type", "?")][tier] += 1
        objs.append((j["obj_id"], sc, edits))

    # ── HTML ──
    style = """
body { font-family: -apple-system, sans-serif; margin: 0; background: #0d1117; color: #c9d1d9; }
header { padding: 16px 24px; background: #161b22; border-bottom: 1px solid #30363d; position: sticky; top:0; z-index:10; }
h1 { margin: 0 0 6px; font-size: 18px; }
.subtle { color: #8b949e; font-size: 13px; }
.summary { display: flex; gap: 16px; flex-wrap: wrap; margin-top: 10px; }
.pill { padding: 4px 10px; border-radius: 12px; font-size: 12px; font-weight: 600; }
.controls { margin-top: 12px; display: flex; gap: 8px; flex-wrap: wrap; }
.controls button { background: #21262d; color: #c9d1d9; border: 1px solid #30363d; padding: 4px 12px; border-radius: 6px; cursor: pointer; font-size: 12px; }
.controls button.active { background: #1f6feb; border-color: #1f6feb; color: white; }
.byType { margin-top: 10px; font-size: 12px; color: #8b949e; }
.byType code { color: #c9d1d9; background: #21262d; padding: 1px 6px; border-radius: 4px; }
.obj { padding: 24px; border-bottom: 1px solid #30363d; }
.objHeader { display:flex; align-items:baseline; gap: 14px; margin-bottom:14px; flex-wrap: wrap; }
.objHeader h2 { margin: 0; font-size: 15px; font-family: monospace; }
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(440px, 1fr)); gap: 16px; }
.card { background: #161b22; border: 1px solid #30363d; border-radius: 8px; overflow: hidden; }
.card img { width: 100%; display: block; background: white; }
.card .meta { padding: 10px 12px; font-size: 12px; }
.card .row { display:flex; justify-content:space-between; align-items:center; margin-bottom:6px; }
.card .prompt { color: #c9d1d9; font-weight: 500; }
.card .desc { color: #8b949e; font-size: 11px; margin: 2px 0 6px; }
.card .scores { display: flex; gap: 6px; flex-wrap: wrap; font-size: 11px; color: #8b949e; }
.card .scores span { background: #21262d; padding: 1px 6px; border-radius: 4px; }
.card .rationale { color: #8b949e; font-size: 11px; margin-top: 6px; font-style: italic; }
.card .issues { margin-top: 4px; font-size: 11px; color: #f85149; }
.tierBadge { padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 700; color: white; }
.hidden { display: none; }
"""

    # build summary header
    summary_pills = "".join(
        f'<span class="pill" style="background:{TIER_COLORS[t]};color:#0d1117;">'
        f'{t}: {g_tier.get(t,0)}</span>'
        for t in ("high", "medium", "low", "reject")
    )
    by_type_html = " &nbsp;|&nbsp; ".join(
        f"<code>{et}</code>: " + " ".join(
            f'<span style="color:{TIER_COLORS[t]}">{c.get(t,0)}</span>'
            for t in ("high", "medium", "low", "reject")
        )
        for et, c in sorted(g_by_type.items())
    )

    parts = [f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Phase 1.5 Score Report</title>
<style>{style}</style></head><body>
<header>
  <h1>Phase 1.5 — VLM scoring report</h1>
  <div class="subtle">{len(files)} objects · <b>{n_total_edits}</b> edits scored ·
       source: <code>{html.escape(str(args.in_dir))}</code></div>
  <div class="summary">{summary_pills}</div>
  <div class="byType">by edit_type → {by_type_html}</div>
  <div class="controls" id="filters">
    <button class="active" data-tier="all">all</button>
    <button data-tier="high">high</button>
    <button data-tier="medium">medium</button>
    <button data-tier="low">low</button>
    <button data-tier="reject">reject</button>
  </div>
</header>"""]

    for obj_id, sc, edits in objs:
        n_h = sc.get("n_high", 0); n_m = sc.get("n_medium", 0)
        n_l = sc.get("n_low", 0);  n_r = sc.get("n_reject", 0)
        parts.append(f'<section class="obj" id="{obj_id}"><div class="objHeader">')
        parts.append(f'<h2>{obj_id}</h2>')
        for t, n in [("high", n_h), ("medium", n_m), ("low", n_l), ("reject", n_r)]:
            if n:
                parts.append(
                    f'<span class="tierBadge" style="background:{TIER_COLORS[t]};">'
                    f'{t} {n}</span>')
        parts.append('</div><div class="grid">')

        for idx, e in enumerate(edits):
            s = e.get("score")
            if not s:
                continue
            tier = s.get("tier", "low")
            color = TIER_COLORS.get(tier, "#888")
            img_path = args.in_dir / "_hl" / f"{obj_id}__e{idx:02d}.png"
            if args.embed:
                # Stitch original (from images npz) + cached highlight on the fly
                import cv2 as _cv2
                hl = _cv2.imread(str(img_path), _cv2.IMREAD_COLOR) \
                    if img_path.is_file() else None
                vi = int(e.get("view_index", 0))
                try:
                    img_npz = args.images_root / args.shard / f"{obj_id}.npz"
                    origs = _orig_views(str(img_npz))
                    orig = origs[vi] if 0 <= vi < len(origs) else None
                except Exception:
                    orig = None
                if orig is not None and hl is not None:
                    panel = stitch_pair(orig, hl,
                                        edit_type=e.get("edit_type", ""),
                                        prompt=(e.get("prompt") or "")[:140])
                    img_rel = _embed_image(panel, args.max_width)
                else:
                    img_rel = _embed_png(img_path, args.max_width)
            else:
                img_rel = f"_hl/{obj_id}__e{idx:02d}.png"
            params = e.get("edit_params") or {}
            param_str = ", ".join(f"{k}={v}" for k, v in params.items())
            issues = s.get("issues") or []
            issues_html = (
                f'<div class="issues">⚠ {", ".join(html.escape(str(i)) for i in issues)}</div>'
                if issues else "")
            parts.append(f'''
<div class="card" data-tier="{tier}">
  <img src="{img_rel}" loading="lazy">
  <div class="meta">
    <div class="row">
      <span style="font-family:monospace;color:#8b949e;font-size:11px;">
        #{idx} · {html.escape(e.get("edit_type",""))}
        {' · ' + html.escape(param_str) if param_str else ''}
      </span>
      <span class="tierBadge" style="background:{color};">{tier}</span>
    </div>
    <div class="prompt">{html.escape(e.get("prompt",""))}</div>
    <div class="desc">target: {html.escape(e.get("target_part_desc",""))}</div>
    <div class="scores">
      <span>sel <b style="color:#c9d1d9">{s.get("selection_correct","?")}</b></span>
      <span>plaus <b style="color:#c9d1d9">{s.get("edit_plausibility","?")}</b></span>
      <span>clar <b style="color:#c9d1d9">{s.get("prompt_clarity","?")}</b></span>
      <span>overall <b style="color:#c9d1d9">{s.get("overall","?")}</b></span>
    </div>
    <div class="rationale">{html.escape(s.get("rationale",""))}</div>
    {issues_html}
  </div>
</div>''')
        parts.append('</div></section>')

    parts.append("""
<script>
const buttons = document.querySelectorAll('#filters button');
buttons.forEach(b => b.addEventListener('click', () => {
  buttons.forEach(x => x.classList.remove('active'));
  b.classList.add('active');
  const tier = b.dataset.tier;
  document.querySelectorAll('.card').forEach(c => {
    c.classList.toggle('hidden', tier !== 'all' && c.dataset.tier !== tier);
  });
  // hide obj sections that have no visible cards
  document.querySelectorAll('.obj').forEach(o => {
    const visible = o.querySelectorAll('.card:not(.hidden)').length;
    o.style.display = visible ? '' : 'none';
  });
}));
</script>
</body></html>""")

    out.write_text("".join(parts), encoding="utf-8")
    print(f"[OK] {out}  ({n_total_edits} edits, {len(files)} objects)")


if __name__ == "__main__":
    main()
