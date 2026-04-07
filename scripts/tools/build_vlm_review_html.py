#!/usr/bin/env python3
"""Build a static HTML page to review VLM cleaning results.

Reads vlm_scores*.jsonl and renders sampled examples per quality tier with
their before/after comparison PNG (relative link into _vlm_render_cache/).

Usage:
    python scripts/tools/build_vlm_review_html.py \\
        --scores outputs/partverse/partverse_pairs/vlm_scores_shard01_gpu*.jsonl \\
        --cache  outputs/partverse/partverse_pairs/_vlm_render_cache \\
        --out    outputs/partverse/vlm_review_shard01.html \\
        --per-tier 40
"""
from __future__ import annotations
import argparse, base64, glob, html, json, os, random
from pathlib import Path

TIERS = ["high", "medium", "low", "negative", "rejected"]
TIER_COLOR = {"high":"#1b8a3a","medium":"#2e7dd7","low":"#d18b16",
              "negative":"#c0392b","rejected":"#7f8c8d"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", nargs="+", required=True,
                    help="One or more jsonl files / globs")
    ap.add_argument("--cache", required=True, type=Path,
                    help="_vlm_render_cache dir containing <edit_id>.png")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--per-tier", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--edit-type", default=None,
                    help="Optional filter, e.g. deletion")
    ap.add_argument("--embed", action="store_true",
                    help="Embed images as base64 (single self-contained HTML)")
    ap.add_argument("--max-img-width", type=int, default=0,
                    help="If --embed, downscale images to this width (px) "
                         "to keep HTML small. 0 = keep original.")
    args = ap.parse_args()

    files: list[str] = []
    for pat in args.scores:
        files.extend(sorted(glob.glob(pat)) or [pat])

    by_tier: dict[str, list[dict]] = {t: [] for t in TIERS}
    n_total = 0
    for f in files:
        if not os.path.isfile(f):
            continue
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if args.edit_type and rec.get("edit_type") != args.edit_type:
                    continue
                t = rec.get("quality_tier", "rejected")
                if t in by_tier:
                    by_tier[t].append(rec)
                n_total += 1

    rng = random.Random(args.seed)
    sampled: dict[str, list[dict]] = {}
    for t in TIERS:
        lst = by_tier[t]
        rng.shuffle(lst)
        sampled[t] = lst[: args.per_tier]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cache_rel = os.path.relpath(args.cache, args.out.parent)

    parts: list[str] = []
    parts.append(f"""<!doctype html><html><head><meta charset="utf-8">
<title>VLM Review</title>
<style>
body{{font-family:-apple-system,Segoe UI,sans-serif;margin:18px;background:#f5f5f5;color:#222}}
h1{{margin:0 0 8px}}
.summary{{margin-bottom:18px;font-size:14px}}
.summary span{{display:inline-block;margin-right:14px;padding:2px 8px;border-radius:4px;color:#fff}}
.tier-section{{margin-top:28px}}
.tier-title{{font-size:20px;padding:6px 12px;color:#fff;border-radius:4px;display:inline-block}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(380px,1fr));gap:14px;margin-top:12px}}
.card{{background:#fff;border:1px solid #ddd;border-radius:6px;padding:10px;font-size:12px}}
.card img{{width:100%;height:auto;border:1px solid #eee;cursor:zoom-in}}
.card .id{{font-family:monospace;font-size:11px;color:#555;word-break:break-all}}
.card .meta{{margin:4px 0;color:#444}}
.card .reason{{color:#333;margin-top:4px}}
.kv{{color:#888}}.kv b{{color:#222}}
</style></head><body>
<h1>VLM cleaning review</h1>
<div class="summary">Total records: {n_total}. """)
    for t in TIERS:
        parts.append(f'<span style="background:{TIER_COLOR[t]}">{t}: {len(by_tier[t])}</span>')
    parts.append("</div>")

    for t in TIERS:
        recs = sampled[t]
        if not recs:
            continue
        parts.append(f'<div class="tier-section"><div class="tier-title" '
                     f'style="background:{TIER_COLOR[t]}">{t} '
                     f'(showing {len(recs)} / {len(by_tier[t])})</div><div class="grid">')
        for r in recs:
            eid = r.get("edit_id", "")
            disk_path = args.cache / f"{eid}.png"
            if not disk_path.is_file():
                img_tag = '<div style="color:#c00">[image missing]</div>'
            elif args.embed:
                if args.max_img_width > 0:
                    try:
                        from PIL import Image
                        import io as _io
                        im = Image.open(disk_path)
                        if im.width > args.max_img_width:
                            ratio = args.max_img_width / im.width
                            im = im.resize(
                                (args.max_img_width, int(im.height * ratio)),
                                Image.LANCZOS,
                            )
                        buf = _io.BytesIO()
                        im.convert("RGB").save(buf, format="JPEG", quality=82)
                        b64 = base64.b64encode(buf.getvalue()).decode()
                        mime = "jpeg"
                    except Exception:
                        b64 = base64.b64encode(disk_path.read_bytes()).decode()
                        mime = "png"
                else:
                    b64 = base64.b64encode(disk_path.read_bytes()).decode()
                    mime = "png"
                img_tag = (f'<img src="data:image/{mime};base64,{b64}" '
                           f'loading="lazy" onclick="window.open(this.src)">')
            else:
                img_path = f"{cache_rel}/{eid}.png"
                img_tag = (f'<img src="{html.escape(img_path)}" loading="lazy" '
                           f'onclick="window.open(this.src)">')
            parts.append(f"""<div class="card">
{img_tag}
<div class="id">{html.escape(eid)}</div>
<div class="meta">
<span class="kv"><b>type</b> {html.escape(r.get('edit_type',''))}</span> ·
<span class="kv"><b>score</b> {r.get('score','')}</span> ·
<span class="kv"><b>vq</b> {r.get('visual_quality','')}</span> ·
<span class="kv"><b>exec</b> {r.get('edit_executed','')}</span> ·
<span class="kv"><b>region</b> {r.get('correct_region','')}</span> ·
<span class="kv"><b>preserve</b> {r.get('preserve_other','')}</span> ·
<span class="kv"><b>artifact_free</b> {r.get('artifact_free','')}</span>
</div>
<div class="reason"><b>reason:</b> {html.escape(str(r.get('reason','')))}</div>
<div class="reason"><b>improved_prompt:</b> {html.escape(str(r.get('improved_prompt','')))}</div>
</div>""")
        parts.append("</div></div>")

    parts.append("</body></html>")
    args.out.write_text("\n".join(parts), encoding="utf-8")
    print(f"[OK] wrote {args.out}  ({n_total} records, {sum(len(v) for v in sampled.values())} sampled)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
