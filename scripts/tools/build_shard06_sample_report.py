#!/usr/bin/env python3
"""
Shard06 sample QC report — reads edit_status.json (new per-edit tracking).

For N sampled objects shows:
  - Object header: overview image + stage completion badges
  - Per edit: type badge · Gate A/E · prompt/reason · BEFORE strip · AFTER 5-view strip

Usage:
    python scripts/tools/build_shard06_sample_report.py \
        --run-dir .../pipeline_v2_shard06 \
        --images-root .../inputs/images \
        --out report.html
"""
from __future__ import annotations
import argparse, base64, collections, html as _html, io, json, random, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# ── image helpers ─────────────────────────────────────────────────────────────

def _cv2_strip(paths, thumb_h=220):
    import cv2, numpy as np
    imgs = []
    for p in paths:
        if not Path(p).is_file():
            continue
        img = cv2.imread(str(p))
        if img is None:
            continue
        h, w = img.shape[:2]
        nw = max(1, int(w * thumb_h / h))
        imgs.append(cv2.resize(img, (nw, thumb_h)))
    if not imgs:
        return None
    strip = np.hstack(imgs)
    ok, buf = cv2.imencode(".jpg", strip, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not ok:
        return None
    data = base64.b64encode(buf).decode()
    return f'<img src="data:image/jpeg;base64,{data}" style="max-width:100%;border-radius:4px">'


def _npz_strip(npz_path, view_indices=(0,1,2,3,4), thumb_h=220):
    if not npz_path or not Path(npz_path).is_file():
        return None
    try:
        from partcraft.render.overview import load_views_from_npz
        import cv2, numpy as np
        imgs, _ = load_views_from_npz(Path(npz_path), list(view_indices))
        resized = []
        for img in imgs:
            h, w = img.shape[:2]
            nw = max(1, int(w * thumb_h / h))
            resized.append(cv2.resize(img, (nw, thumb_h)))
        strip = np.hstack(resized)
        ok, buf = cv2.imencode(".jpg", strip, [cv2.IMWRITE_JPEG_QUALITY, 88])
        if not ok:
            return None
        data = base64.b64encode(buf).decode()
        return f'<img src="data:image/jpeg;base64,{data}" style="max-width:100%;border-radius:4px">'
    except Exception as e:
        return f'<span class="miss">before strip unavailable: {_html.escape(str(e))}</span>'


def _b64_img(path, max_w=1100):
    if not path or not Path(path).is_file():
        return None
    try:
        from PIL import Image
        img = Image.open(path).convert("RGB")
        if img.width > max_w:
            h = int(img.height * max_w / img.width)
            img = img.resize((max_w, h), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=88)
        data = base64.b64encode(buf.getvalue()).decode()
        return f'<img src="data:image/jpeg;base64,{data}" style="max-width:100%;border-radius:4px">'
    except Exception as e:
        return f'<span class="miss">img error: {_html.escape(str(e))}</span>'


# ── constants ─────────────────────────────────────────────────────────────────

TYPE_COLORS = {
    "deletion": "#f85149", "modification": "#1f6feb",
    "material": "#d29922", "scale": "#a371f7",
    "global": "#3fb950",   "addition": "#39d353",
}
STAGE_LABELS = [
    ("gate_a","Gate-A"), ("s4","S4-2D"), ("s5b","S5b-Del"),
    ("s6p","S6p-Preview"), ("gate_e","Gate-E"),
]
ORDER = ["del","mod","scl","mat","glb","add"]

# ── data helpers ──────────────────────────────────────────────────────────────

def _gate_badge(gate_dict, label):
    if not gate_dict:
        return f'<span class="badge bn">Gate {label} —</span>'
    vlm = gate_dict.get("vlm") or {}
    rule = gate_dict.get("rule") or {}
    passed = vlm.get("pass", rule.get("pass", False))
    score = vlm.get("score","")
    reason = (vlm.get("reason") or rule.get("reason") or "")[:300]
    score_str = f" {score:.0%}" if isinstance(score,float) else ""
    icon = "✅" if passed else "❌"
    cls = "bok" if passed else "bfl"
    reason_html = (f'<div class="reason">{_html.escape(reason)}</div>' if reason else "")
    return (f'<span class="badge {cls}" title="{_html.escape(reason)}">Gate {label}: {icon}{score_str}</span>'
            + reason_html)


def _stage_dot(stages, key, label):
    status = stages.get(key, {}).get("status","—")
    cls = {"pass":"bok","done":"bok","fail":"bfl","skip":"bsk"}.get(status,"bn")
    return f'<span class="badge {cls}">{label}: {status}</span>'


def _load_prompt(obj_dir, edit_id, edit_type):
    meta_f = obj_dir / "edits_3d" / edit_id / "meta.json"
    if meta_f.is_file():
        try:
            m = json.loads(meta_f.read_text())
            if m.get("prompt"):
                return m["prompt"], m.get("target_part_desc","")
        except Exception:
            pass
    parsed_f = obj_dir / "phase1" / "parsed.json"
    if not parsed_f.is_file():
        return "", ""
    try:
        data = json.loads(parsed_f.read_text())
        edits = (data.get("parsed") or {}).get("edits") or []
    except Exception:
        return "", ""
    PREFIX_TO_TYPE = {"del":"deletion","mod":"modification","scl":"scale",
                      "mat":"material","glb":"global","add":"addition"}
    FLUX_TYPES = frozenset({"modification","scale","material","global","addition"})
    parts = edit_id.split("_")
    prefix = parts[0]
    try:
        seq = int(parts[-1])
    except Exception:
        return "", ""
    target_type = PREFIX_TO_TYPE.get(prefix,"")
    is_flux = target_type in FLUX_TYPES
    flux_seq = del_seq = 0
    for e in edits:
        et = e.get("edit_type","")
        if et in FLUX_TYPES:
            if is_flux and et == target_type and flux_seq == seq:
                return e.get("prompt",""), e.get("target_part_desc","")
            if is_flux:
                flux_seq += 1
        elif et == "deletion":
            if not is_flux and target_type=="deletion" and del_seq == seq:
                return e.get("prompt",""), e.get("target_part_desc","")
            if not is_flux:
                del_seq += 1
    return "", ""


# ── render ────────────────────────────────────────────────────────────────────

def render_edit_card(edit_id, edata, obj_dir, before_strip_html):
    etype = edata.get("edit_type", edit_id.split("_")[0])
    color = TYPE_COLORS.get(etype,"#888")
    stages = edata.get("stages",{})
    gates  = edata.get("gates",{})
    final  = edata.get("final_pass")
    fail_gate = edata.get("fail_gate","")
    fail_reason = edata.get("fail_reason","")
    prompt, part_desc = _load_prompt(obj_dir, edit_id, etype)
    stage_dots = " ".join(_stage_dot(stages,k,lbl) for k,lbl in STAGE_LABELS)
    ga_html = _gate_badge(gates.get("A"),"A")
    ge_html = _gate_badge(gates.get("E"),"E")
    final_cls = ("bok" if final else "bfl") if final is not None else "bn"
    final_icon= ("✅" if final else "❌") if final is not None else "—"
    final_html = f'<span class="badge {final_cls}">Final: {final_icon}</span>'
    if fail_gate and fail_reason:
        final_html += f'<div class="reason" style="color:#f85149">Failed Gate {_html.escape(fail_gate)}: {_html.escape(fail_reason[:200])}</div>'
    preview_paths = [obj_dir/"edits_3d"/edit_id/f"preview_{i}.png" for i in range(5)]
    preview_html = _cv2_strip(preview_paths) or '<span class="miss">previews missing</span>'
    npz_html = before_strip_html or '<span class="miss">before images N/A</span>'

    # For addition edits: preview_*.png = before-addition state (object without part,
    # copied from source_del); images.npz = after-addition target (original with part).
    # Swap labels so the report shows the correct before→after direction (mirrors sq3 swap).
    if etype == "addition":
        before_html = preview_html
        after_html  = npz_html
        before_label = "BEFORE — object without part (addition source)"
        after_label  = "AFTER — original with part (addition target, from images.npz)"
    else:
        before_html  = npz_html
        after_html   = preview_html
        before_label = "BEFORE (5 views from images.npz)"
        after_label  = "AFTER (5 preview views)"

    prompt_html = (f'<div class="prompt">"{_html.escape(prompt)}"</div>'
                   if prompt else '<div class="miss">(no prompt)</div>')
    part_html = (f'<div class="part-desc">Part: {_html.escape(part_desc)}</div>'
                 if part_desc else "")
    return f"""
<div class="edit-card">
  <div class="edit-hd" style="border-left:5px solid {color}">
    <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px">
      <span class="etype" style="color:{color}">[{etype.upper()}]</span>
      <span class="eid">{edit_id[-18:]}</span>
    </div>
    {prompt_html}{part_html}
    <div class="stages-row">{stage_dots}</div>
    <div class="gate-row">{ga_html}{ge_html}{final_html}</div>
  </div>
  <div class="edit-views">
    <div class="view-label">{before_label}</div>{before_html}
    <div class="view-label" style="margin-top:8px">{after_label}</div>{after_html}
  </div>
</div>"""


def render_object(obj_dir, images_root):
    obj_id = obj_dir.name
    es_file = obj_dir / "edit_status.json"
    if not es_file.exists():
        return f'<div class="obj-card"><div class="obj-hd"><span class="oid">{obj_id}</span> — no edit_status.json</div></div>\n'
    try:
        es = json.loads(es_file.read_text())
    except Exception as e:
        return f'<div class="obj-card"><div class="obj-hd">ERR: {_html.escape(str(e))}</div></div>\n'
    edits   = es.get("edits",{})
    updated = es.get("updated","")
    shard   = es.get("shard","06")
    total   = len(edits)
    stage_agg = collections.Counter()
    n_pass = n_fail = 0
    for edata in edits.values():
        if not isinstance(edata,dict): continue
        for k,_ in STAGE_LABELS:
            if edata.get("stages",{}).get(k,{}).get("status","") in ("pass","done"):
                stage_agg[k] += 1
        fp = edata.get("final_pass")
        if fp is True:  n_pass += 1
        elif fp is False: n_fail += 1
    agg_badges = " ".join(
        f'<span class="badge {"bok" if stage_agg[k]==total else "bsk" if stage_agg[k]>0 else "bn"}">{lbl}: {stage_agg[k]}/{total}</span>'
        for k,lbl in STAGE_LABELS)
    pass_rate = f"{n_pass}/{n_pass+n_fail}" if (n_pass+n_fail) else "—"
    overview_html = _b64_img(obj_dir/"phase1"/"overview.png") or "<em class='miss'>overview.png missing</em>"
    npz_path = images_root / shard / f"{obj_id}.npz"
    try:
        from partcraft.pipeline_v2.specs import VIEW_INDICES
        view_idxs = VIEW_INDICES
    except Exception:
        view_idxs = list(range(5))
    before_html = _npz_strip(npz_path, view_idxs)
    sorted_eids = sorted(edits.keys(),
        key=lambda eid: (ORDER.index(eid.split("_")[0]) if eid.split("_")[0] in ORDER else 99, eid))
    cards_html = "".join(
        render_edit_card(eid, edits[eid], obj_dir, before_html)
        for eid in sorted_eids if isinstance(edits[eid],dict))
    return f"""
<div class="obj-card">
  <div class="obj-hd">
    <span class="oid">{obj_id}</span>
    <div class="sbadges">{agg_badges}</div>
    <span class="pr">Final pass: {pass_rate} &nbsp;<small style="color:#8b949e;font-weight:normal">updated {updated}</small></span>
  </div>
  <div class="ov">{overview_html}</div>
  <div class="edits">{cards_html}</div>
</div>"""


def build_completion_summary(base_dir):
    total = done_s6p = done_gate_e = total_pass = total_fail = 0
    for obj_dir in base_dir.iterdir():
        es_file = obj_dir / "edit_status.json"
        if not es_file.exists(): continue
        try:
            es = json.loads(es_file.read_text())
        except Exception:
            continue
        edits = es.get("edits",{})
        if not edits: continue
        total += 1
        n_s6p = sum(1 for e in edits.values() if isinstance(e,dict)
                    and e.get("stages",{}).get("s6p",{}).get("status")=="done")
        n_ge  = sum(1 for e in edits.values() if isinstance(e,dict)
                    and e.get("stages",{}).get("gate_e",{}).get("status") in ("pass","fail"))
        if n_s6p > 0: done_s6p  += 1
        if n_ge  > 0: done_gate_e += 1
        for e in edits.values():
            if not isinstance(e,dict): continue
            fp = e.get("final_pass")
            if fp is True:  total_pass += 1
            elif fp is False: total_fail += 1
    return {"total":total,"done_s6p":done_s6p,"done_gate_e":done_gate_e,
            "total_pass":total_pass,"total_fail":total_fail}


CSS = """
* { box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
       background: #0d1117; color: #c9d1d9; margin: 0; padding: 20px; font-size: 13px; }
h1   { font-size: 20px; margin: 0 0 4px; color: #e6edf3; }
.subtitle { color: #8b949e; font-size: 12px; margin: 0 0 16px; }
.summary { background: #161b22; border: 1px solid #30363d; border-radius: 8px;
           padding: 14px 20px; margin-bottom: 24px; }
.summary b { color: #e6edf3; }
.summary-grid { display: flex; gap: 28px; flex-wrap: wrap; margin-top: 12px; }
.stat .num { font-size: 26px; font-weight: 700; color: #e6edf3; }
.stat .lbl { font-size: 11px; color: #8b949e; margin-top: 2px; }
.obj-card { background: #161b22; border: 1px solid #30363d; border-radius: 8px;
            margin-bottom: 32px; overflow: hidden; }
.obj-hd   { background: #1c2128; padding: 10px 16px; display: flex; align-items: center;
            gap: 12px; flex-wrap: wrap; border-bottom: 1px solid #30363d; }
.oid      { font-family: monospace; font-size: 12px; color: #58a6ff; font-weight: 600; }
.sbadges  { display: flex; gap: 4px; flex-wrap: wrap; }
.pr       { margin-left: auto; font-weight: 700; color: #3fb950; font-size: 13px; }
.ov       { padding: 12px; background: #0d1117; border-bottom: 1px solid #30363d; }
.edits    { padding: 12px 16px; display: flex; flex-direction: column; gap: 14px; }
.edit-card { border: 1px solid #30363d; border-radius: 6px; overflow: hidden; }
.edit-hd   { padding: 10px 14px; background: #1c2128; border-bottom: 1px solid #30363d; }
.etype     { font-weight: 700; font-size: 13px; text-transform: uppercase; }
.eid       { font-family: monospace; font-size: 10px; color: #8b949e; }
.prompt    { margin: 6px 0 2px; font-style: italic; color: #e6edf3; line-height: 1.5; }
.part-desc { font-size: 11px; color: #8b949e; margin-bottom: 6px; }
.stages-row{ display: flex; gap: 4px; flex-wrap: wrap; margin: 6px 0; }
.gate-row  { display: flex; gap: 6px; flex-wrap: wrap; align-items: flex-start; margin-top: 6px; }
.reason    { font-size: 11px; color: #8b949e; margin-top: 3px; line-height: 1.4; max-width: 900px; }
.edit-views{ padding: 10px 14px; background: #0d1117; }
.view-label{ font-size: 10px; font-weight: 700; color: #8b949e; letter-spacing:.05em; margin-bottom:4px; }
.badge { display:inline-block; padding:2px 8px; border-radius:10px; font-size:11px; white-space:nowrap; }
.bok { background:#1f4a2e; color:#3fb950; border:1px solid #2ea043; }
.bfl { background:#4a1f1f; color:#f85149; border:1px solid #b62324; }
.bsk { background:#3d3000; color:#d29922; border:1px solid #9e6a03; }
.bn  { background:#21262d; color:#8b949e; border:1px solid #30363d; }
.miss{ color:#6e7681; font-size:11px; }
"""


def main():
    ap = argparse.ArgumentParser(description="Shard06 sample QC report from edit_status.json")
    ap.add_argument("--run-dir",     required=True, type=Path)
    ap.add_argument("--images-root", required=True, type=Path)
    ap.add_argument("--out",         default=None,  type=Path)
    ap.add_argument("--obj-ids",     nargs="*",     default=None)
    ap.add_argument("--n",           type=int,      default=10)
    ap.add_argument("--seed",        type=int,      default=2026)
    args = ap.parse_args()

    obj_root = args.run_dir / "objects"
    shard_dirs = [d for d in obj_root.iterdir() if d.is_dir()] if obj_root.is_dir() else []
    if not shard_dirs:
        sys.exit(f"No shard dirs under {obj_root}")
    base = shard_dirs[0]
    out = args.out or (args.run_dir / f"shard06_sample{args.n}.html")
    out.parent.mkdir(parents=True, exist_ok=True)

    print("Scanning completion status…", file=sys.stderr)
    summary = build_completion_summary(base)

    if args.obj_ids:
        obj_dirs = [base/oid for oid in args.obj_ids if (base/oid).is_dir()]
    else:
        by_type: dict[str,list[str]] = {}
        for obj_dir in base.iterdir():
            es_file = obj_dir / "edit_status.json"
            if not es_file.exists(): continue
            try:
                es = json.loads(es_file.read_text())
            except Exception:
                continue
            edits = es.get("edits",{})
            n_s6p = sum(1 for e in edits.values() if isinstance(e,dict)
                        and e.get("stages",{}).get("s6p",{}).get("status")=="done")
            if n_s6p == 0: continue
            for edata in edits.values():
                if isinstance(edata,dict):
                    t = edata.get("edit_type","")
                    if t:
                        by_type.setdefault(t,[])
                        if obj_dir.name not in by_type[t]:
                            by_type[t].append(obj_dir.name)
        rng = random.Random(args.seed)
        selected: set[str] = set()
        for t in ["deletion","modification","material","scale","addition","global"]:
            pool = [oid for oid in by_type.get(t,[]) if oid not in selected]
            rng.shuffle(pool)
            selected.update(pool[:max(1, args.n//6)])
            if len(selected) >= args.n: break
        if len(selected) < args.n:
            all_ids = [d.name for d in base.iterdir()
                       if (d/"edit_status.json").exists() and d.name not in selected]
            rng.shuffle(all_ids)
            selected.update(all_ids[:args.n-len(selected)])
        obj_dirs = [base/oid for oid in list(selected)[:args.n]]

    print(f"Building report for {len(obj_dirs)} objects…", file=sys.stderr)
    cards_html = []
    for od in obj_dirs:
        print(f"  {od.name}", file=sys.stderr)
        cards_html.append(render_object(od, args.images_root))

    total = summary["total"]; s6p = summary["done_s6p"]
    ge = summary["done_gate_e"]; tp = summary["total_pass"]; tf = summary["total_fail"]
    rate_str = f"{tp}/{tp+tf} ({tp/(tp+tf):.1%})" if (tp+tf) else "N/A"

    summary_html = f"""
<div class="summary">
  <b>Shard 06 — overall completion (as of report generation)</b>
  <div class="summary-grid">
    <div class="stat"><div class="num">{total}</div><div class="lbl">objects with edit_status</div></div>
    <div class="stat"><div class="num">{s6p}</div><div class="lbl">s6p preview done</div></div>
    <div class="stat"><div class="num">{ge}</div><div class="lbl">Gate-E evaluated</div></div>
    <div class="stat"><div class="num">{tp}</div><div class="lbl">edits passed (final)</div></div>
    <div class="stat"><div class="num">{tf}</div><div class="lbl">edits failed (final)</div></div>
    <div class="stat"><div class="num" style="color:#3fb950">{rate_str}</div><div class="lbl">overall pass rate</div></div>
  </div>
</div>"""

    html_out = (
        '<!DOCTYPE html><html><head><meta charset="UTF-8">'
        f'<title>Shard06 QC Sample — {len(obj_dirs)} objects</title>'
        f'<style>{CSS}</style></head><body>'
        f'<h1>Shard06 QC Sample Report — {len(obj_dirs)} objects</h1>'
        '<div class="subtitle">Data source: edit_status.json (per-edit stage tracking)</div>'
        + summary_html
        + "".join(cards_html)
        + "</body></html>"
    )
    out.write_text(html_out, encoding="utf-8")
    print(f"\n✓ {out}  ({out.stat().st_size//1024} KB)", file=sys.stderr)


if __name__ == "__main__":
    main()
