#!/usr/bin/env python3
"""Generate a self-contained HTML QC report for pipeline_v2 shard objects.

Usage:
    python scripts/vis/generate_qc_report.py \
        --run-dir /path/to/pipeline_v2_shard02 \
        --out report.html \
        [--obj-ids id1 id2 ...]   # defaults to all with s6p done

The HTML is fully self-contained (images embedded as base64 JPEG).
"""
from __future__ import annotations
import argparse, base64, json, sys, io
from pathlib import Path

TYPE_COLORS = {
    "deletion":     "#b02020",
    "modification": "#1a5c8a",
    "material":     "#6a2080",
    "scale":        "#7a5a10",
    "global":       "#1a6a3a",
    "addition":     "#10706a",
}
GATE_OK = "✅"; GATE_FAIL = "❌"; GATE_NONE = "—"

def b64_img(path, max_px=200):
    if not path or not Path(path).is_file(): return None
    try:
        from PIL import Image
        img = Image.open(path).convert("RGB")
        w, h = img.size
        if w > max_px:
            img = img.resize((max_px, int(h*max_px/w)), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=80)
        data = base64.b64encode(buf.getvalue()).decode()
        return f'<img src="data:image/jpeg;base64,{data}" style="max-width:{max_px}px;border-radius:3px">'
    except Exception as e:
        return f'<span style="color:red;font-size:10px">ERR:{e}</span>'

def gate_badge(gate, label):
    if not gate:
        return f'<span class="badge bn">Gate {label}: {GATE_NONE}</span>'
    vlm = gate.get("vlm") or {}; rule = gate.get("rule") or {}
    ok = vlm.get("pass", rule.get("pass", False))
    score = vlm.get("score", "")
    score_s = f" {score:.0%}" if isinstance(score, float) else ""
    reason = (vlm.get("reason") or rule.get("reason") or "")[:140]
    tip = f'title="{reason}"' if reason else ""
    cls = "bok" if ok else "bfl"
    return f'<span class="badge {cls}" {tip}>Gate {label}: {GATE_OK if ok else GATE_FAIL}{score_s}</span>'

def step_badge(d, label):
    st = d.get("status","—")
    cls = {"ok":"bok","fail":"bfl","skip":"bsk"}.get(st,"bn")
    n = d.get("n_ok", d.get("n_edits", d.get("n","")))
    ns = f"({n})" if n != "" else ""
    return f'<span class="badge {cls}">{label}:{st}{ns}</span>'

def get_prompt(edit_dir, parsed_edits, qc_edits):
    name = edit_dir.name
    parts = name.split("_")
    if len(parts) >= 3:
        try:
            seq = int(parts[-1])
            e = parsed_edits.get(f"edit_{seq:03d}", {})
            if e.get("prompt"): return e["prompt"]
        except ValueError:
            pass
    return qc_edits.get(name, {}).get("prompt", "")

def load_parsed(parsed_path):
    if not parsed_path.is_file(): return {}
    try:
        data = json.loads(parsed_path.read_text())
        edits = (data.get("parsed") or {}).get("edits") or []
        return {f"edit_{i:03d}": e for i, e in enumerate(edits)}
    except Exception:
        return {}

def render_object(obj_dir, status, qc):
    obj_id = obj_dir.name
    steps = status.get("steps", {})
    badges = " ".join(step_badge(steps.get(k,{}), k)
                      for k in ["s1_phase1","sq1_qc_A","s4_flux_2d","s5_trellis",
                                 "s5b_del_mesh","s6p_preview","sq3_qc_E","s6_render_3d"])
    ov = b64_img(obj_dir/"phase1"/"overview.png", 900) or "<em>overview missing</em>"
    qc_edits = qc.get("edits", {})
    parsed_edits = load_parsed(obj_dir/"phase1"/"parsed.json")

    edits_3d = obj_dir / "edits_3d"
    rows = ""; n_pass = n_fail = 0

    if edits_3d.is_dir():
        for ed in sorted(edits_3d.iterdir()):
            if not ed.is_dir(): continue
            eid = ed.name
            etype = eid.split("_")[0]
            color = TYPE_COLORS.get(etype, "#555")
            qe = qc_edits.get(eid, {})
            gates = qe.get("gates", {})
            ga = gates.get("A"); ge = gates.get("E")
            final = qe.get("final_pass")
            prompt = get_prompt(ed, parsed_edits, qc_edits)
            prompt_html = f'<div class="prompt">"{prompt}"</div>' if prompt else ""

            prev_html = "".join(
                b64_img(ed/f"preview_{i}.png", 160) or '<span class="miss">—</span>'
                for i in range(5))

            bf = b64_img(ed/"before.png", 160); af = b64_img(ed/"after.png", 160)
            render_html = ""
            if bf or af:
                render_html = f'<div class="renders"><b>3D render:</b> {bf or "—"} → {af or "—"}</div>'

            if ge:
                if (ge.get("vlm") or {}).get("pass"): n_pass += 1
                else: n_fail += 1

            fcls = ("bok" if final else "bfl") if final is not None else "bn"
            ficon = (GATE_OK if final else GATE_FAIL) if final is not None else GATE_NONE

            rows += f"""<tr>
              <td class="edit-meta" style="border-left:4px solid {color}">
                <b style="color:{color}">[{etype.upper()}]</b>
                <span class="eid">{eid[-8:]}</span>
                {prompt_html}
                <div style="margin-top:6px">
                  {gate_badge(ga,"A")} {gate_badge(ge,"E")}
                  <span class="badge {fcls}">Final:{ficon}</span>
                </div>
              </td>
              <td class="edit-previews">
                <div class="prev-row">{prev_html}</div>
                {render_html}
              </td>
            </tr><tr><td colspan="2" class="sep"></td></tr>"""

    rate = f"{n_pass}/{n_pass+n_fail}" if (n_pass+n_fail) else "—"
    return f"""<div class="card">
      <div class="card-hd">
        <span class="oid">{obj_id}</span>
        <div class="sbadges">{badges}</div>
        <span class="pr">Gate-E: {rate}</span>
      </div>
      <div class="ov">{ov}</div>
      <table class="et"><thead><tr>
        <th style="width:230px">Edit / Gates</th>
        <th>Previews (view 0→4)</th>
      </tr></thead><tbody>{rows}</tbody></table>
    </div>"""

CSS = """
*{box-sizing:border-box}
body{font-family:-apple-system,sans-serif;background:#f0f0f0;margin:0;padding:20px;font-size:13px}
h1{margin:0 0 16px;font-size:22px;color:#222}
.summary{background:#fff;border-radius:8px;padding:14px 18px;margin-bottom:20px;
  box-shadow:0 1px 3px rgba(0,0,0,.12)}
.card{background:#fff;border-radius:8px;margin-bottom:24px;
  box-shadow:0 1px 4px rgba(0,0,0,.12);overflow:hidden}
.card-hd{background:#1e1e2e;color:#fff;padding:10px 14px;
  display:flex;align-items:center;gap:12px;flex-wrap:wrap}
.oid{font-family:monospace;font-size:12px;color:#adf}
.sbadges{display:flex;gap:4px;flex-wrap:wrap}
.pr{margin-left:auto;font-weight:700;color:#afd;font-size:13px}
.ov{padding:10px;background:#fafafa;border-bottom:1px solid #e8e8e8}
.et{width:100%;border-collapse:collapse}
.et th{background:#eee;padding:6px 10px;text-align:left;font-size:11px;border-bottom:2px solid #ccc}
.et td{padding:8px 10px;border-bottom:1px solid #eee;vertical-align:top}
.edit-meta{min-width:200px;font-size:12px}
.edit-previews{vertical-align:top}
.eid{font-family:monospace;font-size:10px;color:#888;margin-left:4px}
.prompt{font-style:italic;color:#555;margin-top:4px;line-height:1.4}
.prev-row{display:flex;gap:3px;flex-wrap:wrap}
.renders{margin-top:6px;display:flex;align-items:center;gap:6px;
  background:#fff8e0;padding:4px 6px;border-radius:4px}
.sep{height:5px;background:#f5f5f5;padding:0}
.badge{display:inline-block;padding:2px 7px;border-radius:10px;font-size:10px;
  margin:2px;cursor:default;white-space:nowrap}
.bok{background:#d4edda;color:#155724}
.bfl{background:#f8d7da;color:#721c24}
.bsk{background:#fff3cd;color:#856404}
.bn{background:#e2e3e5;color:#495057}
.miss{color:#ccc;font-size:11px}
"""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--out", default="report.html", type=Path)
    ap.add_argument("--obj-ids", nargs="*", default=None)
    ap.add_argument("--min-stage", default="s6p", choices=["s6p","sq3"])
    args = ap.parse_args()

    obj_root = args.run_dir / "objects"
    shard_dirs = [d for d in obj_root.iterdir() if d.is_dir()] if obj_root.is_dir() else []
    if not shard_dirs:
        print(f"No shard dirs under {obj_root}", file=sys.stderr); sys.exit(1)
    base = shard_dirs[0]

    if args.obj_ids:
        obj_dirs = [base/oid for oid in args.obj_ids if (base/oid).is_dir()]
    else:
        obj_dirs = []
        for d in sorted(base.iterdir()):
            sp = d/"status.json"
            if not sp.is_file(): continue
            try: steps = json.loads(sp.read_text()).get("steps",{})
            except: continue
            key = "sq3_qc_E" if args.min_stage == "sq3" else "s6p_preview"
            if steps.get(key,{}).get("status") in ("ok","skip"):
                obj_dirs.append(d)

    print(f"Generating report for {len(obj_dirs)} objects…", file=sys.stderr)
    cards = []; np_tot = nf_tot = nobj_sq3 = 0
    for od in obj_dirs:
        sp = od/"status.json"; qp = od/"qc.json"
        status = json.loads(sp.read_text()) if sp.is_file() else {}
        qc     = json.loads(qp.read_text()) if qp.is_file() else {}
        if status.get("steps",{}).get("sq3_qc_E",{}).get("status") == "ok":
            nobj_sq3 += 1
            for eid, ei in qc.get("edits",{}).items():
                ge = (ei.get("gates") or {}).get("E")
                if ge:
                    if (ge.get("vlm") or {}).get("pass"): np_tot += 1
                    else: nf_tot += 1
        cards.append(render_object(od, status, qc))

    tot = np_tot+nf_tot
    rate = f"{np_tot}/{tot} ({np_tot/tot:.0%})" if tot else "N/A"
    summary = (f"<b>Objects:</b> {len(obj_dirs)} &nbsp;|&nbsp; "
               f"<b>with Gate-E:</b> {nobj_sq3} &nbsp;|&nbsp; "
               f"<b>Gate-E pass rate:</b> {rate} "
               f"<br><small>Run dir: {args.run_dir}</small>")

    html = (f'<!DOCTYPE html><html><head><meta charset="UTF-8">'
            f'<title>QC Report — {args.run_dir.name}</title>'
            f'<style>{CSS}</style></head><body>'
            f'<h1>Pipeline QC Report — {args.run_dir.name}</h1>'
            f'<div class="summary">{summary}</div>'
            + "".join(cards) + "</body></html>")

    args.out.write_text(html, encoding="utf-8")
    print(f"✓ {args.out}  ({args.out.stat().st_size//1024} KB)", file=sys.stderr)

if __name__ == "__main__":
    main()
