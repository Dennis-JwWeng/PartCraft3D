#!/usr/bin/env python3
from __future__ import annotations
import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2]))
"""Generate a self-contained HTML QC report for pipeline_v2 shard objects.

Layout (per object):
  - Object header: ID + step status badges + overview image
  - One card per edit:
      ┌─────────────────────────────────────────────────────────────┐
      │ [TYPE] prompt text                                          │
      │ Gate A: ✅/❌  Gate E: ✅/❌  reason                        │
      │ BEFORE (5 views)  ──────────────────────────────────────── │
      │ AFTER  (5 views)  ──────────────────────────────────────── │
      └─────────────────────────────────────────────────────────────┘

Usage:
    python scripts/vis/generate_qc_report.py \
        --run-dir /path/to/pipeline_v2_shard02 \
        --out report.html \
        [--obj-ids id1 id2 ...]
"""
import argparse, base64, json, sys, io
from pathlib import Path

TYPE_COLORS = {
    "deletion":     "#c0392b",
    "modification": "#1565c0",
    "material":     "#6a1b9a",
    "scale":        "#e65100",
    "global":       "#2e7d32",
    "addition":     "#00695c",
}
GATE_OK   = "✅"
GATE_FAIL = "❌"
GATE_NONE = "—"


# ─── image helpers ────────────────────────────────────────────────────────────

def _b64(path, max_w=None):
    """Return base64 <img> tag for a PNG/JPG, optionally resized."""
    if not path or not Path(path).is_file():
        return None
    try:
        from PIL import Image
        img = Image.open(path).convert("RGB")
        if max_w and img.width > max_w:
            h = int(img.height * max_w / img.width)
            img = img.resize((max_w, h), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=82)
        data = base64.b64encode(buf.getvalue()).decode()
        w = img.width
        return f'<img src="data:image/jpeg;base64,{data}" style="width:{w}px;max-width:100%;border-radius:3px">'
    except Exception as e:
        return f'<span style="color:red;font-size:10px">ERR:{e}</span>'


def _strip_row(paths, thumb_h=160):
    """Render a horizontal strip of images, each resized to thumb_h tall."""
    from PIL import Image
    import numpy as np
    import cv2
    imgs = []
    for p in paths:
        if not Path(p).is_file():
            return None
        img = cv2.imread(str(p))
        if img is None:
            return None
        h, w = img.shape[:2]
        nw = int(w * thumb_h / h)
        imgs.append(cv2.resize(img, (nw, thumb_h)))
    strip = np.hstack(imgs)
    ok, buf = cv2.imencode(".jpg", strip, [cv2.IMWRITE_JPEG_QUALITY, 82])
    if not ok:
        return None
    data = base64.b64encode(buf).decode()
    return f'<img src="data:image/jpeg;base64,{data}" style="max-width:100%;border-radius:3px">'


def _before_strip(ctx_image_npz, view_indices, thumb_h=160):
    """Load before-state views from images.npz and return a horizontal strip <img>."""
    if not ctx_image_npz or not Path(ctx_image_npz).is_file():
        return None
    try:
        import numpy as np, cv2
        from partcraft.render.overview import load_views_from_npz
        imgs, _ = load_views_from_npz(Path(ctx_image_npz), view_indices)
        resized = []
        for img in imgs:
            h, w = img.shape[:2]
            nw = int(w * thumb_h / h)
            resized.append(cv2.resize(img, (nw, thumb_h)))
        strip = np.hstack(resized)
        ok, buf = cv2.imencode(".jpg", strip, [cv2.IMWRITE_JPEG_QUALITY, 82])
        if not ok:
            return None
        data = base64.b64encode(buf).decode()
        return f'<img src="data:image/jpeg;base64,{data}" style="max-width:100%;border-radius:3px">'
    except Exception as e:
        return f'<span style="color:#aaa;font-size:11px">before imgs unavailable: {e}</span>'


# ─── data loaders ─────────────────────────────────────────────────────────────

def load_edit_meta(edit_dir: Path) -> dict:
    """Return edit metadata dict from meta.json (all types have one if produced)."""
    meta_f = edit_dir / "meta.json"
    if meta_f.is_file():
        try:
            return json.loads(meta_f.read_text())
        except Exception:
            pass
    return {}


_FLUX_TYPES = frozenset({"modification", "scale", "material", "global"})
_EDIT_PREFIX = {"deletion": "del", "modification": "mod", "scale": "scl",
                "material": "mat", "global": "glb", "addition": "add"}


def load_parsed_edits(obj_dir: Path, obj_id: str) -> dict:
    """Return {edit_id -> parsed_edit_dict} keyed by the ACTUAL edit_id.

    edit_id convention (mirrors specs.py iter_all_specs):
      - deletion:              del_{obj_id}_{del_seq:03d}
      - flux (mod/scl/mat/glb): {prefix}_{obj_id}_{flux_seq:03d}
    where flux_seq is a shared counter across ALL flux types in parsed order.
    """
    p = obj_dir / "phase1" / "parsed.json"
    if not p.is_file():
        return {}
    try:
        data = json.loads(p.read_text())
        edits = (data.get("parsed") or {}).get("edits") or []
    except Exception:
        return {}

    out = {}
    flux_seq = 0
    del_seq  = 0
    for e in edits:
        et = e.get("edit_type", "")
        prefix = _EDIT_PREFIX.get(et)
        if not prefix:
            continue
        if et in _FLUX_TYPES:
            seq = flux_seq;  flux_seq += 1
        elif et == "deletion":
            seq = del_seq;   del_seq  += 1
        else:
            continue
        eid = f"{prefix}_{obj_id}_{seq:03d}"
        out[eid] = e
    return out


def get_prompt_and_desc(edit_id: str, edit_dir: Path, parsed_edits: dict, qc_edits: dict) -> tuple[str, str]:
    """Best-effort: return (prompt, target_part_desc) for any edit type."""
    # 1. meta.json (present for addition, sometimes for others)
    meta = load_edit_meta(edit_dir)
    if meta.get("prompt"):
        return meta["prompt"], meta.get("target_part_desc", "")

    # 2. parsed.json keyed by full edit_id (correct mapping via flux_seq/del_seq)
    e = parsed_edits.get(edit_id, {})
    if e.get("prompt"):
        return e["prompt"], e.get("target_part_desc", "")

    # 3. qc.json edit entry
    qe = qc_edits.get(edit_id, {})
    return qe.get("prompt", ""), ""


def resolve_image_npz(obj_dir: Path, cfg_path: Path) -> Path | None:
    """Resolve the images.npz path for this object using the pipeline config."""
    try:
        import yaml
        cfg = yaml.safe_load(cfg_path.read_text())
        data = cfg.get("data", {})
        images_root = Path(data.get("images_root", ""))
        # infer shard from obj_dir grandparent
        shard = obj_dir.parent.name
        obj_id = obj_dir.name
        npz = images_root / shard / f"{obj_id}.npz"
        if npz.is_file():
            return npz
    except Exception:
        pass
    return None


# ─── badge helpers ─────────────────────────────────────────────────────────────

def gate_html(gate: dict | None, label: str) -> str:
    if not gate:
        return f'<span class="badge bn">Gate {label} {GATE_NONE}</span>'
    vlm = gate.get("vlm") or {}
    rule = gate.get("rule") or {}
    ok  = vlm.get("pass", rule.get("pass", False))
    score = vlm.get("score", "")
    score_s = f" {score:.0%}" if isinstance(score, float) else ""
    reason = (vlm.get("reason") or rule.get("reason") or "")[:200]
    tip = f' title="{reason}"' if reason else ""
    cls = "bok" if ok else "bfl"
    icon = GATE_OK if ok else GATE_FAIL
    return (f'<span class="badge {cls}"{tip}>Gate {label}: {icon}{score_s}</span>'
            + (f'<div class="reason">{reason}</div>' if reason else ""))


def step_badge(d: dict, label: str) -> str:
    st = d.get("status", "—")
    cls = {"ok": "bok", "fail": "bfl", "skip": "bsk"}.get(st, "bn")
    n = d.get("n_ok", d.get("n_edits", d.get("n", "")))
    ns = f"({n})" if n != "" else ""
    return f'<span class="badge {cls}">{label}:{st}{ns}</span>'


# ─── render one edit card ─────────────────────────────────────────────────────

def render_edit_card(
    edit_id: str,
    edit_dir: Path,
    qc_edits: dict,
    parsed_edits: dict,
    before_strip_html: str | None,
) -> str:
    etype = edit_id.split("_")[0]
    color = TYPE_COLORS.get(etype, "#555")

    prompt, part_desc = get_prompt_and_desc(edit_id, edit_dir, parsed_edits, qc_edits)
    qe    = qc_edits.get(edit_id, {})
    gates = qe.get("gates", {})
    ga    = gates.get("A")
    ge    = gates.get("E")
    final = qe.get("final_pass")

    fcls  = ("bok" if final else "bfl") if final is not None else "bn"
    ficon = (GATE_OK if final else GATE_FAIL) if final is not None else GATE_NONE

    # after strip (5 previews)
    preview_paths = [edit_dir / f"preview_{i}.png" for i in range(5)]
    after_html = _strip_row(preview_paths) or '<span class="miss">previews missing</span>'
    before_html = before_strip_html or '<span class="miss">before images unavailable</span>'

    prompt_html = f'<div class="prompt">"{prompt}"</div>' if prompt else '<div class="miss">(no prompt)</div>'
    part_html   = f'<div class="part-desc">Part: {part_desc}</div>' if part_desc else ""

    return f"""
<div class="edit-card">
  <div class="edit-hd" style="border-left:4px solid {color}">
    <span class="etype" style="color:{color}">[{etype.upper()}]</span>
    <span class="eid">{edit_id[-12:]}</span>
    {prompt_html}
    {part_html}
    <div class="gate-row">
      {gate_html(ga, "A")}
      {gate_html(ge, "E")}
      <span class="badge {fcls}">Final: {ficon}</span>
    </div>
  </div>
  <div class="edit-views">
    <div class="view-label">BEFORE</div>{before_html}
    <div class="view-label" style="margin-top:6px">AFTER</div>{after_html}
  </div>
</div>"""


# ─── render one object ────────────────────────────────────────────────────────

def render_object(obj_dir: Path, status: dict, qc: dict, image_npz: Path | None) -> str:
    obj_id = obj_dir.name
    steps  = status.get("steps", {})

    badges = " ".join(
        step_badge(steps.get(k, {}), k)
        for k in ["s1_phase1", "sq1_qc_A", "s4_flux_2d", "s5_trellis",
                  "s5b_del_mesh", "s6p_preview", "sq3_qc_E"]
    )

    overview_html = _b64(obj_dir / "phase1" / "overview.png", max_w=900) \
                    or "<em style='color:#aaa'>overview.png missing</em>"

    qc_edits    = qc.get("edits", {})
    parsed_edits = load_parsed_edits(obj_dir, obj_id)

    # compute before strip once per object (shared across all edits)
    from partcraft.pipeline_v2.specs import VIEW_INDICES
    before_html = _before_strip(image_npz, VIEW_INDICES) if image_npz else None

    # compute pass rate
    n_pass = n_fail = 0
    for eid, ei in qc_edits.items():
        ge = (ei.get("gates") or {}).get("E")
        if ge:
            if (ge.get("vlm") or {}).get("pass"):
                n_pass += 1
            else:
                n_fail += 1
    rate = f"{n_pass}/{n_pass+n_fail}" if (n_pass + n_fail) else "—"

    # edit cards (sorted: del, mod, scl, mat, glb, add)
    ORDER = ["del", "mod", "scl", "mat", "glb", "add"]
    edits_3d = obj_dir / "edits_3d"
    edit_dirs = sorted(
        (d for d in edits_3d.iterdir() if d.is_dir()),
        key=lambda d: (ORDER.index(d.name.split("_")[0]) if d.name.split("_")[0] in ORDER else 99, d.name)
    ) if edits_3d.is_dir() else []

    cards_html = "".join(
        render_edit_card(ed.name, ed, qc_edits, parsed_edits, before_html)
        for ed in edit_dirs
    )

    return f"""
<div class="obj-card">
  <div class="obj-hd">
    <span class="oid">{obj_id}</span>
    <div class="sbadges">{badges}</div>
    <span class="pr">Gate-E: {rate}</span>
  </div>
  <div class="ov">{overview_html}</div>
  <div class="edits">{cards_html}</div>
</div>"""


# ─── CSS ──────────────────────────────────────────────────────────────────────

CSS = """
* { box-sizing: border-box; }
body { font-family: -apple-system, sans-serif; background: #efefef;
       margin: 0; padding: 20px; font-size: 13px; color: #222; }
h1   { font-size: 20px; margin: 0 0 14px; }
.summary { background: #fff; border-radius: 8px; padding: 12px 18px;
           margin-bottom: 20px; box-shadow: 0 1px 3px rgba(0,0,0,.1); }

/* object card */
.obj-card { background: #fff; border-radius: 8px; margin-bottom: 28px;
            box-shadow: 0 1px 4px rgba(0,0,0,.12); overflow: hidden; }
.obj-hd   { background: #1a1a2e; color: #fff; padding: 10px 16px;
            display: flex; align-items: center; gap: 12px; flex-wrap: wrap; }
.oid      { font-family: monospace; font-size: 12px; color: #9ab4ff; }
.sbadges  { display: flex; gap: 4px; flex-wrap: wrap; }
.pr       { margin-left: auto; font-weight: 700; color: #90ee90; }
.ov       { padding: 10px; background: #f8f8f8; border-bottom: 1px solid #e0e0e0; }
.edits    { padding: 10px 14px; display: flex; flex-direction: column; gap: 12px; }

/* edit card */
.edit-card { border: 1px solid #ddd; border-radius: 6px; overflow: hidden; }
.edit-hd   { padding: 10px 14px; background: #fafafa; border-bottom: 1px solid #eee; }
.etype     { font-weight: 700; font-size: 13px; }
.eid       { font-family: monospace; font-size: 10px; color: #888; margin-left: 6px; }
.prompt    { margin: 6px 0 2px; font-style: italic; color: #333; line-height: 1.5; }
.part-desc { font-size: 11px; color: #777; margin-bottom: 6px; }
.gate-row  { display: flex; gap: 6px; flex-wrap: wrap; align-items: flex-start;
             margin-top: 6px; }
.reason    { font-size: 11px; color: #555; margin-top: 2px; line-height: 1.4;
             max-width: 800px; }
.edit-views { padding: 10px 14px; background: #fff; }
.view-label { font-size: 10px; font-weight: 700; color: #888;
              letter-spacing: .05em; margin-bottom: 3px; }

/* badges */
.badge { display: inline-block; padding: 2px 8px; border-radius: 10px;
         font-size: 11px; white-space: nowrap; cursor: default; }
.bok { background: #d4edda; color: #155724; }
.bfl { background: #f8d7da; color: #721c24; }
.bsk { background: #fff3cd; color: #856404; }
.bn  { background: #e2e3e5; color: #495057; }
.miss { color: #bbb; font-size: 11px; }
"""


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--out", default="report.html", type=Path)
    ap.add_argument("--obj-ids", nargs="*", default=None)
    ap.add_argument("--min-stage", default="s6p", choices=["s6p", "sq3"])
    ap.add_argument("--cfg",
                    default="/mnt/zsn/zsn_workspace/PartCraft3D/configs/pipeline_v2_shard02.yaml",
                    type=Path, help="Pipeline config for resolving images.npz paths")
    args = ap.parse_args()

    obj_root = args.run_dir / "objects"
    shard_dirs = [d for d in obj_root.iterdir() if d.is_dir()] if obj_root.is_dir() else []
    if not shard_dirs:
        print(f"No shard dirs under {obj_root}", file=sys.stderr); sys.exit(1)
    base = shard_dirs[0]

    if args.obj_ids:
        obj_dirs = [base / oid for oid in args.obj_ids if (base / oid).is_dir()]
    else:
        obj_dirs = []
        for d in sorted(base.iterdir()):
            sp = d / "status.json"
            if not sp.is_file(): continue
            try:
                steps = json.loads(sp.read_text()).get("steps", {})
            except Exception:
                continue
            key = "sq3_qc_E" if args.min_stage == "sq3" else "s6p_preview"
            if steps.get(key, {}).get("status") in ("ok", "skip"):
                obj_dirs.append(d)

    print(f"Building report for {len(obj_dirs)} objects…", file=sys.stderr)

    np_tot = nf_tot = nobj_sq3 = 0
    cards = []
    for od in obj_dirs:
        sp = od / "status.json"; qp = od / "qc.json"
        status = json.loads(sp.read_text()) if sp.is_file() else {}
        qc     = json.loads(qp.read_text()) if qp.is_file() else {}
        image_npz = resolve_image_npz(od, args.cfg)

        if status.get("steps", {}).get("sq3_qc_E", {}).get("status") == "ok":
            nobj_sq3 += 1
            for eid, ei in qc.get("edits", {}).items():
                ge = (ei.get("gates") or {}).get("E")
                if ge:
                    if (ge.get("vlm") or {}).get("pass"): np_tot += 1
                    else: nf_tot += 1

        cards.append(render_object(od, status, qc, image_npz))

    tot  = np_tot + nf_tot
    rate = f"{np_tot}/{tot} ({np_tot/tot:.0%})" if tot else "N/A"
    summary = (f"<b>Objects:</b> {len(obj_dirs)} &nbsp;|&nbsp;"
               f" <b>Gate-E evaluated:</b> {nobj_sq3} &nbsp;|&nbsp;"
               f" <b>Gate-E pass rate:</b> {rate}"
               f"<br><small>Run: {args.run_dir}</small>")

    html = (f'<!DOCTYPE html><html><head><meta charset="UTF-8">'
            f'<title>QC Report</title><style>{CSS}</style></head><body>'
            f'<h1>Pipeline QC Report — {args.run_dir.name}</h1>'
            f'<div class="summary">{summary}</div>'
            + "".join(cards)
            + "</body></html>")

    args.out.write_text(html, encoding="utf-8")
    print(f"✓ {args.out}  ({args.out.stat().st_size // 1024} KB)", file=sys.stderr)


if __name__ == "__main__":
    main()
