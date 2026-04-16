#!/usr/bin/env python3
"""
visualize_mode_e.py
───────────────────
Self-contained HTML report for Mode E (text_align_gate) results.

For every object shows:
  • Full overview image + parts legend
  • Per-edit card:
      - Rebuilt gate image (5×2 red/grey for non-global, 5×1 RGB for global)
      - PASS / FAIL badge  +  edit-type badge
      - Selected parts chips  +  prompt text
      - VLM reason text
      - Pixel-count mini bar chart (5 views, best view starred)
      - Rule fail codes (if any)

Usage
-----
  python scripts/tools/visualize_mode_e.py \
      --run-dir data/partverse/outputs/partverse/bench_shard08/mode_e_text_align \
      --output  /tmp/mode_e_report.html
"""
from __future__ import annotations
import argparse, base64, io, json, sys
from pathlib import Path

import cv2
import numpy as np

# ── palette (mirrors partcraft/render/overview.py) ───────────────────────────
_PALETTE: list[list[int]] = [
    [220, 30, 30], [255, 140, 0], [255, 220, 0], [130, 220, 30],
    [30, 160, 50], [0, 150, 150], [60, 220, 230], [30, 90, 240],
    [20, 30, 130], [140, 40, 200], [230, 40, 200], [255, 150, 200],
    [130, 70, 30], [220, 180, 130], [30, 30, 30], [130, 130, 130],
]
_PAL_BGR = [[c[2], c[1], c[0]] for c in _PALETTE]
_RED     = np.array([0, 0, 220], dtype=np.uint8)    # BGR
_GREY    = np.array([110, 110, 110], dtype=np.uint8)
_N_VIEWS = 5
_COL_SEP = 4
_ROW_SEP = 6

_TYPE_CSS = {
    "deletion": "#c0392b", "modification": "#2980b9",
    "scale": "#d39e00", "material": "#27ae60",
    "color": "#8e44ad", "global": "#16a085",
}
_EID_PFX = {"deletion": "del", "modification": "mod", "scale": "scl",
            "material": "mat", "color": "col", "global": "glb"}

OVERVIEWS_ROOT = Path("/mnt/zsn/zsn_workspace/PartCraft3D/outputs/partverse/bench_shard08_overviews")
SHARD = "08"

# ── image helpers ─────────────────────────────────────────────────────────────

def _load_bgr(path: Path) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(path)
    return img


def _extract_cell(img: np.ndarray, col: int, row: int) -> np.ndarray:
    H, W = img.shape[:2]
    W_cell = (W - (_N_VIEWS - 1) * _COL_SEP) // _N_VIEWS
    H_cell = (H - _ROW_SEP) // 2
    return img[row * (H_cell + _ROW_SEP): row * (H_cell + _ROW_SEP) + H_cell,
               col * (W_cell + _COL_SEP): col * (W_cell + _COL_SEP) + W_cell].copy()


def _highlight_cell(cell: np.ndarray, sel_slots: set[int]) -> np.ndarray:
    pal    = np.array(_PAL_BGR, dtype=np.int32)
    flat   = cell.reshape(-1, 3).astype(np.int32)
    diffs  = np.linalg.norm(flat[:, None] - pal[None], axis=2)
    nearest = np.argmin(diffs, axis=1)
    is_bg  = np.all(flat > 230, axis=1)
    is_sel = np.array([i in sel_slots for i in nearest])
    out = np.empty_like(flat)
    out[is_bg]             = [255, 255, 255]
    out[~is_bg & is_sel]   = _RED
    out[~is_bg & ~is_sel]  = _GREY
    return out.reshape(cell.shape).astype(np.uint8)


def _hstack(cells: list[np.ndarray]) -> np.ndarray:
    row = cells[0]
    for c in cells[1:]:
        sep = np.full((c.shape[0], _COL_SEP, 3), 255, dtype=np.uint8)
        row = np.concatenate([row, sep, c], axis=1)
    return row


def _build_gate_img(ov_bgr: np.ndarray, sel_ids: list[int], column_map: list[int]) -> np.ndarray:
    sel_slots = {pid % len(_PAL_BGR) for pid in sel_ids}
    tops = [_extract_cell(ov_bgr, v, 0) for v in column_map]
    if not sel_slots:   # global: RGB strip only
        return _hstack(tops)
    bots = [_highlight_cell(_extract_cell(ov_bgr, v, 1), sel_slots) for v in column_map]
    W_total = sum(c.shape[1] for c in tops) + (_N_VIEWS - 1) * _COL_SEP
    sep_h = np.full((_ROW_SEP, W_total, 3), 255, dtype=np.uint8)
    return np.concatenate([_hstack(tops), sep_h, _hstack(bots)], axis=0)


def _bgr_b64(img: np.ndarray, scale: float = 1.0) -> str:
    if scale != 1.0:
        h, w = img.shape[:2]
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 82])
    return base64.b64encode(buf.tobytes()).decode()


# ── SVG pixel bar ─────────────────────────────────────────────────────────────

def _pixel_bar_svg(pixel_counts: list[int], best_col: int, column_map: list[int]) -> str:
    if not pixel_counts or max(pixel_counts) == 0:
        return ""
    W, H, pad = 140, 38, 3
    bw  = (W - pad * (_N_VIEWS + 1)) // _N_VIEWS
    mx  = max(pixel_counts) or 1
    bars = []
    for i, px in enumerate(pixel_counts):
        x  = pad + i * (bw + pad)
        bh = max(1, int((px / mx) * (H - 15)))
        y  = H - 15 - bh
        fc = "#e74c3c" if i == best_col else "#4a7a9b"
        orig = column_map[i] if i < len(column_map) else i
        star = "★" if i == best_col else ""
        bars.append(
            f'<rect x="{x}" y="{y}" width="{bw}" height="{bh}" fill="{fc}" rx="1"/>'
            f'<text x="{x+bw//2}" y="{H-3}" text-anchor="middle" font-size="7" fill="#999">'
            f'v{orig}{star}</text>'
        )
    return (f'<svg width="{W}" height="{H}" xmlns="http://www.w3.org/2000/svg">'
            + "".join(bars) + "</svg>")


# ── HTML helpers ──────────────────────────────────────────────────────────────

def _chips(pids: list[int], names: dict[int, str]) -> str:
    out = []
    for pid in pids:
        r, g, b = _PALETTE[pid % len(_PALETTE)]
        lum = 0.299*r + 0.587*g + 0.114*b
        tc  = "#000" if lum > 150 else "#fff"
        out.append(
            f'<span class="chip" style="background:rgb({r},{g},{b});color:{tc};">'
            f'p{pid} {names.get(pid, "?")}</span>'
        )
    return "".join(out)


def _legend(parts: list[dict]) -> str:
    rows = []
    for p in parts:
        pid = p["part_id"]; name = p.get("name", "?")
        r, g, b = _PALETTE[pid % len(_PALETTE)]
        lum = 0.299*r + 0.587*g + 0.114*b
        tc  = "#000" if lum > 150 else "#fff"
        rows.append(
            f'<div class="part-row">'
            f'<span class="chip" style="background:rgb({r},{g},{b});color:{tc};">p{pid}</span>'
            f'<span class="pname">{name}</span></div>'
        )
    return '<div class="legend">' + "".join(rows) + '</div>'


def _edit_card(eid: str, edit: dict, status: dict, ov_bgr: np.ndarray,
               names: dict[int, str], scale: float) -> str:
    et     = edit.get("edit_type", "?")
    pids   = edit.get("selected_part_ids", [])
    prompt = edit.get("prompt", "")
    tgt    = edit.get("target_part_desc", "")
    after  = edit.get("after_desc") or ""
    ep     = edit.get("edit_params") or {}

    gate_a   = status.get("gates", {}).get("A", {})
    rule     = gate_a.get("rule", {})
    vlm      = gate_a.get("vlm") or {}
    gstatus  = status.get("stages", {}).get("gate_a", {}).get("status", "?")
    passed   = gstatus == "pass"

    vlm_pass    = vlm.get("pass", False)
    reason      = vlm.get("reason", "")
    best_view   = vlm.get("best_view", 0)
    best_col    = vlm.get("best_view_col", 0)
    px_counts   = vlm.get("pixel_counts", [])
    col_map     = vlm.get("column_map", list(range(5)))
    rule_checks = rule.get("checks", {})

    gate_bgr = _build_gate_img(ov_bgr, pids, col_map)
    gate_b64 = _bgr_b64(gate_bgr, scale)
    gate_w   = int(gate_bgr.shape[1] * scale)

    tc   = _TYPE_CSS.get(et, "#777")
    pcss = "#27ae60" if passed else "#c0392b"
    plbl = "PASS" if passed else "FAIL"

    ep_html = "".join(
        f'<span class="ep-tag">{k}: {v}</span>' for k, v in ep.items() if v)
    rule_html = (
        f'<div class="rule-fail">⚠ rule: {", ".join(rule_checks)}</div>'
    ) if not rule.get("pass", True) and rule_checks else ""
    bar = _pixel_bar_svg(px_counts, best_col, col_map)
    chip_html = _chips(pids, names) or '<span class="sub">global — whole object</span>'

    return f"""\
<div class="ecard {'p' if passed else 'f'}">
  <img src="data:image/jpeg;base64,{gate_b64}" width="{gate_w}" style="display:block;width:100%;height:auto;">
  <div class="ebody">
    <div class="ehdr">
      <span class="badge" style="background:{tc};">{et.upper()}</span>
      <span class="badge" style="background:{pcss};">{plbl}</span>
      <span class="eid">{eid}</span>
    </div>
    <div class="chips-row">{chip_html}</div>
    <div class="prompt">"{prompt}"</div>
    {'<div class="tgt">↳ target: ' + tgt + '</div>' if tgt else ''}
    {'<div class="after">after: ' + after + '</div>' if after else ''}
    {ep_html}
    {rule_html}
    <div class="reason {'pr' if vlm_pass else 'fr'}">{'✓' if vlm_pass else '✗'} {reason}</div>
    <div class="bar-row">{bar}<span class="sub bv">best v{best_view} (col {best_col})</span></div>
  </div>
</div>"""


def _obj_section(obj_id: str, parsed: dict, status_doc: dict,
                 ov_bgr: np.ndarray, scale: float) -> str:
    pd     = parsed.get("parsed", parsed) or {}
    parts  = pd.get("object", {}).get("parts", [])
    edits  = pd.get("edits", [])
    names  = {p["part_id"]: p.get("name", "?") for p in parts}
    sedits = status_doc.get("edits", {})

    ov_b64 = _bgr_b64(ov_bgr, scale)
    ov_w   = int(ov_bgr.shape[1] * scale)

    pass_cards, fail_cards = [], []
    for idx, edit in enumerate(edits):
        et  = edit.get("edit_type", "unk")
        eid = f"{_EID_PFX.get(et,'unk')}_{obj_id}_{idx:03d}"
        st  = sedits.get(eid, {})
        card = _edit_card(eid, edit, st, ov_bgr, names, scale)
        if st.get("stages", {}).get("gate_a", {}).get("status") == "pass":
            pass_cards.append(card)
        else:
            fail_cards.append(card)

    n_t, n_p, n_f = len(edits), len(pass_cards), len(fail_cards)
    pct = f"{100*n_p//n_t}%" if n_t else "—"

    ps = (f'<div class="glabel gl-p">✓ PASS ({n_p})</div>'
          + '<div class="egrid">' + "\n".join(pass_cards) + "</div>") if pass_cards else ""
    fs = (f'<div class="glabel gl-f">✗ FAIL ({n_f})</div>'
          + '<div class="egrid">' + "\n".join(fail_cards) + "</div>") if fail_cards else ""

    return f"""\
<section class="osec">
  <div class="ohdr">
    <span class="oid">{obj_id}</span>
    <span class="omode">[text_align_gate]</span>
    <span class="ostats">{n_t} edits · {n_p} pass · {n_f} fail · {pct}</span>
  </div>
  <div class="ov-row">
    <div class="ovlbl">overview</div>
    <img src="data:image/jpeg;base64,{ov_b64}" width="{ov_w}">
    {_legend(parts)}
  </div>
  {ps}{fs}
</section>"""


# ── CSS ───────────────────────────────────────────────────────────────────────

_CSS = """
:root{--bg:#0f0f0f;--s1:#161616;--s2:#1c1c1c;--bd:#252525;
  --tx:#ddd;--sub:#777;--ac:#4a9eff;--pass:#27ae60;--fail:#c0392b;}
*{box-sizing:border-box;margin:0;padding:0;}
body{background:var(--bg);color:var(--tx);font-family:'Courier New',monospace;padding:24px 28px;}
h1{font-size:1.15rem;color:var(--ac);margin-bottom:6px;}
.summary{font-size:.82rem;color:var(--sub);margin-bottom:30px;}

.osec{border:1px solid var(--bd);border-radius:8px;margin-bottom:48px;
  padding:16px;background:var(--s1);}
.ohdr{display:flex;align-items:baseline;gap:10px;margin-bottom:14px;
  padding-bottom:10px;border-bottom:1px solid var(--bd);}
.oid{font-size:.9rem;font-weight:bold;color:var(--ac);}
.omode{font-size:.75rem;color:var(--sub);}
.ostats{font-size:.78rem;color:#aaa;margin-left:auto;}

.ov-row{display:flex;align-items:flex-start;gap:10px;margin-bottom:16px;}
.ovlbl{writing-mode:vertical-rl;font-size:.62rem;color:var(--sub);padding-top:6px;}
.ov-row img{border-radius:4px;}

.legend{display:flex;flex-direction:column;gap:3px;margin-left:10px;
  background:#101010;border:1px solid var(--bd);border-radius:5px;
  padding:6px 10px;min-width:125px;max-height:450px;overflow-y:auto;}
.part-row{display:flex;align-items:center;gap:5px;}
.pname{font-size:.7rem;color:#bbb;white-space:nowrap;}

.glabel{font-size:.75rem;font-weight:bold;letter-spacing:.05em;
  padding:3px 8px;border-radius:3px;margin:14px 0 8px;display:inline-block;}
.gl-p{background:rgba(39,174,96,.12);color:var(--pass);}
.gl-f{background:rgba(192,57,43,.12);color:var(--fail);}

.egrid{display:grid;grid-template-columns:repeat(auto-fill,minmax(520px,1fr));gap:12px;
  margin-bottom:6px;}

.ecard{background:var(--s2);border:1px solid var(--bd);border-radius:6px;
  overflow:hidden;display:flex;flex-direction:column;}
.ecard.p{border-left:3px solid var(--pass);}
.ecard.f{border-left:3px solid var(--fail);}
.ebody{padding:8px 11px;display:flex;flex-direction:column;gap:5px;}
.ehdr{display:flex;align-items:center;gap:5px;flex-wrap:wrap;}
.eid{font-size:.62rem;color:var(--sub);margin-left:auto;}
.badge{font-size:.66rem;font-weight:bold;padding:2px 7px;border-radius:3px;
  color:#fff;white-space:nowrap;}
.chips-row{display:flex;flex-wrap:wrap;gap:4px;}
.chip{font-size:.66rem;font-weight:bold;padding:1px 5px;border-radius:3px;white-space:nowrap;}
.sub{color:var(--sub);font-size:.72rem;}
.prompt{font-size:.8rem;color:#e0e0e0;font-style:italic;}
.tgt{font-size:.7rem;color:var(--sub);}
.after{font-size:.7rem;color:#6ab;}
.ep-tag{font-size:.7rem;color:#7cc;background:#0e1e1e;padding:1px 5px;
  border-radius:3px;margin-right:4px;}
.rule-fail{font-size:.7rem;color:#e67e22;background:rgba(230,126,34,.1);
  padding:2px 6px;border-radius:3px;}
.reason{font-size:.76rem;padding:4px 7px;border-radius:3px;line-height:1.4;}
.pr{color:#a8d5a2;background:rgba(39,174,96,.07);}
.fr{color:#e8a8a8;background:rgba(192,57,43,.07);}
.bar-row{display:flex;align-items:center;gap:6px;}
.bv{font-size:.66rem;}
"""

_HEAD = f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="UTF-8">
<title>Mode E — Alignment Gate Viewer</title>
<style>{_CSS}</style>
</head><body>
<h1>Mode E — Text + Alignment Gate Viewer</h1>
"""
_TAIL = "</body></html>\n"


def _find_overview(obj_id: str) -> Path | None:
    p = OVERVIEWS_ROOT / "objects" / SHARD / obj_id / "phase1" / "overview.png"
    return p if p.exists() else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path,
        default=Path("data/partverse/outputs/partverse/bench_shard08/mode_e_text_align"))
    ap.add_argument("--output",  type=Path, default=Path("/tmp/mode_e_report.html"))
    ap.add_argument("--scale",   type=float, default=0.5)
    ap.add_argument("--limit",   type=int, default=0)
    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve()
    pfiles  = sorted(run_dir.rglob("parsed.json"))
    if not pfiles:
        print(f"No parsed.json under {run_dir}"); sys.exit(1)
    if args.limit:
        pfiles = pfiles[:args.limit]

    print(f"Building HTML for {len(pfiles)} objects …")
    blocks: list[str] = []
    total_e = total_p = 0

    for pf in pfiles:
        obj_id  = pf.parent.parent.name
        # v3 layout: <obj>/edit_status.json  (sibling of phase1/)
        # v2 layout: <obj>/phase1/edit_status.json (sibling of parsed.json)
        sf      = pf.parent / "edit_status.json"
        if not sf.exists():
            sf = pf.parent.parent / "edit_status.json"
        ov_path = _find_overview(obj_id)

        if not sf.exists():
            print(f"  [skip no status] {obj_id}"); continue
        if ov_path is None:
            print(f"  [skip no overview] {obj_id}"); continue

        try:
            parsed  = json.loads(pf.read_text())
            sdoc    = json.loads(sf.read_text())
            ov_bgr  = _load_bgr(ov_path)
        except Exception as e:
            print(f"  [error] {obj_id}: {e}"); continue

        n_e = len((parsed.get("parsed", parsed) or {}).get("edits", []))
        n_p = sum(1 for s in sdoc.get("edits", {}).values()
                  if s.get("stages", {}).get("gate_a", {}).get("status") == "pass")
        total_e += n_e; total_p += n_p
        print(f"  {obj_id}  edits={n_e}  pass={n_p}", flush=True)
        blocks.append(_obj_section(obj_id, parsed, sdoc, ov_bgr, args.scale))

    pct = 100 * total_p // total_e if total_e else 0
    summary = (
        f'<p class="summary">{len(blocks)} objects · {total_e} edits · '
        f'{total_p} pass · {total_e-total_p} fail · yield {pct}%</p>'
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(_HEAD + summary + "\n".join(blocks) + _TAIL, encoding="utf-8")
    print(f"\n✓ Saved → {args.output}  ({len(blocks)} objects, {total_e} edits)")


if __name__ == "__main__":
    main()
