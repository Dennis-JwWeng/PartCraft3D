"""Build an HTML report for the gate_quality (gate E) experiment.

Each edit card now reproduces the EXACT input the VLM saw:
  * a 2x5 BEFORE / AFTER collage (mirror of
    ``partcraft.pipeline_v3.vlm_core._make_before_after_collage``), and
  * the per-edit USER message text (mirror of
    ``partcraft.pipeline_v3.vlm_core.build_quality_judge_prompt``),

so a human can compare what the model received against its verdict and
quickly judge whether the verdict is reasonable.
"""
from __future__ import annotations
import argparse, base64, io, json, sys
from collections import Counter, defaultdict
from html import escape
from pathlib import Path

# ── lazy deps ───────────────────────────────────────────────────────────────
_PIL = None
def _pil():
    global _PIL
    if _PIL is None:
        from PIL import Image  # type: ignore
        _PIL = Image
    return _PIL

_CV2 = None; _NP = None
def _cv():
    global _CV2, _NP
    if _CV2 is None:
        import cv2 as _cv2  # type: ignore
        import numpy as _np  # type: ignore
        _CV2 = _cv2; _NP = _np
    return _CV2, _NP

_THUMB_CACHE: dict[Path, str] = {}
_BEFORE_CACHE: dict[str, list] = {}

# Must match partcraft.pipeline_v3.specs.VIEW_INDICES (single source of
# truth in vlm_core.py).  Hard-coded here to avoid importing the heavy
# pipeline module just to read this constant.
VIEW_INDICES_DEFAULT = [89, 90, 91, 100, 8]

EDIT_TYPE_CHIP_COLOR = {
    "modification": "#4a9eff", "color":  "#e056b9", "material": "#f39c12",
    "scale":  "#16a085", "global": "#9b59b6",
    "deletion": "#888888", "addition": "#27ae60",
}
FLUX_TYPES = {"modification", "color", "material", "scale", "global"}

# Kept verbatim so the report shows the same system prompt the live judge
# uses.  If you bump JUDGE_SYSTEM_PROMPT in vlm_core.py, refresh this too.
JUDGE_SYSTEM_PROMPT = """You are a 3D-edit quality judge.

INPUT:
  A single 2x5 collage image of ONE 3D object.
    TOP row    = BEFORE (5 views)
    BOTTOM row = AFTER  (5 views, same camera per column)
  The user message supplies: edit_type, edit_prompt, object description,
  target part label + description, and one type-specific extra
  (new_part_desc | factor | target_material | target_color | target_style).

WHAT SHOULD CHANGE vs MUST NOT CHANGE (by edit_type):
  modification  change: SHAPE / SILHOUETTE of target part
                keep:   colour, material, position; ALL other parts intact
  scale         change: SIZE of target part (shrinks by factor; still attached)
                keep:   shape of target part; ALL other parts intact
  material      change: SURFACE FINISH of target part (e.g. wood -> steel)
                keep:   geometry of ALL parts; colour broadly similar
  color         change: HUE / SHADE of target part
                keep:   shape + material type of ALL parts; no colour bleed
  global        change: WHOLE-OBJECT art style / rendering aesthetic
                keep:   underlying geometry + structure still recognisable

OUTPUT: ONE valid JSON object only. First character must be "{" and last "}".
{
  "edit_executed":      <true|false>,
  "correct_region":     <true|false>,
  "preserve_other":     <true|false>,
  "visual_quality":     <1-5>,
  "artifact_free":      <true|false>,
  "reason":             "<one sentence explaining your verdict>",
  "prompt_quality":     <1-5>,
  "improved_prompt":    "<imperative rewrite of the original prompt>",
  "improved_after_desc":"<concise description of the AFTER object>"
}
(see vlm_core.JUDGE_SYSTEM_PROMPT for the full rules / R1..R6)
"""

# ── image helpers ───────────────────────────────────────────────────────────
def encode_thumb(path: Path, size: int = 256, quality: int = 80) -> str:
    """Return ``data:image/jpeg;base64,...`` URI for *path* (cached)."""
    if path in _THUMB_CACHE:
        return _THUMB_CACHE[path]
    Image = _pil()
    with Image.open(path) as im:
        im = im.convert("RGB")
        im.thumbnail((size, size), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=quality, optimize=True)
    uri = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
    _THUMB_CACHE[path] = uri
    return uri


def load_before_views(image_npz: Path, view_indices=VIEW_INDICES_DEFAULT) -> list:
    """Return list of 5 BGR arrays loaded from *image_npz* (cached per file).

    Mirrors ``partcraft.render.overview.load_views_from_npz``: NPZ entries
    are encoded PNG bytes (uint8 1-D arrays) keyed by ``"%03d.png" % idx``.
    Returns ``[]`` on any failure so the caller can fall back gracefully.
    """
    key = str(image_npz)
    if key in _BEFORE_CACHE:
        return _BEFORE_CACHE[key]
    if not image_npz.is_file():
        _BEFORE_CACHE[key] = []
        return []
    cv2, np = _cv()
    try:
        d = np.load(image_npz)
        files = set(d.files)
        imgs = []
        for idx in view_indices:
            name = f"{idx:03d}.png"
            if name not in files:
                _BEFORE_CACHE[key] = []
                return []
            arr = d[name]
            buf = np.frombuffer(arr.tobytes(), dtype=np.uint8)
            # Use IMREAD_UNCHANGED + alpha-composite onto WHITE — must mirror
            # partcraft.render.overview.load_views_from_npz exactly, otherwise
            # transparent NPZ PNGs (RGBA) come out black-background and no
            # longer match what the judge VLM actually receives.
            im = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
            if im is None:
                _BEFORE_CACHE[key] = []
                return []
            if im.ndim == 3 and im.shape[2] == 4:
                a = im[:, :, 3:4].astype(np.float32) / 255.0
                rgb = im[:, :, :3].astype(np.float32)
                bg = np.full_like(rgb, 255)
                im = (rgb * a + bg * (1 - a)).astype(np.uint8)
            elif im.ndim == 2:
                im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
            imgs.append(im)
        _BEFORE_CACHE[key] = imgs
        return imgs
    except Exception:
        _BEFORE_CACHE[key] = []
        return []


def build_vlm_collage_uri(before_imgs: list, edit_dir: Path,
                          *, h: int = 192, q: int = 75) -> "str | None":
    """Reproduce the 2x5 BEFORE/AFTER collage VLM judge actually saw.

    Mirrors ``partcraft.pipeline_v3.vlm_core._make_before_after_collage``
    (top row = before, bottom row = after, hstack rows + vstack), then
    JPEG-encodes for inline embedding.  Returns ``None`` if any
    ``preview_*.png`` is missing.

    *h* is the per-row height (192 px keeps a 5-view row ~960 px wide,
    a good size for in-card display while limiting HTML weight).
    """
    if not before_imgs or len(before_imgs) != 5:
        return None
    cv2, np = _cv()
    after_imgs = []
    for i in range(5):
        p = edit_dir / f"preview_{i}.png"
        if not p.is_file():
            return None
        im = cv2.imread(str(p))
        if im is None:
            return None
        after_imgs.append(im)

    def _r(x):
        s = h / x.shape[0]
        return cv2.resize(x, (int(x.shape[1] * s), h))

    try:
        row_b = np.hstack([_r(im) for im in before_imgs])
        row_a = np.hstack([_r(im) for im in after_imgs])
        w = max(row_b.shape[1], row_a.shape[1])
        if row_b.shape[1] < w:
            row_b = np.pad(row_b, ((0, 0), (0, w - row_b.shape[1]), (0, 0)))
        if row_a.shape[1] < w:
            row_a = np.pad(row_a, ((0, 0), (0, w - row_a.shape[1]), (0, 0)))
        coll = np.vstack([row_b, row_a])
        ok, buf = cv2.imencode(".jpg", coll, [cv2.IMWRITE_JPEG_QUALITY, q])
        if not ok:
            return None
        return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")
    except Exception:
        return None


def build_before_only_collage_uri(before_imgs, *, h: int = 160,
                                  q: int = 75) -> "str | None":
    """Render a single-row BEFORE collage (5 source views) as a data URI.

    Used for gate_a_reject / no_output cards where no AFTER previews exist
    so the user can still see what object the rejected edit referred to.
    """
    if not before_imgs or len(before_imgs) != 5:
        return None
    cv2, np = _cv()
    try:
        def _r(x):
            s = h / x.shape[0]
            return cv2.resize(x, (int(x.shape[1] * s), h))
        row = np.hstack([_r(im) for im in before_imgs])
        ok, buf = cv2.imencode(".jpg", row, [cv2.IMWRITE_JPEG_QUALITY, q])
        if not ok:
            return None
        return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")
    except Exception:
        return None


def build_partial_preview_strip_uri(edit_dir: Path, *, h: int = 160,
                                    q: int = 75) -> "tuple[str | None, int]":
    """Render any existing preview_*.png (0-4) as a single-row JPEG.

    Returns ``(data_uri, n_found)``; data_uri is None when no previews exist.
    """
    cv2, np = _cv()
    rows = []
    for i in range(5):
        p = edit_dir / f"preview_{i}.png"
        if not p.is_file():
            continue
        im = cv2.imread(str(p))
        if im is None:
            continue
        s = h / im.shape[0]
        rows.append(cv2.resize(im, (int(im.shape[1] * s), h)))
    if not rows:
        return None, 0
    try:
        row = np.hstack(rows)
        ok, buf = cv2.imencode(".jpg", row, [cv2.IMWRITE_JPEG_QUALITY, q])
        if not ok:
            return None, len(rows)
        return ("data:image/jpeg;base64," +
                base64.b64encode(buf.tobytes()).decode("ascii"),
                len(rows))
    except Exception:
        return None, len(rows)


def encode_overview_uri(obj_dir: Path, *, max_side: int = 1100,
                        q: int = 78) -> "str | None":
    """Embed phase1/overview.png as a data URI (downsized for HTML weight)."""
    p = obj_dir / "phase1" / "overview.png"
    if not p.is_file():
        return None
    Image = _pil()
    try:
        with Image.open(p) as im:
            im = im.convert("RGB")
            im.thumbnail((max_side, max_side), Image.LANCZOS)
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=q, optimize=True)
        return ("data:image/jpeg;base64," +
                base64.b64encode(buf.getvalue()).decode("ascii"))
    except Exception:
        return None


def vlm_user_message(ed: dict) -> str:
    """Reproduce the per-edit USER message the judge VLM actually received.

    Mirrors ``partcraft.pipeline_v3.vlm_core.build_quality_judge_prompt``
    line for line so the rendered text stays a faithful transcript.
    """
    et = (ed.get("edit_type") or "").lower()
    object_desc = ed.get("object_desc") or ""
    prompt = ed.get("prompt") or ""
    part_labels = ed.get("part_labels") or []
    target_part_desc = ed.get("target_part_desc") or ""
    ep = ed.get("edit_params") or {}
    lines = [
        f"edit_type: {et}",
        f"Object: {object_desc}",
        f'Edit instruction: "{prompt}"',
        f"Target part: {', '.join(part_labels)}",
    ]
    if target_part_desc:
        lines.append(f'Target part description (before): "{target_part_desc}"')
    if et == "modification" and ep.get("new_part_desc"):
        lines.append(f'Expected shape after: "{ep["new_part_desc"]}"')
    elif et == "scale" and ep.get("factor"):
        lines.append(f'Scale factor (shrink): {ep["factor"]}')
    elif et == "material" and ep.get("target_material"):
        lines.append(f'Expected material/finish: "{ep["target_material"]}"')
    elif et == "color" and ep.get("target_color"):
        lines.append(f'Expected colour: "{ep["target_color"]}"')
    elif et == "global" and ep.get("target_style"):
        lines.append(f'Expected style: "{ep["target_style"]}"')
    return "\n".join(lines)

# ── styling ─────────────────────────────────────────────────────────────────
CSS = r"""
:root{--bg:#0e0e10;--s1:#16161a;--s2:#1c1c22;--bd:#2a2a32;--tx:#e6e6ea;--sub:#7e7e88;--ac:#4a9eff;--pass:#27ae60;--fail:#c0392b;--inh:#888;}
*{box-sizing:border-box;margin:0;padding:0;}
body{background:var(--bg);color:var(--tx);font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;padding:20px 26px;line-height:1.4;}
h1{font-size:1.4rem;color:var(--ac);margin-bottom:6px;font-weight:600;}
.sub{font-size:.82rem;color:var(--sub);margin-bottom:18px;}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin-bottom:18px;}
.stat{background:var(--s1);border:1px solid var(--bd);border-radius:8px;padding:10px 14px;}
.stat .v{font-size:1.4rem;font-weight:700;color:var(--ac);font-family:'JetBrains Mono',monospace;}
.stat .v.pass{color:var(--pass);} .stat .v.fail{color:var(--fail);} .stat .v.inh{color:var(--inh);}
.stat .l{font-size:.72rem;color:var(--sub);text-transform:uppercase;letter-spacing:.06em;}
.bar-section{background:var(--s1);border:1px solid var(--bd);border-radius:8px;padding:14px 16px;margin-bottom:18px;}
.bar-section h3{font-size:.85rem;color:var(--ac);margin-bottom:10px;font-weight:600;}
.bar-row{display:flex;align-items:center;margin:5px 0;gap:8px;font-size:.78rem;font-family:'JetBrains Mono',monospace;}
.bar-row .lbl{width:110px;color:var(--tx);}
.bar-row .bar{height:18px;background:#0a0a0e;border-radius:3px;overflow:hidden;display:flex;}
.bar-row .bp{background:var(--pass);} .bar-row .bf{background:var(--fail);}
.bar-row .nums{width:170px;color:var(--sub);font-size:.74rem;text-align:right;margin-left:auto;}
.controls{position:sticky;top:0;z-index:20;background:var(--bg);padding:10px 0;border-bottom:1px solid var(--bd);margin-bottom:18px;display:flex;gap:8px;flex-wrap:wrap;align-items:center;}
.btn{background:var(--s2);color:var(--tx);border:1px solid var(--bd);border-radius:5px;padding:6px 12px;font-size:.78rem;cursor:pointer;font-family:inherit;transition:all .15s;}
.btn:hover{border-color:var(--ac);}
.btn.active{background:var(--ac);border-color:var(--ac);color:#fff;}
.btn.pass.active{background:var(--pass);border-color:var(--pass);}
.btn.fail.active{background:var(--fail);border-color:var(--fail);}
.osec{margin-bottom:32px;}
.ohdr{display:flex;align-items:baseline;gap:10px;margin-bottom:10px;padding-bottom:6px;border-bottom:1px solid var(--bd);}
.oid{font-size:.95rem;font-weight:600;color:var(--ac);font-family:'JetBrains Mono',monospace;}
.ostats{font-size:.74rem;color:var(--sub);margin-left:auto;}
.egrid{display:grid;grid-template-columns:repeat(auto-fill,minmax(620px,1fr));gap:12px;}
.ecard{background:var(--s2);border:1px solid var(--bd);border-radius:7px;overflow:hidden;display:flex;flex-direction:column;}
.ecard.p{border-left:3px solid var(--pass);} .ecard.f{border-left:3px solid var(--fail);}
.ecard.hide{display:none;}
.ehdr{padding:8px 12px 6px;display:flex;align-items:center;gap:8px;flex-wrap:wrap;border-bottom:1px solid var(--bd);background:#13131a;}
.badge{font-size:.66rem;font-weight:700;padding:2px 8px;border-radius:3px;color:#fff;letter-spacing:.05em;}
.badge.pass{background:var(--pass);} .badge.fail{background:var(--fail);} .badge.gafail{background:#d39e00;color:#fff;}
.btn.gafail.active{background:#d39e00;border-color:#d39e00;}
.ecard.g{border-left:3px solid #d39e00;}
.gate-a-note{font-size:.72rem;line-height:1.45;padding:6px 10px;margin:0 8px 8px;background:rgba(211,158,0,.08);border:1px solid rgba(211,158,0,.35);border-radius:4px;color:#e8d099;}
.gate-a-note .why{font-size:.65rem;color:#bba56a;text-transform:uppercase;letter-spacing:.06em;display:block;margin-bottom:3px;}
.chip{font-size:.65rem;font-weight:600;padding:1px 7px;border-radius:3px;color:#fff;letter-spacing:.04em;text-transform:uppercase;}
.eid{font-size:.66rem;color:var(--sub);margin-left:auto;font-family:'JetBrains Mono',monospace;}
.score{font-size:.7rem;color:var(--sub);font-family:'JetBrains Mono',monospace;}
.ebody{padding:8px 12px;display:flex;flex-direction:column;gap:6px;}
.prompt{font-size:.84rem;color:#e8e8ec;font-style:italic;line-height:1.45;padding:4px 0;}
.tgt{font-size:.7rem;color:var(--sub);}
.reason{font-size:.76rem;line-height:1.45;padding:5px 9px;border-radius:4px;}
.reason.p{color:#b8e0ae;background:rgba(39,174,96,.08);}
.reason.f{color:#e8b8b8;background:rgba(192,57,43,.08);}
.thumbs{display:grid;grid-template-columns:repeat(4,1fr);gap:4px;padding:0 8px 8px;}
.thumbs img{width:100%;height:auto;border-radius:3px;cursor:pointer;background:#000;}
.thumbs img:hover{outline:1px solid var(--ac);}
.vlm-input{padding:0 10px 10px;display:flex;flex-direction:column;gap:6px;}
.vlm-input .cap{font-size:.66rem;color:var(--sub);text-transform:uppercase;letter-spacing:.06em;display:flex;align-items:center;gap:6px;margin-top:2px;}
.vlm-input .cap .row-tag{padding:0 6px;border-radius:3px;background:#13131a;border:1px solid var(--bd);}
.vlm-input img.collage{width:100%;height:auto;display:block;border-radius:4px;cursor:pointer;background:#000;border:1px solid var(--bd);}
.vlm-input img.collage:hover{outline:1px solid var(--ac);}
.vlm-input details{background:#0a0a0e;border:1px solid var(--bd);border-radius:4px;}
.vlm-input details summary{cursor:pointer;padding:5px 9px;font-size:.7rem;color:var(--sub);user-select:none;}
.vlm-input details[open] summary{color:var(--ac);}
.vlm-input pre{font-family:'JetBrains Mono',monospace;font-size:.72rem;color:#cfd;line-height:1.45;padding:6px 10px;white-space:pre-wrap;word-break:break-word;border-top:1px solid var(--bd);margin:0;}
.sys-prompt{background:var(--s1);border:1px solid var(--bd);border-radius:6px;margin-bottom:14px;}
.sys-prompt summary{cursor:pointer;padding:8px 12px;font-size:.78rem;color:var(--ac);user-select:none;font-weight:600;}
.sys-prompt[open] summary{border-bottom:1px solid var(--bd);}
.sys-prompt pre{font-family:'JetBrains Mono',monospace;font-size:.72rem;color:#cfd;line-height:1.5;padding:10px 16px;white-space:pre-wrap;margin:0;}
#lightbox{position:fixed;inset:0;background:rgba(0,0,0,.94);z-index:100;display:none;align-items:center;justify-content:center;cursor:pointer;}
#lightbox.show{display:flex;}
#lightbox img{max-width:96vw;max-height:96vh;object-fit:contain;}
"""

JS = r"""
const filterState = {verdict:'all', type:'all'};
function applyFilter(){
  const cards = document.querySelectorAll('.ecard');
  let shown=0;
  cards.forEach(c=>{
    const okV = filterState.verdict==='all' || filterState.verdict===c.dataset.verdict;
    const okT = filterState.type==='all' || filterState.type===c.dataset.type;
    if(okV&&okT){c.classList.remove('hide');shown++;} else c.classList.add('hide');
  });
  document.querySelectorAll('.osec').forEach(s=>{
    s.style.display = s.querySelector('.ecard:not(.hide)') ? '' : 'none';
  });
  document.getElementById('shown-count').textContent = shown;
}
function setVerdict(v){filterState.verdict=v;document.querySelectorAll('.btn-v').forEach(b=>b.classList.toggle('active',b.dataset.v===v));applyFilter();}
function setType(t){filterState.type=t;document.querySelectorAll('.btn-t').forEach(b=>b.classList.toggle('active',b.dataset.t===t));applyFilter();}
function lightbox(src){const lb=document.getElementById('lightbox');lb.querySelector('img').src=src;lb.classList.add('show');}
document.addEventListener('DOMContentLoaded',()=>{
  document.getElementById('lightbox').addEventListener('click',e=>e.currentTarget.classList.remove('show'));
  document.addEventListener('keydown',e=>{if(e.key==='Escape')document.getElementById('lightbox').classList.remove('show');});
});
"""

# ── data collection ─────────────────────────────────────────────────────────
def collect(root: Path, today_prefix: str = "2026-04-18", shard: str = "08",
            *, include_gate_a_reject: bool = False):
    """Walk ``<root>/objects/<shard>/<obj_id>/`` and collect today's gate_e
    judgements plus enough parsed-edit context to reconstruct the VLM
    user message.
    """
    objects_dir = root / "objects" / shard
    if not objects_dir.is_dir():
        sys.exit(f"objects/{shard} not found under {root}")
    by_obj: dict[str, list] = defaultdict(list)
    obj_descs: dict[str, str] = {}
    inh = Counter(); vlm_n = Counter(); vlm_p = Counter()
    ga_reject = Counter(); no_output = Counter()

    for od in sorted(objects_dir.iterdir()):
        if not od.is_dir():
            continue
        es_path = od / "edit_status.json"
        if not es_path.is_file():
            continue
        es = json.loads(es_path.read_text())

        # Build flux-seq -> parsed-edit lookup, plus id -> part-name map.
        parsed_path = od / "phase1" / "parsed.json"
        seq2pe: dict[int, dict] = {}
        parts_by_id: dict[int, dict] = {}
        object_desc = ""
        if parsed_path.is_file():
            pd = json.loads(parsed_path.read_text())
            parsed = pd.get("parsed") or {}
            obj = parsed.get("object") or {}
            object_desc = obj.get("full_desc", "") or ""
            parts_by_id = {p["part_id"]: p for p in (obj.get("parts") or [])
                           if isinstance(p, dict) and "part_id" in p}
            plist = parsed.get("edits") or []
            fs = -1
            for pe in plist:
                if pe.get("edit_type", "?") in FLUX_TYPES:
                    fs += 1
                    seq2pe[fs] = pe
        obj_descs[od.name] = object_desc

        for eid, e in (es.get("edits") or {}).items():
            et = e.get("edit_type", "?")
            ge = (e.get("stages") or {}).get("gate_e") or {}
            ts = ge.get("ts", "")
            if not ts.startswith(today_prefix):
                continue
            if ge.get("reason", "") == "inherited_from_gate_a":
                inh[et] += 1
                continue
            st = ge.get("status", "?")
            if st not in ("pass", "fail"):
                continue
            ge_vlm = ((e.get("gates") or {}).get("E") or {}).get("vlm") or {}
            ga_vlm = ((e.get("gates") or {}).get("A") or {}).get("vlm") or {}
            # Classify the source of the verdict so the report can stop
            # mislabelling upstream gate_a rejections as 'missing_previews':
            #   * 'gate_a_reject' — gate_a marked .vlm.pass=false; FLUX/trellis
            #     never ran, no previews exist anywhere, the empty gate_e
            #     reason is just a bookkeeping placeholder.
            #   * 'no_output' — gate_a passed but previews are still missing
            #     (true mid-pipeline drop, e.g. trellis crash).
            #   * 'vlm_judged' — gate_e really called the VLM with a collage.
            preview_dir = od / "edits_3d" / eid
            n_prev = sum(1 for i in range(5) if (preview_dir / f"preview_{i}.png").is_file())
            if n_prev < 5 and ga_vlm.get("pass") is False:
                kind = "gate_a_reject"
            elif n_prev < 5:
                kind = "no_output"
            else:
                kind = "vlm_judged"
            # Quality-focused view: gate_a rejects never reached the
            # quality judge, so when callers opt out (the default) we drop
            # them outright -- no card, no counter, no header stat.
            if kind == "gate_a_reject" and not include_gate_a_reject:
                continue
            # Counters: keep vlm_n/vlm_p meaningful (true gate_e judgements
            # only). gate_a rejections / no-output drops get separate buckets
            # surfaced in the header stats.
            if kind == "vlm_judged":
                vlm_n[et] += 1
                if st == "pass":
                    vlm_p[et] += 1
            else:
                bucket = ga_reject if kind == "gate_a_reject" else no_output
                bucket[et] += 1
            seq = int(eid.rsplit("_", 1)[-1]) if eid.rsplit("_", 1)[-1].isdigit() else -1
            pe = seq2pe.get(seq) or {}
            sel_pids = list(pe.get("selected_part_ids") or [])
            part_labels = [parts_by_id.get(pid, {}).get("name", "") for pid in sel_pids]
            by_obj[od.name].append({
                "eid": eid, "edit_type": et, "status": st, "kind": kind,
                "score": ge_vlm.get("score"), "reason": ge_vlm.get("reason", ""),
                "gate_a_pass": ga_vlm.get("pass"),
                "gate_a_reason": ga_vlm.get("reason", ""),
                "_seq": seq,
                "preview_dir": preview_dir,
                "prompt": pe.get("prompt", ""),
                "target_part_desc": pe.get("target_part_desc", ""),
                "after_desc": pe.get("after_desc") or "",
                "selected_part_ids": sel_pids,
                "part_labels": part_labels,
                "edit_params": dict(pe.get("edit_params") or {}),
                "object_desc": object_desc,
            })
    return by_obj, obj_descs, inh, vlm_n, vlm_p, ga_reject, no_output

# ── rendering ───────────────────────────────────────────────────────────────
def _append_context_strips(p: list, ed: dict, before_imgs) -> None:
    """Embed BEFORE collage + any partial AFTER previews into card *p*.

    Used for gate_a_reject and no_output cards so every card in the
    report carries inline imagery (no broken file:// references).
    """
    before_uri = build_before_only_collage_uri(before_imgs)
    preview_dir = ed.get("preview_dir")
    partial_uri, n_partial = (build_partial_preview_strip_uri(preview_dir)
                              if preview_dir else (None, 0))
    obj_dir = preview_dir.parent.parent if preview_dir else None
    overview_uri = encode_overview_uri(obj_dir) if obj_dir else None

    if before_uri:
        p.append("<div class='vlm-input'>"
                 "<div class='cap'><span class='row-tag'>BEFORE</span> "
                 "5 source views (no AFTER — see card text above)</div>"
                 f"<img class='collage' loading='lazy' src='{before_uri}' "
                 f"onclick=\"lightbox('{before_uri}')\" alt='before views'>"
                 "</div>")
    if partial_uri:
        p.append("<div class='vlm-input'>"
                 f"<div class='cap'><span class='row-tag'>AFTER (partial)</span> "
                 f"only {n_partial}/5 preview frames produced</div>"
                 f"<img class='collage' loading='lazy' src='{partial_uri}' "
                 f"onclick=\"lightbox('{partial_uri}')\" alt='partial after'>"
                 "</div>")
    if not before_uri and overview_uri:
        # Fallback: gate_a's part-overview rendering when source NPZ isn't
        # available (e.g. images_root not provided).
        p.append("<div class='vlm-input'>"
                 "<div class='cap'><span class='row-tag'>OVERVIEW</span> "
                 "phase1 part-segmentation overview (gate_a input)</div>"
                 f"<img class='collage' loading='lazy' src='{overview_uri}' "
                 f"onclick=\"lightbox('{overview_uri}')\" alt='phase1 overview'>"
                 "</div>")


def render(by_obj, obj_descs, inh, vlm_n, vlm_p,
           ga_reject, no_output,
           root: Path, out_path: Path,
           *, embed: bool = False,
           images_root: "Path | None" = None,
           shard: str = "08",
           show_vlm_input: bool = True):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    total = sum(vlm_n.values()); total_p = sum(vlm_p.values()); total_inh = sum(inh.values())
    total_gar = sum(ga_reject.values()); total_no = sum(no_output.values())
    p: list[str] = []
    p.append(f"<!DOCTYPE html><html lang='en'><head><meta charset='UTF-8'><title>Gate E (gate_quality) Report</title><style>{CSS}</style></head><body>")
    p.append("<h1>Gate E (gate_quality) — VLM-input view</h1>")
    rel = root.relative_to(Path("/mnt/zsn/zsn_workspace/PartCraft3D")) if str(root).startswith("/mnt/zsn/zsn_workspace/PartCraft3D") else root
    p.append(f"<p class='sub'>Each card shows the EXACT 2x5 BEFORE/AFTER collage and per-edit user message the judge VLM received, alongside its verdict. Output: <code>{escape(str(rel))}</code></p>")

    # JUDGE_SYSTEM_PROMPT (collapsible — same prefix for every call)
    p.append("<details class='sys-prompt'><summary>JUDGE_SYSTEM_PROMPT (shared across all edits)</summary>"
             f"<pre>{escape(JUDGE_SYSTEM_PROMPT)}</pre></details>")

    # stats
    p.append("<div class='stats'>")
    p.append(f"<div class='stat'><div class='v'>{len(by_obj)}</div><div class='l'>objects</div></div>")
    p.append(f"<div class='stat'><div class='v'>{total}</div><div class='l'>VLM-judged</div></div>")
    p.append(f"<div class='stat'><div class='v pass'>{total_p}</div><div class='l'>pass</div></div>")
    p.append(f"<div class='stat'><div class='v fail'>{total-total_p}</div><div class='l'>fail</div></div>")
    rate = (total_p/total*100) if total else 0
    p.append(f"<div class='stat'><div class='v'>{rate:.1f}%</div><div class='l'>pass rate</div></div>")
    p.append(f"<div class='stat'><div class='v inh'>{total_inh}</div><div class='l'>inherited (no VLM)</div></div>")
    p.append(f"<div class='stat'><div class='v' style='color:#d39e00'>{total_gar}</div>"
             "<div class='l'>gate_a rejected (no gate_e VLM call)</div></div>")
    if total_no:
        p.append(f"<div class='stat'><div class='v fail'>{total_no}</div>"
                 "<div class='l'>no output (mid-pipeline drop)</div></div>")
    p.append("</div>")

    # bars
    p.append("<div class='bar-section'><h3>Pass / Fail by edit_type (today's VLM judgements only)</h3>")
    types_sorted = sorted(vlm_n.keys(), key=lambda t: -vlm_n[t])
    max_n = max(vlm_n.values()) if vlm_n else 1
    for t in types_sorted:
        n = vlm_n[t]; pp = vlm_p[t]; ff = n - pp
        rate2 = (pp/n*100) if n else 0
        bar_w = n / max_n * 60
        p_inner = (pp/n*100) if n else 0
        p.append(f"<div class='bar-row'><span class='lbl'>{escape(t)}</span>"
                 f"<div class='bar' style='width:{bar_w}%;'>"
                 f"<div class='bp' style='width:{p_inner:.1f}%'></div>"
                 f"<div class='bf' style='flex:1'></div></div>"
                 f"<span class='nums'>{pp}p / {ff}f / {n}t · {rate2:.1f}%</span></div>")
    p.append("</div>")

    if inh:
        p.append("<div class='bar-section'><h3>Inherited from Gate A (deletion/addition — no VLM judge)</h3>")
        max_inh = max(inh.values())
        for t, n in sorted(inh.items()):
            p.append(f"<div class='bar-row'><span class='lbl'>{escape(t)}</span>"
                     f"<div class='bar' style='width:{n/max_inh*60}%;background:var(--inh);'></div>"
                     f"<span class='nums'>{n} edits inherited pass</span></div>")
        p.append("</div>")
    if ga_reject:
        p.append("<div class='bar-section'><h3>Rejected upstream at Gate A "
                 "(flux types — gate_e was never called, no previews exist)</h3>")
        max_g = max(ga_reject.values())
        for t, n in sorted(ga_reject.items(), key=lambda kv: -kv[1]):
            p.append(f"<div class='bar-row'><span class='lbl'>{escape(t)}</span>"
                     f"<div class='bar' style='width:{n/max_g*60}%;background:#d39e00;'></div>"
                     f"<span class='nums'>{n} edits rejected at gate_a</span></div>")
        p.append("</div>")

    # filters
    p.append("<div class='controls'>")
    p.append("<span class='btn btn-v active' data-v='all' onclick=\"setVerdict('all')\">All verdicts</span>")
    p.append("<span class='btn btn-v pass' data-v='pass' onclick=\"setVerdict('pass')\">Pass only</span>")
    p.append("<span class='btn btn-v fail' data-v='fail' onclick=\"setVerdict('fail')\">Fail only</span>")
    p.append("<span class='btn btn-v gafail' data-v='gate_a_reject' onclick=\"setVerdict('gate_a_reject')\">Gate-A rejected</span>")
    p.append("<span style='color:var(--sub);font-size:.78rem;margin:0 6px;'>·</span>")
    p.append("<span class='btn btn-t active' data-t='all' onclick=\"setType('all')\">All types</span>")
    for t in types_sorted:
        p.append(f"<span class='btn btn-t' data-t='{escape(t)}' onclick=\"setType('{escape(t)}')\">{escape(t)} ({vlm_n[t]})</span>")
    grand = total + total_gar + total_no
    p.append(f"<span style='margin-left:auto;color:var(--sub);font-size:.78rem;'>"
             f"showing <span id='shown-count'>{grand}</span> / {grand} "
             f"(<span style='color:var(--ac)'>{total}</span> VLM-judged · "
             f"<span style='color:#d39e00'>{total_gar}</span> gate_a-rejected"
             + (f" · <span style='color:var(--fail)'>{total_no}</span> no-output" if total_no else "")
             + ")</span></div>")

    # cards per object
    n_collage = 0
    for obj_id in sorted(by_obj.keys()):
        edits = by_obj[obj_id]
        n = len(edits)
        n_vlm = sum(1 for e in edits if e["kind"] == "vlm_judged")
        np_ = sum(1 for e in edits if e["kind"] == "vlm_judged" and e["status"] == "pass")
        n_gar = sum(1 for e in edits if e["kind"] == "gate_a_reject")
        n_no  = sum(1 for e in edits if e["kind"] == "no_output")
        # Lazy-load BEFORE views once per object (cached in module-level dict).
        before_imgs = []
        if show_vlm_input and images_root is not None:
            before_imgs = load_before_views(images_root / shard / f"{obj_id}.npz")

        p.append(f"<section class='osec'><div class='ohdr'>"
                 f"<span class='oid'>{escape(obj_id)}</span>"
                 f"<span class='ostats'>{n} edits · "
                 f"<span style='color:var(--pass)'>{np_}</span>p / "
                 f"<span style='color:var(--fail)'>{n_vlm-np_}</span>f"
                 + (f" · <span style='color:#d39e00'>{n_gar}</span> gate_a-rej" if n_gar else "")
                 + (f" · {n_no} no-output" if n_no else "")
                 + f" · gate_e {(np_/n_vlm*100 if n_vlm else 0):.0f}%</span></div><div class='egrid'>")
        edits.sort(key=lambda e: (e["edit_type"], e["_seq"]))
        for ed in edits:
            chip = EDIT_TYPE_CHIP_COLOR.get(ed["edit_type"], "#666")
            kind = ed.get("kind", "vlm_judged")
            if kind == "gate_a_reject":
                cls = "g"; badge_cls = "gafail"; badge_text = "GATE-A FAIL"
                # data-verdict='gate_a_reject' so the new filter can target it
                verdict_attr = "gate_a_reject"
            elif kind == "no_output":
                cls = "f"; badge_cls = "fail"; badge_text = "NO OUTPUT"
                verdict_attr = "fail"
            else:
                cls = "p" if ed["status"] == "pass" else "f"
                badge_cls = ed["status"]; badge_text = ed["status"].upper()
                verdict_attr = ed["status"]
            score = f"score={ed['score']:.2f}" if ed.get("score") is not None and kind == "vlm_judged" else ""
            ga = ed.get("gate_a_pass")
            ga_html = ""
            if kind != "gate_a_reject":  # already self-evident in that case
                if ga is True:
                    ga_html = "<span style='color:#888;'>· gate_A: <span style='color:#9c9;'>pass</span></span>"
                elif ga is False:
                    ga_html = "<span style='color:#888;'>· gate_A: <span style='color:#e88;'>fail</span></span>"
            short_eid = ed['eid'].split('_')[0] + '_…_' + ed['eid'].rsplit('_', 1)[-1]
            p.append(f"<div class='ecard {cls}' data-verdict='{verdict_attr}' data-type='{escape(ed['edit_type'])}'>"
                     f"<div class='ehdr'>"
                     f"<span class='badge {badge_cls}'>{badge_text}</span>"
                     f"<span class='chip' style='background:{chip}'>{escape(ed['edit_type'])}</span>"
                     f"<span class='score'>{score} {ga_html}</span>"
                     f"<span class='eid'>{escape(short_eid)}</span>"
                     f"</div><div class='ebody'>"
                     f"<div class='prompt'>“{escape(ed.get('prompt') or '(no prompt)')}”</div>")
            if ed.get("target_part_desc"):
                p.append(f"<div class='tgt'>target: {escape(ed['target_part_desc'])}</div>")
            if ed.get("after_desc"):
                p.append(f"<div class='tgt' style='color:#6ab;'>after: {escape(ed['after_desc'])}</div>")
            if kind == "gate_a_reject":
                ga_reason = ed.get("gate_a_reason") or "(no reason recorded)"
                p.append("</div>"
                         "<div class='gate-a-note'>"
                         "<span class='why'>Why no full collage?</span>"
                         "Upstream <b>gate_a</b> rejected this edit (text↔image alignment failure), "
                         "so FLUX 2D / Trellis 3D were never run and there are no full AFTER previews. "
                         "<b>gate_e</b> was never called for this entry — BEFORE views (and any partial "
                         "AFTER previews that did get produced) are shown below for visual context."
                         f"<div style='margin-top:6px;'><b>gate_a reason:</b> {escape(ga_reason)}</div>"
                         "</div>")
                _append_context_strips(p, ed, before_imgs)
                p.append("</div>")  # close .ecard
                continue
            elif kind == "no_output":
                p.append(f"<div class='reason f'>"
                         f"No (full) preview_*.png produced even though gate_a passed "
                         f"(probable mid-pipeline drop in FLUX or Trellis). "
                         f"gate_e could not be evaluated. BEFORE views and any partial "
                         f"previews are embedded below for context.</div>")
                p.append("</div>")  # close .ebody
                _append_context_strips(p, ed, before_imgs)
                p.append("</div>")  # close .ecard
                continue
            p.append(f"<div class='reason {cls}'>{escape(ed.get('reason') or '(no reason from VLM)')}</div>")
            p.append("</div>")

            preview_dir = ed["preview_dir"]
            collage_uri = (build_vlm_collage_uri(before_imgs, preview_dir)
                           if show_vlm_input and before_imgs else None)
            if collage_uri:
                # Replace the after-only thumb strip with the actual VLM input
                # collage (5 BEFORE on top, 5 AFTER on bottom, same camera).
                n_collage += 1
                user_msg = vlm_user_message(ed)
                p.append("<div class='vlm-input'>"
                         "<div class='cap'><span class='row-tag'>VLM input</span> "
                         "top = BEFORE (5 views) · bottom = AFTER (5 views, same camera per column)</div>"
                         f"<img class='collage' loading='lazy' src='{collage_uri}' "
                         f"onclick=\"lightbox('{collage_uri}')\" alt='vlm collage'>"
                         "<details><summary>VLM user message (per-edit)</summary>"
                         f"<pre>{escape(user_msg)}</pre></details>"
                         "</div>")
            else:
                # Fall back to the original 4-thumb after-only strip.
                p.append("<div class='thumbs'>")
                for v in (0, 1, 2, 3):
                    pp_ = preview_dir / f"preview_{v}.png"
                    if pp_.is_file():
                        img_src = encode_thumb(pp_)
                        p.append(f"<img loading='lazy' src='{escape(img_src)}' "
                                 f"onclick=\"lightbox('{escape(img_src)}')\" alt='view {v}'>")
                    else:
                        p.append(f"<div style='background:#222;aspect-ratio:1;display:flex;align-items:center;justify-content:center;color:#555;font-size:.7rem;'>no v{v}</div>")
                p.append("</div>")
            p.append("</div>")  # close .ecard
        p.append("</div></section>")

    p.append("<div id='lightbox'><img src=''></div>")
    p.append(f"<script>{JS}</script></body></html>")
    out_path.write_text("\n".join(p), encoding="utf-8")
    return n_collage


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True,
                    help="Run output dir, e.g. .../bench_*/mode_e_text_align")
    ap.add_argument("--out", required=True)
    ap.add_argument("--today", default="2026-04-18")
    ap.add_argument("--shard", default="08",
                    help="Shard folder under <root>/objects/ (default: 08)")
    ap.add_argument("--images-root", default="/mnt/zsn/data/partverse/bench/inputs/images",
                    help="Root holding <shard>/<obj_id>.npz NPZs of source views. "
                         "Required to reconstruct the BEFORE row of the VLM collage. "
                         "Pass empty string to disable the VLM-input panel.")
    ap.add_argument("--embed", action="store_true",
                    help="(Deprecated; embedding is always on.) Kept for "
                         "backward compatibility. ALL thumbnails, BEFORE/AFTER "
                         "collages, partial previews, and gate_a-context strips "
                         "are embedded as base64 data URIs unconditionally so "
                         "the report is fully portable as a single .html file.")
    ap.add_argument("--include-gate-a-reject", action="store_true",
                    help="Include cards/stats for edits rejected at gate_a "
                         "(text\u2194image alignment) that never reached "
                         "gate_quality. Off by default so the report focuses "
                         "purely on gate_quality verdicts.")
    ap.add_argument("--no-vlm-input", action="store_true",
                    help="Disable the new VLM-input collage panel and revert to "
                         "the original after-only 4-thumb strip.")
    a = ap.parse_args()
    root = Path(a.root).resolve(); out = Path(a.out).resolve()
    images_root = Path(a.images_root).resolve() if a.images_root and not a.no_vlm_input else None
    show_vlm_input = (not a.no_vlm_input) and images_root is not None
    by_obj, obj_descs, inh, vlm_n, vlm_p, ga_reject, no_output = collect(
        root, a.today, shard=a.shard,
        include_gate_a_reject=a.include_gate_a_reject)
    n_collage = render(by_obj, obj_descs, inh, vlm_n, vlm_p,
                       ga_reject, no_output, root, out,
                       embed=a.embed, images_root=images_root,
                       shard=a.shard, show_vlm_input=show_vlm_input)
    print(f"wrote {out}")
    extras = []
    if show_vlm_input:
        extras.append(f"{n_collage} VLM-input collages")
    if a.embed and _THUMB_CACHE:
        extras.append(f"{len(_THUMB_CACHE)} fallback thumbnails")
    suffix = f"  ({', '.join(extras)})" if extras else ""
    print(f"  {len(by_obj)} objects, {sum(vlm_n.values())} VLM-judged, "
          f"{sum(ga_reject.values())} gate_a-rejected, "
          f"{sum(no_output.values())} no-output, "
          f"{sum(inh.values())} inherited{suffix}")
