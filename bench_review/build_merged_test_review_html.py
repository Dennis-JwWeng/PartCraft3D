#!/usr/bin/env python3
"""Build chunked self-contained HTML files for H3D merged-test review."""
from __future__ import annotations

import argparse
import base64
import html
import json
import re
from pathlib import Path
from typing import Mapping

FLUX_TYPES = {"modification", "scale", "material", "color", "global"}
DEFAULT_MANIFEST = Path("/mnt/zsn/zsn_workspace/PartCraft3D/bench_review/merged_test_2000/h3d_test_merged_2000_manifest.jsonl")
DEFAULT_OUT_DIR = Path("/mnt/zsn/zsn_workspace/PartCraft3D/bench_review/merged_test_2000_html")


def prompt_to_zh(prompt: str) -> str:
    text = prompt.strip().rstrip(".")
    low = text.lower()
    patterns = [
        (r"^add the (.+)$", "添加{0}"),
        (r"^add (.+)$", "添加{0}"),
        (r"^remove the (.+)$", "移除{0}"),
        (r"^remove (.+)$", "移除{0}"),
        (r"^replace the (.+) with (.+)$", "将{0}替换为{1}"),
        (r"^replace (.+) with (.+)$", "将{0}替换为{1}"),
        (r"^change the (.+) to (.+)$", "将{0}改为{1}"),
        (r"^change (.+) to (.+)$", "将{0}改为{1}"),
        (r"^repaint the (.+) (.+)$", "将{0}重新上色为{1}"),
        (r"^render the entire object in (.+)$", "将整个物体渲染为{0}"),
        (r"^render the object as (.+)$", "将物体渲染为{0}"),
        (r"^render the object in (.+)$", "将物体渲染为{0}"),
    ]
    for pattern, template in patterns:
        match = re.match(pattern, low, flags=re.IGNORECASE)
        if match:
            return template.format(*match.groups())
    return f"请按英文指令完成编辑：{prompt.strip()}"


def data_uri(path: str) -> str:
    p = Path(path)
    encoded = base64.b64encode(p.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def load_records(manifest_path: Path, *, start: int, limit: int) -> list[dict]:
    records: list[dict] = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if index < start:
                continue
            if len(records) >= limit:
                break
            if not line.strip():
                continue
            row = json.loads(line)
            meta = json.loads(Path(row["h3d_meta_json"]).read_text(encoding="utf-8"))
            prompt_en = (meta.get("instruction") or {}).get("prompt") or ""
            record = dict(row)
            record["prompt_en"] = prompt_en
            record["prompt_zh"] = prompt_to_zh(prompt_en)
            record["h3d_before_data"] = data_uri(row["h3d_before_png"])
            record["h3d_after_data"] = data_uri(row["h3d_after_png"])
            if row["edit_type"] in FLUX_TYPES:
                record["two_d_input_data"] = data_uri(row["two_d_input_png"])
                record["two_d_edited_data"] = data_uri(row["two_d_edited_png"])
            else:
                record["two_d_input_data"] = ""
                record["two_d_edited_data"] = ""
            records.append(record)
    return records


def esc(value: object) -> str:
    return html.escape(str(value or ""), quote=True)


def figure(label: str, data: str) -> str:
    return f'''<figure><figcaption>{esc(label)}</figcaption><img src="{esc(data)}" alt="{esc(label)}"></figure>'''


def render_card(record: Mapping, index: int) -> str:
    figures = [
        figure("3D before", str(record.get("h3d_before_data", ""))),
        figure("3D after", str(record.get("h3d_after_data", ""))),
    ]
    if record.get("edit_type") in FLUX_TYPES:
        figures.extend([
            figure("2D input", str(record.get("two_d_input_data", ""))),
            figure("2D edited", str(record.get("two_d_edited_data", ""))),
        ])
    grid_class = "grid four" if record.get("edit_type") in FLUX_TYPES else "grid two"
    edit_id = esc(record.get("edit_id"))
    reasons = [
        "图像缺失或加载失败",
        "编辑不符合指令",
        "目标区域错误",
        "变化太弱或无变化",
        "几何/外观严重崩坏",
        "2D edit 可用但 3D 不一致",
        "其他",
    ]
    reason_html = "".join(
        f'<label><input type="checkbox" name="reason-{edit_id}" value="{esc(reason)}"> {esc(reason)}</label>'
        for reason in reasons
    )
    return f'''
<article class="card unreviewed" data-record-card="true" data-edit-id="{edit_id}" data-edit-type="{esc(record.get('edit_type'))}" data-status="unreviewed">
  <div class="meta">
    <span class="pill">#{index + 1}</span>
    <span class="pill type">{esc(record.get('edit_type'))}</span>
    <span class="pill">edit_id: {edit_id}</span>
    <span class="pill">obj_id: {esc(record.get('obj_id'))}</span>
    <span class="pill">shard: {esc(record.get('shard'))}</span>
    <span class="pill">source: {esc(record.get('source_hf_split'))}</span>
  </div>
  <div class="prompt zh"><strong>中文：</strong>{esc(record.get('prompt_zh'))}</div>
  <div class="prompt en"><strong>English:</strong> {esc(record.get('prompt_en'))}</div>
  <div class="{grid_class}">{''.join(figures)}</div>
  <div class="decision" role="radiogroup">
    <label><input type="radio" name="decision-{edit_id}" value="keep"> 要</label>
    <label><input type="radio" name="decision-{edit_id}" value="reject"> 不要</label>
    <label><input type="radio" name="decision-{edit_id}" value="unsure"> 不确定</label>
    <label><input type="radio" name="decision-{edit_id}" value="unreviewed" checked> 未标</label>
  </div>
  <div class="reasons"><div class="reason-title">拒绝/疑问原因：</div>{reason_html}</div>
  <textarea placeholder="备注：记录具体问题、视角疑问或保留理由"></textarea>
</article>'''


def write_html(records: list[dict], output_path: Path, *, chunk_index: int, total_chunks: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps([
        {
            "edit_id": r["edit_id"],
            "edit_type": r["edit_type"],
            "obj_id": r["obj_id"],
            "shard": r["shard"],
            "source_hf_split": r["source_hf_split"],
            "prompt_en": r["prompt_en"],
            "prompt_zh": r["prompt_zh"],
        }
        for r in records
    ], ensure_ascii=False)
    cards = "\n".join(render_card(record, i) for i, record in enumerate(records))
    html_text = f'''<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>H3D Test Review {chunk_index:03d}</title>
  <style>
    :root {{ color-scheme: light dark; --bg:#f6f7fb; --card:#fff; --text:#17202a; --muted:#64748b; --border:#d7dde8; --accent:#2563eb; }}
    @media (prefers-color-scheme: dark) {{ :root {{ --bg:#0f172a; --card:#111827; --text:#e5e7eb; --muted:#94a3b8; --border:#334155; }} }}
    * {{ box-sizing: border-box; }} body {{ margin:0; background:var(--bg); color:var(--text); font-family:system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }}
    header {{ position:sticky; top:0; z-index:3; background:color-mix(in srgb,var(--bg) 90%,transparent); backdrop-filter:blur(10px); border-bottom:1px solid var(--border); }}
    .wrap, main {{ max-width:1500px; margin:0 auto; padding:16px 22px; }} h1 {{ margin:0 0 8px; font-size:24px; }} .subtle {{ color:var(--muted); line-height:1.5; }}
    .toolbar {{ display:flex; flex-wrap:wrap; gap:10px; align-items:center; margin-top:12px; }} button {{ border:1px solid var(--border); border-radius:9px; padding:8px 11px; background:var(--card); color:var(--text); cursor:pointer; }} button.active,button.primary {{ background:var(--accent); border-color:var(--accent); color:white; }}
    .criteria,.card {{ background:var(--card); border:1px solid var(--border); border-radius:14px; padding:16px; margin:16px 0; }} .criteria ul {{ margin:8px 0 0 20px; padding:0; }} .criteria li {{ margin:5px 0; }}
    .meta {{ display:flex; flex-wrap:wrap; gap:8px; margin-bottom:10px; }} .pill {{ border:1px solid var(--border); border-radius:999px; padding:4px 9px; color:var(--muted); font-size:13px; }} .type {{ color:var(--accent); font-weight:700; }}
    .prompt {{ font-size:17px; line-height:1.45; margin:8px 0; }} .zh {{ font-size:19px; }}
    .grid {{ display:grid; gap:12px; margin-top:14px; }} .grid.two {{ grid-template-columns:repeat(2,1fr); }} .grid.four {{ grid-template-columns:repeat(4,1fr); }}
    figure {{ margin:0; border:1px solid var(--border); border-radius:12px; overflow:hidden; background:#0b1020; }} figcaption {{ padding:8px 10px; background:#111827; color:#fff; font-weight:700; }} img {{ display:block; width:100%; height:auto; }}
    .decision,.reasons {{ display:flex; flex-wrap:wrap; gap:14px; align-items:center; margin-top:14px; }} .reason-title {{ width:100%; color:var(--muted); }} label {{ display:inline-flex; gap:6px; align-items:center; }} textarea {{ width:100%; min-height:54px; margin-top:10px; border:1px solid var(--border); border-radius:10px; padding:10px; background:transparent; color:var(--text); }}
    .hidden {{ display:none; }} .stats {{ font-weight:700; }}
    @media (max-width:1100px) {{ .grid.four {{ grid-template-columns:repeat(2,1fr); }} }} @media (max-width:760px) {{ .grid.two,.grid.four {{ grid-template-columns:1fr; }} header {{ position:static; }} }}
  </style>
</head>
<body>
<header><div class="wrap">
  <h1>H3D Test Bench 人工筛选 · {chunk_index + 1}/{total_chunks}</h1>
  <div class="subtle">每页 100 条。del/add 展示 3D before/after；flux 类型额外展示 2D input/edited。</div>
  <div class="toolbar"><span class="stats" id="stats">Total: {len(records)}</span><button class="active" data-filter="all">全部</button><button data-filter="unreviewed">未标</button><button data-filter="keep">要</button><button data-filter="reject">不要</button><button data-filter="unsure">不确定</button><button class="primary" id="downloadJson">导出 review_results_{chunk_index:03d}.json</button><button class="primary" id="downloadKeep">导出 selected_edit_ids_{chunk_index:03d}.txt</button></div>
</div></header>
<main>
<section class="criteria"><h2>标注要求</h2><ul><li><strong>要：</strong>图片清晰，编辑结果和指令一致，主体没有严重崩坏。</li><li><strong>del/add：</strong>只判断 3D before/after；删除应移除目标，添加应出现目标且位置尺度合理。</li><li><strong>flux：</strong>同时参考 2D input/edited 和 3D before/after；2D 表达目标变化，3D after 语义上继承该效果即可，2D/3D 视角不一致可接受。</li><li><strong>不要：</strong>目标错、几乎无变化、严重崩坏、图像异常或指令无法判断。</li><li><strong>不确定：</strong>变化弱、遮挡、视角造成难判时先标不确定。</li></ul></section>
<section id="cards">{cards}</section>
</main>
<script id="records" type="application/json">{payload}</script>
<script>
const records = JSON.parse(document.getElementById('records').textContent);
const state = new Map(records.map(r => [r.edit_id, {{decision:'unreviewed', reasons:[], note:''}}]));
let activeFilter = 'all';
function updateStats() {{ const vals=[...state.values()]; const c=s=>vals.filter(v=>v.decision===s).length; document.getElementById('stats').textContent=`Total: ${{records.length}} · 已标: ${{c('keep')+c('reject')+c('unsure')}} · 要: ${{c('keep')}} · 不要: ${{c('reject')}} · 不确定: ${{c('unsure')}} · 未标: ${{c('unreviewed')}}`; }}
function applyFilter() {{ document.querySelectorAll('.card').forEach(card => {{ const show = activeFilter === 'all' || card.dataset.status === activeFilter; card.classList.toggle('hidden', !show); }}); }}
function collectResults() {{ return records.map(r => ({{...r, decision: state.get(r.edit_id).decision, reject_reasons: state.get(r.edit_id).reasons, note: state.get(r.edit_id).note}})); }}
function download(filename, text, type='text/plain') {{ const blob=new Blob([text],{{type}}); const url=URL.createObjectURL(blob); const a=document.createElement('a'); a.href=url; a.download=filename; document.body.appendChild(a); a.click(); a.remove(); URL.revokeObjectURL(url); }}
document.getElementById('cards').addEventListener('change', e => {{ const card=e.target.closest('.card'); if(!card) return; const id=card.dataset.editId; if(e.target.matches('input[type="radio"]')) {{ state.get(id).decision=e.target.value; card.dataset.status=e.target.value; updateStats(); applyFilter(); }} if(e.target.matches('input[type="checkbox"]')) {{ state.get(id).reasons=[...card.querySelectorAll('input[type="checkbox"]:checked')].map(x=>x.value); }} }});
document.getElementById('cards').addEventListener('input', e => {{ const card=e.target.closest('.card'); if(card && e.target.matches('textarea')) state.get(card.dataset.editId).note=e.target.value; }});
document.querySelectorAll('[data-filter]').forEach(btn => btn.addEventListener('click', () => {{ activeFilter=btn.dataset.filter; document.querySelectorAll('[data-filter]').forEach(b=>b.classList.remove('active')); btn.classList.add('active'); applyFilter(); }}));
document.getElementById('downloadJson').addEventListener('click', () => download('review_results_{chunk_index:03d}.json', JSON.stringify(collectResults(), null, 2), 'application/json'));
document.getElementById('downloadKeep').addEventListener('click', () => {{ const ids=collectResults().filter(x=>x.decision==='keep').map(x=>x.edit_id).join('\n'); download('selected_edit_ids_{chunk_index:03d}.txt', ids + (ids ? '\n' : '')); }});
updateStats();
</script>
</body></html>
'''
    output_path.write_text(html_text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--chunk-size", type=int, default=100)
    ap.add_argument("--chunk-index", type=int, default=0)
    ap.add_argument("--all", action="store_true", help="Generate all chunks instead of one chunk")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    total_records = sum(1 for line in args.manifest.open("r", encoding="utf-8") if line.strip())
    total_chunks = (total_records + args.chunk_size - 1) // args.chunk_size
    chunk_indices = range(total_chunks) if args.all else [args.chunk_index]
    for chunk_index in chunk_indices:
        records = load_records(args.manifest, start=chunk_index * args.chunk_size, limit=args.chunk_size)
        if not records:
            raise ValueError(f"chunk {chunk_index} has no records")
        out = args.out_dir / f"h3d_test_review_{chunk_index:03d}.html"
        write_html(records, out, chunk_index=chunk_index, total_chunks=total_chunks)
        print(f"wrote {out} ({len(records)} records)")


if __name__ == "__main__":
    main()
