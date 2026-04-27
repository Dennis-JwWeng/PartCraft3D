#!/usr/bin/env python3
"""Build drag-and-drop ZIP based review pages for H3D merged test."""
from __future__ import annotations

import argparse
import json
import re
import zipfile
from pathlib import Path
from typing import Iterable

from build_merged_test_review_html import FLUX_TYPES, DEFAULT_MANIFEST, prompt_to_zh

DEFAULT_OUT_DIR = Path("/mnt/zsn/zsn_workspace/PartCraft3D/bench_review/merged_test_2000_zip_review")
DEFAULT_TRANSLATIONS = Path("/mnt/zsn/zsn_workspace/PartCraft3D/bench_review/merged_test_2000/prompt_translations_per_edit_llm_full.jsonl")


def safe_asset_dir(edit_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", edit_id)



def load_translations(path: Path | None) -> dict[str, str]:
    if path is None or not path.is_file():
        return {}
    translations: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            edit_id = row.get("edit_id")
            prompt_zh = row.get("prompt_zh")
            if edit_id and prompt_zh:
                translations[str(edit_id)] = str(prompt_zh)
    return translations

def load_manifest_records(manifest_path: Path, *, start: int, limit: int, translations: dict[str, str] | None = None) -> list[dict]:
    out: list[dict] = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if index < start:
                continue
            if len(out) >= limit:
                break
            if not line.strip():
                continue
            record = json.loads(line)
            meta = json.loads(Path(record["h3d_meta_json"]).read_text(encoding="utf-8"))
            prompt_en = (meta.get("instruction") or {}).get("prompt") or ""
            record["prompt_en"] = prompt_en
            record["prompt_zh"] = (translations or {}).get(record["edit_id"], prompt_to_zh(prompt_en))
            out.append(record)
    return out


def _add_asset(zf: zipfile.ZipFile, source: str, arcname: str) -> None:
    zf.write(source, arcname=arcname, compress_type=zipfile.ZIP_STORED)


def write_assets_zip(records: Iterable[dict], output_path: Path) -> list[dict]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_STORED) as zf:
        for record in records:
            asset_dir = f"assets/{safe_asset_dir(record['edit_id'])}"
            assets = {
                "h3d_before": f"{asset_dir}/3d_before.png",
                "h3d_after": f"{asset_dir}/3d_after.png",
            }
            _add_asset(zf, record["h3d_before_png"], assets["h3d_before"])
            _add_asset(zf, record["h3d_after_png"], assets["h3d_after"])
            if record["edit_type"] in FLUX_TYPES:
                assets["two_d_input"] = f"{asset_dir}/2d_input.png"
                assets["two_d_edited"] = f"{asset_dir}/2d_edited.png"
                _add_asset(zf, record["two_d_input_png"], assets["two_d_input"])
                _add_asset(zf, record["two_d_edited_png"], assets["two_d_edited"])
            manifest.append({
                "bench_split": record.get("bench_split", "test_merged"),
                "source_hf_split": record.get("source_hf_split", ""),
                "edit_id": record["edit_id"],
                "edit_type": record["edit_type"],
                "obj_id": record["obj_id"],
                "shard": record["shard"],
                "prompt_en": record.get("prompt_en", ""),
                "prompt_zh": record.get("prompt_zh", ""),
                "assets": assets,
            })
        zf.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2), compress_type=zipfile.ZIP_STORED)
    return manifest


def write_tool_html(output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    html = f'''<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>H3D Zip Review Tool</title>
<style>
:root {{ color-scheme: light dark; --bg:#f6f7fb; --card:#fff; --text:#17202a; --muted:#64748b; --border:#d7dde8; --accent:#2563eb; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#0f172a; --card:#111827; --text:#e5e7eb; --muted:#94a3b8; --border:#334155; }} }}
* {{ box-sizing:border-box; }} body {{ margin:0; background:var(--bg); color:var(--text); font-family:system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }}
header {{ position:sticky; top:0; z-index:3; background:color-mix(in srgb,var(--bg) 90%,transparent); backdrop-filter:blur(10px); border-bottom:1px solid var(--border); }}
.wrap, main {{ max-width:1500px; margin:0 auto; padding:16px 22px; }} h1 {{ margin:0 0 8px; font-size:24px; }} .subtle {{ color:var(--muted); line-height:1.5; }}
.drop {{ border:2px dashed var(--border); border-radius:16px; background:var(--card); padding:28px; text-align:center; margin:16px 0; }} .drop.drag {{ border-color:var(--accent); }}
.toolbar {{ display:flex; flex-wrap:wrap; gap:10px; align-items:center; margin-top:12px; }} button {{ border:1px solid var(--border); border-radius:9px; padding:8px 11px; background:var(--card); color:var(--text); cursor:pointer; }} button.active,button.primary {{ background:var(--accent); border-color:var(--accent); color:white; }}
.criteria,.card {{ background:var(--card); border:1px solid var(--border); border-radius:14px; padding:16px; margin:16px 0; }} .criteria ul {{ margin:8px 0 0 20px; padding:0; }} .criteria li {{ margin:5px 0; }}
.meta {{ display:flex; flex-wrap:wrap; gap:8px; margin-bottom:10px; }} .pill {{ border:1px solid var(--border); border-radius:999px; padding:4px 9px; color:var(--muted); font-size:13px; }} .type {{ color:var(--accent); font-weight:700; }}
.prompt {{ font-size:17px; line-height:1.45; margin:8px 0; }} .zh {{ font-size:19px; }} .grid {{ display:grid; gap:12px; margin-top:14px; }} .grid.two {{ grid-template-columns:repeat(2,1fr); }} .grid.four {{ grid-template-columns:repeat(4,1fr); }}
figure {{ margin:0; border:1px solid var(--border); border-radius:12px; overflow:hidden; background:#0b1020; }} figcaption {{ padding:8px 10px; background:#111827; color:#fff; font-weight:700; }} img {{ display:block; width:100%; height:auto; }}
.decision,.reasons {{ display:flex; flex-wrap:wrap; gap:14px; align-items:center; margin-top:14px; }} .reason-title {{ width:100%; color:var(--muted); }} label {{ display:inline-flex; gap:6px; align-items:center; }} textarea {{ width:100%; min-height:54px; margin-top:10px; border:1px solid var(--border); border-radius:10px; padding:10px; background:transparent; color:var(--text); }} .hidden {{ display:none; }} .stats {{ font-weight:700; }}
@media (max-width:1100px) {{ .grid.four {{ grid-template-columns:repeat(2,1fr); }} }} @media (max-width:760px) {{ .grid.two,.grid.four {{ grid-template-columns:1fr; }} header {{ position:static; }} }}
</style>
</head>
<body>
<header><div class="wrap">
<h1>H3D Test Bench Zip Review Tool</h1>
<div class="subtle">拖入 assets.zip 后渲染本页样本。不需要手动解压。del/add 展示 3D before/after；flux 类型额外展示 2D input/edited。</div>
<div class="toolbar"><span class="stats" id="stats">请先拖入 assets.zip</span><button class="active" data-filter="all">全部</button><button data-filter="unreviewed">未标</button><button data-filter="keep">要</button><button data-filter="reject">不要</button><button data-filter="unsure">不确定</button><button class="primary" id="downloadJson">导出 review_results_*.json</button><button class="primary" id="downloadKeep">导出 selected_edit_ids_*.txt</button></div>
</div></header>
<main>
<section id="drop" class="drop"><h2>拖入 assets.zip</h2><p>把任意 <code>h3d_test_review_XXX_assets.zip</code> 拖到这里，或拖到页面任意位置；也可以点击选择文件。</p><p id="loadStatus" class="subtle">等待 zip...</p><input id="fileInput" type="file" accept=".zip"></section>
<section class="criteria"><h2>标注要求</h2><ul><li><strong>要：</strong>图片清晰，编辑结果和指令一致，主体没有严重崩坏。</li><li><strong>del/add：</strong>只判断 3D before/after；删除应移除目标，添加应出现目标且位置尺度合理。</li><li><strong>flux：</strong>同时参考 2D input/edited 和 3D before/after；2D 表达目标变化，3D after 语义上继承该效果即可，2D/3D 视角不一致可接受。</li><li><strong>不要：</strong>目标错、几乎无变化、严重崩坏、图像异常或指令无法判断。</li><li><strong>不确定：</strong>变化弱、遮挡、视角造成难判时先标不确定。</li></ul></section>
<section id="cards"></section>
</main>
<script>
const FLUX_TYPES = new Set(['modification','scale','material','color','global']);
let records = [];
let state = new Map();
let activeFilter = 'all';
let objectUrls = [];
let currentZipStem = 'unloaded';
const cards = document.getElementById('cards');
const drop = document.getElementById('drop');
const loadStatus = document.getElementById('loadStatus');
function esc(value) {{ return String(value ?? '').replace(/[&<>"']/g, ch => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[ch])); }}
function u16(view, off) {{ return view.getUint16(off, true); }}
function u32(view, off) {{ return view.getUint32(off, true); }}
async function parseStoredZip(file) {{
  const buffer = await file.arrayBuffer();
  const view = new DataView(buffer);
  let eocd = -1;
  for (let i = buffer.byteLength - 22; i >= 0; i--) {{ if (u32(view, i) === 0x06054b50) {{ eocd = i; break; }} }}
  if (eocd < 0) throw new Error('Cannot find ZIP central directory');
  const count = u16(view, eocd + 10);
  let ptr = u32(view, eocd + 16);
  const files = new Map();
  for (let i = 0; i < count; i++) {{
    if (u32(view, ptr) !== 0x02014b50) throw new Error('Bad central directory entry');
    const method = u16(view, ptr + 10);
    const compressedSize = u32(view, ptr + 20);
    const uncompressedSize = u32(view, ptr + 24);
    const nameLen = u16(view, ptr + 28); const extraLen = u16(view, ptr + 30); const commentLen = u16(view, ptr + 32);
    const localOffset = u32(view, ptr + 42);
    const name = new TextDecoder().decode(new Uint8Array(buffer, ptr + 46, nameLen));
    if (method !== 0) throw new Error(`ZIP entry ${{name}} is compressed; expected ZIP_STORED`);
    if (compressedSize !== uncompressedSize) throw new Error(`ZIP entry ${{name}} size mismatch`);
    const localNameLen = u16(view, localOffset + 26); const localExtraLen = u16(view, localOffset + 28);
    const dataStart = localOffset + 30 + localNameLen + localExtraLen;
    files.set(name, buffer.slice(dataStart, dataStart + uncompressedSize));
    ptr += 46 + nameLen + extraLen + commentLen;
  }}
  return files;
}}
function fileUrl(files, name) {{
  const data = files.get(name);
  if (!data) throw new Error(`Missing ${{name}} in zip`);
  const url = URL.createObjectURL(new Blob([data], {{type:'image/png'}}));
  objectUrls.push(url);
  return url;
}}
function figure(label, src) {{ return `<figure><figcaption>${{esc(label)}}</figcaption><img src="${{src}}" alt="${{esc(label)}}"></figure>`; }}
function renderCard(record, index, files) {{
  const assets = record.assets;
  const figs = [figure('3D before', fileUrl(files, assets.h3d_before)), figure('3D after', fileUrl(files, assets.h3d_after))];
  if (FLUX_TYPES.has(record.edit_type)) {{ figs.push(figure('2D input', fileUrl(files, assets.two_d_input)), figure('2D edited', fileUrl(files, assets.two_d_edited))); }}
  const gridClass = FLUX_TYPES.has(record.edit_type) ? 'grid four' : 'grid two';
  const reasons = ['图像缺失或加载失败','编辑不符合指令','目标区域错误','变化太弱或无变化','几何/外观严重崩坏','2D edit 可用但 3D 不一致','其他'];
  const reasonHtml = reasons.map(r => `<label><input type="checkbox" name="reason-${{esc(record.edit_id)}}" value="${{esc(r)}}"> ${{esc(r)}}</label>`).join('');
  return `<article class="card unreviewed" data-record-card="true" data-edit-id="${{esc(record.edit_id)}}" data-status="unreviewed"><div class="meta"><span class="pill">#${{index + 1}}</span><span class="pill type">${{esc(record.edit_type)}}</span><span class="pill">edit_id: ${{esc(record.edit_id)}}</span><span class="pill">obj_id: ${{esc(record.obj_id)}}</span><span class="pill">shard: ${{esc(record.shard)}}</span><span class="pill">source: ${{esc(record.source_hf_split)}}</span></div><div class="prompt zh"><strong>中文：</strong>${{esc(record.prompt_zh)}}</div><div class="prompt en"><strong>English:</strong> ${{esc(record.prompt_en)}}</div><div class="${{gridClass}}">${{figs.join('')}}</div><div class="decision"><label><input type="radio" name="decision-${{esc(record.edit_id)}}" value="keep"> 要</label><label><input type="radio" name="decision-${{esc(record.edit_id)}}" value="reject"> 不要</label><label><input type="radio" name="decision-${{esc(record.edit_id)}}" value="unsure"> 不确定</label><label><input type="radio" name="decision-${{esc(record.edit_id)}}" value="unreviewed" checked> 未标</label></div><div class="reasons"><div class="reason-title">拒绝/疑问原因：</div>${{reasonHtml}}</div><textarea placeholder="备注：记录具体问题、视角疑问或保留理由"></textarea></article>`;
}}
function updateStats() {{ const vals=[...state.values()]; const c=s=>vals.filter(v=>v.decision===s).length; document.getElementById('stats').textContent=`Total: ${{records.length}} · 已标: ${{c('keep')+c('reject')+c('unsure')}} · 要: ${{c('keep')}} · 不要: ${{c('reject')}} · 不确定: ${{c('unsure')}} · 未标: ${{c('unreviewed')}}`; }}
function applyFilter() {{ document.querySelectorAll('.card').forEach(card => card.classList.toggle('hidden', !(activeFilter === 'all' || card.dataset.status === activeFilter))); }}
function collectResults() {{ return records.map(r => ({{edit_id:r.edit_id, edit_type:r.edit_type, obj_id:r.obj_id, shard:r.shard, source_hf_split:r.source_hf_split, prompt_en:r.prompt_en, prompt_zh:r.prompt_zh, decision:state.get(r.edit_id).decision, reject_reasons:state.get(r.edit_id).reasons, note:state.get(r.edit_id).note}})); }}
function download(filename, text, type='text/plain') {{ const blob=new Blob([text],{{type}}); const url=URL.createObjectURL(blob); const a=document.createElement('a'); a.href=url; a.download=filename; document.body.appendChild(a); a.click(); a.remove(); URL.revokeObjectURL(url); }}
async function loadZip(file) {{
  loadStatus.textContent = `正在读取 ${{file.name}} ...`;
  objectUrls.forEach(URL.revokeObjectURL); objectUrls = [];
  currentZipStem = file.name.replace(/[.]zip$/i, '').replace(/[^A-Za-z0-9_.-]+/g, '_');
  const files = await parseStoredZip(file);
  records = JSON.parse(new TextDecoder().decode(new Uint8Array(files.get('manifest.json'))));
  state = new Map(records.map(r => [r.edit_id, {{decision:'unreviewed', reasons:[], note:''}}]));
  cards.innerHTML = records.map((r, i) => renderCard(r, i, files)).join('');
  loadStatus.textContent = `已加载 ${{file.name}}，共 ${{records.length}} 条。`;
  updateStats(); applyFilter();
}}
drop.addEventListener('dragover', e => {{ e.preventDefault(); drop.classList.add('drag'); }});
drop.addEventListener('dragleave', () => drop.classList.remove('drag'));
function handleDrop(e) {{ e.preventDefault(); drop.classList.remove('drag'); const file=e.dataTransfer.files[0]; if(file) loadZip(file).catch(err => {{ loadStatus.textContent = `加载失败：${{err.message}}`; alert(err.message); }}); }}
drop.addEventListener('drop', handleDrop);
document.addEventListener('dragover', e => {{ e.preventDefault(); drop.classList.add('drag'); }});
document.addEventListener('drop', handleDrop);
document.getElementById('fileInput').addEventListener('change', e => {{ const file=e.target.files[0]; if(file) loadZip(file).catch(err => {{ loadStatus.textContent = `加载失败：${{err.message}}`; alert(err.message); }}); }});
cards.addEventListener('change', e => {{ const card=e.target.closest('.card'); if(!card) return; const id=card.dataset.editId; if(e.target.matches('input[type="radio"]')) {{ state.get(id).decision=e.target.value; card.dataset.status=e.target.value; updateStats(); applyFilter(); }} if(e.target.matches('input[type="checkbox"]')) {{ state.get(id).reasons=[...card.querySelectorAll('input[type="checkbox"]:checked')].map(x=>x.value); }} }});
cards.addEventListener('input', e => {{ const card=e.target.closest('.card'); if(card && e.target.matches('textarea')) state.get(card.dataset.editId).note=e.target.value; }});
document.querySelectorAll('[data-filter]').forEach(btn => btn.addEventListener('click', () => {{ activeFilter=btn.dataset.filter; document.querySelectorAll('[data-filter]').forEach(b=>b.classList.remove('active')); btn.classList.add('active'); applyFilter(); }}));
document.getElementById('downloadJson').addEventListener('click', () => download(`review_results_${{currentZipStem}}.json`, JSON.stringify(collectResults(), null, 2), 'application/json'));
document.getElementById('downloadKeep').addEventListener('click', () => {{ const ids=collectResults().filter(x=>x.decision==='keep').map(x=>x.edit_id).join('\\n'); download(`selected_edit_ids_${{currentZipStem}}.txt`, ids + (ids ? '\\n' : '')); }});
</script>
</body>
</html>
'''
    output_path.write_text(html, encoding="utf-8")


def build_chunk(manifest_path: Path, out_dir: Path, *, chunk_index: int, chunk_size: int, chunk_offset: int = 0, translations: dict[str, str] | None = None) -> Path:
    records = load_manifest_records(manifest_path, start=chunk_index * chunk_size, limit=chunk_size, translations=translations)
    if not records:
        raise ValueError(f"chunk {chunk_index} has no records")
    zip_path = out_dir / f"h3d_test_review_{chunk_index + chunk_offset:03d}_assets.zip"
    write_assets_zip(records, zip_path)
    return zip_path


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--chunk-size", type=int, default=100)
    ap.add_argument("--chunk-index", type=int, default=0)
    ap.add_argument("--chunk-offset", type=int, default=0, help="Add this offset to output chunk filenames; record slicing still uses --chunk-index.")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--translations", type=Path, default=DEFAULT_TRANSLATIONS)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    total_records = sum(1 for line in args.manifest.open("r", encoding="utf-8") if line.strip())
    total_chunks = (total_records + args.chunk_size - 1) // args.chunk_size
    chunk_indices = range(total_chunks) if args.all else [args.chunk_index]
    translations = load_translations(args.translations)
    for chunk_index in chunk_indices:
        zip_path = build_chunk(args.manifest, args.out_dir, chunk_index=chunk_index, chunk_size=args.chunk_size, chunk_offset=args.chunk_offset, translations=translations)
        print(f"wrote {zip_path}")
    tool_path = args.out_dir / "h3d_review_tool.html"
    write_tool_html(tool_path)
    print(f"wrote {tool_path}")


if __name__ == "__main__":
    main()
