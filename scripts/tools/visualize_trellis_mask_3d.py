#!/usr/bin/env python3
"""Interactive 3D HTML visualizer for Trellis SLAT mask.

Generates a self-contained HTML with Three.js point cloud.
  gray  = SLAT voxels preserved (not in mask)
  red   = SLAT voxels in edit mask

Usage:
    python scripts/tools/visualize_trellis_mask_3d.py \
        --obj-id bde1b486ee284e4d94f54bdbb3b3d6d7 --shard 08 \
        --config configs/pipeline_v3_shard08_bench100.yaml \
        [--edit-ids edit_id_1 ...] \
        [--out /tmp/mask3d.html]
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import yaml

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))


# ─── reuse helpers from 2D script ─────────────────────────────────────────────

def _load_slat_stub(before_npz: Path):
    import torch
    class _S: pass
    d   = np.load(str(before_npz))
    s   = _S()
    s.coords = torch.from_numpy(d["slat_coords"])  # [N,4] batch already included
    s.feats  = torch.from_numpy(d["slat_feats"])
    return s

def _stub_refiner(cfg):
    import torch
    from partcraft.trellis.refiner import TrellisRefiner
    r = TrellisRefiner.__new__(TrellisRefiner)
    r.device      = torch.device("cpu")
    r.ckpt_root   = Path(cfg.get("ckpt_root", "checkpoints"))
    r.slat_dir    = Path(cfg["data"].get("slat_dir", ""))
    r.img_enc_dir = None
    r.debug       = False
    r.pipeline = r.image_enc = None
    return r

def _get_edit_part_ids(spec):
    et = spec.edit_type.capitalize()
    if et == "Global":   return []
    if et == "Material": return [spec.selected_part_ids[0]] if spec.selected_part_ids else []
    return list(spec.selected_part_ids) if spec.selected_part_ids else []

def build_mask(refiner, ctx, spec, cfg):
    from partcraft.io.partcraft_loader import PartCraftDataset
    before_npz = ctx.edit_3d_dir(spec.edit_id) / "before.npz"
    if not before_npz.is_file():
        return None, None, None
    slat    = _load_slat_stub(before_npz)
    dataset = PartCraftDataset(render_dir=cfg["data"]["images_root"],
                               mesh_dir=cfg["data"]["mesh_root"])
    obj_rec = dataset.load_object(ctx.shard, ctx.obj_id)
    mask, eff = refiner.build_part_mask(
        ctx.obj_id, obj_rec, _get_edit_part_ids(spec),
        slat, spec.edit_type.capitalize())
    return mask, slat, eff


# ─── HTML template ────────────────────────────────────────────────────────────

HTML_TMPL = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Trellis Mask 3D — {title}</title>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ background:#111; color:#eee; font-family: monospace; overflow:hidden; }}
#container {{ width:100vw; height:100vh; display:flex; flex-direction:column; }}
#toolbar {{
  display:flex; align-items:center; gap:12px;
  padding:8px 14px; background:#1a1a1a; border-bottom:1px solid #333;
  flex-shrink:0; flex-wrap:wrap;
}}
#toolbar h2 {{ font-size:13px; color:#aaa; margin-right:8px; }}
.edit-btn {{
  padding:4px 10px; border-radius:4px; border:1px solid #444;
  background:#222; color:#ccc; cursor:pointer; font-size:11px;
  transition: background 0.15s;
}}
.edit-btn:hover {{ background:#333; }}
.edit-btn.active {{ background:#9b2a2a; border-color:#cc4444; color:#fff; }}
#info {{
  position:absolute; bottom:12px; left:14px;
  background:rgba(0,0,0,0.7); padding:8px 12px; border-radius:6px;
  font-size:11px; line-height:1.7; pointer-events:none;
}}
#canvas-wrap {{ flex:1; position:relative; }}
canvas {{ display:block; }}
</style>
</head>
<body>
<div id="container">
  <div id="toolbar">
    <h2>🎯 Trellis Mask 3D</h2>
    {buttons}
  </div>
  <div id="canvas-wrap">
    <div id="info">Loading…</div>
  </div>
</div>

<script type="importmap">
{{ "imports": {{ "three": "https://esm.sh/three@0.164.1", "three/addons/": "https://esm.sh/three@0.164.1/examples/jsm/" }} }}
</script>

<script type="module">
import * as THREE from 'three';
import {{ OrbitControls }} from 'three/addons/controls/OrbitControls.js';

const EDITS = {edits_json};

// ── scene ──────────────────────────────────────────────────────────────────
const wrap   = document.getElementById('canvas-wrap');
const info   = document.getElementById('info');
const renderer = new THREE.WebGLRenderer({{ antialias: true }});
renderer.setPixelRatio(devicePixelRatio);
renderer.setClearColor(0x111111);
wrap.appendChild(renderer.domElement);

const scene  = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(50, 1, 0.1, 2000);
camera.position.set(96, 80, 140);

const ctrl   = new OrbitControls(camera, renderer.domElement);
ctrl.enableDamping = true; ctrl.dampingFactor = 0.08;
ctrl.target.set(32, 32, 32);

// box outline
const boxGeo  = new THREE.BoxGeometry(64, 64, 64);
const boxEdge = new THREE.EdgesGeometry(boxGeo);
scene.add(new THREE.LineSegments(boxEdge,
  new THREE.LineBasicMaterial({{ color:0x333333 }})));
const boxHelper = new THREE.Mesh(boxGeo,
  new THREE.MeshBasicMaterial({{ color:0x001133, transparent:true, opacity:0.05 }}));
boxHelper.position.set(32,32,32);
scene.add(boxHelper);

// axes labels (X=red, Y=green, Z=blue) at corner
const axLen = 8;
[[[0,0,0],[axLen,0,0],0xff4444],[[0,0,0],[0,axLen,0],0x44ff44],[[0,0,0],[0,0,axLen],0x4444ff]].forEach(([a,b,c])=>{{
  const g = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(...a),new THREE.Vector3(...b)]);
  scene.add(new THREE.Line(g, new THREE.LineBasicMaterial({{color:c,linewidth:2}})));
}});

// ── point cloud helpers ────────────────────────────────────────────────────
let currentMesh = null;

function makeCloud(coords, colors) {{
  const n   = coords.length / 3;
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute(coords, 3));
  geo.setAttribute('color',    new THREE.Float32BufferAttribute(colors,  3));
  const mat = new THREE.PointsMaterial({{
    size: 1.5, vertexColors: true, sizeAttenuation: true,
  }});
  return new THREE.Points(geo, mat);
}}

function showEdit(idx) {{
  if (currentMesh) scene.remove(currentMesh);
  const e    = EDITS[idx];
  const pres = e.preserved;  // [[x,y,z], ...]
  const edit = e.edit;

  const nP = pres.length, nE = edit.length, n = nP + nE;
  const pos = new Float32Array(n * 3);
  const col = new Float32Array(n * 3);

  for (let i=0; i<nP; i++) {{
    pos[i*3]=pres[i][0]; pos[i*3+1]=pres[i][1]; pos[i*3+2]=pres[i][2];
    col[i*3]=0.55; col[i*3+1]=0.55; col[i*3+2]=0.55;
  }}
  for (let i=0; i<nE; i++) {{
    const j=nP+i;
    pos[j*3]=edit[i][0]; pos[j*3+1]=edit[i][1]; pos[j*3+2]=edit[i][2];
    col[j*3]=0.92; col[j*3+1]=0.15; col[j*3+2]=0.15;
  }}

  currentMesh = makeCloud(pos, col);
  scene.add(currentMesh);

  const pct = nE/(nP+nE)*100;
  info.innerHTML =
    `<b style="color:#f55">${{e.edit_id}}</b><br>` +
    `parts: ${{JSON.stringify(e.selected_part_ids)}}<br>` +
    `type: ${{e.edit_type}} → <b>${{e.effective_type}}</b><br>` +
    `<span style="color:#f55">■</span> edit SLAT: ${{nE}} (${{pct.toFixed(1)}}%)<br>` +
    `<span style="color:#888">■</span> preserved: ${{nP}}<br>` +
    `<br><i style="color:#666">drag to rotate · scroll to zoom</i>`;

  // highlight button
  document.querySelectorAll('.edit-btn').forEach((b,i)=>
    b.classList.toggle('active', i===idx));
}}

// ── toolbar buttons ────────────────────────────────────────────────────────
const toolbar = document.getElementById('toolbar');
EDITS.forEach((e, i) => {{
  const btn = document.createElement('button');
  btn.className = 'edit-btn';
  btn.textContent = e.short_label;
  btn.title = e.edit_id + '  parts=' + JSON.stringify(e.selected_part_ids);
  btn.onclick = () => showEdit(i);
  toolbar.appendChild(btn);
}});

showEdit(0);

// ── resize + render ────────────────────────────────────────────────────────
function resize() {{
  const w = wrap.clientWidth, h = wrap.clientHeight;
  renderer.setSize(w, h);
  camera.aspect = w/h; camera.updateProjectionMatrix();
}}
window.addEventListener('resize', resize); resize();

(function animate() {{
  requestAnimationFrame(animate);
  ctrl.update();
  renderer.render(scene, camera);
}})();
</script>
</body>
</html>
"""


# ─── main ─────────────────────────────────────────────────────────────────────

def run(obj_id, shard, cfg, edit_ids, out_path):
    from partcraft.pipeline_v3.paths import PipelineRoot, DatasetRoots
    from partcraft.pipeline_v3.specs import iter_flux_specs

    root  = PipelineRoot(Path(cfg["data"]["output_dir"]))
    roots = DatasetRoots.from_pipeline_cfg(cfg)
    mesh_npz, image_npz = roots.input_npz_paths(shard, obj_id)
    ctx = root.context(shard, obj_id, mesh_npz=mesh_npz, image_npz=image_npz)

    refiner = _stub_refiner(cfg)
    specs   = [s for s in iter_flux_specs(ctx)
               if (edit_ids is None or s.edit_id in set(edit_ids))
               and (ctx.edit_3d_dir(s.edit_id) / "before.npz").is_file()]
    if not specs:
        print("[WARN] no specs with before.npz"); return

    print(f"Building masks for {len(specs)} edits …")
    edits_data = []
    for spec in specs:
        mask, slat, eff = build_mask(refiner, ctx, spec, cfg)
        if mask is None:
            continue
        sc       = slat.coords[:, 1:].numpy()
        mask_np  = mask.numpy()
        in_mask  = mask_np[sc[:,0], sc[:,1], sc[:,2]].astype(bool)
        edit_sc  = sc[in_mask].tolist()
        pres_sc  = sc[~in_mask].tolist()
        et_input = spec.edit_type.capitalize()
        short    = f"{spec.edit_id.split('_')[-2]}_{spec.edit_id.split('_')[-1]}  p={spec.selected_part_ids}"
        edits_data.append({
            "edit_id":          spec.edit_id,
            "short_label":      short,
            "edit_type":        et_input,
            "effective_type":   eff,
            "selected_part_ids": spec.selected_part_ids,
            "preserved":        pres_sc,
            "edit":             edit_sc,
        })
        pct = 100*len(edit_sc)/(len(sc) or 1)
        print(f"  {spec.edit_id}: edit={len(edit_sc)} pres={len(pres_sc)} ({pct:.1f}%) eff={eff}")

    if not edits_data:
        print("[WARN] all masks failed"); return

    buttons_html = ""  # generated by JS
    html = HTML_TMPL.format(
        title=f"{obj_id[:8]}…",
        buttons=buttons_html,
        edits_json=json.dumps(edits_data),
    )
    out_path.write_text(html, encoding="utf-8")
    print(f"Saved → {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obj-id",   required=True)
    ap.add_argument("--shard",    default="08")
    ap.add_argument("--config",   default="configs/pipeline_v3_shard08_bench100.yaml")
    ap.add_argument("--edit-ids", nargs="*", default=None)
    ap.add_argument("--out",      default=None)
    args = ap.parse_args()
    cfg  = yaml.safe_load(Path(args.config).read_text())
    out  = Path(args.out) if args.out else Path(f"/tmp/{args.obj_id}_mask3d.html")
    run(args.obj_id, args.shard.zfill(2), cfg, args.edit_ids, out)

if __name__ == "__main__":
    main()
