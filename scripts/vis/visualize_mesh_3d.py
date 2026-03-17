#!/usr/bin/env python3
"""Launch an interactive 3D viewer for PLY/GLB meshes in the browser.

Generates a self-contained HTML file with three.js that supports:
  - Mouse drag to rotate, scroll to zoom, right-click to pan
  - GLB (textured), PLY (vertex colors), OBJ
  - Auto-rotating turntable mode
  - Optional: side-by-side comparison of two meshes (before/after)

Usage:
    python scripts/visualize_mesh_3d.py /path/to/mesh.glb
    python scripts/visualize_mesh_3d.py /path/to/mesh.ply --port 8899
    python scripts/visualize_mesh_3d.py /path/to/before.glb /path/to/after.glb
    python scripts/visualize_mesh_3d.py /path/to/mesh.glb --no-server  # just save HTML
"""
import argparse
import base64
import http.server
import mimetypes
import os
import sys
import threading
import webbrowser
from pathlib import Path


def mesh_to_data_uri(mesh_path: str) -> tuple[str, str]:
    """Read mesh file and return (data_uri, format)."""
    path = Path(mesh_path)
    suffix = path.suffix.lower()
    mime_map = {
        ".glb": "model/gltf-binary",
        ".gltf": "model/gltf+json",
        ".ply": "application/octet-stream",
        ".obj": "text/plain",
        ".stl": "application/octet-stream",
    }
    mime = mime_map.get(suffix, "application/octet-stream")
    with open(path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{b64}", suffix


def generate_html(mesh_paths: list[str], title: str = "") -> str:
    """Generate a self-contained HTML viewer with embedded mesh data."""

    meshes_js = []
    for i, mp in enumerate(mesh_paths):
        uri, fmt = mesh_to_data_uri(mp)
        name = Path(mp).name
        meshes_js.append(f'{{ uri: "{uri}", fmt: "{fmt}", name: "{name}" }}')
    meshes_array = ",\n        ".join(meshes_js)

    n = len(mesh_paths)
    if not title:
        title = " vs ".join(Path(p).stem for p in mesh_paths)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ background: #1a1a2e; overflow: hidden; font-family: system-ui, sans-serif; }}
  #info {{
    position: fixed; top: 10px; left: 50%; transform: translateX(-50%);
    color: #ccc; font-size: 13px; z-index: 10;
    background: rgba(0,0,0,0.6); padding: 6px 16px; border-radius: 6px;
    pointer-events: none;
  }}
  #controls {{
    position: fixed; bottom: 15px; left: 50%; transform: translateX(-50%);
    z-index: 10; display: flex; gap: 8px;
  }}
  #controls button {{
    background: rgba(255,255,255,0.15); color: #eee; border: 1px solid rgba(255,255,255,0.2);
    padding: 6px 14px; border-radius: 4px; cursor: pointer; font-size: 13px;
  }}
  #controls button:hover {{ background: rgba(255,255,255,0.25); }}
  #controls button.active {{ background: rgba(100,200,255,0.3); border-color: rgba(100,200,255,0.5); }}
  .label {{
    position: absolute; bottom: 8px; left: 50%; transform: translateX(-50%);
    color: #aaa; font-size: 12px; background: rgba(0,0,0,0.5);
    padding: 3px 10px; border-radius: 3px; pointer-events: none;
  }}
</style>
</head>
<body>

<div id="info">Loading...</div>
<div id="controls">
  <button id="btn-rotate" class="active" onclick="toggleRotate()">Auto Rotate</button>
  <button onclick="resetCamera()">Reset View</button>
  <button onclick="toggleWireframe()">Wireframe</button>
  <button onclick="toggleBg()">Toggle BG</button>
</div>

<script type="importmap">
{{
  "imports": {{
    "three": "https://cdn.jsdelivr.net/npm/three@0.170.0/build/three.module.js",
    "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.170.0/examples/jsm/"
  }}
}}
</script>

<script type="module">
import * as THREE from 'three';
import {{ OrbitControls }} from 'three/addons/controls/OrbitControls.js';
import {{ GLTFLoader }} from 'three/addons/loaders/GLTFLoader.js';
import {{ PLYLoader }} from 'three/addons/loaders/PLYLoader.js';
import {{ OBJLoader }} from 'three/addons/loaders/OBJLoader.js';

const meshes = [
    {meshes_array}
];

const N = meshes.length;
const scenes = [], cameras = [], controlsList = [], containers = [];
let renderer, autoRotate = true, wireframeOn = false, darkBg = true;
const loadedMeshes = [];

// Setup renderer
renderer = new THREE.WebGLRenderer({{ antialias: true }});
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setClearColor(0x1a1a2e);
renderer.outputColorSpace = THREE.SRGBColorSpace;
document.body.appendChild(renderer.domElement);

// For multiple meshes, we use scissor rendering
const vpWidth = () => Math.floor(window.innerWidth / N);
const vpHeight = () => window.innerHeight;

for (let i = 0; i < N; i++) {{
  const scene = new THREE.Scene();

  // Lights
  const ambient = new THREE.AmbientLight(0xffffff, 0.6);
  scene.add(ambient);
  const dirLight = new THREE.DirectionalLight(0xffffff, 1.2);
  dirLight.position.set(3, 5, 4);
  scene.add(dirLight);
  const dirLight2 = new THREE.DirectionalLight(0xffffff, 0.5);
  dirLight2.position.set(-3, -2, -4);
  scene.add(dirLight2);

  // Grid
  const grid = new THREE.GridHelper(4, 20, 0x444466, 0x333355);
  grid.position.y = -1;
  scene.add(grid);

  const camera = new THREE.PerspectiveCamera(40, vpWidth() / vpHeight(), 0.01, 100);
  camera.position.set(2.5, 1.5, 2.5);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.autoRotate = true;
  controls.autoRotateSpeed = 2.0;
  controls.target.set(0, 0, 0);

  scenes.push(scene);
  cameras.push(camera);
  controlsList.push(controls);
}}

// Load meshes
async function loadMesh(info, index) {{
  const {{ uri, fmt, name }} = info;
  const scene = scenes[index];

  return new Promise((resolve) => {{
    if (fmt === '.glb' || fmt === '.gltf') {{
      const loader = new GLTFLoader();
      loader.load(uri, (gltf) => {{
        const model = gltf.scene;
        centerAndScale(model);
        scene.add(model);
        loadedMeshes.push(model);
        resolve();
      }});
    }} else if (fmt === '.ply') {{
      const loader = new PLYLoader();
      loader.load(uri, (geometry) => {{
        geometry.computeVertexNormals();
        let material;
        if (geometry.hasAttribute('color')) {{
          material = new THREE.MeshStandardMaterial({{
            vertexColors: true, roughness: 0.7, metalness: 0.1
          }});
        }} else {{
          material = new THREE.MeshStandardMaterial({{
            color: 0x888888, roughness: 0.7, metalness: 0.1
          }});
        }}
        const mesh = new THREE.Mesh(geometry, material);
        centerAndScale(mesh);
        scene.add(mesh);
        loadedMeshes.push(mesh);
        resolve();
      }});
    }} else if (fmt === '.obj') {{
      const loader = new OBJLoader();
      loader.load(uri, (obj) => {{
        centerAndScale(obj);
        scene.add(obj);
        loadedMeshes.push(obj);
        resolve();
      }});
    }} else {{
      resolve();
    }}
  }});
}}

function centerAndScale(obj) {{
  const box = new THREE.Box3().setFromObject(obj);
  const center = box.getCenter(new THREE.Vector3());
  const size = box.getSize(new THREE.Vector3());
  const maxDim = Math.max(size.x, size.y, size.z);
  const scale = 2.0 / maxDim;
  obj.scale.setScalar(scale);
  obj.position.sub(center.multiplyScalar(scale));
}}

// Load all
const infoEl = document.getElementById('info');
Promise.all(meshes.map((m, i) => loadMesh(m, i))).then(() => {{
  const names = meshes.map(m => m.name).join('  vs  ');
  infoEl.textContent = names + '  |  drag: rotate, scroll: zoom, right-drag: pan';

  // Add labels for multi-mesh
  if (N > 1) {{
    meshes.forEach((m, i) => {{
      const label = document.createElement('div');
      label.className = 'label';
      label.textContent = m.name;
      label.style.left = (vpWidth() * i + vpWidth() / 2) + 'px';
      document.body.appendChild(label);
    }});
  }}

  animate();
}});

function animate() {{
  requestAnimationFrame(animate);
  controlsList.forEach(c => c.update());

  renderer.setScissorTest(N > 1);
  for (let i = 0; i < N; i++) {{
    const x = vpWidth() * i;
    renderer.setViewport(x, 0, vpWidth(), vpHeight());
    if (N > 1) renderer.setScissor(x, 0, vpWidth(), vpHeight());
    renderer.render(scenes[i], cameras[i]);
  }}
}}

// Global controls
window.toggleRotate = () => {{
  autoRotate = !autoRotate;
  controlsList.forEach(c => c.autoRotate = autoRotate);
  document.getElementById('btn-rotate').classList.toggle('active', autoRotate);
}};

window.resetCamera = () => {{
  cameras.forEach(cam => cam.position.set(2.5, 1.5, 2.5));
  controlsList.forEach(c => {{ c.target.set(0, 0, 0); c.update(); }});
}};

window.toggleWireframe = () => {{
  wireframeOn = !wireframeOn;
  loadedMeshes.forEach(obj => {{
    obj.traverse(child => {{
      if (child.isMesh && child.material) {{
        const mats = Array.isArray(child.material) ? child.material : [child.material];
        mats.forEach(m => m.wireframe = wireframeOn);
      }}
    }});
  }});
}};

window.toggleBg = () => {{
  darkBg = !darkBg;
  const c = darkBg ? 0x1a1a2e : 0xf0f0f0;
  renderer.setClearColor(c);
  scenes.forEach(s => {{
    s.children.forEach(ch => {{
      if (ch.isGridHelper) ch.visible = darkBg;
    }});
  }});
}};

window.addEventListener('resize', () => {{
  renderer.setSize(window.innerWidth, window.innerHeight);
  cameras.forEach(cam => {{
    cam.aspect = vpWidth() / vpHeight();
    cam.updateProjectionMatrix();
  }});
}});
</script>
</body>
</html>"""
    return html


def main():
    parser = argparse.ArgumentParser(
        description="Interactive 3D viewer for PLY/GLB meshes (opens in browser)")
    parser.add_argument("mesh_paths", nargs="+", type=str,
                        help="Path(s) to mesh file(s). Give 2 for side-by-side comparison.")
    parser.add_argument("--port", type=int, default=8877,
                        help="HTTP server port (default: 8877)")
    parser.add_argument("--save", type=str, default=None,
                        help="Save HTML to this path (always saved, this overrides default)")
    parser.add_argument("--no-server", action="store_true",
                        help="Just save HTML, don't start server")
    args = parser.parse_args()

    for mp in args.mesh_paths:
        if not Path(mp).exists():
            print(f"File not found: {mp}")
            sys.exit(1)

    names = [Path(p).stem for p in args.mesh_paths]
    print(f"Generating 3D viewer for: {', '.join(Path(p).name for p in args.mesh_paths)}")

    html = generate_html(args.mesh_paths)

    save_path = args.save or f"outputs/viewer_{'_vs_'.join(names[:2])}.html"
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        f.write(html)
    print(f"Saved HTML: {save_path}")

    if args.no_server:
        print("Done (--no-server). Open the HTML file in a browser.")
        return

    # Start a simple HTTP server and open browser
    serve_dir = str(Path(save_path).parent)
    html_name = Path(save_path).name

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=serve_dir, **kw)
        def log_message(self, fmt, *a):
            pass  # quiet

    port = args.port
    server = http.server.HTTPServer(("0.0.0.0", port), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    url = f"http://localhost:{port}/{html_name}"
    print(f"Serving at: {url}")
    print("Press Ctrl+C to stop.")

    try:
        webbrowser.open(url)
    except Exception:
        pass

    try:
        thread.join()
    except KeyboardInterrupt:
        print("\nStopped.")
        server.shutdown()


if __name__ == "__main__":
    main()
