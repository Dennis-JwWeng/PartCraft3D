"""End-to-end smoke test for the H3D_v1 CLIs on a synthetic mini-shard.

Constructs one obj with one del + one add + one mod edit (all gates
"pass"), pre-stages the per-edit artefacts that pipeline_v3 would
normally produce, and drives all 5 CLIs in sequence:

  1. pull_deletion --skip-encode   (after.npz pre-staged so no GPU)
  2. pull_flux
  3. pull_addition
  4. build_h3d_v1_index --validate
  5. pack_shard

Final assertions:
  * Per-edit hardlinks land at the spec-defined inodes.
  * manifests/all.jsonl contains exactly the 3 expected records.
  * The tarball lists every expected entry.

Doesn't mock anything that ships with the package — only stubs out
the s6b GPU step by pre-staging after.npz, which is exactly what
``--skip-encode`` is designed to support.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tarfile
from pathlib import Path

import numpy as np
import pytest

from partcraft.cleaning.h3d_v1.layout import H3DLayout, N_VIEWS

SHARD = "08"
OBJ = "obj42"
DEL_ID = f"del_{OBJ}_000"
ADD_ID = f"add_{OBJ}_000"
MOD_ID = f"mod_{OBJ}_007"
PREVIEW_SIDE = 518


def _save_npz(path: Path, *, marker: int) -> None:
    feats = np.full((4, 8), marker, dtype=np.float32)
    coords = np.zeros((4, 4), dtype=np.int32)
    ss = np.zeros((8, 16, 16, 16), dtype=np.float32)
    np.savez(path, slat_feats=feats, slat_coords=coords, ss=ss)


def _save_pngs(dir_: Path, prefix: str, *, n: int = N_VIEWS) -> None:
    cv2 = pytest.importorskip("cv2")
    dir_.mkdir(parents=True, exist_ok=True)
    for k in range(n):
        img = np.full((PREVIEW_SIDE, PREVIEW_SIDE, 3), fill_value=(k + 1) * 30, dtype=np.uint8)
        cv2.imwrite(str(dir_ / f"{prefix}{k}.png"), img)


def _write_yaml(path: Path, output_dir: Path, slat_dir: Path,
                images_root: Path, mesh_root: Path) -> None:
    """Minimal pipeline_v3 yaml the loader will accept."""
    content = (
        f"ckpt_root: {output_dir.parent}/ckpt_unused\n"
        f"data:\n"
        f"  output_dir: {output_dir}\n"
        f"  slat_dir: {slat_dir}\n"
        f"  images_root: {images_root}\n"
        f"  mesh_root: {mesh_root}\n"
        f"pipeline:\n"
        f"  gpus: [0]\n"
    )
    path.write_text(content)


@pytest.fixture()
def synthetic_shard(tmp_path: Path) -> dict:
    """Fully wire a fake pipeline_v3 output dir + cfg yaml."""
    output_dir = tmp_path / "pipeline_out"
    obj_dir = output_dir / "objects" / SHARD / OBJ
    edits_root = obj_dir / "edits_3d"
    del_dir = edits_root / DEL_ID
    add_dir = edits_root / ADD_ID
    mod_dir = edits_root / MOD_ID

    for d in (del_dir, add_dir, mod_dir):
        d.mkdir(parents=True, exist_ok=True)
    _save_npz(del_dir / "after.npz", marker=11)
    (del_dir / "after_new.glb").write_text("fake glb (not used because after.npz exists)")
    _save_pngs(del_dir, "preview_")
    _save_pngs(add_dir, "preview_")
    _save_npz(mod_dir / "before.npz", marker=22)
    _save_npz(mod_dir / "after.npz", marker=33)
    _save_pngs(mod_dir, "preview_")

    edit_status = {
        "obj_id": OBJ, "shard": SHARD, "schema_version": 1,
        "edits": {
            DEL_ID: {"edit_type": "deletion",
                     "stages": {"gate_a": {"status": "pass"}, "gate_e": {"status": "pass"}},
                     "gates": {"A": {"vlm": {"pass": True}}, "C": None,
                                "E": {"vlm": {"pass": True}}},
                     "final_pass": True},
            ADD_ID: {"edit_type": "addition",
                     "stages": {"gate_e": {"status": "pass"}},
                     "gates": {"A": None, "C": None, "E": {"vlm": {"pass": True}}},
                     "final_pass": True},
            MOD_ID: {"edit_type": "modification",
                     "stages": {"gate_a": {"status": "pass"}, "gate_e": {"status": "pass"}},
                     "gates": {"A": {"vlm": {"pass": True}}, "C": None,
                                "E": {"vlm": {"pass": True}}},
                     "final_pass": True},
        },
    }
    (obj_dir / "edit_status.json").write_text(json.dumps(edit_status))

    ckpt_root = tmp_path / "ckpt_unused"
    ckpt_root.mkdir(parents=True, exist_ok=True)
    for d in ("slat_unused", "images_unused", "mesh_unused"):
        (tmp_path / d).mkdir(parents=True, exist_ok=True)
    cfg = tmp_path / "pipeline.yaml"
    _write_yaml(cfg, output_dir,
                slat_dir=tmp_path / "slat_unused",
                images_root=tmp_path / "images_unused",
                mesh_root=tmp_path / "mesh_unused")

    dataset_root = tmp_path / "H3D_v1"
    return {"cfg": cfg, "dataset_root": dataset_root,
            "tmp": tmp_path, "obj_dir": obj_dir}


def _run(*argv: str) -> int:
    """Invoke a CLI module via subprocess for true end-to-end coverage."""
    proc = subprocess.run([sys.executable, "-m", *argv],
                          capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stdout.write(proc.stdout)
        sys.stderr.write(proc.stderr)
    return proc.returncode


def _ino(p: Path) -> int:
    return p.stat().st_ino


def test_e2e_synthetic_shard(synthetic_shard: dict, tmp_path: Path) -> None:
    cfg = synthetic_shard["cfg"]
    dataset_root: Path = synthetic_shard["dataset_root"]
    layout = H3DLayout(root=dataset_root)

    # 1. pull_deletion (skip-encode because after.npz already pre-staged)
    rc = _run("scripts.cleaning.h3d_v1.pull_deletion",
              "--pipeline-cfg", str(cfg), "--shard", SHARD,
              "--dataset-root", str(dataset_root),
              "--skip-encode", "--workers", "2", "--log-level", "WARNING")
    assert rc == 0

    # 2. pull_flux
    rc = _run("scripts.cleaning.h3d_v1.pull_flux",
              "--pipeline-cfg", str(cfg), "--shard", SHARD,
              "--dataset-root", str(dataset_root),
              "--workers", "2", "--log-level", "WARNING")
    assert rc == 0

    # 3. pull_addition
    rc = _run("scripts.cleaning.h3d_v1.pull_addition",
              "--pipeline-cfg", str(cfg), "--shard", SHARD,
              "--dataset-root", str(dataset_root),
              "--workers", "2", "--log-level", "WARNING")
    assert rc == 0

    # ── verify hardlink graph ────────────────────────────────────────
    obj_npz = layout.object_npz(SHARD, OBJ)
    assert obj_npz.is_file()
    del_after = layout.after_npz("deletion", SHARD, OBJ, DEL_ID)
    add_before = layout.before_npz("addition", SHARD, OBJ, ADD_ID)
    add_after = layout.after_npz("addition", SHARD, OBJ, ADD_ID)
    mod_before = layout.before_npz("modification", SHARD, OBJ, MOD_ID)
    mod_after = layout.after_npz("modification", SHARD, OBJ, MOD_ID)

    assert _ino(layout.before_npz("deletion", SHARD, OBJ, DEL_ID)) == _ino(obj_npz)
    assert _ino(mod_before) == _ino(obj_npz)
    assert _ino(add_after) == _ino(obj_npz)
    assert _ino(add_before) == _ino(del_after)

    # ── 4. build index --validate ────────────────────────────────────
    rc = _run("scripts.cleaning.h3d_v1.build_h3d_v1_index",
              "--dataset-root", str(dataset_root), "--validate",
              "--log-level", "WARNING")
    assert rc == 0

    all_path = layout.aggregated_manifest()
    assert all_path.is_file()
    records = [json.loads(line) for line in all_path.read_text().splitlines() if line]
    assert len(records) == 3
    eids = {r["edit_id"] for r in records}
    assert eids == {DEL_ID, ADD_ID, MOD_ID}

    # ── 5. pack_shard ────────────────────────────────────────────────
    out_tar = tmp_path / "shard08.tar"
    rc = _run("scripts.cleaning.h3d_v1.pack_shard",
              "--dataset-root", str(dataset_root), "--shard", SHARD,
              "--out", str(out_tar), "--log-level", "WARNING")
    assert rc == 0
    assert out_tar.is_file() and out_tar.stat().st_size > 0

    with tarfile.open(out_tar) as tf:
        names = tf.getnames()
    expected_subset = {
        f"_assets/{SHARD}/{OBJ}/object.npz",
        f"_assets/{SHARD}/{OBJ}/orig_views/view0.png",
        f"deletion/{SHARD}/{OBJ}/{DEL_ID}/meta.json",
        f"deletion/{SHARD}/{OBJ}/{DEL_ID}/after.npz",
        f"addition/{SHARD}/{OBJ}/{ADD_ID}/meta.json",
        f"modification/{SHARD}/{OBJ}/{MOD_ID}/after.npz",
        f"manifests/deletion/{SHARD}.jsonl",
        f"manifests/addition/{SHARD}.jsonl",
        f"manifests/modification/{SHARD}.jsonl",
        "manifests/all.jsonl",
    }
    missing = expected_subset - set(names)
    assert not missing, f"tarball missing expected entries: {missing}"
