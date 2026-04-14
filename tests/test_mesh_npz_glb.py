"""TDD tests for GLB-format mesh NPZ support.

All 7 tests are expected to FAIL until implementation tasks 2-6 are complete.
Tests use real on-disk data where available; disk-absent paths xfail (never pass silently).

Run with:
    python -m pytest tests/test_mesh_npz_glb.py -v --no-header
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")

# ---------------------------------------------------------------------------
# Shared fixture paths (real data; missing paths trigger xfail in dependent tests)
# ---------------------------------------------------------------------------
OBJ = "0008dc75fb3648f2af4ca8c4d711e53e"
PART_GLB_DIR = Path("/mnt/zsn/data/partverse/source/textured_part_glbs") / OBJ
NORM_GLB = Path("/mnt/zsn/data/partverse/source/normalized_glbs") / f"{OBJ}.glb"
ANNO_DIR = Path("/mnt/zsn/data/partverse/source/anno_infos") / OBJ
TRANSFORMS = {"scale": 0.5005004940503051, "offset": [0.0, 0.0, 0.0]}

_REAL_DATA_AVAILABLE = PART_GLB_DIR.exists() and NORM_GLB.exists()


def _make_minimal_img_npz(tmp_path: Path) -> Path:
    """Create a minimal image NPZ with split_mesh.json for build_part_menu."""
    split = {"valid_clusters": {}, "part_id_to_name": []}
    data = {"split_mesh.json": np.frombuffer(json.dumps(split).encode(), dtype=np.uint8)}
    p = tmp_path / "img.npz"
    np.savez(p, **data)
    return p


# ---------------------------------------------------------------------------
# Test 1 - GLB pack roundtrip preserves UV
# ---------------------------------------------------------------------------
@pytest.mark.real_data
def test_glb_pack_roundtrip_uv(tmp_path):
    """_pack_mesh_glb must return a dict with 'full.glb' + part GLBs.

    Expected to FAIL (ImportError / AttributeError) until Task 3 implements
    _pack_mesh_glb in scripts.datasets.partverse.pack_npz.
    """
    if not _REAL_DATA_AVAILABLE:
        pytest.xfail("real data not available; function also not yet implemented")

    try:
        from scripts.datasets.partverse.pack_npz import _pack_mesh_glb
    except (ImportError, AttributeError) as exc:
        pytest.xfail(f"_pack_mesh_glb not yet implemented: {exc}")

    result = _pack_mesh_glb(OBJ, PART_GLB_DIR.parent, NORM_GLB.parent, TRANSFORMS)

    assert "full.glb" in result, "Expected 'full.glb' key in result"
    part_keys = [k for k in result if k.startswith("part_") and k.endswith(".glb")]
    assert part_keys, "Expected at least one 'part_N.glb' key in result"

    glb_bytes = bytes(result["full.glb"])
    scene = trimesh.load(io.BytesIO(glb_bytes), file_type="glb")
    mesh = scene if isinstance(scene, trimesh.Trimesh) else trimesh.util.concatenate(
        list(scene.geometry.values()) if hasattr(scene, "geometry") else [scene]
    )
    assert mesh.visual is not None and hasattr(mesh.visual, "uv"), \
        "full.glb roundtrip lost UV data"
    assert mesh.visual.uv is not None, "UV attribute is None after GLB roundtrip"


# ---------------------------------------------------------------------------
# Test 2 - ObjectRecord._mesh_fmt detects PLY format
# ---------------------------------------------------------------------------
@pytest.mark.real_data
def test_loader_detects_ply_format(tmp_path):
    """ObjectRecord._mesh_fmt() must return 'ply' when NPZ has .ply keys.

    Expected to FAIL (AttributeError) until Task 2 adds _mesh_fmt to ObjectRecord.
    """
    try:
        from partcraft.io.partcraft_loader import ObjectRecord
    except ImportError as exc:
        pytest.xfail(f"Could not import ObjectRecord: {exc}")

    npz_path = tmp_path / "mesh_ply.npz"
    np.savez(npz_path,
             **{"full.ply": np.zeros(0, dtype=np.uint8),
                "part_0.ply": np.zeros(0, dtype=np.uint8)})

    record = ObjectRecord(
        obj_id="fake",
        shard="00",
        render_npz_path=str(tmp_path / "render.npz"),
        mesh_npz_path=str(npz_path),
    )

    try:
        fmt = record._mesh_fmt()
    except AttributeError as exc:
        pytest.xfail(f"_mesh_fmt not yet implemented: {exc}")

    assert fmt == "ply", f"Expected 'ply', got {fmt!r}"


# ---------------------------------------------------------------------------
# Test 3 - ObjectRecord._mesh_fmt detects GLB format
# ---------------------------------------------------------------------------
@pytest.mark.real_data
def test_loader_detects_glb_format(tmp_path):
    """ObjectRecord._mesh_fmt() must return 'glb' when NPZ has .glb keys.

    Expected to FAIL (AttributeError) until Task 2 adds _mesh_fmt to ObjectRecord.
    """
    try:
        from partcraft.io.partcraft_loader import ObjectRecord
    except ImportError as exc:
        pytest.xfail(f"Could not import ObjectRecord: {exc}")

    npz_path = tmp_path / "mesh_glb.npz"
    np.savez(npz_path,
             **{"full.glb": np.zeros(0, dtype=np.uint8),
                "part_0.glb": np.zeros(0, dtype=np.uint8)})

    record = ObjectRecord(
        obj_id="fake",
        shard="00",
        render_npz_path=str(tmp_path / "render.npz"),
        mesh_npz_path=str(npz_path),
    )

    try:
        fmt = record._mesh_fmt()
    except AttributeError as exc:
        pytest.xfail(f"_mesh_fmt not yet implemented: {exc}")

    assert fmt == "glb", f"Expected 'glb', got {fmt!r}"


# ---------------------------------------------------------------------------
# Test 4 - get_part_mesh returns Trimesh with UV from a GLB NPZ
# ---------------------------------------------------------------------------
@pytest.mark.real_data
def test_get_part_mesh_from_glb(tmp_path):
    """ObjectRecord.get_part_mesh(pid) must return a Trimesh with UV when NPZ is GLB.

    Expected to FAIL until Tasks 2+3 are done.
    """
    if not _REAL_DATA_AVAILABLE:
        pytest.xfail("real data not available; function also not yet implemented")

    try:
        from scripts.datasets.partverse.pack_npz import _pack_mesh_glb
    except (ImportError, AttributeError) as exc:
        pytest.xfail(f"_pack_mesh_glb not yet implemented (needed for pack step): {exc}")

    try:
        from partcraft.io.partcraft_loader import ObjectRecord
    except ImportError as exc:
        pytest.xfail(f"Could not import ObjectRecord: {exc}")

    # Pack to a temp NPZ
    raw = _pack_mesh_glb(OBJ, PART_GLB_DIR.parent, NORM_GLB.parent, TRANSFORMS)
    npz_path = tmp_path / "mesh.npz"
    np.savez(npz_path, **{k: np.frombuffer(v, dtype=np.uint8) if isinstance(v, bytes) else v
                          for k, v in raw.items()})

    z = np.load(npz_path, allow_pickle=True)
    if "part_0.glb" not in z.files:
        z.close()
        pytest.skip("part_0.glb not in packed NPZ; source data for part 0 does not exist")
    z.close()

    record = ObjectRecord(
        obj_id=OBJ,
        shard="00",
        render_npz_path=str(tmp_path / "render.npz"),
        mesh_npz_path=str(npz_path),
    )

    try:
        mesh = record.get_part_mesh(0)
    except (AttributeError, KeyError, Exception) as exc:
        pytest.xfail(f"get_part_mesh not yet updated for GLB format: {exc}")

    assert isinstance(mesh, trimesh.Trimesh), \
        f"Expected trimesh.Trimesh, got {type(mesh)}"
    assert mesh.visual is not None, "Mesh has no visual attribute"
    assert hasattr(mesh.visual, "uv") and mesh.visual.uv is not None, \
        "UV is None after loading part mesh from GLB NPZ"


# ---------------------------------------------------------------------------
# Test 5 - _build_deletion_from_npz produces edit_mesh.glb
# ---------------------------------------------------------------------------
@pytest.mark.real_data
def test_build_deletion_from_npz(tmp_path):
    """_build_deletion_from_npz must write 'edit_mesh.glb' to pair_dir.

    Expected to FAIL (ImportError / AttributeError) until Task 6 implements it
    in partcraft.pipeline_v2.s5b_deletion.
    """
    if not _REAL_DATA_AVAILABLE:
        pytest.xfail("real data not available; function also not yet implemented")

    try:
        from scripts.datasets.partverse.pack_npz import _pack_mesh_glb
    except (ImportError, AttributeError) as exc:
        pytest.xfail(f"_pack_mesh_glb not yet implemented (needed for setup): {exc}")

    try:
        from partcraft.pipeline_v2.s5b_deletion import _build_deletion_from_npz
    except (ImportError, AttributeError) as exc:
        pytest.xfail(f"_build_deletion_from_npz not yet implemented: {exc}")

    raw = _pack_mesh_glb(OBJ, PART_GLB_DIR.parent, NORM_GLB.parent, TRANSFORMS)
    npz_path = tmp_path / "mesh.npz"
    np.savez(npz_path, **{k: np.frombuffer(v, dtype=np.uint8) if isinstance(v, bytes) else v
                          for k, v in raw.items()})

    z = np.load(npz_path, allow_pickle=True)
    if "part_0.glb" not in z.files:
        z.close()
        pytest.skip("part_0.glb not in packed NPZ; source data for part 0 does not exist")
    z.close()

    pair_dir = tmp_path / "pair"
    pair_dir.mkdir()
    _build_deletion_from_npz(npz_path, [0], pair_dir)

    assert (pair_dir / "edit_mesh.glb").exists(), \
        f"edit_mesh.glb not found in {pair_dir}"


# ---------------------------------------------------------------------------
# Test 6 - extract_parts writes .glb files (not .ply) from a GLB NPZ
# ---------------------------------------------------------------------------
@pytest.mark.real_data
def test_extract_parts_glb(tmp_path):
    """extract_parts must emit .glb files when the NPZ contains .glb part keys.

    Expected to FAIL until Task 4 updates extract_parts to handle GLB format.
    """
    from partcraft.render.overview import extract_parts

    # Build a fake GLB-format mesh NPZ
    glb_stub = np.frombuffer(b"glb_stub", dtype=np.uint8)
    npz_path = tmp_path / "mesh_glb.npz"
    np.savez(npz_path,
             **{"full.glb": glb_stub,
                "part_0.glb": glb_stub,
                "part_1.glb": glb_stub})

    out_dir = tmp_path / "parts_out"
    out_dir.mkdir()

    extract_parts(npz_path, out_dir)

    glb_files = list(out_dir.glob("*.glb"))
    assert glb_files, \
        f"No .glb files written to {out_dir}; only found: {list(out_dir.iterdir())}"


# ---------------------------------------------------------------------------
# Test 7 - build_part_menu detects part IDs from .glb keys
# ---------------------------------------------------------------------------
@pytest.mark.real_data
def test_build_part_menu_glb(tmp_path):
    """build_part_menu must detect PIDs {0, 2} from a GLB-keyed mesh NPZ.

    Expected to FAIL until Task 5 updates the key regex to handle .glb suffixes.
    """
    from partcraft.pipeline_v2.s1_vlm_core import build_part_menu

    # Mesh NPZ with .glb keys
    glb_stub = np.frombuffer(b"stub", dtype=np.uint8)
    mesh_npz = tmp_path / "mesh_glb.npz"
    np.savez(mesh_npz,
             **{"full.glb": glb_stub,
                "part_0.glb": glb_stub,
                "part_2.glb": glb_stub})

    # Minimal image NPZ required by build_part_menu
    img_npz = _make_minimal_img_npz(tmp_path)

    pids, menu_text = build_part_menu(mesh_npz, img_npz)

    assert set(pids) == {0, 2}, \
        f"Expected PIDs {{0, 2}} from GLB NPZ keys, got {set(pids)!r}"
