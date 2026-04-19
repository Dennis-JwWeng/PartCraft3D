import json
from pathlib import Path

import pytest
import torch

from partcraft.cleaning.v1.layout import V1Layout
from partcraft.cleaning.v1.linker import LinkMode
from partcraft.cleaning.v1.promoter import PromoterConfig, promote_records
from partcraft.cleaning.v1.source_v2 import iter_records_from_v2_obj
from partcraft.cleaning.v1.pending import DelLatentPending


def _fake_before_assets(tmp_path: Path, obj_id: str = "objA", shard: str = "05") -> Path:
    data = tmp_path / "data_partverse"
    img_enc = data / "img_Enc" / obj_id
    img_enc.mkdir(parents=True)
    for i in [8, 89, 90, 91, 100]:
        (img_enc / f"{i:03d}.png").write_bytes(b"png")
    slat_dir = data / "slat" / shard
    slat_dir.mkdir(parents=True)
    torch.save(torch.zeros(1, 8), slat_dir / f"{obj_id}_feats.pt")
    torch.save(torch.zeros(1, 3, dtype=torch.int32), slat_dir / f"{obj_id}_coords.pt")
    torch.save(torch.zeros(1, 4), slat_dir / f"{obj_id}_ss.pt")
    return data


def _make_cfg(data_root: Path) -> PromoterConfig:
    return PromoterConfig(
        rule={"required_passes": ["gate_text_align", "gate_quality"]},
        link_mode=LinkMode.HARDLINK,
        img_enc_root=data_root / "img_Enc",
        slat_root=data_root / "slat",
        view_indices=[89, 90, 91, 100, 8],
    )


def test_promote_one_v2_obj_creates_layout(tmp_path: Path, v2_obj_dir: Path):
    data_root = _fake_before_assets(tmp_path)
    v1 = V1Layout(root=tmp_path / "v1")
    cfg = _make_cfg(data_root)
    recs = list(iter_records_from_v2_obj(v2_obj_dir, run_tag="pipeline_v2_shard05"))
    pending = DelLatentPending(v1.pending_del_latent_file())

    summary = promote_records(recs, layout=v1, cfg=cfg, pending=pending)

    # del_objA_000 passes (gate_quality.pass=True); mod_objA_001 fails (pass=False).
    assert summary.promoted == 1
    assert summary.deferred == 0
    assert summary.failed == 1

    assert v1.before_ss_npz("05", "objA").is_file()
    assert v1.before_slat_npz("05", "objA").is_file()
    for p in v1.before_view_paths("05", "objA"):
        assert p.is_file()

    edit_dir = v1.edit_dir("05", "objA", "del_objA_000")
    assert (edit_dir / "spec.json").is_file()
    assert (edit_dir / "qc.json").is_file()
    assert v1.after_pending_marker("05", "objA", "del_objA_000").is_file()
    assert not v1.after_npz_path("05", "objA", "del_objA_000").exists()
    for p in v1.after_view_paths("05", "objA", "del_objA_000"):
        assert p.is_file()

    assert any(e.edit_id == "del_objA_000" for e in pending.iter_entries())


def test_rerunning_same_source_skips_existing(tmp_path: Path, v2_obj_dir: Path):
    data_root = _fake_before_assets(tmp_path)
    v1 = V1Layout(root=tmp_path / "v1")
    cfg = _make_cfg(data_root)
    recs = list(iter_records_from_v2_obj(v2_obj_dir, run_tag="pipeline_v2_shard05"))
    pending = DelLatentPending(v1.pending_del_latent_file())
    promote_records(recs, layout=v1, cfg=cfg, pending=pending)
    summary2 = promote_records(recs, layout=v1, cfg=cfg, pending=pending)
    assert summary2.promoted == 0
    assert summary2.skipped_existing == 1


def test_collision_from_different_run_appends_r2_suffix(tmp_path: Path, v2_obj_dir: Path):
    data_root = _fake_before_assets(tmp_path)
    v1 = V1Layout(root=tmp_path / "v1")
    cfg = _make_cfg(data_root)
    recs1 = list(iter_records_from_v2_obj(v2_obj_dir, run_tag="pipeline_v2_shard05"))
    pending = DelLatentPending(v1.pending_del_latent_file())
    promote_records(recs1, layout=v1, cfg=cfg, pending=pending)

    recs2 = list(iter_records_from_v2_obj(v2_obj_dir, run_tag="pipeline_v2_shard05_rerun"))
    summary = promote_records(recs2, layout=v1, cfg=cfg, pending=pending)

    assert summary.promoted == 1
    edit_dir_r2 = v1.edit_dir("05", "objA", "del_objA_000", suffix="__r2")
    assert edit_dir_r2.is_dir()
    qc = json.loads(v1.qc_json("05", "objA", "del_objA_000", suffix="__r2").read_text())
    assert qc["source"]["run_tag"] == "pipeline_v2_shard05_rerun"



def _fake_image_npz(tmp_path: Path, obj_id: str = "objA", shard: str = "05",
                    indices: list[int] = [8, 89, 90, 91, 100],
                    *, mode: str = "RGB", alpha: int = 255) -> Path:
    """Build a per-object image NPZ matching the real partverse format.

    ``mode``/``alpha`` let tests cover both opaque-RGB and the real-world
    RGBA-with-transparency case (which the promoter must alpha-composite
    onto white before writing).
    """
    import io
    import numpy as np
    from PIL import Image
    npz_root = tmp_path / "data_images"
    shard_dir = npz_root / shard
    shard_dir.mkdir(parents=True)
    arrs = {}
    for idx in indices:
        if mode == "RGBA":
            img = Image.new("RGBA", (4, 4), color=(idx % 256, 0, 0, alpha))
        else:
            img = Image.new("RGB", (4, 4), color=(idx % 256, 0, 0))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        arrs[f"{idx:03d}.png"] = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    np.savez(shard_dir / f"{obj_id}.npz", **arrs)
    return npz_root


def test_promote_with_image_npz_root_writes_decoded_pngs(
    tmp_path: Path, v2_obj_dir: Path,
):
    """When image_npz_root is set, before/views are written from the NPZ
    (PNG bytes preserved verbatim), not from img_enc_root."""
    img_npz_root = _fake_image_npz(tmp_path)
    data_root = _fake_before_assets(tmp_path)
    v1 = V1Layout(root=tmp_path / "v1")
    cfg = PromoterConfig(
        rule={"required_passes": ["gate_text_align", "gate_quality"]},
        link_mode=LinkMode.HARDLINK,
        slat_root=data_root / "slat",
        view_indices=[89, 90, 91, 100, 8],
        image_npz_root=img_npz_root,
    )
    pending = DelLatentPending(v1.pending_del_latent_file())
    summary = promote_records(
        list(iter_records_from_v2_obj(v2_obj_dir, run_tag="t")),
        layout=v1, cfg=cfg, pending=pending,
    )
    assert summary.promoted == 1, f"summary={summary}"
    # The v1 view should round-trip back to the same RGB pixels we'd get
    # from naive PIL decode of the source PNG (this fake NPZ has no alpha).
    import io
    import numpy as np
    from PIL import Image
    z = np.load(img_npz_root / "05" / "objA.npz")
    src_pixels = np.array(Image.open(io.BytesIO(bytes(z["089.png"].tobytes()))).convert("RGB"))
    actual = np.array(Image.open(v1.before_view_paths("05", "objA")[0]).convert("RGB"))
    assert np.array_equal(actual, src_pixels), "before/view_0 pixels differ from NPZ source"


def test_promote_without_image_npz_falls_back_to_img_enc(
    tmp_path: Path, v2_obj_dir: Path,
):
    """When image_npz_root is None, the old img_enc_root path still works."""
    data_root = _fake_before_assets(tmp_path)
    v1 = V1Layout(root=tmp_path / "v1")
    cfg = PromoterConfig(
        rule={"required_passes": ["gate_text_align", "gate_quality"]},
        link_mode=LinkMode.HARDLINK,
        slat_root=data_root / "slat",
        view_indices=[89, 90, 91, 100, 8],
        img_enc_root=data_root / "img_Enc",
    )
    pending = DelLatentPending(v1.pending_del_latent_file())
    summary = promote_records(
        list(iter_records_from_v2_obj(v2_obj_dir, run_tag="t")),
        layout=v1, cfg=cfg, pending=pending,
    )
    assert summary.promoted == 1
    assert (v1.before_view_paths("05", "objA")[0]).read_bytes() == b"png"



def test_promote_npz_rgba_is_composited_onto_white(
    tmp_path: Path, v2_obj_dir: Path,
):
    """When the NPZ entry is RGBA with transparency, the v1 ``before`` view
    must be composited onto white (matching ``load_views_from_npz``), not
    written with alpha intact (which would render as a checkerboard)."""
    # alpha=0 → fully transparent everywhere → after compositing onto white,
    # every pixel must be (255, 255, 255).
    img_npz_root = _fake_image_npz(tmp_path, mode="RGBA", alpha=0)
    data_root = _fake_before_assets(tmp_path)
    v1 = V1Layout(root=tmp_path / "v1")
    cfg = PromoterConfig(
        rule={"required_passes": ["gate_text_align", "gate_quality"]},
        link_mode=LinkMode.HARDLINK,
        slat_root=data_root / "slat",
        view_indices=[89, 90, 91, 100, 8],
        image_npz_root=img_npz_root,
    )
    pending = DelLatentPending(v1.pending_del_latent_file())
    summary = promote_records(
        list(iter_records_from_v2_obj(v2_obj_dir, run_tag="t")),
        layout=v1, cfg=cfg, pending=pending,
    )
    assert summary.promoted == 1
    import numpy as np
    from PIL import Image
    out = np.array(Image.open(v1.before_view_paths("05", "objA")[0]).convert("RGB"))
    assert out.shape == (4, 4, 3)
    assert (out == 255).all(), \
        "transparent RGBA NPZ entry should composite to all-white, not stay transparent"



def test_blocklist_excludes_records_from_named_run(
    tmp_path: Path, v2_obj_dir: Path,
):
    """Records whose source_run_tag is in cfg.source_blocklist must be
    counted as ``blocklisted`` (not ``promoted``, ``filtered`` or
    ``failed``) and must never materialize files on disk.

    Regression guard for the shard05 upside-down-overview poisoning:
    even if a future caller forgets to drop shard05 from --source-runs,
    the YAML-level blocklist refuses to admit any of its records.
    """
    img_npz_root = _fake_image_npz(tmp_path)
    data_root = _fake_before_assets(tmp_path)
    v1 = V1Layout(root=tmp_path / "v1")
    cfg = PromoterConfig(
        rule={"required_passes": ["gate_text_align", "gate_quality"]},
        link_mode=LinkMode.HARDLINK,
        slat_root=data_root / "slat",
        view_indices=[89, 90, 91, 100, 8],
        image_npz_root=img_npz_root,
        source_blocklist=frozenset({"pipeline_v2_shard05"}),
    )
    pending = DelLatentPending(v1.pending_del_latent_file())
    # Tag the records with the blocklisted run name.
    recs = list(iter_records_from_v2_obj(v2_obj_dir, run_tag="pipeline_v2_shard05"))
    assert len(recs) >= 1, "fixture should yield at least one record"
    summary = promote_records(recs, layout=v1, cfg=cfg, pending=pending)

    assert summary.blocklisted == len(recs), \
        f"expected {len(recs)} blocklisted, got summary={summary}"
    assert summary.promoted == 0
    assert summary.failed == 0
    assert summary.filtered == 0
    # No before/views should be on disk.
    assert not (v1.before_view_paths("05", "objA")[0]).exists()
    # And no edit dir.
    assert not (v1.edit_dir("05", "objA", "del_objA_000")).exists()


def test_blocklist_lets_other_runs_through(tmp_path: Path, v2_obj_dir: Path):
    """A blocklist on one run name must not affect records from another."""
    img_npz_root = _fake_image_npz(tmp_path)
    data_root = _fake_before_assets(tmp_path)
    v1 = V1Layout(root=tmp_path / "v1")
    cfg = PromoterConfig(
        rule={"required_passes": ["gate_text_align", "gate_quality"]},
        link_mode=LinkMode.HARDLINK,
        slat_root=data_root / "slat",
        view_indices=[89, 90, 91, 100, 8],
        image_npz_root=img_npz_root,
        source_blocklist=frozenset({"some_other_bad_run"}),
    )
    pending = DelLatentPending(v1.pending_del_latent_file())
    recs = list(iter_records_from_v2_obj(v2_obj_dir, run_tag="pipeline_v2_shard06"))
    summary = promote_records(recs, layout=v1, cfg=cfg, pending=pending)
    assert summary.blocklisted == 0
    assert summary.promoted >= 1
