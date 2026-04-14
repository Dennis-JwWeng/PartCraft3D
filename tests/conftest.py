"""pytest configuration: HTML report metadata and shared fixtures."""
from __future__ import annotations

import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "smoke: marks tests as smoke/import tests (fast, no IO)",
    )
    config.addinivalue_line(
        "markers",
        "unit: marks tests as unit tests (no network, no GPU)",
    )
    config.addinivalue_line(
        "markers",
        "real_data: marks tests that require real on-disk data (may skip on CI)",
    )


@pytest.fixture
def tmp_obj_ctx(tmp_path):
    """Return a lightweight mock ObjectContext backed by a real temp directory."""
    from unittest.mock import MagicMock

    obj_id = "test_obj_001"
    ctx = MagicMock()
    ctx.obj_id = obj_id
    ctx.shard = "00"
    ctx.dir = tmp_path / obj_id
    ctx.dir.mkdir(parents=True, exist_ok=True)
    ctx.qc_path = ctx.dir / "qc.json"
    ctx.status_path = ctx.dir / "status.json"
    ctx.parsed_path = ctx.dir / "phase1" / "parsed.json"
    ctx.overview_path = ctx.dir / "phase1" / "overview.png"
    (ctx.dir / "phase1").mkdir(parents=True, exist_ok=True)
    ctx.highlight_path = lambda idx: ctx.dir / "highlights" / f"e{idx:02d}.png"
    ctx.edit_2d_output = lambda eid: ctx.dir / "edits_2d" / f"{eid}_edited.png"
    ctx.edit_3d_png = lambda eid, tag: ctx.dir / "edits_3d" / eid / f"{tag}.png"
    return ctx
