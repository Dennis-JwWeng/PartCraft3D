"""Regression test: concurrent update_step must not lose step entries."""
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest


def test_concurrent_update_step_no_data_loss(tmp_path):
    """Two threads writing different step keys must never overwrite each other."""
    from partcraft.pipeline_v2.paths import PipelineRoot
    from partcraft.pipeline_v2.status import STATUS_OK, load_status, update_step

    root = PipelineRoot(tmp_path)
    ctx = root.context("05", "concurrent_test_obj")
    ctx.dir.mkdir(parents=True, exist_ok=True)

    lost = 0
    for _ in range(200):
        if ctx.status_path.exists():
            ctx.status_path.unlink()

        def write_s5():
            time.sleep(random.random() * 0.002)
            update_step(ctx, "s5_trellis", status=STATUS_OK, n_ok=1)

        def write_s5b():
            time.sleep(random.random() * 0.002)
            update_step(ctx, "s5b_del_mesh", status=STATUS_OK, n_ok=1)

        with ThreadPoolExecutor(max_workers=2) as ex:
            ex.submit(write_s5)
            ex.submit(write_s5b)

        steps = load_status(ctx).get("steps") or {}
        if "s5_trellis" not in steps or "s5b_del_mesh" not in steps:
            lost += 1

    assert lost == 0, (
        f"Lost {lost}/200 status updates — concurrent write is not safe"
    )
