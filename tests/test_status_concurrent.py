"""Regression test: concurrent update_step must not lose step entries."""
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def _write_step(tmp_path_str: str, step_name: str) -> None:
    """Worker for multiprocessing test — must be top-level for pickling."""
    import random, time
    from pathlib import Path
    from partcraft.pipeline_v2.paths import PipelineRoot
    from partcraft.pipeline_v2.status import STATUS_OK, update_step
    root = PipelineRoot(Path(tmp_path_str))
    ctx = root.context("05", "concurrent_test_obj")
    time.sleep(random.random() * 0.002)
    update_step(ctx, step_name, status=STATUS_OK, n_ok=1)


def test_concurrent_update_step_multiprocess(tmp_path):
    """Two OS processes writing different step keys must not overwrite each other."""
    from multiprocessing import Process
    from partcraft.pipeline_v2.paths import PipelineRoot
    from partcraft.pipeline_v2.status import load_status

    root = PipelineRoot(tmp_path)
    ctx = root.context("05", "concurrent_test_obj")
    ctx.dir.mkdir(parents=True, exist_ok=True)

    lost = 0
    for _ in range(50):
        if ctx.status_path.exists():
            ctx.status_path.unlink()

        p1 = Process(target=_write_step, args=(str(tmp_path), "s5_trellis"))
        p2 = Process(target=_write_step, args=(str(tmp_path), "s5b_del_mesh"))
        p1.start(); p2.start()
        p1.join(); p2.join()

        steps = load_status(ctx).get("steps") or {}
        if "s5_trellis" not in steps or "s5b_del_mesh" not in steps:
            lost += 1

    assert lost == 0, (
        f"Lost {lost}/50 status updates across OS processes"
    )


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


def test_dump_stage_batches_groups_d_and_d2():
    """Stages with the same parallel_group must end up in the same batch."""
    import yaml
    from partcraft.pipeline_v2.scheduler import dump_stage_batches

    # Minimal inline config: D and D2 share a parallel_group, others do not.
    cfg = yaml.safe_load("""
pipeline:
  stages:
    - {name: A, servers: none, steps: [s1]}
    - {name: C, servers: none, steps: [s4]}
    - name: D
      desc: TRELLIS 3D edit
      servers: none
      steps: [s5]
      use_gpus: true
      parallel_group: "D+D2"
    - name: D2
      desc: deletion PLY
      servers: none
      steps: [s5b]
      parallel_group: "D+D2"
    - {name: E, servers: none, steps: [s6], use_gpus: true}
""")
    result = dump_stage_batches(cfg, ["A", "C", "D", "D2", "E"])
    assert result == [["A"], ["C"], ["D", "D2"], ["E"]], f"Unexpected batches: {result}"

    # No parallel_group → every stage in its own batch (serial)
    cfg_no_group = yaml.safe_load("""
pipeline:
  stages:
    - {name: D, servers: none, steps: [s5]}
    - {name: D2, servers: none, steps: [s5b]}
""")
    assert dump_stage_batches(cfg_no_group, ["D", "D2"]) == [["D"], ["D2"]]


def test_dump_stage_batches_allows_one_server_stage_per_group():
    """A group may contain one server-backed stage plus non-server stages."""
    import yaml
    from partcraft.pipeline_v2.scheduler import dump_stage_batches

    cfg = yaml.safe_load("""
pipeline:
  stages:
    - {name: gate_a, servers: vlm, steps: [s1, sq1]}
    - {name: flux_branch, servers: flux, steps: [s4], parallel_group: branches}
    - {name: del_branch, servers: none, steps: [s5b], parallel_group: branches}
    - {name: gate_e_qc, servers: vlm, steps: [sq3]}
""")

    result = dump_stage_batches(cfg, ["gate_a", "flux_branch", "del_branch", "gate_e_qc"])
    assert result == [["gate_a"], ["flux_branch", "del_branch"], ["gate_e_qc"]]


def test_dump_stage_batches_splits_when_two_server_stages_share_group():
    """Two server-backed stages in one group must not run in parallel."""
    import yaml
    from partcraft.pipeline_v2.scheduler import dump_stage_batches

    cfg = yaml.safe_load("""
pipeline:
  stages:
    - {name: v1, servers: vlm, steps: [sq1], parallel_group: g}
    - {name: v2, servers: flux, steps: [s4], parallel_group: g}
""")

    result = dump_stage_batches(cfg, ["v1", "v2"])
    assert result == [["v1"], ["v2"]]
