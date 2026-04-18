from pathlib import Path

from partcraft.cleaning.v1.source_v3 import (
    iter_records_from_v3_obj, iter_records_from_v3_run,
)


def test_v3_adapter_extracts_record(v3_obj_dir: Path):
    recs = list(iter_records_from_v3_obj(v3_obj_dir, run_tag="shard08_run"))
    assert len(recs) == 1
    rec = recs[0]
    assert rec.source_pipeline == "v3"
    assert rec.source_run_tag == "shard08_run"
    assert rec.shard == "08"
    assert rec.obj_id == "objB"
    assert rec.passes["gate_text_align"].passed is True
    assert rec.passes["gate_quality"].passed is True
    assert "v3.gate_quality" in rec.passes["gate_quality"].producer


def test_v3_run_walker_finds_obj(v3_obj_dir: Path):
    run_root = v3_obj_dir.parents[3]
    recs = list(iter_records_from_v3_run(run_root, run_tag="shard08_run"))
    assert len(recs) == 1
    assert recs[0].edit_id == "del_objB_000"
