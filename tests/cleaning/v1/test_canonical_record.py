from pathlib import Path

import pytest

from partcraft.cleaning.v1.canonical_record import (
    PromotionRecord, PassResult, evaluate_rule,
)


def _ok_passes() -> dict[str, PassResult]:
    return {
        "gate_text_align": PassResult(passed=True, score=1.0,
                                      producer="v3.gate_text_align@x", reason="", ts="t"),
        "gate_quality":    PassResult(passed=True, score=0.9,
                                      producer="v3.gate_quality@x", reason="", ts="t"),
    }


def test_record_holds_source_paths():
    rec = PromotionRecord(
        obj_id="abc", shard="05", edit_id="del_abc_000", edit_type="deletion",
        source_pipeline="v2", source_run_tag="pipeline_v2_shard05",
        source_run_dir=Path("/tmp/run/abc"),
        spec={"prompt": "Remove handle", "selected_part_ids": [0]},
        passes=_ok_passes(),
        after_glb=Path("/tmp/run/abc/edits_3d/del_abc_000/after_new.glb"),
        after_npz=None,
        preview_pngs=[Path(f"/tmp/run/abc/edits_3d/del_abc_000/preview_{k}.png") for k in range(5)],
    )
    assert rec.is_deletion() is True
    assert rec.is_flux_branch() is False


def test_evaluate_rule_passes_when_all_required_pass():
    rule = {"required_passes": ["gate_text_align", "gate_quality"]}
    assert evaluate_rule(_ok_passes(), rule) == (True, "")


def test_evaluate_rule_fails_when_required_pass_failed():
    p = _ok_passes()
    p["gate_quality"] = PassResult(passed=False, score=0.1,
                                   producer="x", reason="bad", ts="t")
    ok, reason = evaluate_rule(p, {"required_passes": ["gate_text_align", "gate_quality"]})
    assert ok is False
    assert "gate_quality" in reason


def test_evaluate_rule_defers_when_required_pass_missing():
    rule = {"required_passes": ["gate_text_align", "gate_quality", "future_pass"]}
    ok, reason = evaluate_rule(_ok_passes(), rule)
    assert ok is False
    assert "future_pass" in reason
    assert "missing" in reason


def test_pass_result_to_json_roundtrip():
    pr = PassResult(passed=True, score=0.5, producer="x", reason="ok", ts="t",
                    extra={"metric_a": 0.3})
    d = pr.to_json()
    pr2 = PassResult.from_json(d)
    assert pr2 == pr


def test_record_qc_dict_contains_source_and_passes():
    rec = PromotionRecord(
        obj_id="abc", shard="05", edit_id="del_abc_000", edit_type="deletion",
        source_pipeline="v2", source_run_tag="pipeline_v2_shard05",
        source_run_dir=Path("/tmp/run/abc"),
        spec={}, passes=_ok_passes(),
        after_glb=Path("/x"), after_npz=None, preview_pngs=[],
    )
    qc = rec.to_qc_json(promoted_at="2026-04-19T12:00:00Z")
    assert qc["edit_id"] == "del_abc_000"
    assert qc["source"]["pipeline_version"] == "v2"
    assert qc["source"]["run_tag"] == "pipeline_v2_shard05"
    assert qc["source"]["promoted_at"] == "2026-04-19T12:00:00Z"
    assert set(qc["passes"].keys()) == {"gate_text_align", "gate_quality"}
