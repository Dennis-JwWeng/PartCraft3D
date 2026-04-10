from __future__ import annotations
import tempfile, unittest
from pathlib import Path
from unittest.mock import MagicMock

def _ctx(tmp, oid="obj001"):
    c = MagicMock(); c.obj_id = oid; c.shard = "00"
    c.dir = tmp / oid; c.dir.mkdir(parents=True, exist_ok=True)
    c.qc_path = c.dir / "qc.json"; return c

class TestQcIo(unittest.TestCase):
    def setUp(self):
        self._t = tempfile.TemporaryDirectory(); self.p = Path(self._t.name)
    def tearDown(self): self._t.cleanup()

    def test_load_missing_skeleton(self):
        from partcraft.pipeline_v2.qc_io import load_qc
        self.assertEqual(load_qc(_ctx(self.p))["edits"], {})

    def test_save_load_roundtrip(self):
        from partcraft.pipeline_v2.qc_io import load_qc, save_qc
        ctx = _ctx(self.p); qc = load_qc(ctx)
        qc["edits"]["del_000"] = {"final_pass": True}; save_qc(ctx, qc)
        self.assertTrue(load_qc(ctx)["edits"]["del_000"]["final_pass"])

    def test_gate_rule_fail(self):
        from partcraft.pipeline_v2.qc_io import update_edit_gate, load_qc
        ctx = _ctx(self.p)
        update_edit_gate(ctx, "del_000", "deletion", "A",
                         rule_result={"pass": False, "checks": {"prompt_too_short": True}})
        e = load_qc(ctx)["edits"]["del_000"]
        self.assertFalse(e["final_pass"])
        self.assertEqual(e["fail_gate"], "A"); self.assertEqual(e["fail_reason"], "prompt_too_short")

    def test_all_pass(self):
        from partcraft.pipeline_v2.qc_io import update_edit_gate, load_qc
        ctx = _ctx(self.p)
        update_edit_gate(ctx, "del_000", "deletion", "A",
                         rule_result={"pass": True, "checks": {}},
                         vlm_result={"pass": True, "score": 0.9, "reason": ""})
        update_edit_gate(ctx, "del_000", "deletion", "E",
                         vlm_result={"pass": True, "score": 0.85, "reason": ""})
        self.assertTrue(load_qc(ctx)["edits"]["del_000"]["final_pass"])

    def test_not_failed_before_qc(self):
        from partcraft.pipeline_v2.qc_io import is_edit_qc_failed
        self.assertFalse(is_edit_qc_failed(_ctx(self.p), "del_000"))

    def test_failed_after_fail(self):
        from partcraft.pipeline_v2.qc_io import update_edit_gate, is_edit_qc_failed
        ctx = _ctx(self.p)
        update_edit_gate(ctx, "del_000", "deletion", "A",
                         rule_result={"pass": False, "checks": {"parts_missing": True}})
        self.assertTrue(is_edit_qc_failed(ctx, "del_000"))

    def test_null_c_gate_counts_as_pass(self):
        from partcraft.pipeline_v2.qc_io import update_edit_gate, load_qc
        ctx = _ctx(self.p)
        update_edit_gate(ctx, "del_000", "deletion", "A",
                         rule_result={"pass": True, "checks": {}},
                         vlm_result={"pass": True, "score": 0.9, "reason": ""})
        update_edit_gate(ctx, "del_000", "deletion", "E",
                         vlm_result={"pass": True, "score": 0.9, "reason": ""})
        self.assertTrue(load_qc(ctx)["edits"]["del_000"]["final_pass"])


class TestGateAFailed(unittest.TestCase):
    """is_gate_a_failed only blocks on Gate A, never on Gate C."""

    def setUp(self):
        self._t = tempfile.TemporaryDirectory()
        self.p = Path(self._t.name)

    def tearDown(self):
        self._t.cleanup()

    def test_no_qc_file_returns_false(self):
        from partcraft.pipeline_v2.qc_io import is_gate_a_failed
        self.assertFalse(is_gate_a_failed(_ctx(self.p), "del_001"))

    def test_gate_c_fail_does_not_block(self):
        """Gate C failure must NOT cause is_gate_a_failed to return True."""
        from partcraft.pipeline_v2.qc_io import update_edit_gate, is_gate_a_failed
        ctx = _ctx(self.p)
        update_edit_gate(ctx, "mod_001", "modification", "C",
                         vlm_result={"pass": False, "reason": "wrong region"})
        self.assertFalse(is_gate_a_failed(ctx, "mod_001"))

    def test_gate_a_fail_blocks(self):
        """Gate A failure must cause is_gate_a_failed to return True."""
        from partcraft.pipeline_v2.qc_io import update_edit_gate, is_gate_a_failed
        ctx = _ctx(self.p)
        update_edit_gate(ctx, "mod_002", "modification", "A",
                         vlm_result={"pass": False, "reason": "wrong part"})
        self.assertTrue(is_gate_a_failed(ctx, "mod_002"))

    def test_gate_a_pass_gate_c_fail_does_not_block(self):
        """Gate A pass + Gate C fail => is_gate_a_failed False."""
        from partcraft.pipeline_v2.qc_io import update_edit_gate, is_gate_a_failed
        ctx = _ctx(self.p)
        update_edit_gate(ctx, "mod_003", "modification", "A",
                         vlm_result={"pass": True})
        update_edit_gate(ctx, "mod_003", "modification", "C",
                         vlm_result={"pass": False, "reason": "global edit"})
        self.assertFalse(is_gate_a_failed(ctx, "mod_003"))

    def test_unknown_edit_id_returns_false(self):
        from partcraft.pipeline_v2.qc_io import update_edit_gate, is_gate_a_failed
        ctx = _ctx(self.p)
        update_edit_gate(ctx, "mod_004", "modification", "A",
                         vlm_result={"pass": False})
        self.assertFalse(is_gate_a_failed(ctx, "nonexistent_edit"))


if __name__ == "__main__": unittest.main()
