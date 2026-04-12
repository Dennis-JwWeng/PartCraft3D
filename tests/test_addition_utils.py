"""Unit tests for addition_utils.invert_delete_prompt."""
from __future__ import annotations
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from partcraft.pipeline_v2.addition_utils import invert_delete_prompt


class TestInvertDeletePrompt:
    def test_remove_basic(self):
        assert invert_delete_prompt("Remove the wheel") == "Add the wheel"

    def test_delete_basic(self):
        assert invert_delete_prompt("Delete the engine") == "Add the engine"

    def test_remove_from_locative(self):
        # Critical: 'from' must become 'to'
        assert invert_delete_prompt("Remove the antenna from the robot") == \
            "Add the antenna to the robot"

    def test_delete_from_locative(self):
        assert invert_delete_prompt("Delete the handle from the door") == \
            "Add the handle to the door"

    def test_strip_verb(self):
        assert invert_delete_prompt("Strip the paint") == "Add the paint"

    def test_erase_verb(self):
        assert invert_delete_prompt("Erase the text") == "Add the text"

    def test_get_rid_of(self):
        assert invert_delete_prompt("Get rid of the bumper") == "Add the bumper"

    def test_take_away_from(self):
        assert invert_delete_prompt("Take away the wheel from the car") == \
            "Add the wheel to the car"

    def test_mid_sentence_verb(self):
        result = invert_delete_prompt("Please remove the door handle from the body")
        assert "add" in result.lower()
        assert " to " in result

    def test_fallback_no_known_verb(self):
        assert invert_delete_prompt("Cut off the fin") == "Add back Cut off the fin"

    def test_empty_string(self):
        assert invert_delete_prompt("") == ""

    def test_whitespace_only(self):
        assert invert_delete_prompt("   ") == ""

    def test_no_from_clause(self):
        assert invert_delete_prompt("Remove the rear spoiler") == "Add the rear spoiler"

    def test_from_not_replaced_twice(self):
        result = invert_delete_prompt("Remove the piece from the top from below")
        assert result.count(" to ") == 1
        assert result.count(" from ") == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
