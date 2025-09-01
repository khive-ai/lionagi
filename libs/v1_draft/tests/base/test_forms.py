# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for base/forms.py - Assignment parsing and form validation.

Focus: Parse assignment validation, input/output checks, and edge cases.
"""

import pytest

from lionagi.base.forms import Form, parse_assignment


class TestAssignmentParsing:
    """Test parse_assignment function with various input patterns."""

    def test_parse_assignment_single_step(self):
        """Test parsing single step assignment."""
        initial, final, steps = parse_assignment("a,b->c")

        assert initial == ["a", "b"]
        assert final == ["c"]
        assert steps == [(["a", "b"], ["c"])]

    def test_parse_assignment_multi_step(self):
        """Test parsing multi-step assignment."""
        initial, final, steps = parse_assignment("a,b->c; c->d,e")

        assert initial == ["a", "b"]
        assert final == ["d", "e"]
        assert steps == [(["a", "b"], ["c"]), (["c"], ["d", "e"])]

    def test_parse_assignment_complex_flow(self):
        """Test parsing complex flow with intermediate values."""
        initial, final, steps = parse_assignment("x,y->z; z,w->a,b; a->c")

        assert initial == ["x", "y", "w"]  # w is needed but not produced
        assert final == ["b", "c"]  # b and c are produced but never consumed
        assert steps == [(["x", "y"], ["z"]), (["z", "w"], ["a", "b"]), (["a"], ["c"])]

    def test_parse_assignment_with_whitespace(self):
        """Test parsing handles whitespace correctly."""
        initial, final, steps = parse_assignment("  a , b  ->  c ; c  ->  d  , e  ")

        assert initial == ["a", "b"]
        assert final == ["d", "e"]
        assert steps == [(["a", "b"], ["c"]), (["c"], ["d", "e"])]

    def test_parse_assignment_invalid_segments(self):
        """Test parse_assignment rejects segments without arrow."""
        with pytest.raises(ValueError, match="Invalid segment"):
            parse_assignment("a,b c")  # Missing ->

        with pytest.raises(ValueError, match="Invalid segment"):
            parse_assignment("a,b->c; d,e")  # Second segment missing ->

    def test_parse_assignment_empty_segments_ignored(self):
        """Test that empty segments are ignored."""
        initial, final, steps = parse_assignment("a->b;; ;c->d")

        assert initial == ["a", "c"]
        assert final == ["b", "d"]
        assert steps == [(["a"], ["b"]), (["c"], ["d"])]

    def test_parse_assignment_duplicate_handling(self):
        """Test handling of duplicate inputs/outputs."""
        # Same output produced multiple times
        initial, final, steps = parse_assignment("a->x; b->x")

        assert "x" in final  # x is final output
        assert steps == [(["a"], ["x"]), (["b"], ["x"])]

    def test_parse_assignment_cycle_detection(self):
        """Test parsing doesn't fail on cycles (logic should handle)."""
        # This creates a cycle but parsing should work
        initial, final, steps = parse_assignment("a->b; b->a")

        assert steps == [(["a"], ["b"]), (["b"], ["a"])]
        # In this case, both a and b are needed as inputs and produced as outputs
        # The logic determines a is initial input, b is not final output


class TestFormClass:
    """Test Form class functionality."""

    def test_form_parse_integration(self):
        """Test Form.parse() integrates with parse_assignment."""
        form = Form(assignment="a,b->c; c->d,e")
        form.parse()

        assert form.input_fields == ["a", "b"]
        assert form.output_fields == ["d", "e"]
        assert form.steps == [(["a", "b"], ["c"]), (["c"], ["d", "e"])]

    def test_form_check_inputs_success(self):
        """Test Form.check_inputs() passes when all inputs present."""
        form = Form(assignment="a,b->c", values={"a": 1, "b": 2, "extra": 3})
        form.parse()

        # Should not raise
        form.check_inputs()

    def test_form_check_inputs_failure(self):
        """Test Form.check_inputs() raises on missing inputs."""
        form = Form(assignment="a,b,c->d", values={"a": 1, "b": 2})  # Missing 'c'
        form.parse()

        with pytest.raises(ValueError, match="Missing inputs"):
            form.check_inputs()

    def test_form_check_outputs_success(self):
        """Test Form.check_outputs() passes when all outputs present."""
        form = Form(assignment="a->b,c", values={"a": 1, "b": 2, "c": 3})
        form.parse()

        # Should not raise
        form.check_outputs()

    def test_form_check_outputs_failure(self):
        """Test Form.check_outputs() raises on missing outputs."""
        form = Form(assignment="a->b,c", values={"a": 1, "b": 2})  # Missing 'c'
        form.parse()

        with pytest.raises(ValueError, match="Missing outputs"):
            form.check_outputs()

    def test_form_to_instructions(self):
        """Test Form.to_instructions() returns expected structure."""
        form = Form(assignment="a->b", guidance="Test guidance", task="Test task")

        instructions = form.to_instructions()

        assert instructions == {
            "assignment": "a->b",
            "guidance": "Test guidance",
            "task": "Test task",
        }

    def test_form_get_results(self):
        """Test Form.get_results() returns output field values."""
        form = Form(assignment="a,b->c,d", values={"a": 1, "b": 2, "c": 3, "d": 4, "extra": 5})
        form.parse()

        results = form.get_results()

        # Should only return output fields
        assert results == {"c": 3, "d": 4}

    def test_form_get_results_missing_values(self):
        """Test Form.get_results() handles missing values gracefully."""
        form = Form(
            assignment="a->b,c",
            values={"a": 1, "b": 2},  # Missing 'c'
        )
        form.parse()

        results = form.get_results()

        # Should return None for missing values
        assert results == {"b": 2, "c": None}

    def test_form_default_values(self):
        """Test Form default values and initialization."""
        form = Form()

        # Test defaults
        assert form.assignment == ""
        assert form.input_fields == []
        assert form.output_fields == []
        assert form.steps == []
        assert form.values == {}
        assert form.guidance == ""
        assert form.task == ""
        assert form.has_processed is False

        # Should have a generated ID
        assert len(form.id) > 0

    def test_form_msgspec_serialization(self):
        """Test Form serialization/deserialization with msgspec."""
        import msgspec

        original = Form(assignment="a->b", values={"a": 1}, guidance="test", task="task")
        original.parse()

        # Serialize and deserialize
        json_data = msgspec.json.encode(original)
        restored = msgspec.json.decode(json_data, type=Form)

        # Verify restoration
        assert restored.assignment == original.assignment
        assert restored.input_fields == original.input_fields
        assert restored.output_fields == original.output_fields
        assert restored.values == original.values
        assert restored.guidance == original.guidance
        assert restored.task == original.task

    def test_form_workflow_integration(self):
        """Test complete Form workflow: parse -> validate -> execute -> check."""
        form = Form(
            assignment="name,age->greeting,category",
            values={"name": "Alice", "age": 25},
            task="Generate greeting and categorize person",
        )

        # Step 1: Parse assignment
        form.parse()
        assert form.input_fields == ["name", "age"]
        assert form.output_fields == ["greeting", "category"]

        # Step 2: Check inputs (should pass)
        form.check_inputs()

        # Step 3: Simulate processing (add outputs)
        form.values["greeting"] = f"Hello, {form.values['name']}!"
        form.values["category"] = "adult" if form.values["age"] >= 18 else "minor"
        form.has_processed = True

        # Step 4: Check outputs (should pass)
        form.check_outputs()

        # Step 5: Get final results
        results = form.get_results()
        assert results == {"greeting": "Hello, Alice!", "category": "adult"}

    def test_form_complex_assignment_workflow(self):
        """Test Form with complex multi-step assignment."""
        form = Form(
            assignment="raw_data,config->processed; processed,template->formatted,metadata",
            values={
                "raw_data": [1, 2, 3],
                "config": {"format": "json"},
                "template": "{{data}}",
            },
        )

        form.parse()

        # Verify parsing
        assert form.input_fields == ["raw_data", "config", "template"]
        assert form.output_fields == ["formatted", "metadata"]
        assert len(form.steps) == 2

        # Check inputs
        form.check_inputs()

        # Simulate step 1: raw_data,config->processed
        form.values["processed"] = {
            "data": form.values["raw_data"],
            "fmt": form.values["config"]["format"],
        }

        # Simulate step 2: processed,template->formatted,metadata
        form.values["formatted"] = f"Formatted: {form.values['processed']}"
        form.values["metadata"] = {"steps": 2, "template_used": True}

        # Check outputs
        form.check_outputs()

        # Verify final results
        results = form.get_results()
        assert "formatted" in results
        assert "metadata" in results
        assert results["metadata"]["steps"] == 2


class TestFormEdgeCases:
    """Test Form behavior with edge cases and error conditions."""

    def test_form_empty_assignment(self):
        """Test Form behavior with empty assignment."""
        form = Form(assignment="")
        form.parse()

        assert form.input_fields == []
        assert form.output_fields == []
        assert form.steps == []

        # check_inputs should pass (no inputs required)
        form.check_inputs()

        # check_outputs should pass (no outputs required)
        form.check_outputs()

    def test_form_self_referential_assignment(self):
        """Test Form with self-referential assignment."""
        form = Form(assignment="a->a")
        form.parse()

        # 'a' is both input and output, so it shouldn't be in final_outputs
        # This tests the logic of: final = produced but never consumed
        assert "a" in form.input_fields
        # 'a' is produced but also consumed (as input), so not final
        assert form.output_fields == []

    def test_form_validation_order_independence(self):
        """Test that input/output validation works regardless of parsing order."""
        form = Form(assignment="a,b->c", values={"a": 1, "b": 2})

        # Should be able to check inputs before parsing
        with pytest.raises(ValueError, match="Missing inputs"):
            form.check_inputs()  # No input_fields set yet

        form.parse()

        # Now input check should work
        form.check_inputs()

    def test_form_partial_values(self):
        """Test Form behavior with partial value assignment."""
        form = Form(
            assignment="a,b,c->x,y,z",
            values={"a": 1, "x": 10},  # Partial inputs and outputs
        )
        form.parse()

        # Input validation should fail (missing b, c)
        with pytest.raises(ValueError, match="Missing inputs"):
            form.check_inputs()

        # Output validation should fail (missing y, z)
        with pytest.raises(ValueError, match="Missing outputs"):
            form.check_outputs()

        # But get_results should work and return available values
        results = form.get_results()
        assert results == {"x": 10, "y": None, "z": None}
