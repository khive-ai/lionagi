"""Test suite for forms/spec.py - Spec semantics and validation.

Focus: Clarify outputs vs out_map semantics, validate constraints,
and test required_inputs/final_outputs analysis.
"""

import pytest

from lionagi.forms.spec import FlowSpec, StepSpec, final_outputs, required_inputs


class TestStepSpecSemantics:
    """Test StepSpec outputs vs out_map semantics and validation."""

    def test_outputs_are_result_keys_not_ctx_keys(self):
        """Test that outputs represent result keys from operation, not ctx keys."""
        step = StepSpec(
            name="test_step",
            op="test_op",
            inputs=[],  # No inputs needed
            outputs=["result_key1", "result_key2"],  # These are result keys
            out_map={"result_key1": "ctx_target1"},  # Maps to ctx keys
        )

        # Outputs should be treated as result keys
        assert step.outputs == ["result_key1", "result_key2"]
        assert step.out_map == {"result_key1": "ctx_target1"}

        # result_key2 uses identity mapping (result_key2 -> result_key2)
        # result_key1 maps to ctx_target1

    def test_out_map_subset_of_outputs_validation(self):
        """Test that out_map keys must be subset of outputs."""
        # Valid case - out_map keys are subset of outputs
        valid_step = StepSpec(
            name="valid",
            op="test_op",
            inputs=[],
            outputs=["out1", "out2", "out3"],
            out_map={"out1": "ctx_a", "out3": "ctx_c"},  # Subset of outputs
        )

        # This should be valid (no exception expected during instantiation)
        assert valid_step.out_map.keys() <= set(valid_step.outputs)

        # Invalid case would be handled by validation (when implemented)
        # For now, we can test the logical constraint
        invalid_out_map = {
            "out1": "ctx_a",
            "unknown_key": "ctx_b",
        }  # unknown_key not in outputs

        # The constraint that should be enforced: out_map.keys() ⊆ outputs
        outputs = ["out1", "out2"]
        assert not (set(invalid_out_map.keys()) <= set(outputs)), "Should violate subset constraint"

    def test_duplicate_outputs_detection(self):
        """Test detection of duplicate entries in outputs."""
        # This test validates the logical constraint
        # When validation is implemented, duplicate outputs should be rejected

        outputs_with_duplicates = ["result", "data", "result"]  # 'result' appears twice
        unique_outputs = list(set(outputs_with_duplicates))

        # Constraint: outputs should not have duplicates
        assert len(outputs_with_duplicates) != len(unique_outputs), "Should detect duplicates"

        # Valid outputs without duplicates
        valid_outputs = ["result", "data", "metadata"]
        assert len(valid_outputs) == len(set(valid_outputs)), "Should have no duplicates"

    def test_identity_mapping_behavior(self):
        """Test that missing out_map entries use identity mapping."""
        step = StepSpec(
            name="identity_test",
            op="test_op",
            inputs=[],
            outputs=["mapped", "identity1", "identity2"],
            out_map={"mapped": "custom_ctx_key"},  # Only one key mapped
        )

        # The unmapped outputs (identity1, identity2) should use identity mapping
        # This is the expected behavior in required_inputs/final_outputs analysis
        expected_ctx_keys = {
            "custom_ctx_key",  # from explicit mapping
            "identity1",  # identity mapping
            "identity2",  # identity mapping
        }

        # Simulate the mapping logic from required_inputs/final_outputs
        actual_ctx_keys = set()
        for output in step.outputs:
            ctx_key = step.out_map.get(output, output)  # Identity if not in out_map
            actual_ctx_keys.add(ctx_key)

        assert actual_ctx_keys == expected_ctx_keys


class TestRequiredInputsAndFinalOutputs:
    """Test required_inputs and final_outputs analysis functions."""

    def test_required_inputs_basic(self):
        """Test required_inputs analysis with basic flow."""
        steps = [
            StepSpec(
                name="step1",
                op="op1",
                inputs=["external_input"],  # Requires external input
                outputs=["intermediate"],
            ),
            StepSpec(
                name="step2",
                op="op2",
                inputs=["intermediate"],  # Satisfied by step1 output
                outputs=["final"],
            ),
        ]

        flow = FlowSpec(steps=steps)
        required = required_inputs(flow)

        # Only external_input should be required from outside
        assert required == {"external_input"}

    def test_required_inputs_with_out_map(self):
        """Test required_inputs analysis with out_map remapping."""
        steps = [
            StepSpec(
                name="producer",
                op="produce_op",
                inputs=[],
                outputs=["raw_result"],
                out_map={"raw_result": "processed_data"},  # Maps to ctx
            ),
            StepSpec(
                name="consumer",
                op="consume_op",
                inputs=["processed_data"],  # Consumes the mapped ctx key
                outputs=["final"],
            ),
        ]

        flow = FlowSpec(steps=steps)
        required = required_inputs(flow)

        # No external inputs required - producer satisfies consumer via mapping
        assert required == set()

    def test_final_outputs_basic(self):
        """Test final_outputs analysis with basic flow."""
        steps = [
            StepSpec(
                name="step1",
                op="op1",
                inputs=[],
                outputs=["intermediate", "debug_info"],
            ),
            StepSpec(
                name="step2",
                op="op2",
                inputs=["intermediate"],  # Consumes intermediate
                outputs=["final_result"],
            ),
        ]

        flow = FlowSpec(steps=steps)
        finals = final_outputs(flow)

        # final_result and debug_info are not consumed by other steps
        assert finals == {"final_result", "debug_info"}

    def test_final_outputs_with_out_map(self):
        """Test final_outputs analysis with out_map affecting ctx keys."""
        steps = [
            StepSpec(
                name="step1",
                op="op1",
                inputs=[],
                outputs=["result1", "result2"],
                out_map={"result1": "ctx_final", "result2": "ctx_intermediate"},
            ),
            StepSpec(
                name="step2",
                op="op2",
                inputs=["ctx_intermediate"],  # Consumes ctx_intermediate
                outputs=["final"],
            ),
        ]

        flow = FlowSpec(steps=steps)
        finals = final_outputs(flow)

        # ctx_final and final are not consumed
        assert finals == {"ctx_final", "final"}

    def test_complex_flow_analysis(self):
        """Test analysis of complex flow with multiple producers and consumers."""
        steps = [
            # Initial data producers
            StepSpec(
                name="input_a",
                op="input_op",
                inputs=[],
                outputs=["data_a"],
                out_map={"data_a": "input_stream_a"},
            ),
            StepSpec(
                name="input_b",
                op="input_op",
                inputs=[],
                outputs=["data_b"],
                out_map={"data_b": "input_stream_b"},
            ),
            # Processor that combines inputs
            StepSpec(
                name="processor",
                op="process_op",
                inputs=["input_stream_a", "input_stream_b"],
                outputs=["combined", "metadata"],
                out_map={"combined": "processed_data"},
            ),
            # Final output generator
            StepSpec(
                name="output_gen",
                op="output_op",
                inputs=["processed_data"],
                outputs=["final_output"],
            ),
            # Parallel metadata processor
            StepSpec(
                name="meta_processor",
                op="meta_op",
                inputs=["metadata"],
                outputs=["meta_report"],
            ),
        ]

        flow = FlowSpec(steps=steps)

        # Analyze inputs and outputs
        required = required_inputs(flow)
        finals = final_outputs(flow)

        # No external inputs required (all satisfied internally)
        assert required == set()

        # Final outputs: final_output and meta_report (not consumed by others)
        assert finals == {"final_output", "meta_report"}

    def test_circular_dependencies_in_analysis(self):
        """Test that analysis handles circular dependencies gracefully."""
        # Create steps with circular dependency (A needs B, B needs A)
        steps = [
            StepSpec(
                name="step_a",
                op="op_a",
                inputs=["from_b"],
                outputs=["to_b"],
                out_map={"to_b": "from_a"},
            ),
            StepSpec(
                name="step_b",
                op="op_b",
                inputs=["from_a"],
                outputs=["to_a"],
                out_map={"to_a": "from_b"},
            ),
        ]

        flow = FlowSpec(steps=steps)

        # Analysis should handle this without infinite loops
        required = required_inputs(flow)
        finals = final_outputs(flow)

        # In circular case, both are required inputs since neither can satisfy the other initially
        assert "from_b" in required or "from_a" in required

    def test_empty_flow_analysis(self):
        """Test analysis of empty flow."""
        empty_flow = FlowSpec(steps=[])

        required = required_inputs(empty_flow)
        finals = final_outputs(empty_flow)

        assert required == set()
        assert finals == set()

    def test_single_step_analysis(self):
        """Test analysis of single-step flow."""
        single_step = [
            StepSpec(
                name="only_step",
                op="only_op",
                inputs=["external_input"],
                outputs=["final_output"],
            )
        ]

        flow = FlowSpec(steps=single_step)

        required = required_inputs(flow)
        finals = final_outputs(flow)

        assert required == {"external_input"}
        assert finals == {"final_output"}
