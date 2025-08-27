"""Test suite for forms/compile.py - Compile-time dataflow integrity checks.

Focus: Reject ambiguous producers, validate out_map consistency,
and ensure registry products are proper morphisms.
"""

from uuid import uuid4

import pytest

from lionagi.forms.compile import compile_flow_to_graph
from lionagi.forms.spec import FlowSpec, StepSpec


class MockMorphism:
    """Mock morphism that satisfies Morphism protocol."""

    def __init__(self, name: str = "mock", result_keys: set[str] = None):
        self.name = name
        self.requires = set()
        self.io = False
        self.result_keys = result_keys or {"result"}

    async def pre(self, br, **kw) -> bool:
        return True

    async def apply(self, br, **kw) -> dict:
        return {k: f"value_{k}" for k in self.result_keys}

    async def post(self, br, res) -> bool:
        return True


class NonMorphism:
    """Invalid object that doesn't satisfy Morphism protocol."""

    def __init__(self):
        self.name = "invalid"


def create_mock_factory(morphism: MockMorphism):
    """Create a factory function that returns the given morphism."""

    def factory(step_spec: StepSpec):
        return morphism

    return factory


class TestCompileTimeDataflowChecks:
    """Test compile-time validation of dataflow integrity."""

    def test_compile_single_step_identity_mapping(self):
        """Test compilation of single step with identity output mapping."""
        # Create simple flow with one step
        step = StepSpec(
            name="test_step",
            op="mock_op",
            inputs=[],  # No inputs needed
            outputs=["result"],
            out_map={},  # Identity mapping
        )
        flow = FlowSpec(steps=[step])

        # Create registry
        morphism = MockMorphism(result_keys={"result"})
        registry = {"mock_op": create_mock_factory(morphism)}

        # Compile should succeed
        graph, required_inputs, final_outputs = compile_flow_to_graph(flow, registry)

        assert len(graph.nodes) == 1
        assert len(graph.roots) == 1
        assert len(final_outputs) == 1
        assert "result" in final_outputs

    def test_compile_out_map_and_dependencies(self):
        """Test compilation with out_map creating dependencies between steps."""
        # Step A produces 'data' -> maps to ctx key 'processed_data'
        step_a = StepSpec(
            name="step_a",
            op="producer",
            inputs=[],  # No inputs
            outputs=["data"],
            out_map={"data": "processed_data"},
        )

        # Step B consumes 'processed_data' from context
        step_b = StepSpec(
            name="step_b", op="consumer", inputs=["processed_data"], outputs=["final"]
        )

        flow = FlowSpec(steps=[step_a, step_b])

        # Create registry
        producer = MockMorphism(result_keys={"data"})
        consumer = MockMorphism(result_keys={"final"})
        registry = {
            "producer": create_mock_factory(producer),
            "consumer": create_mock_factory(consumer),
        }

        # Compile should succeed and create dependency
        graph, required_inputs, final_outputs = compile_flow_to_graph(flow, registry)

        assert len(graph.nodes) == 2

        # Find nodes by step name
        node_a = None
        node_b = None
        for node in graph.nodes.values():
            if node.params.get("step_name") == "step_a":
                node_a = node
            elif node.params.get("step_name") == "step_b":
                node_b = node

        assert node_a is not None
        assert node_b is not None

        # Verify dependency exists (step_b depends on step_a)
        assert node_a.id in node_b.deps

    def test_duplicate_producer_rejected(self):
        """Test that multiple steps producing same ctx key are rejected."""
        # Two steps both map their output to same ctx key
        step_a = StepSpec(
            name="step_a",
            op="op1",
            inputs=[],
            outputs=["result"],
            out_map={"result": "shared_key"},
        )

        step_b = StepSpec(
            name="step_b",
            op="op2",
            inputs=[],
            outputs=["result"],
            out_map={"result": "shared_key"},  # Same target as step_a
        )

        flow = FlowSpec(steps=[step_a, step_b])

        registry = {
            "op1": create_mock_factory(MockMorphism(result_keys={"result"})),
            "op2": create_mock_factory(MockMorphism(result_keys={"result"})),
        }

        # Should raise ValueError about ambiguous producer
        with pytest.raises(ValueError, match="Ambiguous producer.*shared_key"):
            compile_flow_to_graph(flow, registry)

    def test_out_map_keys_not_in_outputs_rejected(self):
        """Test that out_map keys not in outputs are rejected."""
        step = StepSpec(
            name="invalid_step",
            op="mock_op",
            inputs=[],
            outputs=["valid_output"],
            out_map={
                "valid_output": "ctx_key1",
                "invalid_output": "ctx_key2",  # Not in outputs!
            },
        )

        flow = FlowSpec(steps=[step])

        registry = {"mock_op": create_mock_factory(MockMorphism())}

        # Should raise ValueError about foreign keys
        with pytest.raises(ValueError, match="out_map keys not in outputs.*invalid_output"):
            compile_flow_to_graph(flow, registry)

    def test_registry_object_shape_validated(self):
        """Test that registry factories must return proper morphisms."""
        step = StepSpec(name="test_step", op="invalid_op", inputs=[], outputs=["result"])
        flow = FlowSpec(steps=[step])

        # Registry returns non-morphism
        def invalid_factory(step_spec):
            return NonMorphism()

        registry = {"invalid_op": invalid_factory}

        # Should raise TypeError about invalid morphism
        with pytest.raises(TypeError, match="did not return a Morphism-like object"):
            compile_flow_to_graph(flow, registry)

    def test_registry_missing_morphism_methods(self):
        """Test validation of morphism protocol methods."""

        class IncompleteMorphism:
            name = "incomplete"

            # Missing pre, apply, post methods

        step = StepSpec(name="test_step", op="incomplete_op", inputs=[], outputs=["result"])
        flow = FlowSpec(steps=[step])

        def incomplete_factory(step_spec):
            return IncompleteMorphism()

        registry = {"incomplete_op": incomplete_factory}

        # Should raise TypeError about missing methods
        with pytest.raises(TypeError, match="did not return a Morphism-like object"):
            compile_flow_to_graph(flow, registry)

    def test_step_name_preserved_for_debugging(self):
        """Test that step names are preserved in node params for debugging."""
        step = StepSpec(name="debug_step", op="debug_op", inputs=[], outputs=["result"])
        flow = FlowSpec(steps=[step])

        registry = {"debug_op": create_mock_factory(MockMorphism())}

        graph, _, _ = compile_flow_to_graph(flow, registry)

        # Find the node and verify step name is preserved
        node = list(graph.nodes.values())[0]
        assert node.params.get("step_name") == "debug_step"

    def test_complex_dataflow_validation(self):
        """Test complex multi-step flow with proper validation."""
        # Create a complex flow: Input -> Process -> Transform -> Output
        steps = [
            StepSpec(
                name="input_step",
                op="input_op",
                inputs=[],
                outputs=["raw_data"],
                out_map={"raw_data": "input_data"},
            ),
            StepSpec(
                name="process_step",
                op="process_op",
                inputs=["input_data"],
                outputs=["processed"],
                out_map={"processed": "processed_data"},
            ),
            StepSpec(
                name="transform_step",
                op="transform_op",
                inputs=["processed_data"],
                outputs=["transformed"],
                out_map={"transformed": "final_data"},
            ),
            StepSpec(
                name="output_step",
                op="output_op",
                inputs=["final_data"],
                outputs=["result"],
            ),
        ]

        flow = FlowSpec(steps=steps)

        registry = {
            "input_op": create_mock_factory(MockMorphism(result_keys={"raw_data"})),
            "process_op": create_mock_factory(MockMorphism(result_keys={"processed"})),
            "transform_op": create_mock_factory(MockMorphism(result_keys={"transformed"})),
            "output_op": create_mock_factory(MockMorphism(result_keys={"result"})),
        }

        # Should compile successfully
        graph, required_inputs, final_outputs = compile_flow_to_graph(flow, registry)

        assert len(graph.nodes) == 4
        assert len(final_outputs) == 1
        assert "result" in final_outputs

        # Verify dependency chain exists
        nodes_by_name = {}
        for node in graph.nodes.values():
            step_name = node.params.get("step_name")
            if step_name:
                nodes_by_name[step_name] = node

        # Check dependencies: input -> process -> transform -> output
        assert nodes_by_name["input_step"].id in nodes_by_name["process_step"].deps
        assert nodes_by_name["process_step"].id in nodes_by_name["transform_step"].deps
        assert nodes_by_name["transform_step"].id in nodes_by_name["output_step"].deps

    def test_duplicate_step_names_handling(self):
        """Test handling of duplicate step names."""
        # Create flow with duplicate step names
        steps = [
            StepSpec(name="duplicate", op="op1", inputs=[], outputs=["out1"]),
            StepSpec(name="duplicate", op="op2", inputs=[], outputs=["out2"]),  # Same name!
        ]

        flow = FlowSpec(steps=steps)

        registry = {
            "op1": create_mock_factory(MockMorphism(result_keys={"out1"})),
            "op2": create_mock_factory(MockMorphism(result_keys={"out2"})),
        }

        # Should still work - step names are for debugging, node IDs provide uniqueness
        graph, _, _ = compile_flow_to_graph(flow, registry)

        assert len(graph.nodes) == 2

        # Both nodes should have same step name but different IDs
        step_names = [node.params.get("step_name") for node in graph.nodes.values()]
        assert step_names.count("duplicate") == 2
