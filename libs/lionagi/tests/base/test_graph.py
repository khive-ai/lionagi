"""Test suite for graph.py (OpGraph, OpNode) - TDD Specification Implementation.

Focus: DAG structure, cycle detection, and correct traversal order.
"""

from uuid import uuid4

import pytest

from lionagi.base.graph import OpGraph, OpNode
from lionagi.base.morphism import Morphism


# Mock morphism for testing
class MockMorphism:
    def __init__(self, name: str):
        self.name = name
        self.requires = set()

    async def pre(self, br, **kw) -> bool:
        return True

    async def apply(self, br, **kw) -> dict:
        return {"result": f"from_{self.name}"}

    async def post(self, br, result) -> bool:
        return True


class TestOpGraphStructure:
    """TestSuite: OpGraphStructure - DAG construction, root detection, and validation."""

    def test_graph_construction_and_root_detection(self):
        """Test: GraphConstructionAndRootDetection

        GIVEN Nodes A, B (deps: A), C (deps: A), D (deps: B, C)
        WHEN OpGraph is created
        THEN Graph.roots should contain only A.id
        AND dependencies are correctly stored in each node.
        """
        # Create nodes with dependencies
        node_a = OpNode(id=uuid4(), m=MockMorphism("A"))
        node_b = OpNode(id=uuid4(), m=MockMorphism("B"), deps={node_a.id})
        node_c = OpNode(id=uuid4(), m=MockMorphism("C"), deps={node_a.id})
        node_d = OpNode(id=uuid4(), m=MockMorphism("D"), deps={node_b.id, node_c.id})

        # Create graph
        graph = OpGraph(
            nodes={node_a.id: node_a, node_b.id: node_b, node_c.id: node_c, node_d.id: node_d},
            roots={node_a.id},
        )

        # Verify root detection
        assert graph.roots == {node_a.id}, "Graph should have only A as root"

        # Verify dependencies are correctly stored
        assert node_a.deps == set(), "Node A should have no dependencies"
        assert node_b.deps == {node_a.id}, "Node B should depend on A"
        assert node_c.deps == {node_a.id}, "Node C should depend on A"
        assert node_d.deps == {node_b.id, node_c.id}, "Node D should depend on B and C"

    def test_cycle_detection(self):
        """Test: CycleDetection (CRITICAL)

        GIVEN Nodes A (deps: C), B (deps: A), C (deps: B) # Cyclic dependency
        WHEN attempting to validate DAG
        THEN it should raise ValueError with cycle detection.
        """
        # Create cyclic dependencies: A -> C -> B -> A
        node_a = OpNode(id=uuid4(), m=MockMorphism("A"))
        node_b = OpNode(id=uuid4(), m=MockMorphism("B"))
        node_c = OpNode(id=uuid4(), m=MockMorphism("C"))

        # Set up circular dependencies
        node_a.deps = {node_c.id}
        node_b.deps = {node_a.id}
        node_c.deps = {node_b.id}

        graph = OpGraph(
            nodes={node_a.id: node_a, node_b.id: node_b, node_c.id: node_c},
            roots={node_a.id},  # This will be problematic due to cycle
        )

        # Should detect cycle and raise error
        with pytest.raises(ValueError, match="Cycle detected"):
            graph.validate_dag()

    def test_missing_dependency_detection(self):
        """Test: MissingDependencyDetection

        GIVEN Node B depends on A, but A is not included in the graph initialization
        WHEN attempting to validate DAG
        THEN it should raise ValueError for missing dependency.
        """
        missing_id = uuid4()
        node_b = OpNode(id=uuid4(), m=MockMorphism("B"), deps={missing_id})

        graph = OpGraph(nodes={node_b.id: node_b}, roots=set())

        with pytest.raises(ValueError, match="Missing dependency node"):
            graph.validate_dag()

    def test_topological_sort(self):
        """Test: TopologicalSort (PBT Recommended)

        GIVEN a complex, valid DAG structure (e.g., a diamond shape)
        WHEN performing a topological sort (used internally by Runner)
        THEN the resulting order must ensure every node appears after all its dependencies.
        """
        # Diamond graph: A -> B, A -> C, B -> D, C -> D
        node_a = OpNode(id=uuid4(), m=MockMorphism("A"))
        node_b = OpNode(id=uuid4(), m=MockMorphism("B"), deps={node_a.id})
        node_c = OpNode(id=uuid4(), m=MockMorphism("C"), deps={node_a.id})
        node_d = OpNode(id=uuid4(), m=MockMorphism("D"), deps={node_b.id, node_c.id})

        graph = OpGraph(
            nodes={node_a.id: node_a, node_b.id: node_b, node_c.id: node_c, node_d.id: node_d},
            roots={node_a.id},
        )

        # Get topological order
        topo_order = graph.validate_dag()

        # Verify every node appears after its dependencies
        position = {node_id: i for i, node_id in enumerate(topo_order)}

        # A should come before B and C
        assert position[node_a.id] < position[node_b.id], "A must come before B"
        assert position[node_a.id] < position[node_c.id], "A must come before C"

        # B and C should come before D
        assert position[node_b.id] < position[node_d.id], "B must come before D"
        assert position[node_c.id] < position[node_d.id], "C must come before D"

        # All nodes should be present
        assert len(topo_order) == 4, "All nodes should be in topological order"
        assert set(topo_order) == {
            node_a.id,
            node_b.id,
            node_c.id,
            node_d.id,
        }, "All nodes should be present"

    def test_empty_graph_handling(self):
        """Test handling of empty graphs."""
        empty_graph = OpGraph()
        assert empty_graph.validate_dag() == [], "Empty graph should return empty order"

    def test_single_node_graph(self):
        """Test handling of single node graphs."""
        node = OpNode(id=uuid4(), m=MockMorphism("Single"))
        graph = OpGraph(nodes={node.id: node}, roots={node.id})

        order = graph.validate_dag()
        assert order == [node.id], "Single node graph should return that node"

    def test_complex_dag_validation(self):
        """Test validation of a more complex DAG structure."""
        # Create a more complex graph:
        # A -> B -> D
        # A -> C -> D
        # B -> E
        # C -> F
        # D -> G

        nodes = {}
        for name in ["A", "B", "C", "D", "E", "F", "G"]:
            nodes[name] = OpNode(id=uuid4(), m=MockMorphism(name))

        # Set dependencies
        nodes["B"].deps = {nodes["A"].id}
        nodes["C"].deps = {nodes["A"].id}
        nodes["D"].deps = {nodes["B"].id, nodes["C"].id}
        nodes["E"].deps = {nodes["B"].id}
        nodes["F"].deps = {nodes["C"].id}
        nodes["G"].deps = {nodes["D"].id}

        graph = OpGraph(nodes={node.id: node for node in nodes.values()}, roots={nodes["A"].id})

        order = graph.validate_dag()
        position = {node_id: i for i, node_id in enumerate(order)}

        # Verify all dependency relationships are respected
        assert position[nodes["A"].id] < position[nodes["B"].id]
        assert position[nodes["A"].id] < position[nodes["C"].id]
        assert position[nodes["B"].id] < position[nodes["D"].id]
        assert position[nodes["C"].id] < position[nodes["D"].id]
        assert position[nodes["B"].id] < position[nodes["E"].id]
        assert position[nodes["C"].id] < position[nodes["F"].id]
        assert position[nodes["D"].id] < position[nodes["G"].id]

        # All nodes should be present
        assert len(order) == 7
        assert set(order) == set(node.id for node in nodes.values())

    def test_auto_root_detection(self):
        """Test automatic root detection for nodes with no dependencies."""
        node_a = OpNode(id=uuid4(), m=MockMorphism("A"))
        node_b = OpNode(id=uuid4(), m=MockMorphism("B"), deps={node_a.id})

        # Create graph without explicitly setting roots
        graph = OpGraph(nodes={node_a.id: node_a, node_b.id: node_b})

        # Should automatically detect A as root (indegree 0)
        order = graph.validate_dag()
        assert order[0] == node_a.id, "Node with no dependencies should be detected as root"
