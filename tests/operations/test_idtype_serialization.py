# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""
Test IDType JSON serialization in flow operations.

This test ensures that IDType objects are properly serialized to strings
when passed as parameters to operations, preventing JSON serialization errors.
"""

import json

import pytest

from lionagi.operations.builder import OperationGraphBuilder
from lionagi.operations.flow import flow
from lionagi.session.branch import Branch
from lionagi.session.session import Session


@pytest.mark.asyncio
async def test_aggregation_with_idtype_serialization():
    """
    Test that aggregation operations properly serialize IDType objects.

    Previously, aggregation_sources contained IDType objects that caused
    JSON serialization errors when passed to API calls.
    """
    # Create session and builder
    session = Session()
    builder = OperationGraphBuilder(session)

    # Create multiple operations
    source1 = builder.add_operation("chat", instruction="Source 1")
    source2 = builder.add_operation("chat", instruction="Source 2")
    source3 = builder.add_operation("chat", instruction="Source 3")

    # Create aggregation that combines results
    aggregator = builder.add_aggregation(
        "chat",
        source_node_ids=[source1, source2, source3],
        instruction="Aggregate the results from all sources",
    )

    # Get the graph
    graph = builder.graph

    # Find the aggregation operation
    agg_op = None
    for node in graph.internal_nodes.values():
        if node.metadata.get("aggregation"):
            agg_op = node
            break

    assert agg_op is not None, "Aggregation operation not found"

    # Check that aggregation_sources are strings, not IDType
    sources = agg_op.parameters.get("aggregation_sources", [])
    assert len(sources) == 3, "Should have 3 sources"

    # Verify all sources are JSON serializable strings
    for source in sources:
        assert isinstance(
            source, str
        ), f"Source should be string, got {type(source)}"
        # Should be able to JSON serialize
        json.dumps(source)  # This would fail if it were IDType

    # Verify the entire parameters dict is JSON serializable
    try:
        json.dumps(agg_op.parameters)
    except TypeError as e:
        pytest.fail(f"Parameters should be JSON serializable: {e}")


@pytest.mark.asyncio
async def test_context_with_idtype_keys():
    """
    Test that predecessor context uses string keys for IDType objects.

    Previously, pred.id was used directly as a key, which could cause issues.
    """
    from lionagi.operations.node import Operation
    from lionagi.protocols.graph.edge import Edge
    from lionagi.protocols.graph.graph import Graph

    # Create operations
    op1 = Operation(operation="chat", parameters={"instruction": "First"})
    op2 = Operation(operation="chat", parameters={"instruction": "Second"})

    # Build graph
    graph = Graph()
    graph.add_node(op1)
    graph.add_node(op2)
    graph.add_edge(Edge(head=op1.id, tail=op2.id))

    # Create session
    session = Session()
    branch = Branch(user="test", name="test")
    session.branches.include(branch)
    session.default_branch = branch

    # Execute flow
    result = await flow(session, graph, verbose=False)

    # Check that context keys are strings
    if "context" in op2.parameters:
        context = op2.parameters["context"]
        if isinstance(context, dict):
            for key in context.keys():
                # All keys should be strings
                assert isinstance(
                    key, str
                ), f"Context key should be string, got {type(key)}"
                # Should be JSON serializable
                json.dumps({key: "test"})


@pytest.mark.asyncio
async def test_full_aggregation_flow():
    """
    Test a complete aggregation flow to ensure no JSON serialization errors.
    """
    session = Session()
    builder = OperationGraphBuilder(session)

    # Create a fan-out/fan-in pattern
    start = builder.add_operation("chat", instruction="Start analysis")

    # Fan out to multiple parallel operations
    analysis1 = builder.add_operation(
        "chat", instruction="Analyze aspect 1", branch=session.default_branch
    )
    analysis2 = builder.add_operation(
        "chat", instruction="Analyze aspect 2", branch=session.default_branch
    )
    analysis3 = builder.add_operation(
        "chat", instruction="Analyze aspect 3", branch=session.default_branch
    )

    # Connect start to all analyses
    from lionagi.protocols.graph.edge import Edge

    builder.graph.add_edge(Edge(head=start, tail=analysis1))
    builder.graph.add_edge(Edge(head=start, tail=analysis2))
    builder.graph.add_edge(Edge(head=start, tail=analysis3))

    # Aggregate results
    summary = builder.add_aggregation(
        "chat",
        source_node_ids=[analysis1, analysis2, analysis3],
        instruction="Summarize all analyses",
    )

    # Test that the graph can be executed without JSON errors
    try:
        result = await flow(
            session, builder.graph, context={"test": "value"}, verbose=False
        )

        # Basic checks
        assert "completed_operations" in result
        assert "operation_results" in result

    except TypeError as e:
        if "not JSON serializable" in str(e):
            pytest.fail(f"JSON serialization error should be fixed: {e}")
        raise


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
