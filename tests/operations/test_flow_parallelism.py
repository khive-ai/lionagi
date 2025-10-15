# Copyright (c) 2023-2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""
Tests for flow parallelism and incremental execution patterns.

These tests ensure that:
1. Operations run truly in parallel without locking bottlenecks
2. Completed operations are not re-executed
3. Flows can be expanded and re-run incrementally
4. Context handling works correctly with various types
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from lionagi.fields import Instruct
from lionagi.operations.flow import flow
from lionagi.operations.node import Operation
from lionagi.protocols.graph.edge import Edge
from lionagi.protocols.graph.graph import Graph
from lionagi.session.session import Session


@pytest.mark.asyncio
async def test_flow_true_parallelism():
    """Test that operations truly run in parallel without locking."""
    # Track execution times
    execution_times = {}
    start_time = time.time()

    async def slow_operation(**kwargs):
        """Simulate a slow operation that takes 0.5 seconds."""
        op_id = kwargs.get("op_id")
        execution_times[op_id] = {
            "start": time.time() - start_time,
            "end": None,
        }
        await asyncio.sleep(0.5)
        execution_times[op_id]["end"] = time.time() - start_time
        return f"Result from {op_id}"

    # Create multiple independent operations
    graph = Graph()
    operations = []

    for i in range(5):
        op = Operation(operation="chat", parameters={"op_id": f"op_{i}"})
        graph.add_node(op)
        operations.append(op)

    # Mock branch with chat operation
    branch = MagicMock()
    branch.id = str(uuid4())  # Use valid UUID
    branch.chat = AsyncMock(side_effect=slow_operation)

    # Mock get_operation to return the correct async method
    def mock_get_operation(operation: str):
        if operation == "chat":
            return branch.chat
        return None

    branch.get_operation = MagicMock(side_effect=mock_get_operation)

    session = Session()
    session.branches.include(branch)
    session.default_branch = branch

    # Execute flow with high concurrency
    start_time = time.time()
    result = await flow(
        session,
        graph,
        max_concurrent=10,  # Allow all to run in parallel
        verbose=False,
    )
    total_time = time.time() - start_time

    # Verify all operations completed
    assert len(result["completed_operations"]) == 5

    # If truly parallel, total time should be ~0.5s (plus overhead)
    # If serialized, it would be ~2.5s
    assert (
        total_time < 1.0
    ), f"Operations took {total_time}s - likely serialized!"

    # Check that operations started nearly simultaneously
    start_times = [execution_times[f"op_{i}"]["start"] for i in range(5)]
    max_start_diff = max(start_times) - min(start_times)
    assert (
        max_start_diff < 0.2
    ), f"Operations didn't start together: {max_start_diff}s spread"


@pytest.mark.asyncio
async def test_flow_incremental_execution():
    """Test that flows can be expanded and re-run without re-executing completed operations."""
    execution_count = {}

    async def counting_operation(**kwargs):
        """Count how many times each operation is executed."""
        op_name = kwargs.get("name", "unknown")
        execution_count[op_name] = execution_count.get(op_name, 0) + 1
        return f"Result from {op_name}"

    # Create initial graph
    graph = Graph()
    root = Operation(operation="chat", parameters={"name": "root"})
    graph.add_node(root)

    # Mock branch
    branch = MagicMock()
    branch.id = str(uuid4())
    branch.chat = AsyncMock(side_effect=counting_operation)

    # Mock get_operation to return the correct async method
    def mock_get_operation(operation: str):
        if operation == "chat":
            return branch.chat
        return None

    branch.get_operation = MagicMock(side_effect=mock_get_operation)

    # Mock clone to return a proper branch
    def mock_clone(sender=None):
        cloned = MagicMock()
        cloned.id = str(uuid4())
        cloned.chat = AsyncMock(side_effect=counting_operation)
        cloned.clone = MagicMock(side_effect=mock_clone)
        cloned._message_manager = MagicMock()
        cloned._message_manager.pile = MagicMock()
        cloned._message_manager.pile.clear = MagicMock()

        # Mock get_operation for cloned branch too
        def cloned_get_operation(operation: str):
            if operation == "chat":
                return cloned.chat
            return None

        cloned.get_operation = MagicMock(side_effect=cloned_get_operation)
        return cloned

    branch.clone = MagicMock(side_effect=mock_clone)

    session = Session()
    session.branches.include(branch)
    session.default_branch = branch

    # First execution
    result1 = await flow(session, graph, verbose=False)
    assert execution_count["root"] == 1
    assert len(result1["completed_operations"]) == 1

    # Add more operations to the same graph
    child1 = Operation(operation="chat", parameters={"name": "child1"})
    child2 = Operation(operation="chat", parameters={"name": "child2"})
    graph.add_node(child1)
    graph.add_node(child2)
    graph.add_edge(Edge(head=root.id, tail=child1.id))
    graph.add_edge(Edge(head=root.id, tail=child2.id))

    # Second execution - root should NOT be re-executed
    result2 = await flow(session, graph, verbose=False)

    # Verify execution counts
    assert execution_count["root"] == 1, "Root was re-executed!"
    assert execution_count["child1"] == 1
    assert execution_count["child2"] == 1
    assert len(result2["completed_operations"]) == 3

    # Add a third layer
    grandchild = Operation(operation="chat", parameters={"name": "grandchild"})
    graph.add_node(grandchild)
    graph.add_edge(Edge(head=child1.id, tail=grandchild.id))
    graph.add_edge(Edge(head=child2.id, tail=grandchild.id))

    # Third execution - only grandchild should execute
    result3 = await flow(session, graph, verbose=False)

    assert execution_count["root"] == 1, "Root was re-executed!"
    assert execution_count["child1"] == 1, "Child1 was re-executed!"
    assert execution_count["child2"] == 1, "Child2 was re-executed!"
    assert execution_count["grandchild"] == 1
    assert len(result3["completed_operations"]) == 4


@pytest.mark.asyncio
async def test_flow_context_type_handling():
    """Test that context handles both string and dict types correctly."""

    async def context_checker(**kwargs):
        """Verify context is properly formatted."""
        context = kwargs.get("context")
        if isinstance(context, str):
            return {"context_was": "string", "value": context}
        elif isinstance(context, dict):
            return {"context_was": "dict", "keys": list(context.keys())}
        else:
            return {"context_was": "other", "type": type(context).__name__}

    # Test with string context
    graph = Graph()
    op1 = Operation(
        operation="chat",
        parameters={
            "instruction": "test",
            "context": "string-context-value",  # String context
        },
    )
    op2 = Operation(operation="chat", parameters={"name": "child"})
    graph.add_node(op1)
    graph.add_node(op2)
    graph.add_edge(Edge(head=op1.id, tail=op2.id))

    branch = MagicMock()
    branch.id = str(uuid4())
    branch.chat = AsyncMock(side_effect=context_checker)

    # Mock get_operation to return the correct async method
    def mock_get_operation(operation: str):
        if operation == "chat":
            return branch.chat
        return None

    branch.get_operation = MagicMock(side_effect=mock_get_operation)

    # Mock clone to return a proper branch
    def mock_clone(sender=None):
        cloned = MagicMock()
        cloned.id = str(uuid4())
        cloned.chat = AsyncMock(side_effect=context_checker)
        cloned.clone = MagicMock(side_effect=mock_clone)
        cloned._message_manager = MagicMock()
        cloned._message_manager.pile = MagicMock()
        cloned._message_manager.pile.clear = MagicMock()
        cloned.metadata = {}

        # Mock get_operation for cloned branch too
        def cloned_get_operation(operation: str):
            if operation == "chat":
                return cloned.chat
            return None

        cloned.get_operation = MagicMock(side_effect=cloned_get_operation)
        return cloned

    branch.clone = MagicMock(side_effect=mock_clone)

    session = Session()
    session.branches.include(branch)
    session.default_branch = branch

    # Execute with additional flow context
    result = await flow(
        session, graph, context={"flow_level": "data"}, verbose=False
    )

    # The second operation should have merged contexts
    op2_result = result["operation_results"][op2.id]
    assert op2_result["context_was"] == "dict"
    assert (
        "original_context" in op2_result["keys"]
        or "flow_level" in op2_result["keys"]
    )


@pytest.mark.asyncio
async def test_flow_dynamic_branch_allocation():
    """Test that branches are pre-allocated correctly for all operations."""
    branch_creation_count = 0

    def counting_clone(sender=None):
        """Count branch clones."""
        nonlocal branch_creation_count
        branch_creation_count += 1

        # Create a simple mock branch
        new_branch = MagicMock()
        new_branch.id = str(uuid4())
        new_branch.clone = MagicMock(
            side_effect=lambda sender=None: counting_clone(sender)
        )
        new_branch._message_manager = MagicMock()
        new_branch._message_manager.pile = MagicMock()
        new_branch._message_manager.pile.clear = MagicMock()
        return new_branch

    # Create a complex graph
    graph = Graph()

    # Create a tree structure
    root = Operation(operation="operate", parameters={"instruction": "root"})
    graph.add_node(root)

    branches = []
    for i in range(3):
        branch_op = Operation(
            operation="operate", parameters={"instruction": f"branch_{i}"}
        )
        graph.add_node(branch_op)
        graph.add_edge(Edge(head=root.id, tail=branch_op.id))
        branches.append(branch_op)

        # Add children to each branch
        for j in range(2):
            leaf_op = Operation(
                operation="operate",
                parameters={"instruction": f"leaf_{i}_{j}"},
            )
            graph.add_node(leaf_op)
            graph.add_edge(Edge(head=branch_op.id, tail=leaf_op.id))

    # Mock branch and session
    default_branch = MagicMock()
    default_branch.id = str(uuid4())
    default_branch._message_manager = MagicMock()
    default_branch._message_manager.pile = MagicMock()
    default_branch._message_manager.pile.clear = MagicMock()
    default_branch.metadata = {}
    default_branch.clone = MagicMock(
        side_effect=lambda sender=None: counting_clone(sender)
    )

    async def mock_operate(**kwargs):
        return "result"

    default_branch.operate = AsyncMock(side_effect=mock_operate)

    session = Session()
    session.default_branch = default_branch

    # Execute flow
    result = await flow(session, graph, verbose=True)

    # Count total operations (1 root + 3 branches + 6 leaves = 10)
    total_ops = 10
    assert len(result["completed_operations"]) == total_ops

    # All operations should get branches pre-allocated
    # The count might be less than total_ops due to optimizations
    assert branch_creation_count > 0, "No branches were created!"


@pytest.mark.asyncio
async def test_flow_aggregation_pattern():
    """Test the aggregation pattern with dynamic fan-out."""

    async def list_generator(**kwargs):
        """Generate a list of sub-tasks."""
        # Create a mock result that has instruct_model attribute
        result = MagicMock()
        result.instruct_model = [
            Instruct(instruction="Research topic A", context="context_a"),
            Instruct(instruction="Research topic B", context="context_b"),
            Instruct(instruction="Research topic C", context="context_c"),
        ]
        return result

    async def researcher(**kwargs):
        """Simulate research operation."""
        instruction = kwargs.get("instruction", "")
        return f"Research results for: {instruction}"

    async def synthesizer(**kwargs):
        """Synthesize results from multiple sources."""
        sources = kwargs.get("aggregation_sources", [])
        # In real implementation, this would access results from sources
        return f"Synthesized {len(sources)} research results"

    # Phase 1: Generate tasks
    graph1 = Graph()
    root = Operation(operation="chat", parameters={})
    graph1.add_node(root)

    # Mock branch
    branch = MagicMock()
    branch.id = str(uuid4())
    branch.chat = AsyncMock(
        side_effect=list_generator
    )  # Use chat instead of generate
    branch.operate = AsyncMock(side_effect=researcher)
    branch.communicate = AsyncMock(side_effect=synthesizer)

    # Mock get_operation to return the correct async method
    def mock_get_operation(operation: str):
        operation_map = {
            "chat": branch.chat,
            "operate": branch.operate,
            "communicate": branch.communicate,
        }
        return operation_map.get(operation)

    branch.get_operation = MagicMock(side_effect=mock_get_operation)

    # Mock clone to return a proper branch
    def mock_clone(sender=None):
        cloned = MagicMock()
        cloned.id = str(uuid4())
        cloned.chat = AsyncMock(
            side_effect=list_generator
        )  # Use chat instead of generate
        cloned.operate = AsyncMock(side_effect=researcher)
        cloned.communicate = AsyncMock(side_effect=synthesizer)
        cloned.clone = MagicMock(side_effect=mock_clone)
        cloned._message_manager = MagicMock()
        cloned._message_manager.pile = MagicMock()
        cloned._message_manager.pile.clear = MagicMock()
        cloned.metadata = {}

        # Mock get_operation for cloned branch too
        def cloned_get_operation(operation: str):
            operation_map = {
                "chat": cloned.chat,
                "operate": cloned.operate,
                "communicate": cloned.communicate,
            }
            return operation_map.get(operation)

        cloned.get_operation = MagicMock(side_effect=cloned_get_operation)
        return cloned

    branch.clone = MagicMock(side_effect=mock_clone)

    session = Session()
    session.branches.include(branch)
    session.default_branch = branch

    # Execute phase 1
    result1 = await flow(session, graph1, verbose=False)
    instruct_model = result1["operation_results"][root.id].instruct_model

    # Phase 2: Create researcher nodes based on results
    graph2 = Graph()
    research_nodes = []

    for i, instruct in enumerate(instruct_model):
        node = Operation(
            operation="operate",
            parameters=instruct.to_dict(),  # Put instruct fields in parameters
        )
        graph2.add_node(node)
        research_nodes.append(node)

    # Add aggregation
    synthesis = Operation(
        operation="communicate",
        parameters={
            "aggregation_sources": [n.id for n in research_nodes],
            "instruction": "Synthesize the research",
        },
        metadata={"aggregation": True},
    )
    graph2.add_node(synthesis)
    for node in research_nodes:
        graph2.add_edge(Edge(head=node.id, tail=synthesis.id))

    # Execute phase 2
    result2 = await flow(session, graph2, verbose=False)

    # Verify results
    assert len(research_nodes) == 3
    assert synthesis.id in result2["operation_results"]
    synthesis_result = result2["operation_results"][synthesis.id]
    assert "Synthesized 3 research results" in synthesis_result


@pytest.mark.asyncio
async def test_flow_lock_contention_measurement():
    """Test that measures actual lock contention during parallel execution."""
    lock_wait_times = []
    operation_timings = {}

    # Create a more realistic branch mock that simulates lock contention
    class TimingLock:
        def __init__(self):
            self._lock = asyncio.Lock()

        async def __aenter__(self):
            start = time.time()
            await self._lock.__aenter__()
            wait_time = time.time() - start
            if wait_time > 0.001:  # Only record meaningful waits
                lock_wait_times.append(wait_time)

        async def __aexit__(self, *args):
            await self._lock.__aexit__(*args)

    async def timed_operation(**kwargs):
        """Operation that tracks its execution time."""
        op_id = kwargs.get("op_id")
        start = time.time()
        await asyncio.sleep(0.1)  # Simulate work
        operation_timings[op_id] = time.time() - start
        return f"Result {op_id}"

    # Create many parallel operations
    graph = Graph()
    num_operations = 20

    for i in range(num_operations):
        op = Operation(operation="chat", parameters={"op_id": f"op_{i}"})
        graph.add_node(op)

    # Setup session with our timing lock
    branch = MagicMock()
    branch.id = str(uuid4())
    branch.chat = AsyncMock(side_effect=timed_operation)

    # Mock clone
    def mock_clone(sender=None):
        cloned = MagicMock()
        cloned.id = str(uuid4())
        cloned.chat = AsyncMock(side_effect=timed_operation)
        cloned.clone = MagicMock(side_effect=mock_clone)
        cloned._message_manager = MagicMock()
        cloned._message_manager.pile = MagicMock()
        cloned._message_manager.pile.clear = MagicMock()
        cloned.metadata = {}
        return cloned

    branch.clone = MagicMock(side_effect=mock_clone)

    session = Session()
    session.branches.include(branch)
    session.default_branch = branch

    # Replace the branches lock with our timing lock
    # We can't directly set async_lock, so we'll patch it
    original_lock = session.branches.async_lock
    timing_lock = TimingLock()

    # Monkey patch the async_lock property
    type(session.branches).async_lock = property(lambda self: timing_lock)

    # Execute with high concurrency
    start = time.time()
    result = await flow(session, graph, max_concurrent=20, verbose=False)
    total_time = time.time() - start

    # Analysis
    assert len(result["completed_operations"]) == num_operations

    # With our fix, lock contention should be minimal
    # We should see very few lock waits during execution
    significant_waits = [w for w in lock_wait_times if w > 0.01]
    assert (
        len(significant_waits) < 5
    ), f"Too many lock waits: {len(significant_waits)}"

    # Total execution time should be close to single operation time
    # since they run in parallel
    assert (
        total_time < 0.5
    ), f"Execution took {total_time}s - likely serialized!"


@pytest.mark.asyncio
async def test_flow_error_recovery_with_parallelism():
    """Test that errors in parallel operations don't break the flow."""

    async def flaky_operation(**kwargs):
        """Operation that fails for specific inputs."""
        op_id = kwargs.get("op_id")
        if "fail" in op_id:
            raise ValueError(f"Simulated failure in {op_id}")
        return f"Success {op_id}"

    # Create mixed operations
    graph = Graph()
    success_ops = []
    fail_ops = []

    for i in range(5):
        op = Operation(operation="chat", parameters={"op_id": f"success_{i}"})
        graph.add_node(op)
        success_ops.append(op)

    for i in range(3):
        op = Operation(operation="chat", parameters={"op_id": f"fail_{i}"})
        graph.add_node(op)
        fail_ops.append(op)

    # Add operations that depend on both success and failure ops
    mixed_dep = Operation(
        operation="chat", parameters={"op_id": "mixed_dependency"}
    )
    graph.add_node(mixed_dep)
    for dep in success_ops[:2]:  # Only depend on successful ones
        graph.add_edge(Edge(head=dep.id, tail=mixed_dep.id))

    branch = MagicMock()
    branch.id = str(uuid4())
    branch.chat = AsyncMock(side_effect=flaky_operation)

    # Mock get_operation to return the correct async method
    def mock_get_operation(operation: str):
        if operation == "chat":
            return branch.chat
        return None

    branch.get_operation = MagicMock(side_effect=mock_get_operation)

    # Mock clone
    def mock_clone(sender=None):
        cloned = MagicMock()
        cloned.id = str(uuid4())
        cloned.chat = AsyncMock(side_effect=flaky_operation)
        cloned.clone = MagicMock(side_effect=mock_clone)
        cloned._message_manager = MagicMock()
        cloned._message_manager.pile = MagicMock()
        cloned._message_manager.pile.clear = MagicMock()
        cloned.metadata = {}

        # Mock get_operation for cloned branch too
        def cloned_get_operation(operation: str):
            if operation == "chat":
                return cloned.chat
            return None

        cloned.get_operation = MagicMock(side_effect=cloned_get_operation)
        return cloned

    branch.clone = MagicMock(side_effect=mock_clone)

    session = Session()
    session.branches.include(branch)
    session.default_branch = branch

    # Execute flow
    result = await flow(session, graph, verbose=False)

    # Verify partial success - all operations should be in results even if failed
    assert (
        len(result["operation_results"]) == 9
    )  # All ops should have results (some with errors)

    # Check specific results
    for op in success_ops:
        assert "Success" in str(result["operation_results"][op.id])

    for op in fail_ops:
        # Check that the result is either None or contains error
        op_result = result["operation_results"][op.id]
        assert op_result is None or (
            isinstance(op_result, dict) and "error" in op_result
        )

    # Mixed dependency should succeed since it only depends on successful ops
    assert (
        "Success mixed_dependency" in result["operation_results"][mixed_dep.id]
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
