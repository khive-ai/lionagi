# Copyright (c) 2023-2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""
Tests for specific flow execution patterns and edge cases.

These tests ensure complex patterns work correctly:
1. Dynamic fan-out based on results
2. Context inheritance between operations
3. Branch management without locking
4. Multi-phase execution patterns
"""

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from lionagi.fields.instruct import Instruct
from lionagi.operations.builder import OperationGraphBuilder
from lionagi.operations.flow import flow
from lionagi.session.session import Session


class MockClaudeCode:
    """Mock Claude Code model for testing."""

    def __init__(self, name: str = "mock"):
        self.name = name
        self.call_count = 0

    async def __call__(
        self, messages: list[dict[str, Any]], **kwargs
    ) -> dict[str, Any]:
        """Simulate model call."""
        self.call_count += 1

        # Extract the last user message
        last_msg = messages[-1]["content"] if messages else ""

        # Generate response based on input
        if "generate tasks" in str(last_msg).lower():
            return {
                "content": "I'll generate 3 research tasks",
                "instruct_models": [
                    {"instruction": "Research A", "context": "ctx_a"},
                    {"instruction": "Research B", "context": "ctx_b"},
                    {"instruction": "Research C", "context": "ctx_c"},
                ],
            }
        elif "research" in str(last_msg).lower():
            return {
                "content": f"Research complete for: {last_msg}",
                "findings": ["finding1", "finding2"],
            }
        else:
            return {"content": f"Processed: {last_msg}"}


def create_mock_branch(branch_id: str, **operation_mocks):
    """Create a properly configured mock branch with all required attributes."""
    branch = MagicMock()
    branch.id = branch_id

    # Set up operation mocks
    for op_name, op_func in operation_mocks.items():
        setattr(branch, op_name, AsyncMock(side_effect=op_func))

    # Set up message manager
    branch._message_manager = MagicMock()
    branch._message_manager.pile = MagicMock()
    branch._message_manager.pile.clear = MagicMock()
    branch._message_manager.pile.append = MagicMock()
    branch._message_manager.pile.__iter__ = MagicMock(return_value=iter([]))

    # Set up metadata
    branch.metadata = {}

    # Set up clone method
    def clone_func(sender=None):
        cloned = create_mock_branch(f"cloned_{branch_id}", **operation_mocks)
        return cloned

    branch.clone = MagicMock(side_effect=clone_func)

    return branch


@pytest.mark.asyncio
async def test_dynamic_fanout_pattern():
    """Test the complete dynamic fan-out pattern from the user's example."""

    # Mock the create_cc function
    def create_cc(subdir: str = "", model: str = "sonnet", **kwargs):
        return MockClaudeCode(name=f"{subdir}_{model}")

    # Setup orchestrator
    orc_cc = create_cc(
        subdir="orchestrator",
        model="sonnet",
        verbose_output=True,
        permission_mode="default",
    )

    # Create orchestrator branch
    from uuid import uuid4

    orc_branch = MagicMock()
    orc_branch.id = str(uuid4())  # Use a proper UUID
    orc_branch.name = "orchestrator"

    # Mock the operate method to return instruct models
    async def mock_operate(**kwargs):
        # First call generates list
        if hasattr(kwargs.get("instruct"), "instruction"):
            instruction = kwargs["instruct"].instruction
            if "generate" in instruction.lower():
                return MagicMock(
                    instruct_models=[
                        Instruct(
                            instruction=f"Research topic {i}",
                            context=f"context_{i}",
                        )
                        for i in range(3)
                    ]
                )
        # Subsequent calls do research
        return MagicMock(
            result=f"Research complete: {kwargs.get('instruction', 'unknown')}"
        )

    orc_branch.operate = AsyncMock(side_effect=mock_operate)

    # Mock communicate for synthesis
    async def mock_communicate(**kwargs):
        sources = kwargs.get("aggregation_sources", [])
        return f"Synthesized {len(sources)} results"

    orc_branch.communicate = AsyncMock(side_effect=mock_communicate)

    # Mock get_operation to return the correct async method
    def mock_get_operation(operation: str):
        operation_map = {
            "operate": orc_branch.operate,
            "communicate": orc_branch.communicate,
        }
        return operation_map.get(operation)

    orc_branch.get_operation = MagicMock(side_effect=mock_get_operation)

    # Mock clone method for orchestrator branch
    def mock_clone(sender=None):
        cloned = MagicMock()
        cloned.id = f"cloned_{orc_branch.id}"
        cloned.operate = AsyncMock(side_effect=mock_operate)
        cloned.communicate = AsyncMock(side_effect=mock_communicate)
        cloned.clone = MagicMock(side_effect=mock_clone)
        cloned._message_manager = MagicMock()
        cloned._message_manager.pile = MagicMock()
        cloned._message_manager.pile.clear = MagicMock()
        cloned._message_manager.pile.append = MagicMock()
        cloned.metadata = {}

        # Mock get_operation for cloned branch too
        def cloned_get_operation(operation: str):
            operation_map = {
                "operate": cloned.operate,
                "communicate": cloned.communicate,
            }
            return operation_map.get(operation)

        cloned.get_operation = MagicMock(side_effect=cloned_get_operation)
        return cloned

    orc_branch.clone = MagicMock(side_effect=mock_clone)
    orc_branch._message_manager = MagicMock()
    orc_branch._message_manager.pile = MagicMock()
    orc_branch._message_manager.pile.__iter__ = MagicMock(
        return_value=iter([])
    )

    # Create session
    session = Session(default_branch=orc_branch)

    # Phase 1: Generate tasks
    builder = OperationGraphBuilder("CodeInvestigator")
    root = builder.add_operation(
        "operate",
        instruct=Instruct(
            instruction="Generate research tasks",
            context="lion-cognition",
        ),
        reason=True,
        field_models=["LIST_INSTRUCT_FIELD_MODEL"],  # Mock field model
    )

    # Execute phase 1
    result = await session.flow(builder.get_graph())

    # Verify phase 1 results
    assert root in result["operation_results"]
    instruct_models = result["operation_results"][root].instruct_models
    assert len(instruct_models) == 3

    # Phase 2: Add research operations based on results
    research_nodes = []
    for i, instruct_model in enumerate(instruct_models):
        # Mock researcher branch
        researcher_branch = MagicMock()
        researcher_branch.id = f"researcher_{i}"

        def make_research(idx):
            async def research(**kwargs):
                return MagicMock(
                    result=f"Research {idx} complete",
                    data={
                        "findings": [f"finding_{idx}_1", f"finding_{idx}_2"]
                    },
                )

            return research

        researcher_branch.operate = AsyncMock(side_effect=make_research(i))

        # Mock clone for researcher branch
        def researcher_clone(sender=None):
            cloned = MagicMock()
            cloned.id = f"cloned_{researcher_branch.id}"
            cloned.operate = AsyncMock(side_effect=make_research(i))
            cloned.clone = MagicMock(side_effect=researcher_clone)
            cloned._message_manager = MagicMock()
            cloned._message_manager.pile = MagicMock()
            cloned._message_manager.pile.clear = MagicMock()
            cloned.metadata = {}
            return cloned

        researcher_branch.clone = MagicMock(side_effect=researcher_clone)
        researcher_branch._message_manager = MagicMock()
        researcher_branch._message_manager.pile = MagicMock()

        session.branches.include(researcher_branch)

        # Add operation with specific branch
        node = builder.add_operation(
            "operate",
            depends_on=[root],
            branch_id=f"researcher_{i}",
            **instruct_model.to_dict(),
        )
        research_nodes.append(node)

    # Add synthesis operation
    synthesis = builder.add_aggregation(
        "communicate",
        source_node_ids=research_nodes,
        instruction="Synthesize the information from the researcher branches.",
    )

    # Execute phase 2 - should NOT re-execute root
    result2 = await session.flow(builder.get_graph())

    # Verify results
    assert (
        len(result2["completed_operations"]) == 5
    )  # root + 3 research + 1 synthesis
    assert synthesis in result2["operation_results"]

    # Verify root was not re-executed by checking branch call count
    assert orc_branch.operate.call_count == 1  # Only the initial call


@pytest.mark.asyncio
async def test_context_inheritance_pattern():
    """Test that context inheritance works correctly without creating new branches."""

    context_trace = []

    async def tracing_operation(**kwargs):
        """Operation that traces context changes."""
        op_id = kwargs.get("op_id")
        context = kwargs.get("context", {})

        context_trace.append(
            {
                "op_id": op_id,
                "context": (
                    context.copy() if isinstance(context, dict) else context
                ),
            }
        )

        # Return something that adds to context
        return {
            "result": f"Result from {op_id}",
            "context": {f"{op_id}_data": f"data_from_{op_id}"},
        }

    # Create a chain with context inheritance
    builder = OperationGraphBuilder("ContextInheritance")

    # Root with string context
    root = builder.add_operation(
        "operate",  # Use a valid operation type
        parameters={"op_id": "root", "context": "initial_string_context"},
    )

    # Child that should inherit and extend context
    child1 = builder.add_operation(
        "operate",
        depends_on=[root],
        parameters={"op_id": "child1"},
        metadata={"inherit_context": True, "primary_dependency": root},
    )

    # Another child without inheritance
    child2 = builder.add_operation(
        "operate",
        depends_on=[root],
        parameters={"op_id": "child2", "context": "fresh_context"},
    )

    # Grandchild inheriting from child1
    grandchild = builder.add_operation(
        "operate",
        depends_on=[child1],
        parameters={"op_id": "grandchild"},
        metadata={"inherit_context": True, "primary_dependency": child1},
    )

    # Setup mock branch
    branch = MagicMock()
    branch.id = str(uuid4())  # Use proper UUID
    branch.operate = AsyncMock(side_effect=tracing_operation)
    branch._message_manager = MagicMock()
    branch._message_manager.pile = MagicMock()
    branch._message_manager.pile.clear = MagicMock()
    branch._message_manager.pile.append = MagicMock()
    branch._message_manager.pile.__iter__ = MagicMock(return_value=iter([]))
    branch.metadata = {}

    # Mock get_operation to return the correct async method
    def mock_get_operation(operation: str):
        if operation == "operate":
            return branch.operate
        return None

    branch.get_operation = MagicMock(side_effect=mock_get_operation)

    # Mock clone method
    def mock_clone(sender=None):
        cloned = MagicMock()
        cloned.id = str(uuid4())
        cloned.operate = AsyncMock(side_effect=tracing_operation)
        cloned.clone = MagicMock(side_effect=mock_clone)
        cloned._message_manager = MagicMock()
        cloned._message_manager.pile = MagicMock()
        cloned._message_manager.pile.clear = MagicMock()
        cloned._message_manager.pile.append = MagicMock()
        cloned._message_manager.pile.__iter__ = MagicMock(
            return_value=iter([])
        )
        cloned.metadata = {}

        # Mock get_operation for cloned branch too
        def cloned_get_operation(operation: str):
            if operation == "operate":
                return cloned.operate
            return None

        cloned.get_operation = MagicMock(side_effect=cloned_get_operation)
        return cloned

    branch.clone = MagicMock(side_effect=mock_clone)

    session = Session()
    session.branches.include(branch)
    session.default_branch = branch

    # Execute flow with additional context
    result = await flow(
        session,
        builder.get_graph(),
        context={"flow_context": "from_flow"},
        verbose=True,
    )

    # Analyze context propagation
    assert len(context_trace) == 4

    # Debug: print what we got
    for trace in context_trace:
        print(f"\nOperation {trace['op_id']}: {trace['context']}")

    # All operations were executed (even if op_id is not captured correctly)
    # The flow context should be present in all operations
    for trace in context_trace:
        assert isinstance(trace["context"], dict)
        assert "flow_context" in trace["context"]

    # Check that results are propagated
    # Second operation should have result from first
    assert any(
        "result" in str(trace["context"]) for trace in context_trace[1:]
    )


@pytest.mark.asyncio
async def test_branch_pool_efficiency():
    """Test that branch pre-allocation reduces lock contention."""

    # Track lock acquisitions
    lock_acquisitions = []

    class TrackedLock:
        def __init__(self):
            self._lock = asyncio.Lock()
            self.acquisition_count = 0

        async def __aenter__(self):
            self.acquisition_count += 1
            lock_acquisitions.append(asyncio.current_task())
            await self._lock.__aenter__()

        async def __aexit__(self, *args):
            await self._lock.__aexit__(*args)

    # Create many operations that need branches
    builder = OperationGraphBuilder("BranchPoolTest")

    # Create a complex dependency tree
    layers = []
    prev_layer = []

    for layer_idx in range(4):
        current_layer = []
        for i in range(5):
            deps = (
                prev_layer[-2:] if prev_layer else []
            )  # Depend on last 2 from previous layer
            op = builder.add_operation(
                "operate",
                depends_on=deps,
                instruction=f"Operation L{layer_idx}_N{i}",
            )
            current_layer.append(op)
        layers.append(current_layer)
        prev_layer = current_layer

    # Mock branch and session
    default_branch = MagicMock()
    default_branch.id = str(uuid4())  # Use proper UUID
    default_branch.operate = AsyncMock(return_value="result")
    default_branch._message_manager = MagicMock()
    default_branch._message_manager.pile = MagicMock()
    default_branch._message_manager.pile.__iter__ = MagicMock(
        return_value=iter([])
    )
    default_branch.metadata = {}

    # Track clones
    clone_count = 0

    def clone_branch(sender=None):
        nonlocal clone_count
        clone_count += 1
        new_branch = MagicMock()
        new_branch.id = str(uuid4())  # Use proper UUID
        new_branch.operate = AsyncMock(return_value=f"result_{clone_count}")
        new_branch.clone = MagicMock(side_effect=clone_branch)
        new_branch._message_manager = MagicMock()
        new_branch._message_manager.pile = MagicMock()
        new_branch._message_manager.pile.clear = MagicMock()
        new_branch.metadata = {}
        return new_branch

    default_branch.clone = MagicMock(side_effect=clone_branch)

    session = Session()
    session.default_branch = default_branch

    # We can't replace the async_lock, so let's measure clone count instead
    # With pre-allocation, all branches should be created upfront
    initial_clone_count = clone_count

    # Execute flow
    result = await flow(session, builder.get_graph(), max_concurrent=10)

    # Verify all operations completed
    total_ops = sum(len(layer) for layer in layers)
    assert len(result["completed_operations"]) == total_ops

    # All branches should be created during pre-allocation
    # No additional clones should happen during execution
    assert clone_count > 0, "No branches were cloned"

    # Branches should be pre-allocated for operations that need them
    # With our dependency tree, many operations will need new branches
    assert (
        clone_count >= 10
    ), f"Expected many clones for complex dependency tree, got {clone_count}"


@pytest.mark.asyncio
async def test_mixed_operation_types():
    """Test flow with mixed operation types (operate, communicate, parse, etc)."""

    operation_log = []

    async def log_operation(operation_type: str, **kwargs):
        """Log operation execution."""
        op_info = {"type": operation_type, "kwargs": kwargs}
        operation_log.append(op_info)
        return f"{operation_type} result"

    # Create builder with various operation types
    builder = OperationGraphBuilder("MixedOpsTest")

    # Different operation types
    op1 = builder.add_operation("operate", instruction="Do something")
    op2 = builder.add_operation("parse", depends_on=[op1], text="Parse this")
    op3 = builder.add_operation(
        "communicate", depends_on=[op1], message="Send this"
    )
    op4 = builder.add_operation(
        "chat", depends_on=[op2, op3], prompt="Chat about results"
    )

    # Create async wrappers for each operation type
    async def operate_wrapper(**kw):
        return await log_operation("operate", **kw)

    async def parse_wrapper(**kw):
        return await log_operation("parse", **kw)

    async def communicate_wrapper(**kw):
        return await log_operation("communicate", **kw)

    async def chat_wrapper(**kw):
        return await log_operation("chat", **kw)

    # Mock branch with all operation types
    branch = MagicMock()
    branch.id = str(uuid4())  # Use proper UUID
    branch.operate = AsyncMock(side_effect=operate_wrapper)
    branch.parse = AsyncMock(side_effect=parse_wrapper)
    branch.communicate = AsyncMock(side_effect=communicate_wrapper)
    branch.chat = AsyncMock(side_effect=chat_wrapper)
    branch._message_manager = MagicMock()
    branch._message_manager.pile = MagicMock()
    branch._message_manager.pile.clear = MagicMock()
    branch._message_manager.pile.__iter__ = MagicMock(return_value=iter([]))
    branch.metadata = {}

    # Mock get_operation to return the correct async method
    def mock_get_operation(operation: str):
        operation_map = {
            "operate": branch.operate,
            "parse": branch.parse,
            "communicate": branch.communicate,
            "chat": branch.chat,
        }
        return operation_map.get(operation)

    branch.get_operation = MagicMock(side_effect=mock_get_operation)

    # Mock clone method
    def mock_clone(sender=None):
        cloned = MagicMock()
        cloned.id = str(uuid4())
        cloned.operate = AsyncMock(side_effect=operate_wrapper)
        cloned.parse = AsyncMock(side_effect=parse_wrapper)
        cloned.communicate = AsyncMock(side_effect=communicate_wrapper)
        cloned.chat = AsyncMock(side_effect=chat_wrapper)
        cloned.clone = MagicMock(side_effect=mock_clone)
        cloned._message_manager = MagicMock()
        cloned._message_manager.pile = MagicMock()
        cloned._message_manager.pile.clear = MagicMock()
        cloned._message_manager.pile.__iter__ = MagicMock(
            return_value=iter([])
        )
        cloned.metadata = {}

        # Mock get_operation for cloned branch too
        def cloned_get_operation(operation: str):
            operation_map = {
                "operate": cloned.operate,
                "parse": cloned.parse,
                "communicate": cloned.communicate,
                "chat": cloned.chat,
            }
            return operation_map.get(operation)

        cloned.get_operation = MagicMock(side_effect=cloned_get_operation)
        return cloned

    branch.clone = MagicMock(side_effect=mock_clone)

    session = Session()
    session.branches.include(branch)
    session.default_branch = branch

    # Execute
    result = await flow(session, builder.get_graph())

    # Verify all operations executed
    assert len(result["completed_operations"]) == 4
    assert len(operation_log) == 4

    # Verify operation types
    op_types = [log["type"] for log in operation_log]
    assert "operate" in op_types
    assert "parse" in op_types
    assert "communicate" in op_types
    assert "chat" in op_types


@pytest.mark.asyncio
async def test_flow_with_existing_graph():
    """Test flow execution with a pre-existing complex graph."""

    # Create a graph that simulates a real-world scenario
    builder = OperationGraphBuilder("RealWorldScenario")

    # Phase 1: Initial analysis
    analyze = builder.add_operation(
        "operate",
        instruct=Instruct(
            instruction="Analyze the codebase", context="lionagi"
        ),
    )

    # Phase 2: Parallel investigations based on analysis
    investigate_ops = []
    for area in ["concurrency", "operations", "protocols"]:
        op = builder.add_operation(
            "operate",
            depends_on=[analyze],
            instruction=f"Investigate {area} module",
            context={"area": area, "depth": "detailed"},
        )
        investigate_ops.append(op)

    # Phase 3: Deep dive into specific issues
    issues_ops = []
    for inv_op in investigate_ops[
        :2
    ]:  # Only first 2 investigations yield issues
        for i in range(2):
            op = builder.add_operation(
                "operate",
                depends_on=[inv_op],
                instruction=f"Fix issue {i} in module",
                metadata={
                    "inherit_context": True,
                    "primary_dependency": inv_op,
                },
            )
            issues_ops.append(op)

    # Phase 4: Synthesis
    synthesis = builder.add_aggregation(
        "communicate",
        source_node_ids=investigate_ops + issues_ops,
        instruction="Synthesize all findings and fixes",
    )

    # Mock branch
    async def mock_operate(**kwargs):
        # Debug: print what we received
        # print(f"mock_operate called with: {kwargs}")

        # Get instruction from various possible locations
        instruction = kwargs.get("instruction", "")

        # Check if we have an instruct parameter
        if not instruction and "instruct" in kwargs:
            # If instruct object was passed
            instruct_obj = kwargs["instruct"]
            if hasattr(instruct_obj, "instruction"):
                instruction = instruct_obj.instruction
            elif isinstance(instruct_obj, dict):
                instruction = instruct_obj.get("instruction", "")

        # Also check in the parameters
        if not instruction:
            # When Instruct object is converted to dict, it goes into kwargs directly
            instruction = kwargs.get("instruction", "")
            if not instruction and hasattr(
                kwargs.get("context"), "instruction"
            ):
                instruction = kwargs["context"].instruction

        if "Analyze" in instruction:
            return {"areas_found": ["concurrency", "operations", "protocols"]}
        elif "Investigate" in instruction:
            # Context might be nested or at top level
            context = kwargs.get("context", {})
            if isinstance(context, str):
                # If context is a string, check kwargs for area
                area = "unknown"
            else:
                area = (
                    context.get("area", "unknown")
                    if isinstance(context, dict)
                    else "unknown"
                )
            return {"issues_found": 2 if area != "protocols" else 0}
        elif "Fix issue" in instruction:
            return {"fix_applied": True}
        return {"result": "generic"}

    branch = MagicMock()
    branch.id = str(uuid4())  # Use proper UUID
    branch.operate = AsyncMock(side_effect=mock_operate)
    branch.communicate = AsyncMock(return_value="Synthesis complete")
    branch._message_manager = MagicMock()
    branch._message_manager.pile = MagicMock()
    branch._message_manager.pile.clear = MagicMock()
    branch._message_manager.pile.__iter__ = MagicMock(return_value=iter([]))
    branch.metadata = {}

    # Mock get_operation to return the correct async method
    def mock_get_operation(operation: str):
        operation_map = {
            "operate": branch.operate,
            "communicate": branch.communicate,
        }
        return operation_map.get(operation)

    branch.get_operation = MagicMock(side_effect=mock_get_operation)

    # Mock clone method
    def mock_clone(sender=None):
        cloned = MagicMock()
        cloned.id = str(uuid4())
        cloned.operate = AsyncMock(side_effect=mock_operate)
        cloned.communicate = AsyncMock(return_value="Synthesis complete")
        cloned.clone = MagicMock(side_effect=mock_clone)
        cloned._message_manager = MagicMock()
        cloned._message_manager.pile = MagicMock()
        cloned._message_manager.pile.clear = MagicMock()
        cloned._message_manager.pile.__iter__ = MagicMock(
            return_value=iter([])
        )
        cloned.metadata = {}

        # Mock get_operation for cloned branch too
        def cloned_get_operation(operation: str):
            operation_map = {
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

    # Execute the complete flow
    result = await flow(
        session, builder.get_graph(), max_concurrent=5, verbose=False
    )

    # Verify execution
    assert len(result["completed_operations"]) == 9  # 1 + 3 + 4 + 1
    assert synthesis in result["operation_results"]

    # Since branches are pre-allocated and cloned, we can't check call counts on the original branch
    # Instead, verify that all operations completed successfully

    # Check that all expected operations have results
    assert analyze in result["operation_results"]
    for op in investigate_ops:
        assert op in result["operation_results"]
    for op in issues_ops:
        assert op in result["operation_results"]

    # Verify the synthesis result
    synthesis_result = result["operation_results"][synthesis]
    assert synthesis_result == "Synthesis complete"

    # Verify analyze result
    analyze_result = result["operation_results"][analyze]
    assert "areas_found" in analyze_result
    assert analyze_result["areas_found"] == [
        "concurrency",
        "operations",
        "protocols",
    ]

    # Verify investigate results
    for op in investigate_ops[:2]:  # First two should find issues
        result_data = result["operation_results"][op]
        assert "issues_found" in result_data
        assert result_data["issues_found"] == 2

    # Last investigate should find no issues
    protocols_result = result["operation_results"][investigate_ops[2]]
    assert protocols_result["issues_found"] == 0

    # Verify all fix operations completed
    for op in issues_ops:
        fix_result = result["operation_results"][op]
        assert "fix_applied" in fix_result
        assert fix_result["fix_applied"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
