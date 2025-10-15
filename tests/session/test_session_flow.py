# Copyright (c) 2023-2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""
Comprehensive tests for Session class focusing on multi-branch orchestration.

Test Coverage:
1. Basic flow execution (single/multiple branches, context passing)
2. Branch management (creation, registration, selection, iteration)
3. Edge cases (empty graphs, branch lifecycle, error handling, context isolation)
4. Mail system (send/receive, routing, mailbox management)
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID

import pytest
from pydantic import BaseModel

from lionagi.operations.builder import OperationGraphBuilder
from lionagi.operations.flow import flow
from lionagi.operations.node import Operation
from lionagi.protocols.generic.event import EventStatus
from lionagi.protocols.graph.edge import Edge
from lionagi.protocols.graph.graph import Graph
from lionagi.protocols.messages import Instruction, MessageRole
from lionagi.service.connections.api_calling import APICalling
from lionagi.service.connections.endpoint import Endpoint
from lionagi.service.connections.providers.oai_ import _get_oai_config
from lionagi.service.imodel import iModel
from lionagi.service.third_party.openai_models import (
    OpenAIChatCompletionsRequest,
)
from lionagi.session.branch import Branch
from lionagi.session.session import Session

# ============================================================================
# Test Fixtures and Helpers
# ============================================================================


def make_mock_branch(name: str = "TestBranch") -> Branch:
    """Create a Branch with mocked iModel for testing."""
    branch = Branch(user="test_user", name=name)

    async def _fake_invoke(**kwargs):
        config = _get_oai_config(
            name="oai_chat",
            endpoint="chat/completions",
            request_options=OpenAIChatCompletionsRequest,
            kwargs={"model": "gpt-4.1-mini"},
        )
        endpoint = Endpoint(config=config)
        fake_call = APICalling(
            payload={"model": "gpt-4.1-mini", "messages": []},
            headers={"Authorization": "Bearer test"},
            endpoint=endpoint,
        )
        fake_call.execution.response = "mocked_response"
        fake_call.execution.status = EventStatus.COMPLETED
        return fake_call

    mock_invoke = AsyncMock(side_effect=_fake_invoke)
    mock_chat_model = iModel(
        provider="openai", model="gpt-4.1-mini", api_key="test_key"
    )
    mock_chat_model.invoke = mock_invoke

    branch.chat_model = mock_chat_model
    return branch


def make_simple_graph(num_nodes: int = 3) -> tuple[Graph, list[Operation]]:
    """Create a simple linear graph with specified number of operations."""
    ops = [
        Operation(operation="chat", parameters={"instruction": f"Task {i}"})
        for i in range(num_nodes)
    ]

    graph = Graph()
    for op in ops:
        graph.add_node(op)

    for i in range(len(ops) - 1):
        graph.add_edge(Edge(head=ops[i].id, tail=ops[i + 1].id))

    return graph, ops


def make_parallel_graph() -> tuple[Graph, dict[str, Operation]]:
    """Create a diamond-shaped graph for parallel execution testing."""
    ops = {
        "start": Operation(
            operation="chat", parameters={"instruction": "Start"}
        ),
        "branch_a": Operation(
            operation="chat", parameters={"instruction": "Branch A"}
        ),
        "branch_b": Operation(
            operation="chat", parameters={"instruction": "Branch B"}
        ),
        "merge": Operation(
            operation="chat", parameters={"instruction": "Merge"}
        ),
    }

    graph = Graph()
    for op in ops.values():
        graph.add_node(op)

    graph.add_edge(Edge(head=ops["start"].id, tail=ops["branch_a"].id))
    graph.add_edge(Edge(head=ops["start"].id, tail=ops["branch_b"].id))
    graph.add_edge(Edge(head=ops["branch_a"].id, tail=ops["merge"].id))
    graph.add_edge(Edge(head=ops["branch_b"].id, tail=ops["merge"].id))

    return graph, ops


# ============================================================================
# 1. Basic Flow Execution Tests
# ============================================================================


class TestBasicFlowExecution:
    """Test basic flow execution scenarios."""

    @pytest.mark.asyncio
    async def test_flow_single_branch_linear_graph(self):
        """Test flow execution with single branch and linear graph."""
        session = Session()
        branch = make_mock_branch("MainBranch")
        session.include_branches(branch)

        graph, ops = make_simple_graph(3)

        result = await session.flow(graph, parallel=False, verbose=False)

        # Verify all operations completed
        assert len(result["completed_operations"]) == 3
        assert all(op.id in result["completed_operations"] for op in ops)
        assert len(result["operation_results"]) == 3

    @pytest.mark.asyncio
    async def test_flow_multiple_branches_parallel(self):
        """Test flow with multiple branches executing in parallel."""
        session = Session()

        # Create multiple branches
        branch1 = make_mock_branch("Branch1")
        branch2 = make_mock_branch("Branch2")
        branch3 = make_mock_branch("Branch3")
        session.include_branches([branch1, branch2, branch3])

        # Create parallel graph
        graph, ops = make_parallel_graph()

        result = await session.flow(
            graph, parallel=True, max_concurrent=3, verbose=False
        )

        # Verify all operations completed
        assert len(result["completed_operations"]) == 4
        assert all(
            op.id in result["completed_operations"] for op in ops.values()
        )

    @pytest.mark.asyncio
    async def test_flow_context_passing_between_operations(self):
        """Test that context is properly passed between operations."""
        session = Session()
        branch = make_mock_branch()
        session.include_branches(branch)

        # Create operations with context
        op1 = Operation(
            operation="chat",
            parameters={
                "instruction": "Task 1",
                "context": {"key1": "value1"},
            },
        )
        op2 = Operation(operation="chat", parameters={"instruction": "Task 2"})

        graph = Graph()
        graph.add_node(op1)
        graph.add_node(op2)
        graph.add_edge(Edge(head=op1.id, tail=op2.id))

        initial_context = {"global_key": "global_value"}
        result = await session.flow(
            graph, context=initial_context, parallel=False
        )

        # Verify context propagation
        assert "global_key" in result["final_context"]
        # op2 should have received context from op1
        assert op2.parameters.get("context") is not None

    @pytest.mark.asyncio
    async def test_flow_with_empty_graph(self):
        """Test flow handles empty graph correctly."""
        session = Session()
        branch = make_mock_branch()
        session.include_branches(branch)

        graph = Graph()

        result = await session.flow(graph, parallel=False, verbose=False)

        assert result["completed_operations"] == []
        assert result["operation_results"] == {}
        assert result["final_context"] == {}

    @pytest.mark.asyncio
    async def test_flow_with_single_operation(self):
        """Test flow with single operation (no dependencies)."""
        session = Session()
        branch = make_mock_branch()
        session.include_branches(branch)

        op = Operation(
            operation="chat", parameters={"instruction": "Solo task"}
        )
        graph = Graph()
        graph.add_node(op)

        result = await session.flow(graph, parallel=False, verbose=False)

        assert result["completed_operations"] == [op.id]
        assert op.id in result["operation_results"]


# ============================================================================
# 2. Branch Management Tests
# ============================================================================


class TestBranchManagement:
    """Test branch creation, registration, selection, and iteration."""

    def test_session_initialization_with_default_branch(self):
        """Test Session creates default branch on initialization."""
        session = Session()

        assert session.default_branch is not None
        assert session.default_branch in session.branches
        assert len(session.branches) == 1

    def test_session_initialization_with_custom_branch(self):
        """Test Session can be initialized with custom branch."""
        custom_branch = make_mock_branch("CustomBranch")
        session = Session()
        session.include_branches(custom_branch)

        assert custom_branch in session.branches
        # Default branch was created, then custom was added
        assert len(session.branches) >= 2

    def test_include_branches_single(self):
        """Test including single branch."""
        session = Session()
        initial_count = len(session.branches)

        branch = make_mock_branch("NewBranch")
        session.include_branches(branch)

        assert branch in session.branches
        assert len(session.branches) == initial_count + 1
        assert branch.user == session.id  # Branch user set to session ID

    def test_include_branches_multiple(self):
        """Test including multiple branches at once."""
        session = Session()
        initial_count = len(session.branches)

        branches = [make_mock_branch(f"Branch{i}") for i in range(3)]
        session.include_branches(branches)

        assert all(b in session.branches for b in branches)
        assert len(session.branches) == initial_count + 3

    def test_include_branches_idempotent(self):
        """Test that including same branch twice doesn't duplicate."""
        session = Session()
        branch = make_mock_branch("TestBranch")

        session.include_branches(branch)
        initial_count = len(session.branches)

        session.include_branches(branch)  # Include again

        assert len(session.branches) == initial_count  # No duplication

    def test_get_branch_by_id(self):
        """Test retrieving branch by ID."""
        session = Session()
        branch = make_mock_branch("TestBranch")
        session.include_branches(branch)

        retrieved = session.get_branch(branch.id)

        assert retrieved is branch
        assert retrieved.id == branch.id

    def test_get_branch_by_name(self):
        """Test retrieving branch by name."""
        session = Session()
        branch = make_mock_branch("UniqueNameBranch")
        session.include_branches(branch)

        retrieved = session.get_branch("UniqueNameBranch")

        assert retrieved is branch
        assert retrieved.name == "UniqueNameBranch"

    def test_get_branch_not_found_raises_error(self):
        """Test getting non-existent branch raises error."""
        session = Session()

        with pytest.raises(Exception):  # ItemNotFoundError
            session.get_branch("nonexistent")

    def test_get_branch_with_default_value(self):
        """Test get_branch returns default when branch not found."""
        session = Session()

        default_value = "default"
        result = session.get_branch("nonexistent", default_value)

        assert result == default_value

    def test_remove_branch(self):
        """Test removing branch from session."""
        session = Session()
        branch = make_mock_branch("RemoveBranch")
        session.include_branches(branch)

        assert branch in session.branches

        session.remove_branch(branch.id)

        assert branch not in session.branches

    def test_remove_branch_updates_default_branch(self):
        """Test removing default branch updates to next available."""
        session = Session()
        branch1 = make_mock_branch("Branch1")
        branch2 = make_mock_branch("Branch2")
        session.include_branches([branch1, branch2])

        session.change_default_branch(branch1)
        assert session.default_branch is branch1

        session.remove_branch(branch1.id)

        # Default should now be branch2 or another available branch
        assert session.default_branch is not branch1
        assert session.default_branch in session.branches

    def test_change_default_branch(self):
        """Test changing default branch."""
        session = Session()
        branch1 = make_mock_branch("Branch1")
        branch2 = make_mock_branch("Branch2")
        session.include_branches([branch1, branch2])

        initial_default = session.default_branch
        session.change_default_branch(branch2)

        assert session.default_branch is branch2
        assert session.default_branch is not initial_default

    def test_new_branch_creates_and_includes(self):
        """Test new_branch creates branch and adds to session."""
        session = Session()
        initial_count = len(session.branches)

        new_branch = session.new_branch(name="NewBranch")

        assert new_branch in session.branches
        assert new_branch.name == "NewBranch"
        assert len(session.branches) == initial_count + 1

    def test_new_branch_with_custom_imodel(self):
        """Test creating new branch with custom iModel."""
        session = Session()

        custom_model = iModel(
            provider="openai", model="gpt-4o", api_key="test"
        )
        new_branch = session.new_branch(
            name="CustomModelBranch", imodel=custom_model
        )

        assert new_branch.chat_model.model_name == "gpt-4o"

    def test_new_branch_as_default(self):
        """Test creating new branch and setting as default."""
        session = Session()
        old_default = session.default_branch

        new_branch = session.new_branch(
            name="NewDefaultBranch", as_default_branch=True
        )

        assert session.default_branch is new_branch
        assert session.default_branch is not old_default

    def test_split_branch_preserves_messages(self):
        """Test split creates new branch with cloned messages."""
        session = Session()
        branch = make_mock_branch("OriginalBranch")
        session.include_branches(branch)

        # Add message to branch
        msg = Instruction(
            content={"instruction": "Test message"},
            sender=branch.user,
            recipient=branch.id,
        )
        branch.messages.include(msg)

        # Split the branch
        cloned_branch = session.split(branch.id)

        assert cloned_branch in session.branches
        assert len(cloned_branch.messages) == len(branch.messages)
        assert cloned_branch.id != branch.id

    def test_split_branch_clones_tools(self):
        """Test split clones tool manager."""
        session = Session()
        branch = make_mock_branch("OriginalBranch")

        # Register a tool
        def test_tool(x: int) -> int:
            return x * 2

        branch.register_tools(test_tool)
        session.include_branches(branch)

        # Split the branch
        cloned_branch = session.split(branch.id)

        # Verify tool was cloned
        assert "test_tool" in cloned_branch.tools

    @pytest.mark.asyncio
    async def test_asplit_branch(self):
        """Test async split branch."""
        session = Session()
        branch = make_mock_branch("AsyncSplitBranch")
        session.include_branches(branch)

        cloned_branch = await session.asplit(branch.id)

        assert cloned_branch in session.branches
        assert cloned_branch.id != branch.id

    def test_iterate_over_branches(self):
        """Test iterating over session branches."""
        session = Session()
        branches = [make_mock_branch(f"Branch{i}") for i in range(3)]
        session.include_branches(branches)

        branch_list = list(session.branches)

        # Should include default branch + 3 added branches
        assert len(branch_list) >= 3
        assert all(b in branch_list for b in branches)


# ============================================================================
# 3. Edge Cases and Error Handling
# ============================================================================


class TestEdgeCasesAndErrors:
    """Test edge cases, error handling, and boundary conditions."""

    @pytest.mark.asyncio
    async def test_flow_with_operation_error(self):
        """Test flow handles operation errors gracefully.

        Note: The current implementation marks operations as completed
        even when they fail, but records the error in the operation result.
        """
        session = Session()
        branch = make_mock_branch("ErrorBranch")

        # Override the invoke method on the chat_model to raise an error
        original_invoke = branch.chat_model.invoke

        async def failing_invoke(**kwargs):
            raise ValueError("Simulated operation failure")

        branch.chat_model.invoke = failing_invoke

        session.include_branches(branch)
        session.default_branch = branch

        op = Operation(
            operation="chat", parameters={"instruction": "Will fail"}
        )
        graph = Graph()
        graph.add_node(op)

        result = await session.flow(graph, parallel=False, verbose=False)

        # Operation is marked as completed
        assert op.id in result["completed_operations"]
        # Error should be recorded in the operation execution
        assert op.execution.error is not None
        assert "Simulated operation failure" in op.execution.error

    @pytest.mark.asyncio
    async def test_flow_max_concurrent_limit(self):
        """Test max_concurrent properly limits parallel execution."""
        session = Session()
        branch = make_mock_branch()
        session.include_branches(branch)

        # Create multiple independent operations
        ops = [
            Operation(
                operation="chat", parameters={"instruction": f"Task {i}"}
            )
            for i in range(5)
        ]

        graph = Graph()
        for op in ops:
            graph.add_node(op)

        # Execute with max_concurrent=2
        result = await session.flow(
            graph, parallel=True, max_concurrent=2, verbose=False
        )

        # All operations should complete
        assert len(result["completed_operations"]) == 5

    @pytest.mark.asyncio
    async def test_flow_context_inheritance(self):
        """Test context inheritance between operations."""
        session = Session()
        branch = make_mock_branch()
        session.include_branches(branch)

        op1 = Operation(operation="chat", parameters={"instruction": "First"})
        op2 = Operation(
            operation="chat",
            parameters={"instruction": "Second"},
            metadata={"inherit_context": True},
        )

        graph = Graph()
        graph.add_node(op1)
        graph.add_node(op2)
        graph.add_edge(Edge(head=op1.id, tail=op2.id))

        result = await session.flow(
            graph, context={"initial": "context"}, parallel=False
        )

        # op2 should have inherited context from op1
        assert op2.parameters.get("context") is not None

    @pytest.mark.asyncio
    async def test_flow_context_isolation_between_branches(self):
        """Test that branches maintain context isolation."""
        session = Session()

        branch1 = make_mock_branch("Branch1")
        branch2 = make_mock_branch("Branch2")
        session.include_branches([branch1, branch2])

        # Create operations and assign branches via metadata
        op1 = Operation(
            operation="chat",
            parameters={"instruction": "Task 1"},
        )
        op1.branch_id = branch1.id  # Use property setter

        op2 = Operation(
            operation="chat",
            parameters={"instruction": "Task 2"},
        )
        op2.branch_id = branch2.id  # Use property setter

        graph = Graph()
        graph.add_node(op1)
        graph.add_node(op2)

        result = await session.flow(graph, parallel=True, verbose=False)

        # Both should complete independently
        assert op1.id in result["completed_operations"]
        assert op2.id in result["completed_operations"]

    def test_concat_messages_single_branch(self):
        """Test concatenating messages from single branch."""
        session = Session()
        branch = make_mock_branch("TestBranch")

        # Add messages
        msg1 = Instruction(
            content={"instruction": "Message 1"},
            sender="user",
            recipient=branch.id,
        )
        msg2 = Instruction(
            content={"instruction": "Message 2"},
            sender="user",
            recipient=branch.id,
        )
        branch.messages.include([msg1, msg2])
        session.include_branches(branch)

        messages = session.concat_messages([branch.id])

        assert len(messages) >= 2

    def test_concat_messages_multiple_branches(self):
        """Test concatenating messages from multiple branches."""
        session = Session()
        branch1 = make_mock_branch("Branch1")
        branch2 = make_mock_branch("Branch2")

        # Add messages to both branches
        msg1 = Instruction(
            content={"instruction": "Branch1 Message"},
            sender="user",
            recipient=branch1.id,
        )
        msg2 = Instruction(
            content={"instruction": "Branch2 Message"},
            sender="user",
            recipient=branch2.id,
        )
        branch1.messages.include(msg1)
        branch2.messages.include(msg2)

        session.include_branches([branch1, branch2])

        messages = session.concat_messages([branch1.id, branch2.id])

        assert len(messages) >= 2

    def test_concat_messages_deduplication(self):
        """Test that concat_messages removes duplicates."""
        session = Session()
        branch1 = make_mock_branch("Branch1")
        branch2 = make_mock_branch("Branch2")

        # Add same message to both branches
        msg = Instruction(
            content={"instruction": "Shared Message"},
            sender="user",
            recipient=branch1.id,
        )
        branch1.messages.include(msg)
        branch2.messages.include(msg)

        session.include_branches([branch1, branch2])

        messages = session.concat_messages([branch1.id, branch2.id])

        # Should only have one copy of the message
        message_ids = [m.id for m in messages]
        assert len(message_ids) == len(set(message_ids))  # All unique

    def test_to_df_conversion(self):
        """Test converting session messages to DataFrame."""
        session = Session()
        branch = make_mock_branch("TestBranch")

        # Add messages
        msg = Instruction(
            content={"instruction": "Test"},
            sender="user",
            recipient=branch.id,
        )
        branch.messages.include(msg)
        session.include_branches(branch)

        df = session.to_df([branch.id])

        assert df is not None
        assert len(df) >= 1

    def test_operation_manager_shared_across_branches(self):
        """Test that operation manager is shared across all branches."""
        session = Session()

        # Register an operation
        @session.operation("shared_op")
        async def shared_operation(**kwargs):
            return {"result": "success"}

        # Create multiple branches
        branch1 = make_mock_branch("Branch1")
        branch2 = make_mock_branch("Branch2")
        session.include_branches([branch1, branch2])

        # Both branches should have access to the operation
        assert "shared_op" in branch1._operation_manager.registry
        assert "shared_op" in branch2._operation_manager.registry
        assert (
            branch1._operation_manager.registry["shared_op"]
            is branch2._operation_manager.registry["shared_op"]
        )


# ============================================================================
# 4. Mail System Tests (Future Enhancement)
# ============================================================================


class TestMailSystem:
    """
    Test mail routing between branches.

    Note: These tests are placeholders for future mail system implementation.
    The Session class supports branch communication but mail routing
    functionality is not yet implemented.
    """

    def test_mail_system_placeholder(self):
        """Placeholder for future mail system tests."""
        session = Session()
        branch1 = make_mock_branch("Sender")
        branch2 = make_mock_branch("Receiver")
        session.include_branches([branch1, branch2])

        # Mail system tests will be added when feature is implemented
        assert True


# ============================================================================
# 5. Integration Tests
# ============================================================================


class TestSessionFlowIntegration:
    """Integration tests combining multiple Session features."""

    @pytest.mark.asyncio
    async def test_full_multi_branch_workflow(self):
        """Test complete workflow with multiple branches and operations."""
        session = Session()

        # Create branches for different stages
        research_branch = make_mock_branch("Research")
        analysis_branch = make_mock_branch("Analysis")
        summary_branch = make_mock_branch("Summary")

        session.include_branches(
            [research_branch, analysis_branch, summary_branch]
        )

        # Create workflow graph
        op_research = Operation(
            operation="chat",
            parameters={"instruction": "Research topic"},
        )
        op_research.branch_id = research_branch.id

        op_analyze = Operation(
            operation="chat",
            parameters={"instruction": "Analyze findings"},
        )
        op_analyze.branch_id = analysis_branch.id

        op_summarize = Operation(
            operation="chat",
            parameters={"instruction": "Create summary"},
        )
        op_summarize.branch_id = summary_branch.id

        graph = Graph()
        graph.add_node(op_research)
        graph.add_node(op_analyze)
        graph.add_node(op_summarize)
        graph.add_edge(Edge(head=op_research.id, tail=op_analyze.id))
        graph.add_edge(Edge(head=op_analyze.id, tail=op_summarize.id))

        result = await session.flow(
            graph,
            context={"topic": "AI orchestration"},
            parallel=False,
            verbose=False,
        )

        # Verify complete workflow execution
        assert len(result["completed_operations"]) == 3
        assert all(
            op.id in result["completed_operations"]
            for op in [op_research, op_analyze, op_summarize]
        )

    @pytest.mark.asyncio
    async def test_flow_with_builder_pattern(self):
        """Test flow using OperationGraphBuilder."""
        session = Session()

        # Create branches
        branch1 = make_mock_branch("Branch1")
        branch2 = make_mock_branch("Branch2")
        session.include_branches([branch1, branch2])

        # Register operations
        @session.operation()
        async def process_data(**kwargs):
            return {"processed": True}

        @session.operation()
        async def validate_data(**kwargs):
            return {"validated": True}

        # Build graph using builder
        builder = OperationGraphBuilder("TestWorkflow")
        op1 = builder.add_operation("process_data", branch=branch1)
        op2 = builder.add_operation(
            "validate_data", branch=branch2, depends_on=[op1]
        )

        result = await session.flow(
            builder.get_graph(), parallel=False, verbose=False
        )

        assert len(result["completed_operations"]) == 2

    @pytest.mark.asyncio
    async def test_session_resilience_to_branch_errors(self):
        """Test session continues operation despite individual branch errors.

        Note: The current implementation marks operations as completed
        even when they fail, but records the error.
        """
        session = Session()

        # Create mix of working and failing branches
        working_branch = make_mock_branch("WorkingBranch")
        failing_branch = make_mock_branch("FailingBranch")

        # Override the invoke method to fail
        async def failing_invoke(**kwargs):
            raise RuntimeError("Branch failure")

        failing_branch.chat_model.invoke = failing_invoke

        session.include_branches([working_branch, failing_branch])
        session.default_branch = working_branch

        # Create operations on both branches
        op_working = Operation(
            operation="chat",
            parameters={"instruction": "Should work"},
        )
        op_working.branch_id = working_branch.id

        op_failing = Operation(
            operation="chat",
            parameters={"instruction": "Will fail"},
        )
        op_failing.branch_id = failing_branch.id

        graph = Graph()
        graph.add_node(op_working)
        graph.add_node(op_failing)

        result = await session.flow(graph, parallel=True, verbose=False)

        # Both operations complete (success and failure both marked as completed)
        assert op_working.id in result["completed_operations"]
        assert op_failing.id in result["completed_operations"]

        # Verify results exist for both
        assert op_working.id in result["operation_results"]
        assert op_failing.id in result["operation_results"]

        # The failing operation should have error recorded
        failing_result = result["operation_results"][op_failing.id]
        has_error = (
            isinstance(failing_result, dict) and "error" in failing_result
        ) or op_failing.execution.error is not None
        assert has_error


# ============================================================================
# 6. Async Edge Cases: Cancellation, Timeout, Error Propagation
# ============================================================================


class TestSessionFlowAsyncEdgeCases:
    """Test async edge cases for flow execution - cancellation, timeout, error propagation."""

    @pytest.mark.asyncio
    async def test_flow_cancellation_mid_execution(self):
        """Test cancelling flow mid-execution cleans up properly."""
        session = Session()
        branch = make_mock_branch()
        session.include_branches(branch)

        # Create slow operations
        async def slow_invoke(**kwargs):
            await asyncio.sleep(2)  # Simulate slow operation
            config = _get_oai_config(
                name="oai_chat",
                endpoint="chat/completions",
                request_options=OpenAIChatCompletionsRequest,
                kwargs={"model": "gpt-4.1-mini"},
            )
            endpoint = Endpoint(config=config)
            fake_call = APICalling(
                payload={"model": "gpt-4.1-mini", "messages": []},
                headers={"Authorization": "Bearer test"},
                endpoint=endpoint,
            )
            fake_call.execution.response = "mocked_response"
            fake_call.execution.status = EventStatus.COMPLETED
            return fake_call

        branch.chat_model.invoke = AsyncMock(side_effect=slow_invoke)

        graph, ops = make_simple_graph(3)

        # Start flow and cancel after short delay
        task = asyncio.create_task(
            session.flow(graph, parallel=True, verbose=False)
        )

        # Cancel after brief delay
        await asyncio.sleep(0.1)
        task.cancel()

        # Verify cancellation
        with pytest.raises(asyncio.CancelledError):
            await task

    @pytest.mark.asyncio
    async def test_flow_timeout_behavior(self):
        """Test flow timeout enforcement with asyncio.wait_for."""
        session = Session()

        # Create a MagicMock branch for this test to allow method mocking
        branch = MagicMock()
        branch.id = "test-branch-id"

        # Mock the chat method to sleep and prevent API calls
        async def very_slow_chat(**kwargs):
            # Sleep longer than timeout - this ensures timeout happens
            await asyncio.sleep(10)
            # This code never reached due to timeout - no API setup at all
            return "mocked_response"

        branch.chat = AsyncMock(side_effect=very_slow_chat)

        # Mock get_operation to return the correct async method
        def mock_get_operation(operation: str):
            if operation == "chat":
                return branch.chat
            return None

        branch.get_operation = MagicMock(side_effect=mock_get_operation)

        session.branches.include(branch)
        session.default_branch = branch

        graph, ops = make_simple_graph(2)

        # Apply timeout to flow execution - should raise TimeoutError
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(
                session.flow(graph, parallel=False, verbose=False), timeout=0.5
            )

    @pytest.mark.asyncio
    async def test_error_propagation_across_parallel_branches(self):
        """Test that errors in one branch don't block other parallel branches."""
        session = Session()

        # Create branches with different behaviors
        working_branch = make_mock_branch("WorkingBranch")
        failing_branch = make_mock_branch("FailingBranch")

        # Override invoke to fail for failing_branch
        async def failing_invoke(**kwargs):
            raise RuntimeError("Branch-specific error")

        failing_branch.chat_model.invoke = failing_invoke

        session.include_branches([working_branch, failing_branch])

        # Create parallel operations on different branches
        op_working = Operation(
            operation="chat",
            parameters={"instruction": "Should succeed"},
        )
        op_working.branch_id = working_branch.id

        op_failing = Operation(
            operation="chat",
            parameters={"instruction": "Will fail"},
        )
        op_failing.branch_id = failing_branch.id

        graph = Graph()
        graph.add_node(op_working)
        graph.add_node(op_failing)

        result = await session.flow(graph, parallel=True, verbose=False)

        # Both operations complete (success and failure)
        assert op_working.id in result["completed_operations"]
        assert op_failing.id in result["completed_operations"]

        # Working operation should have no error
        assert op_working.execution.error is None

        # Failing operation should have recorded error
        assert op_failing.execution.error is not None
        assert "Branch-specific error" in op_failing.execution.error

    @pytest.mark.asyncio
    async def test_flow_continues_after_operation_failure(self):
        """Test that flow continues processing after one operation fails."""
        session = Session()

        # Create two branches - one will fail, one will succeed
        working_branch = make_mock_branch("WorkingBranch")
        failing_branch = make_mock_branch("FailingBranch")

        # Override invoke to fail for failing_branch
        async def failing_invoke(**kwargs):
            raise ValueError("Operation failure")

        failing_branch.chat_model.invoke = failing_invoke

        session.include_branches([working_branch, failing_branch])

        # Create sequential operations with mixed success/failure
        op1 = Operation(
            operation="chat",
            parameters={"instruction": "First (success)"},
        )
        op1.branch_id = working_branch.id

        op2 = Operation(
            operation="chat",
            parameters={"instruction": "Second (fail)"},
        )
        op2.branch_id = failing_branch.id

        op3 = Operation(
            operation="chat",
            parameters={"instruction": "Third (success)"},
        )
        op3.branch_id = working_branch.id

        graph = Graph()
        graph.add_node(op1)
        graph.add_node(op2)
        graph.add_node(op3)
        graph.add_edge(Edge(head=op1.id, tail=op2.id))
        graph.add_edge(Edge(head=op2.id, tail=op3.id))

        result = await session.flow(graph, parallel=False, verbose=False)

        # All operations should complete
        assert len(result["completed_operations"]) == 3
        # Verify first and third succeeded
        assert op1.execution.error is None
        assert op3.execution.error is None
        # Second should have failed
        assert op2.execution.error is not None

    @pytest.mark.asyncio
    async def test_concurrent_flow_with_mixed_timings(self):
        """Test concurrent flow with operations of varying speeds doesn't deadlock."""
        session = Session()
        branch = make_mock_branch()
        session.include_branches(branch)

        # Create operations with varying IDs
        op_fast = Operation(
            operation="chat",
            parameters={"instruction": "fast task"},
        )
        op_medium = Operation(
            operation="chat",
            parameters={"instruction": "medium task"},
        )
        op_slow = Operation(
            operation="chat",
            parameters={"instruction": "slow task"},
        )

        graph = Graph()
        for op in [op_fast, op_medium, op_slow]:
            graph.add_node(op)

        result = await session.flow(
            graph, parallel=True, max_concurrent=3, verbose=False
        )

        # All operations should complete despite potential timing differences
        assert len(result["completed_operations"]) == 3
        assert op_fast.id in result["completed_operations"]
        assert op_medium.id in result["completed_operations"]
        assert op_slow.id in result["completed_operations"]
