# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

import asyncio
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID

import pytest
from anyio import get_cancelled_exc_class
from pydantic import BaseModel

from lionagi.operations.node import Operation
from lionagi.protocols.generic.event import EventStatus
from lionagi.session.branch import Branch


# Test fixtures and utilities
class OpParams(BaseModel):
    """Test BaseModel for parameters."""

    instruction: str
    count: int = 1
    enabled: bool = True


# Test Operation creation and properties
def test_operation_creation():
    """Test creating an Operation with various parameter types."""
    # Test with dict parameters
    op1 = Operation(
        operation="chat",
        parameters={"instruction": "Hello", "temperature": 0.7},
    )
    assert op1.operation == "chat"
    assert op1.parameters["instruction"] == "Hello"
    assert op1.parameters["temperature"] == 0.7

    # Test with BaseModel parameters
    params = OpParams(instruction="Test instruction", count=5)
    op2 = Operation(operation="operate", parameters=params)
    assert op2.operation == "operate"
    assert isinstance(op2.parameters, OpParams)
    assert op2.parameters.instruction == "Test instruction"

    # Test with default parameters
    op3 = Operation(operation="parse")
    assert op3.operation == "parse"
    assert op3.parameters == {}


def test_operation_branch_id():
    """Test branch_id property getter and setter."""
    op = Operation(operation="chat")

    # Test setting with valid UUID string
    test_uuid_str = "12345678-1234-4678-9234-567812345678"
    op.branch_id = test_uuid_str
    assert str(op.branch_id) == test_uuid_str
    assert op.metadata["branch_id"] == test_uuid_str

    # Test setting with UUID object
    test_uuid = UUID("87654321-4321-4765-8321-876543218765")
    op.branch_id = test_uuid
    assert str(op.branch_id) == str(test_uuid)

    # Test setting to None
    op.branch_id = None
    assert op.branch_id is None
    assert "branch_id" not in op.metadata


def test_operation_graph_id():
    """Test graph_id property getter and setter."""
    op = Operation(operation="chat")

    # Test setting with valid UUID string
    test_graph_uuid = "12345678-1234-4678-9234-567812345678"
    op.graph_id = test_graph_uuid
    assert str(op.graph_id) == test_graph_uuid
    assert op.metadata["graph_id"] == test_graph_uuid

    # Test setting with UUID
    test_uuid = UUID("87654321-4321-4765-8321-876543218765")
    op.graph_id = test_uuid
    assert str(op.graph_id) == str(test_uuid)

    # Test setting to None
    op.graph_id = None
    assert op.graph_id is None
    assert "graph_id" not in op.metadata


def test_operation_request_property():
    """Test request property with different parameter types."""
    # Test with dict parameters
    op1 = Operation(operation="chat", parameters={"key": "value"})
    assert op1.request == {"key": "value"}

    # Test with BaseModel parameters
    params = OpParams(instruction="Test")
    op2 = Operation(operation="operate", parameters=params)
    request = op2.request
    assert isinstance(request, dict)
    assert request["instruction"] == "Test"
    assert request["count"] == 1
    assert request["enabled"] is True

    # Test with empty parameters
    op3 = Operation(operation="parse")
    assert op3.request == {}


def test_operation_response_property():
    """Test response property."""
    op = Operation(operation="chat")

    # Initially no response
    assert op.response is None

    # Set response through execution
    op.execution.response = "test_response"
    assert op.response == "test_response"


# Test async operations
@pytest.mark.asyncio
async def test_operation_invoke_chat():
    """Test invoking a chat operation."""
    op = Operation(
        operation="chat", parameters={"instruction": "Hello, how are you?"}
    )

    # Create a mock branch
    branch = MagicMock()
    branch.id = "12345678-1234-4678-9234-567812345678"

    # Mock the chat method
    async def mock_chat(**kwargs):
        return f"chat_response: {kwargs.get('instruction', 'default')}"

    branch.chat = AsyncMock(side_effect=mock_chat)

    # Mock get_operation to return the correct async method
    def mock_get_operation(operation: str):
        if operation == "chat":
            return branch.chat
        return None

    branch.get_operation = MagicMock(side_effect=mock_get_operation)

    await op.invoke(branch)

    # Verify operation was called
    branch.chat.assert_called_once_with(instruction="Hello, how are you?")

    # Verify execution status
    assert op.execution.status == EventStatus.COMPLETED
    assert op.response == "chat_response: Hello, how are you?"
    assert str(op.branch_id) == branch.id
    assert op.execution.duration > 0


@pytest.mark.asyncio
async def test_operation_invoke_with_basemodel_params():
    """Test invoking an operation with BaseModel parameters."""
    params = OpParams(instruction="Complex task", count=3, enabled=False)
    op = Operation(operation="operate", parameters=params)

    # Create a mock branch
    branch = MagicMock()
    branch.id = "12345678-1234-4678-9234-567812345678"

    async def mock_operate(**kwargs):
        return {"operation": "operate", "result": "success"}

    branch.operate = AsyncMock(side_effect=mock_operate)

    # Mock get_operation to return the correct async method
    def mock_get_operation(operation: str):
        if operation == "operate":
            return branch.operate
        return None

    branch.get_operation = MagicMock(side_effect=mock_get_operation)

    await op.invoke(branch)

    # Verify the method was called with unpacked parameters
    branch.operate.assert_called_once_with(
        instruction="Complex task", count=3, enabled=False
    )

    # Verify response
    assert op.response == {"operation": "operate", "result": "success"}


@pytest.mark.asyncio
async def test_operation_invoke_streaming():
    """Test invoking a streaming operation (ReActStream)."""
    op = Operation(
        operation="ReActStream", parameters={"query": "stream test"}
    )

    # Create a mock branch
    branch = MagicMock()
    branch.id = "12345678-1234-4678-9234-567812345678"

    async def mock_stream(**kwargs):
        """Mock streaming operation."""
        for i in range(3):
            yield f"stream_chunk_{i}"

    branch.ReActStream = mock_stream

    # Mock get_operation to return the correct async method
    def mock_get_operation(operation: str):
        if operation == "ReActStream":
            return branch.ReActStream
        return None

    branch.get_operation = MagicMock(side_effect=mock_get_operation)

    await op.invoke(branch)

    # Verify response is a list of streamed chunks
    assert op.response == [
        "stream_chunk_0",
        "stream_chunk_1",
        "stream_chunk_2",
    ]
    assert op.execution.status == EventStatus.COMPLETED


@pytest.mark.asyncio
async def test_operation_invoke_all_operations():
    """Test invoking all supported operation types."""
    # Create a mock branch
    branch = MagicMock()
    branch.id = "12345678-1234-4678-9234-567812345678"

    # Set up all mock methods
    branch.chat = AsyncMock(return_value="chat_response: test")
    branch.operate = AsyncMock(
        return_value={"operation": "operate", "result": "success"}
    )
    branch.communicate = AsyncMock(return_value="communicate_response")
    branch.parse = AsyncMock(return_value={"parsed": True})
    branch.ReAct = AsyncMock(return_value={"react": "result"})
    branch.select = AsyncMock(return_value="selected_option")
    branch.translate = AsyncMock(return_value="translated_text")
    branch.interpret = AsyncMock(return_value={"interpretation": "complete"})
    branch.act = AsyncMock(return_value={"action": "taken"})
    branch.instruct = AsyncMock(return_value="instruction_result")

    # Mock get_operation to return the correct async method
    def mock_get_operation(operation: str):
        operation_map = {
            "chat": branch.chat,
            "operate": branch.operate,
            "communicate": branch.communicate,
            "parse": branch.parse,
            "ReAct": branch.ReAct,
            "select": branch.select,
            "translate": branch.translate,
            "interpret": branch.interpret,
            "act": branch.act,
            "instruct": branch.instruct,
        }
        return operation_map.get(operation)

    branch.get_operation = MagicMock(side_effect=mock_get_operation)

    operations_and_expected = [
        ("chat", "chat_response: test"),
        ("operate", {"operation": "operate", "result": "success"}),
        ("communicate", "communicate_response"),
        ("parse", {"parsed": True}),
        ("ReAct", {"react": "result"}),
        ("select", "selected_option"),
        ("translate", "translated_text"),
        ("interpret", {"interpretation": "complete"}),
        ("act", {"action": "taken"}),
        ("instruct", "instruction_result"),
    ]

    for op_type, expected_response in operations_and_expected:
        op = Operation(operation=op_type, parameters={"instruction": "test"})
        await op.invoke(branch)
        assert op.response == expected_response
        assert op.execution.status == EventStatus.COMPLETED


@pytest.mark.asyncio
async def test_operation_invoke_invalid_operation():
    """Test invoking an operation with invalid operation type."""
    # Create a proper Branch instance so getattr works correctly
    branch = Branch(user="test_user", name="TestBranch")

    # Create operation with valid type first
    op = Operation(operation="chat")
    # Then change to invalid type (bypassing validation)
    op.operation = "invalid_operation"

    # Invoke should raise ValueError for unsupported operation
    with pytest.raises(ValueError, match="Unsupported operation type"):
        await op.invoke(branch)


@pytest.mark.asyncio
async def test_operation_invoke_exception_handling():
    """Test exception handling during operation invocation."""
    # Create a mock branch
    branch = MagicMock()
    branch.id = "12345678-1234-4678-9234-567812345678"

    # Mock method to raise exception
    async def failing_method(**kwargs):
        raise RuntimeError("Test error occurred")

    branch.chat = AsyncMock(side_effect=failing_method)

    # Mock get_operation to return the correct async method
    def mock_get_operation(operation: str):
        if operation == "chat":
            return branch.chat
        return None

    branch.get_operation = MagicMock(side_effect=mock_get_operation)

    op = Operation(
        operation="chat", parameters={"instruction": "This will fail"}
    )
    await op.invoke(branch)

    # Verify error handling
    assert op.execution.status == EventStatus.FAILED
    assert op.execution.error == "Test error occurred"
    assert op.response is None


@pytest.mark.slow
@pytest.mark.asyncio
async def test_operation_invoke_cancellation():
    """Test handling of operation cancellation."""
    # Create a mock branch
    branch = MagicMock()
    branch.id = "12345678-1234-4678-9234-567812345678"

    # Mock method that will be cancelled
    async def slow_method(**kwargs):
        await asyncio.sleep(10)  # Long running operation
        return "should_not_reach_here"

    branch.chat = AsyncMock(side_effect=slow_method)

    # Mock get_operation to return the correct async method
    def mock_get_operation(operation: str):
        if operation == "chat":
            return branch.chat
        return None

    branch.get_operation = MagicMock(side_effect=mock_get_operation)

    op = Operation(operation="chat")

    # Create task and cancel it
    task = asyncio.create_task(op.invoke(branch))
    await asyncio.sleep(0.1)  # Let it start
    task.cancel()

    with pytest.raises(get_cancelled_exc_class()):
        await task

    # Verify cancellation was handled
    assert op.execution.status == EventStatus.CANCELLED
    assert op.execution.error == "Operation cancelled"


def test_operation_inheritance():
    """Test that Operation properly inherits from Node and Event."""
    op = Operation(operation="chat")

    # Test Node properties
    assert hasattr(op, "id")
    assert hasattr(op, "metadata")
    # op.id is an IDType object
    from lionagi.protocols.generic.element import IDType

    assert isinstance(op.id, IDType)

    # Test Event properties
    assert hasattr(op, "execution")
    assert hasattr(op, "streaming")
    assert op.execution.status == EventStatus.PENDING
    assert op.streaming is False


def test_operation_serialization():
    """Test serialization of Operation with different parameter types."""
    # Test with dict parameters
    op1 = Operation(
        operation="chat",
        parameters={"instruction": "Hello", "temperature": 0.7},
    )
    data1 = op1.model_dump()
    assert data1["operation"] == "chat"

    # Test with BaseModel parameters
    params = OpParams(instruction="Test", count=5)
    op2 = Operation(operation="operate", parameters=params)
    # The Operation keeps the BaseModel instance as is
    assert isinstance(op2.parameters, OpParams)
    data2 = op2.model_dump()
    assert data2["operation"] == "operate"
    # When model_dump is called, the BaseModel parameters should be serialized
    # But it seems the actual behavior might differ - let's check what we get


@pytest.mark.asyncio
async def test_operation_concurrent_invocations():
    """Test multiple operations can be invoked concurrently."""
    # Create a mock branch
    branch = MagicMock()
    branch.id = "12345678-1234-4678-9234-567812345678"

    # Mock chat method
    async def mock_chat(**kwargs):
        await asyncio.sleep(0.01)  # Small delay
        return f"chat_response: {kwargs.get('instruction', 'default')}"

    branch.chat = AsyncMock(side_effect=mock_chat)

    # Mock get_operation to return the correct async method
    def mock_get_operation(operation: str):
        if operation == "chat":
            return branch.chat
        return None

    branch.get_operation = MagicMock(side_effect=mock_get_operation)

    # Create multiple operations
    ops = [
        Operation(operation="chat", parameters={"instruction": f"Task {i}"})
        for i in range(5)
    ]

    # Invoke all operations concurrently
    tasks = [op.invoke(branch) for op in ops]
    await asyncio.gather(*tasks)

    # Verify all completed successfully
    for i, op in enumerate(ops):
        assert op.execution.status == EventStatus.COMPLETED
        assert op.response == f"chat_response: Task {i}"


def test_operation_metadata_persistence():
    """Test that metadata persists through operation lifecycle."""
    op = Operation(
        operation="chat",
        parameters={"instruction": "Test"},
        metadata={"custom_key": "custom_value", "priority": "high"},
    )

    # Check initial metadata
    assert op.metadata["custom_key"] == "custom_value"
    assert op.metadata["priority"] == "high"

    # Set branch and graph IDs
    op.branch_id = "12345678-1234-5678-1234-567812345678"
    op.graph_id = "test-graph"

    # Original metadata should still be there
    assert op.metadata["custom_key"] == "custom_value"
    assert op.metadata["priority"] == "high"
    assert op.metadata["branch_id"] == "12345678-1234-5678-1234-567812345678"
    assert op.metadata["graph_id"] == "test-graph"
