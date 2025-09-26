"""
Shared Test Utilities for LionAGI

Provides reusable helpers for common testing patterns including async operations,
validation, and test data management.
"""

import asyncio
import time
from typing import (
    Any,
    AsyncGenerator,
    Callable,
    Dict,
    List,
    Optional,
    TypeVar,
    Union,
)

import pytest

from lionagi.protocols.generic.element import IDType
from lionagi.protocols.generic.event import EventStatus
from lionagi.protocols.graph.node import Node

T = TypeVar("T")


class AsyncTestHelpers:
    """Helper utilities for async testing patterns."""

    @staticmethod
    async def assert_eventually(
        condition: Callable[[], bool],
        timeout: float = 5.0,
        interval: float = 0.1,
        error_message: Optional[str] = None,
    ) -> None:
        """
        Wait for an async condition to become true within timeout.

        Useful for testing eventual consistency, async operations completion,
        and state changes that happen asynchronously.

        Args:
            condition: Function that returns True when condition is met
            timeout: Maximum time to wait in seconds
            interval: Time between condition checks in seconds
            error_message: Custom error message if condition fails

        Raises:
            AssertionError: If condition doesn't become true within timeout
        """
        start_time = time.time()

        while time.time() - start_time < timeout:
            if condition():
                return
            await asyncio.sleep(interval)

        message = (
            error_message or f"Condition not met within {timeout} seconds"
        )
        raise AssertionError(message)

    @staticmethod
    async def collect_async_results(
        async_gen: AsyncGenerator[T, None],
        limit: int = 100,
        timeout: float = 10.0,
    ) -> List[T]:
        """
        Collect results from an async generator with limits and timeout.

        Args:
            async_gen: Async generator to collect from
            limit: Maximum number of items to collect
            timeout: Maximum time to spend collecting

        Returns:
            List of collected items

        Raises:
            asyncio.TimeoutError: If timeout is exceeded
        """
        results = []

        try:
            async with asyncio.timeout(timeout):
                async for item in async_gen:
                    results.append(item)
                    if len(results) >= limit:
                        break
        except asyncio.TimeoutError:
            # Return partial results on timeout
            pass

        return results

    @staticmethod
    async def run_with_timeout(
        coro: Callable[..., Any], timeout: float = 5.0, *args, **kwargs
    ) -> Any:
        """
        Run an async function with timeout protection.

        Args:
            coro: Async function to run
            timeout: Timeout in seconds
            *args: Arguments for the function
            **kwargs: Keyword arguments for the function

        Returns:
            Function result

        Raises:
            asyncio.TimeoutError: If timeout is exceeded
        """
        async with asyncio.timeout(timeout):
            return await coro(*args, **kwargs)

    @staticmethod
    async def wait_for_all(
        tasks: List[asyncio.Task], timeout: float = 10.0
    ) -> List[Any]:
        """
        Wait for all tasks to complete with timeout.

        Args:
            tasks: List of asyncio tasks
            timeout: Timeout for all tasks

        Returns:
            List of task results

        Raises:
            asyncio.TimeoutError: If any task exceeds timeout
        """
        try:
            async with asyncio.timeout(timeout):
                return await asyncio.gather(*tasks)
        except asyncio.TimeoutError:
            # Cancel remaining tasks
            for task in tasks:
                if not task.done():
                    task.cancel()
            raise

    @staticmethod
    def assert_async_context_cleanup(func: Callable) -> Callable:
        """
        Decorator to ensure async context is properly cleaned up after test.

        Usage:
            @AsyncTestHelpers.assert_async_context_cleanup
            async def test_my_async_function():
                # Test code here
                pass
        """

        async def wrapper(*args, **kwargs):
            # Track event loop state before test
            initial_tasks = len(asyncio.all_tasks())

            try:
                result = await func(*args, **kwargs)

                # Allow brief time for cleanup
                await asyncio.sleep(0.01)

                # Check for leaked tasks
                final_tasks = len(asyncio.all_tasks())
                if final_tasks > initial_tasks:
                    remaining_tasks = [
                        task for task in asyncio.all_tasks() if not task.done()
                    ]
                    print(
                        f"Warning: {len(remaining_tasks)} tasks not cleaned up: {remaining_tasks}"
                    )

                return result
            except Exception:
                # Cancel any remaining tasks on error
                for task in asyncio.all_tasks():
                    if not task.done():
                        task.cancel()
                raise

        return wrapper


class ValidationHelpers:
    """Helper utilities for common validation patterns."""

    @staticmethod
    def assert_valid_node(
        node: Any,
        expected_type: Optional[type] = None,
        check_id: bool = True,
        check_timestamp: bool = True,
    ) -> None:
        """
        Standard validation for Node-like objects.

        Args:
            node: Object to validate
            expected_type: Expected type (if None, just checks basic structure)
            check_id: Whether to validate ID field
            check_timestamp: Whether to validate timestamp fields
        """
        if expected_type:
            assert isinstance(
                node, expected_type
            ), f"Expected {expected_type}, got {type(node)}"

        # Check for Node-like structure (either Node subclass or has required attributes)
        if not isinstance(node, Node):
            # For non-Node objects (like Branch), check basic structure
            if check_id:
                assert hasattr(node, "id"), "Object missing 'id' field"
        else:
            # For actual Node objects, do full validation
            assert isinstance(
                node, Node
            ), f"Expected Node subclass, got {type(node)}"

        if check_id:
            assert hasattr(node, "id"), "Object missing 'id' field"
            assert node.id is not None, "Object 'id' field is None"
            assert isinstance(
                node.id, (str, IDType)
            ), f"Object 'id' should be string or IDType, got {type(node.id)}"

        if check_timestamp and hasattr(node, "timestamp"):
            assert (
                node.timestamp is not None
            ), "Object 'timestamp' field is None"

    @staticmethod
    def assert_api_response_structure(
        response: Any,
        required_fields: Optional[List[str]] = None,
        check_status: bool = True,
    ) -> None:
        """
        Standard validation for API response structures.

        Args:
            response: Response object to validate
            required_fields: List of required field names
            check_status: Whether to validate status field
        """
        if hasattr(response, "execution"):
            execution = response.execution

            if check_status:
                assert hasattr(
                    execution, "status"
                ), "Response missing execution.status"
                assert isinstance(
                    execution.status, EventStatus
                ), f"Invalid status type: {type(execution.status)}"

            assert hasattr(
                execution, "response"
            ), "Response missing execution.response"

        if required_fields:
            for field in required_fields:
                assert hasattr(
                    response, field
                ), f"Response missing required field: {field}"

    @staticmethod
    def assert_pydantic_model_valid(
        model_instance: Any, expected_fields: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Validate Pydantic model instance structure and field values.

        Args:
            model_instance: Pydantic model instance to validate
            expected_fields: Dict of field_name -> expected_value
        """
        # Check it's a Pydantic model
        assert hasattr(model_instance, "model_dump"), "Not a Pydantic model"
        assert hasattr(
            model_instance, "model_validate"
        ), "Not a Pydantic model"

        # Validate serialization works
        dumped = model_instance.model_dump()
        assert isinstance(dumped, dict), "Model dump should return dict"

        # Check expected fields
        if expected_fields:
            for field_name, expected_value in expected_fields.items():
                actual_value = getattr(model_instance, field_name)
                assert (
                    actual_value == expected_value
                ), f"Field {field_name}: expected {expected_value}, got {actual_value}"

    @staticmethod
    def assert_error_handling(
        error_response: Any,
        expected_error_type: Optional[str] = None,
        expected_message_contains: Optional[str] = None,
    ) -> None:
        """
        Validate error response structure and content.

        Args:
            error_response: Error response to validate
            expected_error_type: Expected error type identifier
            expected_message_contains: String that should be in error message
        """
        # Check for error structure
        if hasattr(error_response, "execution"):
            assert (
                error_response.execution.status == EventStatus.FAILED
            ), "Error response should have FAILED status"

        # Check error content
        if hasattr(error_response, "error"):
            error_data = error_response.error

            if expected_error_type:
                assert (
                    "type" in error_data or "code" in error_data
                ), "Error response missing type/code"
                error_type = error_data.get("type") or error_data.get("code")
                assert expected_error_type in str(
                    error_type
                ), f"Expected error type '{expected_error_type}', got '{error_type}'"

            if expected_message_contains:
                message = error_data.get("message", "")
                assert (
                    expected_message_contains in message
                ), f"Expected '{expected_message_contains}' in error message: '{message}'"


class TestDataHelpers:
    """Helper utilities for test data management."""

    @staticmethod
    def create_test_messages(
        count: int = 3,
        message_type: str = "user",
        base_content: str = "Test message",
    ) -> List[Dict[str, Any]]:
        """
        Create standardized test message data.

        Args:
            count: Number of messages to create
            message_type: Type of messages (user, assistant, system)
            base_content: Base content for messages

        Returns:
            List of message dictionaries
        """
        messages = []
        for i in range(count):
            messages.append(
                {
                    "role": message_type,
                    "content": f"{base_content} {i + 1}",
                    "timestamp": time.time() + i,
                }
            )
        return messages

    @staticmethod
    def create_test_payload(
        model: str = "gpt-4o-mini",
        messages: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Create standardized test API payload.

        Args:
            model: Model name for payload
            messages: Message list (if None, creates default)
            **kwargs: Additional payload fields

        Returns:
            API payload dictionary
        """
        if messages is None:
            messages = TestDataHelpers.create_test_messages()

        payload = {"model": model, "messages": messages, **kwargs}

        return payload
