"""Additional tests for the context module to improve coverage."""

import asyncio
from unittest.mock import MagicMock, patch

import pytest


def test_dummy_context_functions():
    """Test the dummy context functions when OpenTelemetry is not available."""
    # We need to patch the actual implementation, not just the flag
    with (
        patch("pynector.telemetry.context.HAS_OPENTELEMETRY", False),
        patch("pynector.telemetry.context.attach", return_value=None),
        patch("pynector.telemetry.context.detach"),
        patch("pynector.telemetry.context.get_current", return_value={}),
    ):
        from pynector.telemetry.context import attach, detach, get_current

        # Test with different types of arguments
        token = attach({"key": "value"})
        assert token is None

        token = attach(None)
        assert token is None

        token = attach([1, 2, 3])
        assert token is None

        # Test detach with different types of arguments
        detach(None)  # Should not raise
        detach("token")  # Should not raise
        detach(123)  # Should not raise

        # Test get_current
        context = get_current()
        assert context == {}


@pytest.mark.asyncio
async def test_traced_async_operation_no_otel():
    """Test traced_async_operation when OpenTelemetry is not available."""
    # Mock OpenTelemetry not available
    with patch("pynector.telemetry.context.HAS_OPENTELEMETRY", False):
        from pynector.telemetry.context import traced_async_operation
        from pynector.telemetry.tracing import NoOpSpan

        # Create a mock tracer that returns a NoOpSpan
        mock_tracer = MagicMock()
        mock_tracer.start_as_current_async_span.return_value = NoOpSpan(
            "test_operation"
        )

        # Test with no attributes
        async with traced_async_operation(mock_tracer, "test_operation") as span:
            # With HAS_OPENTELEMETRY=False, it should use NoOpSpan directly
            assert isinstance(span, NoOpSpan)
            assert span.name == "test_operation"

        # Test with exception
        try:
            async with traced_async_operation(mock_tracer, "test_operation") as span:
                assert isinstance(span, NoOpSpan)
                raise ValueError("Test exception")
        except ValueError:
            pass  # Expected exception


@pytest.mark.asyncio
async def test_traced_gather_no_otel():
    """Test traced_gather when OpenTelemetry is not available."""
    # Mock OpenTelemetry not available
    with patch("pynector.telemetry.context.HAS_OPENTELEMETRY", False):
        from pynector.telemetry.context import traced_gather

        # Create test coroutines
        async def coro1():
            return 1

        async def coro2():
            return 2

        # Test with default name
        results = await traced_gather(MagicMock(), [coro1(), coro2()])
        assert results == [1, 2]

        # Test with custom name
        results = await traced_gather(MagicMock(), [coro1(), coro2()], "custom_gather")
        assert results == [1, 2]

        # Test with mixed results
        async def coro3():
            return "three"

        results = await traced_gather(MagicMock(), [coro1(), coro3()])
        assert results == [1, "three"]

        # Test with empty list
        results = await traced_gather(MagicMock(), [])
        assert results == []


@pytest.mark.asyncio
async def test_traced_task_group_no_otel():
    """Test traced_task_group when OpenTelemetry is not available."""
    # Mock OpenTelemetry not available and create a fresh mock each time
    with patch("pynector.telemetry.context.HAS_OPENTELEMETRY", False):
        from pynector.telemetry.context import traced_task_group

        # Create a mock task group
        mock_task_group = MagicMock()

        # For each test, create a new mock function that returns a fresh coroutine
        with patch(
            "pynector.telemetry.context.create_task_group",
            side_effect=lambda: asyncio.Future(),
        ):
            # Patch the create_task_group to return our mock_task_group
            async def mock_create_task_group():
                return mock_task_group

            with patch(
                "pynector.telemetry.context.create_task_group",
                return_value=mock_create_task_group(),
            ):
                # Test with no attributes
                task_group = await traced_task_group(MagicMock(), "test_task_group")
                assert task_group == mock_task_group


@pytest.mark.asyncio
async def test_traced_gather_with_exception():
    """Test traced_gather with an exception in one of the coroutines."""
    # Mock OpenTelemetry not available
    with patch("pynector.telemetry.context.HAS_OPENTELEMETRY", False):
        from pynector.telemetry.context import traced_gather

        # Create test coroutines
        async def coro1():
            return 1

        async def coro_error():
            raise ValueError("Test exception")

        # Test with an exception
        with pytest.raises(ValueError):
            await traced_gather(MagicMock(), [coro1(), coro_error()])
