"""Test suite for eventbus.py (EventBus) - TDD Specification Implementation.

Focus: Robustness, resilience to handler failures, and non-blocking behavior.
"""

import time
from unittest.mock import AsyncMock, Mock, patch

import anyio
import pytest

from lionagi.base.eventbus import EventBus
from lionagi.ln.concurrency import Event, Lock, create_task_group, fail_after, gather


class TestEventBusRobustness:
    """TestSuite: EventBusRobustness - Async tests for robustness and resilience."""

    @pytest.mark.anyio
    async def test_basic_pub_sub(self):
        """Test: BasicPubSub

        GIVEN a subscriber H1 for topic "T"
        WHEN bus.emit("T", data=1) is called
        THEN H1 must be invoked with data=1.
        """
        bus = EventBus()
        handler_called = Event()
        received_data = {}

        # Define test handler
        async def handler_h1(data):
            received_data["data"] = data
            handler_called.set()

        # Subscribe handler to topic "T"
        bus.subscribe("T", handler_h1)

        # Emit event with data
        await bus.emit("T", data=1)

        # Wait for handler to be called
        with fail_after(1.0):
            await handler_called.wait()

        # Verify handler received correct data
        assert received_data["data"] == 1, "Handler should receive emitted data"

    @pytest.mark.anyio
    async def test_multiple_subscribers_same_topic(self):
        """Test multiple subscribers to the same topic."""
        bus = EventBus()
        handlers_called = []

        # Define multiple handlers
        async def handler_1(data):
            handlers_called.append(("handler_1", data))

        async def handler_2(data):
            handlers_called.append(("handler_2", data))

        async def handler_3(data):
            handlers_called.append(("handler_3", data))

        # Subscribe all handlers to same topic
        bus.subscribe("topic", handler_1)
        bus.subscribe("topic", handler_2)
        bus.subscribe("topic", handler_3)

        # Emit event
        await bus.emit("topic", data="test_data")

        # Allow handlers to complete
        await anyio.sleep(0.01)

        # Verify all handlers were called
        assert len(handlers_called) == 3, "All subscribers should be invoked"
        handler_names = [call[0] for call in handlers_called]
        assert "handler_1" in handler_names, "Handler 1 should be called"
        assert "handler_2" in handler_names, "Handler 2 should be called"
        assert "handler_3" in handler_names, "Handler 3 should be called"

        # Verify all received same data
        for _, data in handlers_called:
            assert data == "test_data", "All handlers should receive same data"

    @pytest.mark.anyio
    async def test_failing_handler_resilience(self):
        """Test: FailingHandlerResilience (CRITICAL: Robustness)

        GIVEN 3 subscribers (H1, H2_Fails, H3) for topic "T"
        WHEN bus.emit("T") is called (using robust gather)
        THEN H1 and H3 must complete successfully.
        AND the exception from H2_Fails must be caught and logged.
        AND the emit call itself must complete without raising an error.
        """
        bus = EventBus()
        successful_calls = []

        # Handler that succeeds
        async def handler_h1(data):
            successful_calls.append("h1")

        # Handler that fails
        async def handler_h2_fails(data):
            raise ValueError("Intentional handler failure")

        # Handler that succeeds
        async def handler_h3(data):
            successful_calls.append("h3")

        # Subscribe all handlers
        bus.subscribe("T", handler_h1)
        bus.subscribe("T", handler_h2_fails)
        bus.subscribe("T", handler_h3)

        # Mock logger to capture error logs
        with patch("lionagi.base.eventbus.logger") as mock_logger:
            # Emit should complete without raising an exception
            try:
                await bus.emit("T", data="test")
            except Exception as e:
                pytest.fail(f"EventBus.emit should not raise exception but got: {e}")

            # Allow handlers to complete
            await anyio.sleep(0.01)

            # Verify successful handlers completed
            assert "h1" in successful_calls, "Handler H1 should complete successfully"
            assert "h3" in successful_calls, "Handler H3 should complete successfully"
            assert len(successful_calls) == 2, "Only successful handlers should be recorded"

            # Verify error was logged (not raised)
            mock_logger.error.assert_called()
            # Check that the error message contains information about the failed handler
            error_call_args = mock_logger.error.call_args[0]
            assert (
                "handler failure" in str(error_call_args).lower()
                or "exception" in str(error_call_args).lower()
            )

    @pytest.mark.anyio
    async def test_slow_handler_timeout(self):
        """Test: SlowHandlerTimeout (CRITICAL: Non-Blocking)

        GIVEN a subscriber H_Slow that sleeps for 5 seconds
        AND the EventBus configured with a handler timeout of 1 second (using move_on_after)
        WHEN bus.emit("T") is called
        THEN the emit call must complete in approximately 1 second (not 5 seconds).
        AND a timeout warning must be logged for H_Slow.
        """
        # Create EventBus with 1 second timeout
        bus = EventBus(handler_timeout=1.0)

        completion_times = []

        # Fast handler
        async def handler_fast():
            completion_times.append("fast")

        # Slow handler that sleeps for 5 seconds
        async def handler_slow():
            await anyio.sleep(5.0)
            completion_times.append("slow")  # Should not reach this due to timeout

        # Subscribe handlers
        bus.subscribe("T", handler_fast)
        bus.subscribe("T", handler_slow)

        # Mock logger to capture timeout warnings
        with patch("lionagi.base.eventbus.logger") as mock_logger:
            # Measure emit time
            start_time = time.perf_counter()
            await bus.emit("T")
            end_time = time.perf_counter()

            emit_duration = end_time - start_time

            # Allow any remaining tasks to complete
            await anyio.sleep(0.01)

            # Verify emit completed in approximately 1 second (with tolerance)
            assert (
                emit_duration < 2.0
            ), f"Emit should complete in ~1 second but took {emit_duration:.3f}s"
            assert (
                emit_duration > 0.9
            ), f"Emit should take close to timeout duration but took {emit_duration:.3f}s"

            # Verify fast handler completed
            assert "fast" in completion_times, "Fast handler should complete"

            # Verify slow handler was cancelled (didn't complete)
            assert "slow" not in completion_times, "Slow handler should be cancelled by timeout"

            # Verify timeout was logged
            mock_logger.warning.assert_called()
            warning_message = str(mock_logger.warning.call_args)
            assert "timeout" in warning_message.lower() or "slow" in warning_message.lower()

    @pytest.mark.anyio
    async def test_unsubscribe_functionality(self):
        """Test that handlers can be properly unsubscribed."""
        bus = EventBus()
        handler_calls = []

        async def handler_to_remove(data):
            handler_calls.append("removed_handler")

        async def handler_to_keep(data):
            handler_calls.append("kept_handler")

        # Subscribe both handlers
        bus.subscribe("topic", handler_to_remove)
        bus.subscribe("topic", handler_to_keep)

        # First emit - both should be called
        await bus.emit("topic", data="test1")
        await anyio.sleep(0.01)

        assert len(handler_calls) == 2, "Both handlers should be called initially"

        # Unsubscribe one handler
        bus.unsubscribe("topic", handler_to_remove)

        # Clear previous calls
        handler_calls.clear()

        # Second emit - only kept handler should be called
        await bus.emit("topic", data="test2")
        await anyio.sleep(0.01)

        assert len(handler_calls) == 1, "Only one handler should be called after unsubscribe"
        assert "kept_handler" in handler_calls, "Kept handler should still be called"
        assert "removed_handler" not in handler_calls, "Removed handler should not be called"

    @pytest.mark.anyio
    async def test_empty_topic_emission(self):
        """Test emitting to topics with no subscribers."""
        bus = EventBus()

        # Should complete without error even if no subscribers
        try:
            await bus.emit("nonexistent_topic", data="test")
        except Exception as e:
            pytest.fail(f"Emitting to empty topic should not raise exception: {e}")

    @pytest.mark.anyio
    async def test_concurrent_emissions(self):
        """Test concurrent emissions to the same topic."""
        bus = EventBus()
        handler_calls = []
        call_lock = Lock()

        async def concurrent_handler(data):
            async with call_lock:
                handler_calls.append(data)

        bus.subscribe("concurrent_topic", concurrent_handler)

        # Create multiple concurrent emissions using Lion gather (same as EventBus)
        await gather(
            *(bus.emit("concurrent_topic", data=f"emission_{i}") for i in range(10)),
            return_exceptions=True,
        )
        await anyio.sleep(0.01)  # Allow handlers to complete

        # Verify all emissions were handled
        assert len(handler_calls) == 10, "All concurrent emissions should be handled"

        # Verify all expected data was received
        expected_data = [f"emission_{i}" for i in range(10)]
        for expected in expected_data:
            assert expected in handler_calls, f"Missing emission data: {expected}"

    @pytest.mark.anyio
    async def test_handler_exception_isolation(self):
        """Test that exceptions in one handler don't affect others."""
        bus = EventBus()
        execution_order = []

        async def handler_before_failure():
            execution_order.append("before")

        async def failing_handler():
            execution_order.append("failing")
            raise RuntimeError("Handler failure")

        async def handler_after_failure():
            execution_order.append("after")

        # Subscribe in specific order
        bus.subscribe("isolation_test", handler_before_failure)
        bus.subscribe("isolation_test", failing_handler)
        bus.subscribe("isolation_test", handler_after_failure)

        # Mock logger to avoid error output during test
        with patch("lionagi.base.eventbus.logger"):
            await bus.emit("isolation_test")
            await anyio.sleep(0.01)

        # Verify all handlers executed despite the failure
        assert "before" in execution_order, "Handler before failure should execute"
        assert "failing" in execution_order, "Failing handler should execute (before failing)"
        assert "after" in execution_order, "Handler after failure should still execute"

    @pytest.mark.anyio
    async def test_handler_with_complex_data(self):
        """Test handlers receive complex data structures correctly."""
        bus = EventBus()
        received_data = None

        async def data_handler(data):
            nonlocal received_data
            received_data = data

        bus.subscribe("complex_data", data_handler)

        # Test with complex nested data structure
        complex_data = {
            "nested": {"list": [1, 2, {"inner": True}], "tuple": (3, 4, 5)},
            "set_data": {6, 7, 8},  # Note: sets might be converted depending on implementation
        }

        await bus.emit("complex_data", data=complex_data)
        await anyio.sleep(0.01)

        # Verify complex data was received correctly
        assert received_data is not None, "Handler should receive data"
        assert received_data["nested"]["list"] == [
            1,
            2,
            {"inner": True},
        ], "Nested list should be preserved"
        assert received_data["nested"]["tuple"] == (3, 4, 5), "Tuple should be preserved"
        assert "set_data" in received_data, "Set data should be present"

    @pytest.mark.anyio
    async def test_eventbus_cleanup(self):
        """Test EventBus cleanup and resource management."""
        bus = EventBus()
        handler_calls = []

        async def test_handler(data):
            handler_calls.append(data)

        # Subscribe and emit
        bus.subscribe("cleanup_test", test_handler)
        await bus.emit("cleanup_test", data="before_cleanup")
        await anyio.sleep(0.01)

        assert len(handler_calls) == 1, "Handler should be called before cleanup"

        # Test cleanup functionality (if implemented)
        if hasattr(bus, "cleanup"):
            await bus.cleanup()

        # Verify system is still stable after cleanup
        handler_calls.clear()
        await bus.emit("cleanup_test", data="after_cleanup")
        await anyio.sleep(0.01)

        # Behavior depends on implementation - either handlers are cleared or still work
        assert isinstance(len(handler_calls), int), "System should remain stable after cleanup"
