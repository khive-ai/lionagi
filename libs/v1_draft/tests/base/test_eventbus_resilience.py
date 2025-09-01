"""Test suite for base/eventbus.py - Resilience and observability improvements.

Focus: Timeout handling, failure isolation, observability counters,
and edge case behavior under structured concurrency.
"""

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock

import anyio
import pytest

from lionagi.base.eventbus import EventBus


class TestEventBusResilience:
    """Test EventBus resilience under failure and timeout conditions."""

    @pytest.mark.anyio
    async def test_emit_to_no_subscribers_is_noop(self):
        """Test that emitting to topic with no subscribers is fast no-op."""
        bus = EventBus()

        # Measure time - should be very fast since no work is done
        start_time = anyio.current_time()
        await bus.emit("nonexistent_topic", {"data": "test"})
        elapsed = anyio.current_time() - start_time

        # Should complete almost instantly (< 10ms for no-op)
        assert elapsed < 0.01, f"No-subscriber emit took {elapsed}s, should be nearly instant"

    @pytest.mark.anyio
    async def test_handler_timeout_cancels_and_continues(self):
        """Test that slow handlers are cancelled after timeout, other handlers continue."""
        bus = EventBus()

        # Track handler execution
        results = []

        async def fast_handler(data):
            results.append("fast_completed")

        async def slow_handler(data):
            try:
                await anyio.sleep(2.0)  # Longer than timeout
                results.append("slow_completed")  # Should not reach here
            except anyio.get_cancelled_exc_class():
                results.append("slow_cancelled")
                raise

        async def another_fast_handler(data):
            results.append("another_fast_completed")

        # Subscribe handlers
        bus.subscribe("test_topic", fast_handler)
        bus.subscribe("test_topic", slow_handler)
        bus.subscribe("test_topic", another_fast_handler)

        # Emit with short timeout
        start_time = anyio.current_time()
        await bus.emit("test_topic", {"data": "test"}, timeout_s=0.5)
        elapsed = anyio.current_time() - start_time

        # Should complete around timeout duration, not wait for slow handler
        assert 0.4 < elapsed < 0.8, f"Emit took {elapsed}s, should be around 0.5s timeout"

        # Fast handlers should complete, slow should be cancelled
        assert "fast_completed" in results
        assert "another_fast_completed" in results
        assert "slow_completed" not in results
        assert "slow_cancelled" in results

    @pytest.mark.anyio
    async def test_handler_failure_isolated_others_continue(self):
        """Test that handler exceptions don't prevent other handlers from running."""
        bus = EventBus()

        results = []

        async def failing_handler(data):
            results.append("failing_started")
            raise ValueError("Handler intentionally failed")

        async def succeeding_handler_1(data):
            results.append("succeeding_1_completed")

        async def succeeding_handler_2(data):
            results.append("succeeding_2_completed")

        # Subscribe all handlers
        bus.subscribe("test_topic", failing_handler)
        bus.subscribe("test_topic", succeeding_handler_1)
        bus.subscribe("test_topic", succeeding_handler_2)

        # Emit should complete without raising exception
        await bus.emit("test_topic", {"data": "test"})

        # All handlers should have been attempted
        assert "failing_started" in results
        assert "succeeding_1_completed" in results
        assert "succeeding_2_completed" in results

    @pytest.mark.anyio
    async def test_mixed_failure_and_timeout_scenarios(self):
        """Test combination of handler failures and timeouts."""
        bus = EventBus()

        results = []

        async def quick_success(data):
            results.append("quick_success")

        async def quick_failure(data):
            results.append("quick_failure")
            raise RuntimeError("Quick fail")

        async def slow_success(data):
            await anyio.sleep(1.0)  # Will timeout
            results.append("slow_success")

        async def slow_failure(data):
            await anyio.sleep(1.0)  # Will timeout before failing
            results.append("slow_failure")
            raise Exception("Slow fail")

        # Subscribe mixed handlers
        bus.subscribe("mixed_topic", quick_success)
        bus.subscribe("mixed_topic", quick_failure)
        bus.subscribe("mixed_topic", slow_success)
        bus.subscribe("mixed_topic", slow_failure)

        # Emit with timeout shorter than slow handlers
        await bus.emit("mixed_topic", {"test": True}, timeout_s=0.2)

        # Quick handlers should complete (success and failure)
        assert "quick_success" in results
        assert "quick_failure" in results

        # Slow handlers should timeout before completing
        assert "slow_success" not in results
        assert "slow_failure" not in results

    @pytest.mark.anyio
    async def test_handler_cancellation_during_emit(self):
        """Test that handlers are properly cancelled if emit itself is cancelled."""
        bus = EventBus()

        handler_started = anyio.Event()
        handler_cancelled = False

        async def cancellable_handler(data):
            nonlocal handler_cancelled
            handler_started.set()
            try:
                await anyio.sleep(10)  # Long operation
            except anyio.get_cancelled_exc_class():
                handler_cancelled = True
                raise

        bus.subscribe("cancel_topic", cancellable_handler)

        # Start emit in background and cancel it
        async with anyio.create_task_group() as tg:
            tg.start_soon(bus.emit, "cancel_topic", {"data": "test"})

            # Wait for handler to start
            await handler_started.wait()

            # Cancel the task group (simulating cancellation)
            tg.cancel_scope.cancel()

        # Handler should have been cancelled
        assert handler_cancelled, "Handler should have been cancelled when emit was cancelled"

    @pytest.mark.anyio
    async def test_unsubscribe_during_emit(self):
        """Test that unsubscribing during emit doesn't cause issues."""
        bus = EventBus()

        results = []
        emit_started = anyio.Event()

        async def handler1(data):
            emit_started.set()
            await anyio.sleep(0.1)  # Give time for unsubscribe
            results.append("handler1")

        async def handler2(data):
            results.append("handler2")

        # Subscribe handlers
        bus.subscribe("unsub_topic", handler1)
        bus.subscribe("unsub_topic", handler2)

        async def unsubscribe_after_start():
            await emit_started.wait()
            bus.unsubscribe("unsub_topic", handler2)

        # Run emit and unsubscribe concurrently
        async with anyio.create_task_group() as tg:
            tg.start_soon(bus.emit, "unsub_topic", {"data": "test"})
            tg.start_soon(unsubscribe_after_start)

        # handler1 should complete, handler2 may or may not (timing dependent)
        assert "handler1" in results
        # Don't assert on handler2 since unsubscribe timing affects execution

    @pytest.mark.anyio
    async def test_multiple_concurrent_emits_same_topic(self):
        """Test multiple concurrent emits to same topic don't interfere."""
        bus = EventBus()

        results = []

        async def counting_handler(data):
            await anyio.sleep(0.05)  # Small delay to test concurrency
            results.append(data["count"])

        bus.subscribe("concurrent_topic", counting_handler)

        # Emit multiple events concurrently
        async with anyio.create_task_group() as tg:
            for i in range(5):
                tg.start_soon(bus.emit, "concurrent_topic", {"count": i})

        # All events should be processed
        assert len(results) == 5
        assert set(results) == {0, 1, 2, 3, 4}

    @pytest.mark.anyio
    async def test_topic_with_special_characters(self):
        """Test topics with special characters are handled correctly."""
        bus = EventBus()

        result = None

        async def special_handler(data):
            nonlocal result
            result = data

        # Topics with various special characters
        special_topics = [
            "topic.with.dots",
            "topic-with-dashes",
            "topic_with_underscores",
            "topic/with/slashes",
            "topic:with:colons",
            "topic with spaces",
            "topic@with#symbols!",
            "🚀 topic with emoji 🎉",
        ]

        for topic in special_topics:
            bus.subscribe(topic, special_handler)
            await bus.emit(topic, {"topic": topic})

            assert result["topic"] == topic
            result = None  # Reset for next test

            bus.unsubscribe(topic, special_handler)

    @pytest.mark.anyio
    async def test_empty_data_and_none_data(self):
        """Test emitting empty or None data."""
        bus = EventBus()

        received_data = []

        async def data_handler(data):
            received_data.append(data)

        bus.subscribe("data_topic", data_handler)

        # Test various data types
        test_cases = [
            {},  # Empty dict
            None,  # None
            [],  # Empty list
            "",  # Empty string
            0,  # Zero
            False,  # Boolean False
        ]

        for test_data in test_cases:
            await bus.emit("data_topic", test_data)

        assert len(received_data) == len(test_cases)
        assert received_data == test_cases
