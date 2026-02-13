"""Tests for lionagi.service.broadcaster module."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from lionagi.protocols.generic.event import Event
from lionagi.service.broadcaster import Broadcaster


class SampleEvent(Event):
    """Sample event class for broadcaster tests."""

    event_type: str = "test_event"


class TestBroadcaster:
    """Test suite for Broadcaster class."""

    @pytest.fixture(autouse=True)
    def reset_broadcaster(self):
        """Reset broadcaster state before each test."""
        # Clear subscribers before each test
        Broadcaster._subscribers.clear()
        Broadcaster._instance = None
        yield
        # Clean up after test
        Broadcaster._subscribers.clear()
        Broadcaster._instance = None

    def test_broadcaster_singleton(self):
        """Test that Broadcaster follows singleton pattern."""

        # Create a subclass for testing
        class TestBroadcaster(Broadcaster):
            _event_type = SampleEvent

        broadcaster1 = TestBroadcaster()
        broadcaster2 = TestBroadcaster()

        assert broadcaster1 is broadcaster2
        assert TestBroadcaster._instance is broadcaster1

    def test_subscribe_adds_callback(self):
        """Test that subscribe adds callback to subscribers list."""

        class TestBroadcaster(Broadcaster):
            _event_type = SampleEvent

        callback = MagicMock()

        TestBroadcaster.subscribe(callback)

        assert TestBroadcaster.get_subscriber_count() == 1

    def test_subscribe_prevents_duplicates(self):
        """Test that subscribing same callback twice doesn't duplicate."""

        class TestBroadcaster(Broadcaster):
            _event_type = SampleEvent

        callback = MagicMock()

        TestBroadcaster.subscribe(callback)
        TestBroadcaster.subscribe(callback)

        assert TestBroadcaster.get_subscriber_count() == 1

    def test_unsubscribe_removes_callback(self):
        """Test that unsubscribe removes callback from subscribers."""

        class TestBroadcaster(Broadcaster):
            _event_type = SampleEvent

        callback = MagicMock()

        TestBroadcaster.subscribe(callback)
        assert TestBroadcaster.get_subscriber_count() == 1

        TestBroadcaster.unsubscribe(callback)
        assert TestBroadcaster.get_subscriber_count() == 0

    def test_unsubscribe_nonexistent_callback_no_error(self):
        """Test that unsubscribing nonexistent callback doesn't raise error."""

        class TestBroadcaster(Broadcaster):
            _event_type = SampleEvent

        callback = MagicMock()

        # Should not raise error
        TestBroadcaster.unsubscribe(callback)
        assert TestBroadcaster.get_subscriber_count() == 0

    @pytest.mark.asyncio
    async def test_broadcast_calls_sync_callback(self):
        """Test that broadcast calls synchronous callbacks."""

        class TestBroadcaster(Broadcaster):
            _event_type = SampleEvent

        callback = MagicMock()
        event = SampleEvent()

        TestBroadcaster.subscribe(callback)
        await TestBroadcaster.broadcast(event)

        callback.assert_called_once_with(event)

    @pytest.mark.asyncio
    async def test_broadcast_calls_async_callback(self):
        """Test that broadcast awaits asynchronous callbacks."""

        class TestBroadcaster(Broadcaster):
            _event_type = SampleEvent

        callback = AsyncMock()
        event = SampleEvent()

        TestBroadcaster.subscribe(callback)
        await TestBroadcaster.broadcast(event)

        callback.assert_awaited_once_with(event)

    @pytest.mark.asyncio
    async def test_broadcast_calls_multiple_subscribers(self):
        """Test that broadcast calls all registered subscribers."""

        class TestBroadcaster(Broadcaster):
            _event_type = SampleEvent

        callback1 = MagicMock()
        callback2 = MagicMock()
        callback3 = AsyncMock()
        event = SampleEvent()

        TestBroadcaster.subscribe(callback1)
        TestBroadcaster.subscribe(callback2)
        TestBroadcaster.subscribe(callback3)

        await TestBroadcaster.broadcast(event)

        callback1.assert_called_once_with(event)
        callback2.assert_called_once_with(event)
        callback3.assert_awaited_once_with(event)

    @pytest.mark.asyncio
    async def test_broadcast_validates_event_type(self):
        """Test that broadcast raises error for wrong event type."""

        class SpecificBroadcaster(Broadcaster):
            _event_type = SampleEvent

        class OtherEvent(Event):
            event_type: str = "other"

        callback = MagicMock()
        wrong_event = OtherEvent()

        SpecificBroadcaster.subscribe(callback)

        with pytest.raises(
            ValueError, match="Event must be of type SampleEvent"
        ):
            await SpecificBroadcaster.broadcast(wrong_event)

        # Callback should not have been called
        callback.assert_not_called()

    @pytest.mark.asyncio
    async def test_broadcast_handles_callback_exception(self):
        """Test that broadcast catches and logs callback exceptions."""

        class TestBroadcaster(Broadcaster):
            _event_type = SampleEvent

        failing_callback = MagicMock(
            side_effect=RuntimeError("Callback error")
        )
        successful_callback = MagicMock()
        event = SampleEvent()

        TestBroadcaster.subscribe(failing_callback)
        TestBroadcaster.subscribe(successful_callback)

        # Should not raise, but log the error
        await TestBroadcaster.broadcast(event)

        # Both callbacks should be attempted
        failing_callback.assert_called_once_with(event)
        successful_callback.assert_called_once_with(event)

    @pytest.mark.asyncio
    async def test_broadcast_handles_async_callback_exception(self):
        """Test that broadcast catches and logs async callback exceptions."""

        class TestBroadcaster(Broadcaster):
            _event_type = SampleEvent

        failing_callback = AsyncMock(
            side_effect=RuntimeError("Async callback error")
        )
        successful_callback = AsyncMock()
        event = SampleEvent()

        TestBroadcaster.subscribe(failing_callback)
        TestBroadcaster.subscribe(successful_callback)

        # Should not raise, but log the error
        await TestBroadcaster.broadcast(event)

        # Both callbacks should be attempted
        assert failing_callback.await_count == 1
        successful_callback.assert_awaited_once_with(event)

    @pytest.mark.asyncio
    async def test_broadcast_with_no_subscribers(self):
        """Test that broadcasting with no subscribers doesn't error."""

        class TestBroadcaster(Broadcaster):
            _event_type = SampleEvent

        event = SampleEvent()

        # Should not raise error
        await TestBroadcaster.broadcast(event)
        assert TestBroadcaster.get_subscriber_count() == 0

    def test_get_subscriber_count_accuracy(self):
        """Test that get_subscriber_count returns accurate count."""

        class TestBroadcaster(Broadcaster):
            _event_type = SampleEvent

        assert TestBroadcaster.get_subscriber_count() == 0

        callback1 = MagicMock()
        callback2 = MagicMock()
        callback3 = MagicMock()

        TestBroadcaster.subscribe(callback1)
        assert TestBroadcaster.get_subscriber_count() == 1

        TestBroadcaster.subscribe(callback2)
        TestBroadcaster.subscribe(callback3)
        assert TestBroadcaster.get_subscriber_count() == 3

        TestBroadcaster.unsubscribe(callback2)
        assert TestBroadcaster.get_subscriber_count() == 2

    def test_multiple_broadcaster_subclasses_independent(self):
        """Test that different Broadcaster subclasses maintain independent state."""

        class BroadcasterA(Broadcaster):
            _event_type = SampleEvent
            _subscribers = []
            _instance = None

        class TestEvent2(Event):
            event_type: str = "test2"

        class BroadcasterB(Broadcaster):
            _event_type = TestEvent2
            _subscribers = []
            _instance = None

        callback_a = MagicMock()
        callback_b = MagicMock()

        BroadcasterA.subscribe(callback_a)
        BroadcasterB.subscribe(callback_b)

        assert BroadcasterA.get_subscriber_count() == 1
        assert BroadcasterB.get_subscriber_count() == 1

    @pytest.mark.asyncio
    async def test_broadcast_mixed_sync_async_callbacks(self):
        """Test broadcasting to mix of sync and async callbacks."""

        class TestBroadcaster(Broadcaster):
            _event_type = SampleEvent

        sync_callback1 = MagicMock()
        async_callback = AsyncMock()
        sync_callback2 = MagicMock()
        event = SampleEvent()

        TestBroadcaster.subscribe(sync_callback1)
        TestBroadcaster.subscribe(async_callback)
        TestBroadcaster.subscribe(sync_callback2)

        await TestBroadcaster.broadcast(event)

        sync_callback1.assert_called_once_with(event)
        async_callback.assert_awaited_once_with(event)
        sync_callback2.assert_called_once_with(event)
