"""Tests for the tracing module."""

from unittest.mock import patch

import pytest


def test_noop_span_init():
    """Test NoOpSpan initialization."""
    from pynector.telemetry.tracing import NoOpSpan

    span = NoOpSpan("test_span", {"key": "value"})

    assert span.name == "test_span"
    assert span.attributes == {"key": "value"}

    # Test with default values
    span = NoOpSpan()
    assert span.name == ""
    assert span.attributes == {}


def test_noop_span_context_manager():
    """Test NoOpSpan as a context manager."""
    from pynector.telemetry.tracing import NoOpSpan

    with NoOpSpan("test_span") as span:
        assert isinstance(span, NoOpSpan)
        span.set_attribute("key", "value")
        assert span.attributes["key"] == "value"


@pytest.mark.asyncio
async def test_noop_span_async_context_manager():
    """Test NoOpSpan as an async context manager."""
    from pynector.telemetry.tracing import NoOpSpan

    async with NoOpSpan("test_span") as span:
        assert isinstance(span, NoOpSpan)
        span.set_attribute("key", "value")
        assert span.attributes["key"] == "value"


def test_noop_span_methods():
    """Test NoOpSpan methods."""
    from pynector.telemetry import Status, StatusCode
    from pynector.telemetry.tracing import NoOpSpan

    span = NoOpSpan("test_span")

    # Test set_attribute
    span.set_attribute("key", "value")
    assert span.attributes["key"] == "value"

    # Test add_event (no-op, should not raise)
    span.add_event("test_event", {"event_key": "event_value"})

    # Test record_exception (no-op, should not raise)
    try:
        raise ValueError("Test exception")
    except ValueError as e:
        span.record_exception(e)

    # Test set_status (no-op, should not raise)
    span.set_status(Status(StatusCode.ERROR))


@pytest.mark.asyncio
async def test_async_span_wrapper():
    """Test AsyncSpanWrapper."""
    from pynector.telemetry.tracing import AsyncSpanWrapper

    # Create a mock span
    class MockSpan:
        def __init__(self):
            self.entered = False
            self.exited = False

        def __enter__(self):
            self.entered = True
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.exited = True

    mock_span = MockSpan()
    wrapper = AsyncSpanWrapper(mock_span)

    async with wrapper as span:
        assert span is mock_span
        assert mock_span.entered is True
        assert mock_span.exited is False

    assert mock_span.exited is True


@pytest.mark.asyncio
async def test_async_span_wrapper_with_token():
    """Test AsyncSpanWrapper with a token."""
    from pynector.telemetry.tracing import AsyncSpanWrapper

    # Create a mock span
    class MockSpan:
        def __init__(self):
            self.entered = False
            self.exited = False

        def __enter__(self):
            self.entered = True
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.exited = True

    mock_span = MockSpan()
    mock_token = "test_token"

    # Mock the opentelemetry.context.detach function
    detach_called = False
    detach_token = None

    def mock_detach(token):
        nonlocal detach_called, detach_token
        detach_called = True
        detach_token = token

    # Patch the opentelemetry.context.detach import in AsyncSpanWrapper.__aexit__
    with patch("pynector.telemetry.tracing.detach", mock_detach):
        wrapper = AsyncSpanWrapper(mock_span, mock_token)

        async with wrapper as span:
            assert span is mock_span
            assert mock_span.entered is True
            assert mock_span.exited is False

        assert mock_span.exited is True
        assert detach_called is True
        assert detach_token == mock_token
