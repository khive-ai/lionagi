# Copyright (c) 2023-2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for Event lifecycle wrapper (invoke/stream template method pattern)."""

from __future__ import annotations

import asyncio

import pytest

from lionagi.protocols.generic.event import Event, EventStatus

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


class SuccessEvent(Event):
    """Event subclass that succeeds via _invoke()."""

    async def _invoke(self) -> None:
        self.execution.response = "ok"


class FailingEvent(Event):
    """Event subclass that raises via _invoke()."""

    async def _invoke(self) -> None:
        raise ValueError("boom")


class SlowEvent(Event):
    """Event subclass with a measurable delay."""

    async def _invoke(self) -> None:
        await asyncio.sleep(0.05)
        self.execution.response = "done"


class DirectOverrideEvent(Event):
    """Subclass that overrides invoke() directly (backwards compat)."""

    async def invoke(self) -> None:
        self.execution.status = EventStatus.COMPLETED
        self.execution.response = "direct"


class StreamSuccessEvent(Event):
    """Event subclass that streams successfully via _stream()."""

    async def _stream(self):
        for chunk in ["a", "b", "c"]:
            yield chunk


class StreamFailEvent(Event):
    """Event subclass whose _stream() raises mid-iteration."""

    async def _stream(self):
        yield "first"
        raise RuntimeError("stream failed")


class StreamDirectOverrideEvent(Event):
    """Subclass that overrides stream() directly (backwards compat)."""

    async def stream(self):
        self.execution.status = EventStatus.COMPLETED
        self.execution.response = "direct-stream"
        yield "direct-chunk"


# ---------------------------------------------------------------------------
# invoke() lifecycle tests
# ---------------------------------------------------------------------------


class TestInvokeLifecycle:
    """Tests for the invoke() template method wrapper."""

    @pytest.mark.asyncio
    async def test_invoke_calls_inner_invoke(self):
        """_invoke() is called by invoke()."""
        event = SuccessEvent()
        await event.invoke()
        assert event.execution.response == "ok"

    @pytest.mark.asyncio
    async def test_status_pending_to_completed(self):
        """Status transitions PENDING -> PROCESSING -> COMPLETED on success."""
        event = SuccessEvent()
        assert event.execution.status == EventStatus.PENDING
        await event.invoke()
        assert event.execution.status == EventStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_status_pending_to_failed(self):
        """Status transitions PENDING -> PROCESSING -> FAILED on error."""
        event = FailingEvent()
        assert event.execution.status == EventStatus.PENDING
        with pytest.raises(ValueError, match="boom"):
            await event.invoke()
        assert event.execution.status == EventStatus.FAILED

    @pytest.mark.asyncio
    async def test_error_captured_on_failure(self):
        """Error is captured in execution.error via add_error()."""
        event = FailingEvent()
        with pytest.raises(ValueError):
            await event.invoke()
        assert event.execution.error is not None
        assert isinstance(event.execution.error, ValueError)
        assert "boom" in str(event.execution.error)

    @pytest.mark.asyncio
    async def test_error_is_reraised(self):
        """The original exception is re-raised after being captured."""
        event = FailingEvent()
        with pytest.raises(ValueError, match="boom"):
            await event.invoke()

    @pytest.mark.asyncio
    async def test_idempotency_completed(self):
        """Calling invoke() on a COMPLETED event is a no-op."""
        event = SuccessEvent()
        await event.invoke()
        assert event.execution.status == EventStatus.COMPLETED
        first_duration = event.execution.duration

        # Invoke again -- should be a no-op
        await event.invoke()
        assert event.execution.status == EventStatus.COMPLETED
        assert event.execution.duration == first_duration
        assert event.execution.response == "ok"

    @pytest.mark.asyncio
    async def test_idempotency_failed(self):
        """Calling invoke() on a FAILED event is a no-op."""
        event = FailingEvent()
        with pytest.raises(ValueError):
            await event.invoke()
        assert event.execution.status == EventStatus.FAILED
        first_duration = event.execution.duration

        # Invoke again -- should be a no-op (no exception)
        await event.invoke()
        assert event.execution.status == EventStatus.FAILED
        assert event.execution.duration == first_duration

    @pytest.mark.asyncio
    async def test_duration_recorded(self):
        """Duration is recorded in execution.duration."""
        event = SlowEvent()
        await event.invoke()
        assert event.execution.duration is not None
        # Slept 50ms, so duration should be at least 0.04s (allow for timing jitter)
        assert event.execution.duration >= 0.04

    @pytest.mark.asyncio
    async def test_duration_recorded_on_failure(self):
        """Duration is recorded even when _invoke() fails."""
        event = FailingEvent()
        with pytest.raises(ValueError):
            await event.invoke()
        assert event.execution.duration is not None
        assert event.execution.duration >= 0

    @pytest.mark.asyncio
    async def test_backwards_compat_direct_override(self):
        """Subclass overriding invoke() directly bypasses lifecycle wrapper."""
        event = DirectOverrideEvent()
        await event.invoke()
        assert event.execution.status == EventStatus.COMPLETED
        assert event.execution.response == "direct"
        # No duration set because the direct override does not use the wrapper
        assert event.execution.duration is None

    @pytest.mark.asyncio
    async def test_base_event_invoke_raises(self):
        """Calling invoke() on bare Event raises NotImplementedError."""
        event = Event()
        with pytest.raises(NotImplementedError):
            await event.invoke()
        assert event.execution.status == EventStatus.FAILED

    @pytest.mark.asyncio
    async def test_response_preserved_on_success(self):
        """Response set in _invoke() is preserved after lifecycle completes."""
        event = SuccessEvent()
        await event.invoke()
        assert event.response == "ok"
        assert event.execution.response == "ok"


# ---------------------------------------------------------------------------
# stream() lifecycle tests
# ---------------------------------------------------------------------------


class TestStreamLifecycle:
    """Tests for the stream() template method wrapper."""

    @pytest.mark.asyncio
    async def test_stream_yields_chunks(self):
        """_stream() chunks are yielded by stream()."""
        event = StreamSuccessEvent()
        chunks = []
        async for chunk in event.stream():
            chunks.append(chunk)
        assert chunks == ["a", "b", "c"]

    @pytest.mark.asyncio
    async def test_stream_status_completed(self):
        """Status transitions to COMPLETED after successful streaming."""
        event = StreamSuccessEvent()
        async for _ in event.stream():
            pass
        assert event.execution.status == EventStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_stream_status_failed_on_error(self):
        """Status transitions to FAILED when _stream() raises."""
        event = StreamFailEvent()
        chunks = []
        with pytest.raises(RuntimeError, match="stream failed"):
            async for chunk in event.stream():
                chunks.append(chunk)
        assert event.execution.status == EventStatus.FAILED
        # First chunk was yielded before the error
        assert chunks == ["first"]

    @pytest.mark.asyncio
    async def test_stream_error_captured(self):
        """Error is captured in execution.error during streaming."""
        event = StreamFailEvent()
        with pytest.raises(RuntimeError):
            async for _ in event.stream():
                pass
        assert event.execution.error is not None
        assert isinstance(event.execution.error, RuntimeError)

    @pytest.mark.asyncio
    async def test_stream_duration_recorded(self):
        """Duration is recorded after streaming completes."""
        event = StreamSuccessEvent()
        async for _ in event.stream():
            pass
        assert event.execution.duration is not None
        assert event.execution.duration >= 0

    @pytest.mark.asyncio
    async def test_stream_duration_recorded_on_failure(self):
        """Duration is recorded even when streaming fails."""
        event = StreamFailEvent()
        with pytest.raises(RuntimeError):
            async for _ in event.stream():
                pass
        assert event.execution.duration is not None
        assert event.execution.duration >= 0

    @pytest.mark.asyncio
    async def test_stream_idempotency_completed(self):
        """Calling stream() on a COMPLETED event yields nothing."""
        event = StreamSuccessEvent()
        async for _ in event.stream():
            pass
        assert event.execution.status == EventStatus.COMPLETED

        # Stream again -- should yield nothing
        chunks = []
        async for chunk in event.stream():
            chunks.append(chunk)
        assert chunks == []

    @pytest.mark.asyncio
    async def test_stream_idempotency_failed(self):
        """Calling stream() on a FAILED event yields nothing."""
        event = StreamFailEvent()
        with pytest.raises(RuntimeError):
            async for _ in event.stream():
                pass
        assert event.execution.status == EventStatus.FAILED

        # Stream again -- should yield nothing (no exception)
        chunks = []
        async for chunk in event.stream():
            chunks.append(chunk)
        assert chunks == []

    @pytest.mark.asyncio
    async def test_stream_backwards_compat_direct_override(self):
        """Subclass overriding stream() directly bypasses lifecycle wrapper."""
        event = StreamDirectOverrideEvent()
        chunks = []
        async for chunk in event.stream():
            chunks.append(chunk)
        assert chunks == ["direct-chunk"]
        assert event.execution.status == EventStatus.COMPLETED
        assert event.execution.response == "direct-stream"

    @pytest.mark.asyncio
    async def test_base_event_stream_raises(self):
        """Calling stream() on bare Event raises NotImplementedError."""
        event = Event()
        with pytest.raises(NotImplementedError):
            async for _ in event.stream():
                pass
        assert event.execution.status == EventStatus.FAILED


# File: tests/protocols/generic/test_event_lifecycle.py
