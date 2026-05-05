# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""
Tests for lionagi.beta.resource.processor — Processor and Executor classes.
"""

import asyncio
from typing import ClassVar
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from lionagi._errors import QueueFullError, ValidationError
from lionagi.beta.resource.backend import (
    Calling,
    Normalized,
    ResourceBackend,
    ResourceConfig,
)
from lionagi.beta.resource.processor import Executor, Processor
from lionagi.protocols.generic.event import Event, EventStatus
from lionagi.protocols.generic.pile import Pile

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


class SimpleBackend(ResourceBackend):
    @property
    def event_type(self):
        return SimpleCalling

    def create_payload(self, request=None, **kwargs):
        return dict(request or {}, **kwargs)

    async def call(self, request, **kwargs):
        return self.normalize_response({"result": "ok"})

    async def stream(self, request, **kwargs):
        yield b"chunk"

    @property
    def endpoint(self):
        return None


class SimpleCalling(Calling):
    backend: SimpleBackend

    @property
    def call_args(self):
        return {"request": self.payload}


class SimpleProcessor(Processor):
    event_type: ClassVar = SimpleCalling


class SimpleExecutor(Executor):
    processor_type: ClassVar = SimpleProcessor


def make_backend():
    return SimpleBackend(config=ResourceConfig(provider="test", name="test-model"))


def make_calling():
    backend = make_backend()
    return SimpleCalling(backend=backend, payload={"test": True})


def make_pile():
    return Pile(item_type=Event)


def make_processor(capacity=10, refresh=1.0, pile=None):
    p = pile or make_pile()
    return SimpleProcessor(
        queue_capacity=capacity,
        capacity_refresh_time=refresh,
        pile=p,
    )


def make_executor(config=None):
    config = config or {"queue_capacity": 10, "capacity_refresh_time": 1.0}
    return SimpleExecutor(processor_config=config)


# ---------------------------------------------------------------------------
# Processor validation tests
# ---------------------------------------------------------------------------


class TestProcessorValidation:
    def test_queue_capacity_zero_raises(self):
        with pytest.raises(ValidationError, match="Queue capacity"):
            SimpleProcessor(
                queue_capacity=0, capacity_refresh_time=1.0, pile=make_pile()
            )

    def test_queue_capacity_too_large_raises(self):
        with pytest.raises(ValidationError, match="10000"):
            SimpleProcessor(
                queue_capacity=10001, capacity_refresh_time=1.0, pile=make_pile()
            )

    def test_capacity_refresh_too_small_raises(self):
        with pytest.raises(ValidationError, match="0.01"):
            SimpleProcessor(
                queue_capacity=10, capacity_refresh_time=0.001, pile=make_pile()
            )

    def test_capacity_refresh_too_large_raises(self):
        with pytest.raises(ValidationError, match="3600"):
            SimpleProcessor(
                queue_capacity=10, capacity_refresh_time=3601, pile=make_pile()
            )

    def test_concurrency_limit_zero_raises(self):
        with pytest.raises(ValidationError, match="Concurrency"):
            SimpleProcessor(
                queue_capacity=10,
                capacity_refresh_time=1.0,
                pile=make_pile(),
                concurrency_limit=0,
            )

    def test_max_queue_size_zero_raises(self):
        with pytest.raises(ValidationError, match="Max queue size"):
            SimpleProcessor(
                queue_capacity=10,
                capacity_refresh_time=1.0,
                pile=make_pile(),
                max_queue_size=0,
            )

    def test_max_denial_tracking_zero_raises(self):
        with pytest.raises(ValidationError, match="denial"):
            SimpleProcessor(
                queue_capacity=10,
                capacity_refresh_time=1.0,
                pile=make_pile(),
                max_denial_tracking=0,
            )

    def test_valid_processor_creation(self):
        proc = make_processor()
        assert proc.queue_capacity == 10
        assert proc.capacity_refresh_time == 1.0
        assert proc.available_capacity == 10
        assert not proc.is_stopped()
        assert not proc.execution_mode


# ---------------------------------------------------------------------------
# Processor property accessors
# ---------------------------------------------------------------------------


class TestProcessorProperties:
    def test_available_capacity_getter_setter(self):
        proc = make_processor(capacity=5)
        assert proc.available_capacity == 5
        proc.available_capacity = 3
        assert proc.available_capacity == 3

    def test_execution_mode_getter_setter(self):
        proc = make_processor()
        assert proc.execution_mode is False
        proc.execution_mode = True
        assert proc.execution_mode is True


# ---------------------------------------------------------------------------
# Processor enqueue / dequeue
# ---------------------------------------------------------------------------


class TestProcessorEnqueue:
    async def test_enqueue_basic(self):
        pile = make_pile()
        event = make_calling()
        pile.include(event)
        proc = SimpleProcessor(queue_capacity=10, capacity_refresh_time=1.0, pile=pile)
        await proc.enqueue(event.id)
        assert proc.queue.qsize() == 1

    async def test_enqueue_with_priority(self):
        pile = make_pile()
        event = make_calling()
        pile.include(event)
        proc = SimpleProcessor(queue_capacity=10, capacity_refresh_time=1.0, pile=pile)
        await proc.enqueue(event.id, priority=5.0)
        assert proc.queue.qsize() == 1

    async def test_enqueue_queue_full_raises(self):
        pile = make_pile()
        events = [make_calling() for _ in range(3)]
        for e in events:
            pile.include(e)
        proc = SimpleProcessor(
            queue_capacity=10, capacity_refresh_time=1.0, pile=pile, max_queue_size=2
        )
        await proc.enqueue(events[0].id)
        await proc.enqueue(events[1].id)
        with pytest.raises(QueueFullError):
            await proc.enqueue(events[2].id)

    async def test_enqueue_nan_priority_raises(self):
        pile = make_pile()
        event = make_calling()
        pile.include(event)
        proc = SimpleProcessor(queue_capacity=10, capacity_refresh_time=1.0, pile=pile)
        import math

        with pytest.raises(ValueError, match="finite"):
            await proc.enqueue(event.id, priority=float("nan"))

    async def test_enqueue_inf_priority_raises(self):
        pile = make_pile()
        event = make_calling()
        pile.include(event)
        proc = SimpleProcessor(queue_capacity=10, capacity_refresh_time=1.0, pile=pile)
        with pytest.raises(ValueError, match="finite"):
            await proc.enqueue(event.id, priority=float("inf"))

    async def test_dequeue_returns_event(self):
        pile = make_pile()
        event = make_calling()
        pile.include(event)
        proc = SimpleProcessor(queue_capacity=10, capacity_refresh_time=1.0, pile=pile)
        await proc.enqueue(event.id)
        dequeued = await proc.dequeue()
        assert dequeued.id == event.id


# ---------------------------------------------------------------------------
# Processor join / stop / start
# ---------------------------------------------------------------------------


class TestProcessorLifecycle:
    async def test_stop_sets_event(self):
        proc = make_processor()
        await proc.stop()
        assert proc.is_stopped()

    async def test_start_clears_stop_event(self):
        proc = make_processor()
        await proc.stop()
        assert proc.is_stopped()
        await proc.start()
        assert not proc.is_stopped()

    async def test_start_when_not_stopped_is_noop(self):
        proc = make_processor()
        assert not proc.is_stopped()
        await proc.start()
        assert not proc.is_stopped()

    async def test_join_empty_queue(self):
        proc = make_processor()
        # Should complete immediately since queue is empty
        await proc.join()

    async def test_stop_clears_denial_counts(self):
        proc = make_processor()
        from uuid import uuid4

        proc._denial_counts[uuid4()] = 2
        await proc.stop()
        assert len(proc._denial_counts) == 0


# ---------------------------------------------------------------------------
# Processor.create factory
# ---------------------------------------------------------------------------


class TestProcessorCreate:
    async def test_create_factory(self):
        pile = make_pile()
        proc = await SimpleProcessor.create(
            queue_capacity=5,
            capacity_refresh_time=0.5,
            pile=pile,
        )
        assert proc.queue_capacity == 5
        assert isinstance(proc, SimpleProcessor)

    async def test_create_with_executor(self):
        pile = make_pile()
        executor = make_executor()
        proc = await SimpleProcessor.create(
            queue_capacity=5,
            capacity_refresh_time=0.5,
            pile=pile,
            executor=executor,
        )
        assert proc.executor is executor


# ---------------------------------------------------------------------------
# Processor.process
# ---------------------------------------------------------------------------


class TestProcessorProcess:
    async def test_process_empty_queue(self):
        proc = make_processor()
        await proc.process()  # Should not raise

    async def test_process_resets_capacity(self):
        pile = make_pile()
        event = make_calling()
        pile.include(event)
        proc = SimpleProcessor(queue_capacity=5, capacity_refresh_time=1.0, pile=pile)
        await proc.enqueue(event.id)
        initial = proc.available_capacity
        await proc.process()
        # After processing an event, capacity resets to queue_capacity
        assert proc.available_capacity == proc.queue_capacity

    async def test_process_event_not_in_pile_skips(self):
        pile = make_pile()
        event = make_calling()
        # Do NOT add event to pile
        proc = SimpleProcessor(queue_capacity=5, capacity_refresh_time=1.0, pile=pile)
        # Manually put an unknown id into queue
        await proc.queue.put((1.0, event.id))
        await proc.process()  # Should not raise; silently skips missing event

    async def test_process_permission_denied_backoff(self):
        """When request_permission returns False, event gets backoff and re-queued."""
        pile = make_pile()
        event = make_calling()
        pile.include(event)

        class DenyingProcessor(SimpleProcessor):
            async def request_permission(self, **kwargs):
                return False

        proc = DenyingProcessor(queue_capacity=5, capacity_refresh_time=1.0, pile=pile)
        await proc.enqueue(event.id)
        # First denial: re-queued with backoff
        await proc.process()
        assert proc._denial_counts.get(event.id, 0) >= 1

    async def test_process_permission_denied_three_times_aborts(self):
        """After 3 denials, event is aborted."""
        from lionagi.protocols.generic.event import EventStatus

        pile = make_pile()
        event = make_calling()
        pile.include(event)

        class DenyingProcessor(SimpleProcessor):
            async def request_permission(self, **kwargs):
                return False

        proc = DenyingProcessor(queue_capacity=5, capacity_refresh_time=1.0, pile=pile)
        executor = make_executor()
        # We need an executor for _update_progression to work
        # Give proc a fake executor ref that tracks state
        proc.executor = executor
        await executor.start()

        # Pre-load the event into executor states
        executor.states.add_item(event, progressions="pending")

        # Force 3 denials by manually setting denial count to 2 and processing
        proc._denial_counts[event.id] = 2
        await proc.enqueue(event.id)
        await proc.process()
        # After 3rd denial, event should be in aborted progression
        aborted = executor.get_events_by_status(EventStatus.ABORTED)
        assert any(e.id == event.id for e in aborted)

    async def test_process_denial_tracking_overflow(self):
        """When denial tracking is at max, oldest entry is evicted."""
        pile = make_pile()
        event = make_calling()
        pile.include(event)

        class DenyingProcessor(SimpleProcessor):
            async def request_permission(self, **kwargs):
                return False

        proc = DenyingProcessor(
            queue_capacity=5,
            capacity_refresh_time=1.0,
            pile=pile,
            max_denial_tracking=1,
        )
        from uuid import uuid4

        old_key = uuid4()
        proc._denial_counts[old_key] = 1  # fill to max

        await proc.enqueue(event.id)
        await proc.process()
        # Old key should have been evicted
        assert old_key not in proc._denial_counts


# ---------------------------------------------------------------------------
# Processor request_permission (base)
# ---------------------------------------------------------------------------


class TestProcessorRequestPermission:
    async def test_request_permission_returns_true(self):
        proc = make_processor()
        result = await proc.request_permission()
        assert result is True


# ---------------------------------------------------------------------------
# Processor._with_semaphore
# ---------------------------------------------------------------------------


class TestProcessorWithSemaphore:
    async def test_with_semaphore_executes_coro(self):
        proc = make_processor()
        result_holder = []

        async def coro():
            result_holder.append(42)
            return 42

        await proc._with_semaphore(coro())
        assert result_holder == [42]


# ---------------------------------------------------------------------------
# Executor creation and properties
# ---------------------------------------------------------------------------


class TestExecutorCreation:
    def test_basic_creation(self):
        exc = make_executor()
        assert exc.processor is None
        assert exc.processor_config == {
            "queue_capacity": 10,
            "capacity_refresh_time": 1.0,
        }

    def test_event_type_property(self):
        exc = make_executor()
        assert exc.event_type is SimpleCalling

    def test_strict_event_type_default_false(self):
        exc = make_executor()
        assert exc.strict_event_type is False

    def test_strict_event_type_true(self):
        exc = SimpleExecutor(
            processor_config={"queue_capacity": 10, "capacity_refresh_time": 1.0},
            strict_event_type=True,
        )
        assert exc.strict_event_type is True

    def test_repr(self):
        exc = make_executor()
        r = repr(exc)
        assert "Executor" in r
        assert "total=" in r

    def test_contains_event(self):
        exc = make_executor()
        event = make_calling()
        assert event not in exc
        exc.states.add_item(event)
        assert event in exc

    def test_status_counts_all_zero(self):
        exc = make_executor()
        counts = exc.status_counts()
        assert all(v == 0 for v in counts.values())

    def test_inspect_state(self):
        exc = make_executor()
        state_str = exc.inspect_state()
        assert "Executor State" in state_str
        assert "pending" in state_str

    def test_name_default(self):
        exc = make_executor()
        assert exc.states.name == "executor_states"

    def test_name_custom(self):
        exc = SimpleExecutor(
            processor_config={"queue_capacity": 10, "capacity_refresh_time": 1.0},
            name="my-executor",
        )
        assert exc.states.name == "my-executor"


# ---------------------------------------------------------------------------
# Executor event status properties
# ---------------------------------------------------------------------------


class TestExecutorStatusProperties:
    def test_pending_events_empty(self):
        exc = make_executor()
        assert exc.pending_events == []

    def test_completed_events_empty(self):
        exc = make_executor()
        assert exc.completed_events == []

    def test_failed_events_empty(self):
        exc = make_executor()
        assert exc.failed_events == []

    def test_processing_events_empty(self):
        exc = make_executor()
        assert exc.processing_events == []

    async def test_append_adds_to_pending(self):
        exc = make_executor()
        event = make_calling()
        await exc.append(event)
        assert len(exc.pending_events) == 1
        assert exc.pending_events[0].id == event.id

    async def test_append_with_processor_enqueues(self):
        exc = make_executor()
        await exc.start()
        event = make_calling()
        await exc.append(event)
        # Event should be in pending
        assert len(exc.pending_events) == 1


# ---------------------------------------------------------------------------
# Executor _update_progression
# ---------------------------------------------------------------------------


class TestExecutorUpdateProgression:
    async def test_update_progression_moves_event(self):
        exc = make_executor()
        event = make_calling()
        # Add to pending
        await exc.append(event)
        # Move to completed
        await exc._update_progression(event, EventStatus.COMPLETED)
        assert len(exc.completed_events) == 1
        assert exc.completed_events[0].id == event.id

    async def test_update_progression_uses_event_status(self):
        exc = make_executor()
        event = make_calling()
        exc.states.add_item(event, progressions="pending")
        event.execution.status = EventStatus.FAILED
        await exc._update_progression(event)
        assert len(exc.failed_events) == 1

    async def test_update_progression_invalid_status_raises(self):
        exc = make_executor()
        event = make_calling()
        exc.states.add_item(event)
        # Use an invalid status by force-setting a mock
        mock_status = MagicMock()
        mock_status.value = "nonexistent_status"
        with pytest.raises(Exception):
            await exc._update_progression(event, force_status=mock_status)


# ---------------------------------------------------------------------------
# Executor start / stop / forward
# ---------------------------------------------------------------------------


class TestExecutorLifecycle:
    async def test_start_creates_processor(self):
        exc = make_executor()
        assert exc.processor is None
        await exc.start()
        assert exc.processor is not None

    async def test_start_twice_reuses_processor(self):
        exc = make_executor()
        await exc.start()
        proc_first = exc.processor
        await exc.start()
        assert exc.processor is proc_first  # same instance

    async def test_stop_stops_processor(self):
        exc = make_executor()
        await exc.start()
        await exc.stop()
        assert exc.processor.is_stopped()

    async def test_stop_without_start(self):
        exc = make_executor()
        # Should not raise even though processor is None
        await exc.stop()

    async def test_forward_without_processor(self):
        exc = make_executor()
        # Should not raise
        await exc.forward()

    async def test_forward_with_processor(self):
        exc = make_executor()
        await exc.start()
        # forward just calls process() which is safe on empty queue
        await exc.forward()


# ---------------------------------------------------------------------------
# Executor get_events_by_status
# ---------------------------------------------------------------------------


class TestExecutorGetEventsByStatus:
    def test_get_by_status_string(self):
        exc = make_executor()
        events = exc.get_events_by_status("pending")
        assert events == []

    def test_get_by_status_enum(self):
        exc = make_executor()
        events = exc.get_events_by_status(EventStatus.PENDING)
        assert events == []


# ---------------------------------------------------------------------------
# Executor cleanup_events
# ---------------------------------------------------------------------------


class TestExecutorCleanup:
    async def test_cleanup_removes_completed(self):
        exc = make_executor()
        event = make_calling()
        exc.states.add_item(event, progressions="pending")
        await exc._update_progression(event, EventStatus.COMPLETED)
        assert len(exc.completed_events) == 1
        removed = await exc.cleanup_events()
        assert removed == 1
        assert len(exc.completed_events) == 0

    async def test_cleanup_removes_failed(self):
        exc = make_executor()
        event = make_calling()
        exc.states.add_item(event, progressions="pending")
        await exc._update_progression(event, EventStatus.FAILED)
        removed = await exc.cleanup_events()
        assert removed >= 1

    async def test_cleanup_removes_aborted(self):
        exc = make_executor()
        event = make_calling()
        exc.states.add_item(event, progressions="pending")
        await exc._update_progression(event, EventStatus.ABORTED)
        removed = await exc.cleanup_events([EventStatus.ABORTED])
        assert removed == 1

    async def test_cleanup_custom_statuses(self):
        exc = make_executor()
        event = make_calling()
        exc.states.add_item(event, progressions="pending")
        await exc._update_progression(event, EventStatus.PROCESSING)
        removed = await exc.cleanup_events([EventStatus.PROCESSING])
        assert removed == 1

    async def test_cleanup_with_processor_clears_denial(self):
        exc = make_executor()
        await exc.start()
        event = make_calling()
        exc.states.add_item(event, progressions="pending")
        exc.processor._denial_counts[event.id] = 2
        await exc._update_progression(event, EventStatus.COMPLETED)
        await exc.cleanup_events([EventStatus.COMPLETED])
        assert event.id not in exc.processor._denial_counts

    async def test_cleanup_no_events_returns_zero(self):
        exc = make_executor()
        removed = await exc.cleanup_events()
        assert removed == 0


# ---------------------------------------------------------------------------
# Executor backfill on start
# ---------------------------------------------------------------------------


class TestExecutorBackfillOnStart:
    async def test_start_backfills_pending_events(self):
        exc = make_executor()
        event = make_calling()
        # Add event to pending before starting
        exc.states.add_item(event, progressions="pending")
        await exc.start()
        # After start, pending event should be enqueued in processor
        assert exc.processor.queue.qsize() == 1
