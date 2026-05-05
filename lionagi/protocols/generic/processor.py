# Copyright (c) 2023-2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

import asyncio
import itertools
import math
from typing import TYPE_CHECKING, Any, ClassVar

import anyio

from lionagi._errors import QueueFullError, ValidationError
from lionagi.ln.concurrency import ConcurrencyEvent, Semaphore, create_task_group

from .._concepts import Observer
from .element import ID
from .event import Event, EventStatus
from .pile import Pile
from .progression import Progression

if TYPE_CHECKING:
    from uuid import UUID

__all__ = (
    "Processor",
    "Executor",
)


class Processor(Observer):
    """Manages a priority queue of events with capacity-limited, async processing.

    Subclass this to provide custom event handling logic or permission
    checks. The processor can enqueue events, handle them in batches, and
    respect a capacity limit that is refreshed periodically.

    Events are stored in a min-heap priority queue. Lower priority values
    are processed first. Default priority is ``event.created_at`` (FIFO by
    insertion time).

    Permission denials are tracked per event. After 3 denials the event is
    aborted (status → ABORTED) and removed from the queue.
    """

    event_type: ClassVar[type[Event]]

    def __init__(
        self,
        queue_capacity: int,
        capacity_refresh_time: float,
        concurrency_limit: int,
        max_queue_size: int = 0,
        max_denial_tracking: int = 10000,
    ) -> None:
        """Initializes a Processor instance.

        Args:
            queue_capacity (int):
                The maximum number of events processed in one batch.
            capacity_refresh_time (float):
                The time in seconds after which processing capacity is reset.
            concurrency_limit (int):
                Maximum concurrent event processing tasks.
            max_queue_size (int):
                Maximum queue size for backpressure. 0 means unlimited.
                When the queue is full, ``enqueue()`` raises ``QueueFullError``
                (non-blocking) or blocks (blocking put). ``try_enqueue``
                returns False.
            max_denial_tracking (int):
                Maximum number of distinct event IDs tracked for permission
                denials. When exceeded, the oldest entry is evicted (LRU-style).
                Defaults to 10000.

        Raises:
            ValidationError: If ``queue_capacity`` < 1, or
                ``capacity_refresh_time`` <= 0.
        """
        super().__init__()
        if queue_capacity < 1:
            raise ValidationError("Queue capacity must be greater than 0.")
        if capacity_refresh_time <= 0:
            raise ValidationError("Capacity refresh time must be larger than 0.")
        if max_denial_tracking < 1:
            raise ValidationError("max_denial_tracking must be >= 1.")

        self.queue_capacity = queue_capacity
        self.capacity_refresh_time = capacity_refresh_time
        self.max_queue_size = max_queue_size
        self.max_denial_tracking = max_denial_tracking
        # Priority queue: (priority, seq, event) tuples stored in asyncio.PriorityQueue.
        # Lower priority value = processed first. Default priority = event.created_at.
        # The sequence counter (seq) is a monotonically increasing tiebreaker so that
        # Event objects are never compared directly (Event does not support __lt__).
        self.queue: asyncio.PriorityQueue[tuple[float, int, Event]] = (
            asyncio.PriorityQueue(maxsize=max_queue_size)
        )
        self._enqueue_counter = itertools.count()
        self._available_capacity = queue_capacity
        self._execution_mode = False
        self._stop_event = ConcurrencyEvent()
        # Per-event permission denial tracking: {event_id: denial_count}
        self._denial_counts: dict[UUID, int] = {}
        if concurrency_limit:
            self._concurrency_sem = Semaphore(concurrency_limit)
        else:
            self._concurrency_sem = None

    @property
    def available_capacity(self) -> int:
        """int: The current capacity available for processing."""
        return self._available_capacity

    @available_capacity.setter
    def available_capacity(self, value: int) -> None:
        self._available_capacity = value

    @property
    def execution_mode(self) -> bool:
        """bool: Indicates if the processor is actively executing events."""
        return self._execution_mode

    @execution_mode.setter
    def execution_mode(self, value: bool) -> None:
        self._execution_mode = value

    async def enqueue(self, event: Event, priority: float | None = None) -> None:
        """Adds an event to the priority queue asynchronously.

        Blocks if the queue is full (backpressure) until space is available.

        Args:
            event (Event): The event to enqueue.
            priority (float | None):
                Sort key for min-heap ordering. Lower values are processed
                first. Defaults to ``event.created_at`` (FIFO by arrival
                time). Must be finite and not NaN.

        Raises:
            ValueError: If ``priority`` is NaN or infinite.
        """
        if priority is None:
            priority = float(event.created_at)
        if not math.isfinite(priority) or math.isnan(priority):
            raise ValueError(f"Priority must be finite and not NaN, got {priority}")
        seq = next(self._enqueue_counter)
        await self.queue.put((priority, seq, event))

    def try_enqueue(self, event: Event, priority: float | None = None) -> bool:
        """Non-blocking enqueue. Returns False if queue is full.

        Args:
            event (Event): The event to enqueue.
            priority (float | None):
                Sort key. Defaults to ``event.created_at``.

        Returns:
            True if enqueued successfully, False if the queue is full.

        Raises:
            QueueFullError: Never raised here — use the return value instead.
        """
        if priority is None:
            priority = float(event.created_at)
        seq = next(self._enqueue_counter)
        try:
            self.queue.put_nowait((priority, seq, event))
            return True
        except asyncio.QueueFull:
            return False

    @property
    def queue_full(self) -> bool:
        """True if the queue is at capacity (backpressure active)."""
        if self.max_queue_size == 0:
            return False
        return self.queue.qsize() >= self.max_queue_size

    async def dequeue(self) -> Event:
        """Retrieves the highest-priority event from the queue.

        Returns:
            Event: The event with the lowest priority value (min-heap ordering).
        """
        _, _seq, event = await self.queue.get()
        return event

    async def join(self) -> None:
        """Blocks until the queue is empty and all tasks are done."""
        await self.queue.join()

    async def stop(self) -> None:
        """Signals the processor to stop processing events and clears denial tracking."""
        self._stop_event.set()
        self._denial_counts.clear()

    async def start(self) -> None:
        """Clears the stop signal, allowing event processing to resume."""
        # Create a new event since ConcurrencyEvent doesn't have clear()
        if self._stop_event.is_set():
            self._stop_event = ConcurrencyEvent()

    def is_stopped(self) -> bool:
        """Checks whether the processor is in a stopped state.

        Returns:
            bool: True if the processor has been signaled to stop.
        """
        return self._stop_event.is_set()

    @classmethod
    async def create(cls, **kwargs: Any) -> "Processor":
        """Asynchronously constructs a new Processor instance.

        Args:
            **kwargs:
                Additional initialization arguments passed to the constructor.

        Returns:
            Processor: A newly instantiated processor.
        """
        return cls(**kwargs)

    async def process(self) -> None:
        """Dequeues and processes events up to the available capacity.

        Dequeues events from the priority queue (lowest value first), checks
        permissions, and executes them with semaphore-limited concurrency.

        Permission denials are tracked per event. After 3 consecutive denials
        the event's status is set to ABORTED and it is removed from the queue.
        On the first or second denial the event is re-enqueued with a backoff
        added to its priority, and batch processing stops for this cycle.

        Resets capacity after processing if any events were handled.
        """
        events_processed = 0

        async with create_task_group() as tg:
            while self.available_capacity > 0 and not self.queue.empty():
                priority, _seq, next_event = await self.queue.get()

                if await self.request_permission(**next_event.request):
                    # Clear any prior denial count on success.
                    self._denial_counts.pop(next_event.id, None)

                    if next_event.streaming:
                        # Consume async generator; catch exceptions to avoid
                        # aborting the TaskGroup — status recorded by Event.stream().
                        async def consume_stream(event):
                            try:
                                async for _ in event.stream():
                                    pass
                            except Exception:
                                pass  # Status already recorded by Event.stream()

                        if self._concurrency_sem:

                            async def stream_with_sem(event):
                                async with self._concurrency_sem:
                                    await consume_stream(event)

                            tg.start_soon(stream_with_sem, next_event)
                        else:
                            tg.start_soon(consume_stream, next_event)
                    else:
                        # Invoke non-streaming event; catch exceptions to avoid
                        # aborting the TaskGroup — status recorded by Event.invoke().
                        async def _invoke_safe(event):
                            try:
                                await event.invoke()
                            except Exception:
                                pass  # Status already recorded by Event.invoke()

                        if self._concurrency_sem:

                            async def invoke_with_sem(event):
                                async with self._concurrency_sem:
                                    await _invoke_safe(event)

                            tg.start_soon(invoke_with_sem, next_event)
                        else:
                            tg.start_soon(_invoke_safe, next_event)

                    events_processed += 1
                    self._available_capacity -= 1
                else:
                    # Permission denied: track denials and apply 3-strike abort.
                    # Evict oldest entry when tracking dict is full (LRU-style).
                    if len(self._denial_counts) >= self.max_denial_tracking:
                        oldest_key = next(iter(self._denial_counts))
                        self._denial_counts.pop(oldest_key)

                    denial_count = self._denial_counts.get(next_event.id, 0) + 1
                    self._denial_counts[next_event.id] = denial_count

                    if denial_count >= 3:
                        # Three strikes: abort the event.
                        next_event.execution.status = EventStatus.ABORTED
                        self._denial_counts.pop(next_event.id, None)
                    else:
                        # Back off and re-enqueue with increased priority value.
                        # A new sequence number preserves heap stability.
                        backoff = denial_count * 1.0
                        new_seq = next(self._enqueue_counter)
                        await self.queue.put((priority + backoff, new_seq, next_event))

                    # Stop this processing cycle after a denial.
                    break

        if events_processed > 0:
            self.available_capacity = self.queue_capacity

    async def request_permission(self, **kwargs: Any) -> bool:
        """Determines if an event may proceed.

        Override this method for custom checks (e.g., rate limits, user
        permissions).

        Args:
            **kwargs: Additional request parameters.

        Returns:
            bool: True if the event is allowed, False otherwise.
        """
        return True

    async def execute(self) -> None:
        """Continuously processes events until `stop()` is called.

        Respects the capacity refresh time between processing cycles.
        """
        self.execution_mode = True
        await self.start()

        while not self.is_stopped():
            await self.process()
            await anyio.sleep(self.capacity_refresh_time)

        self.execution_mode = False


class Executor(Observer):
    """Manages events via a Processor and stores them in a `Pile`.

    Subclass this to customize how events are forwarded or tracked.
    Typically, you configure an internal Processor, then add events to
    the Pile, which eventually are passed along to the Processor for
    execution.
    """

    processor_type: ClassVar[type[Processor]]

    def __init__(
        self,
        processor_config: dict[str, Any] | None = None,
        strict_event_type: bool = False,
    ) -> None:
        """Initializes the Executor.

        Args:
            processor_config (dict[str, Any] | None):
                Configuration parameters for creating the Processor.
            strict_event_type (bool):
                If True, the underlying Pile enforces exact type matching
                for Event objects.
        """
        self.processor_config = processor_config or {}
        self.pending = Progression()
        self.processor: Processor | None = None
        self.pile: Pile[Event] = Pile(
            item_type=self.processor_type.event_type,
            strict_type=strict_event_type,
        )

    @property
    def event_type(self) -> type[Event]:
        """type[Event]: The Event subclass handled by the processor."""
        return self.processor_type.event_type

    @property
    def strict_event_type(self) -> bool:
        """bool: Indicates if the Pile enforces exact event type matching."""
        return self.pile.strict_type

    async def forward(self) -> None:
        """Forwards all pending events from the pile to the processor.

        After all events are enqueued, it calls `processor.process()` for
        immediate handling.
        """
        while len(self.pending) > 0:
            id_ = self.pending.popleft()
            event = self.pile[id_]
            await self.processor.enqueue(event)

        await self.processor.process()

    async def start(self) -> None:
        """Initializes and starts the processor if it has not been created."""
        if not self.processor:
            await self._create_processor()
        await self.processor.start()

    async def stop(self) -> None:
        """Stops the processor if it exists."""
        if self.processor:
            await self.processor.stop()

    async def _create_processor(self) -> None:
        """Instantiates the processor using the stored config."""
        self.processor = await self.processor_type.create(**self.processor_config)

    async def append(self, event: Event) -> None:
        """Adds a new Event to the pile and marks it as pending.

        Args:
            event (Event): The event to add.
        """
        # Use async methods to avoid deadlock between sync/async locks
        await self.pile.ainclude(event)
        self.pending.include(event)

    @property
    def completed_events(self) -> Pile[Event]:
        """Pile[Event]: All events in COMPLETED status."""
        return Pile(
            collections=[e for e in self.pile if e.status == EventStatus.COMPLETED],
            item_type=self.processor_type.event_type,
            strict_type=self.strict_event_type,
        )

    @property
    def pending_events(self) -> Pile[Event]:
        """Pile[Event]: All events currently in PENDING status."""
        return Pile(
            collections=[e for e in self.pile if e.status == EventStatus.PENDING],
            item_type=self.processor_type.event_type,
            strict_type=self.strict_event_type,
        )

    @property
    def failed_events(self) -> Pile[Event]:
        """Pile[Event]: All events whose status is FAILED."""
        return Pile(
            collections=[e for e in self.pile if e.status == EventStatus.FAILED],
            item_type=self.processor_type.event_type,
            strict_type=self.strict_event_type,
        )

    @property
    def cancelled_events(self) -> Pile[Event]:
        """Pile[Event]: All events whose status is CANCELLED."""
        return Pile(
            collections=[e for e in self.pile if e.status == EventStatus.CANCELLED],
            item_type=self.processor_type.event_type,
            strict_type=self.strict_event_type,
        )

    @property
    def skipped_events(self) -> Pile[Event]:
        """Pile[Event]: All events whose status is SKIPPED."""
        return Pile(
            collections=[e for e in self.pile if e.status == EventStatus.SKIPPED],
            item_type=self.processor_type.event_type,
            strict_type=self.strict_event_type,
        )

    @property
    def aborted_events(self) -> Pile[Event]:
        """Pile[Event]: All events whose status is ABORTED (3-strike denial)."""
        return Pile(
            collections=[e for e in self.pile if e.status == EventStatus.ABORTED],
            item_type=self.processor_type.event_type,
            strict_type=self.strict_event_type,
        )

    def status_counts(self) -> dict[str, int]:
        """Return a count of events by status.

        Returns:
            dict mapping status value strings to counts.
        """
        counts: dict[str, int] = {}
        for event in self.pile:
            key = event.status.value
            counts[key] = counts.get(key, 0) + 1
        return counts

    def cleanup_events(self, statuses: list[EventStatus] | None = None) -> int:
        """Remove terminal events from the pile to free memory.

        Also clears denial tracking entries for removed events.

        Args:
            statuses: Event statuses to remove. Defaults to
                ``[COMPLETED, FAILED, ABORTED]``.

        Returns:
            Number of events removed.
        """
        if statuses is None:
            statuses = [
                EventStatus.COMPLETED,
                EventStatus.FAILED,
                EventStatus.ABORTED,
            ]
        target_ids = [e.id for e in self.pile if e.status in statuses]
        for eid in target_ids:
            self.pile.pop(eid)
            if self.processor:
                self.processor._denial_counts.pop(eid, None)
        return len(target_ids)

    def cleanup_completed(self) -> int:
        """Remove completed events from the pile to free memory.

        .. deprecated::
            Use :meth:`cleanup_events` instead, which also handles FAILED and
            ABORTED events and clears denial tracking.

        Returns:
            Number of events removed.
        """
        return self.cleanup_events(statuses=[EventStatus.COMPLETED])

    def inspect_state(self) -> dict:
        """Return a summary of executor state for debugging.

        Returns:
            dict with event counts, queue size, processor status.
        """
        return {
            "total_events": len(self.pile),
            "status_counts": self.status_counts(),
            "pending_queue": len(self.pending),
            "processor_running": (
                self.processor.execution_mode if self.processor else False
            ),
            "processor_stopped": (
                self.processor.is_stopped() if self.processor else True
            ),
        }

    def __contains__(self, ref: ID[Event].Ref) -> bool:
        """Checks if a given Event or ID reference is present in the pile.

        Args:
            ref (ID[Event].Ref):
                A reference to an Event (e.g., the Event object, its ID, etc.).

        Returns:
            bool: True if the referenced event is in the pile, False otherwise.
        """
        return ref in self.pile
