# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, ClassVar

from typing_extensions import Self

from lionagi._errors import (
    ConfigurationError,
    NotFoundError,
    QueueFullError,
    ValidationError,
)
from lionagi.ln import concurrency
from lionagi.protocols.generic.event import Event, EventStatus
from lionagi.protocols.generic.progression import Progression

from lionagi.protocols.generic.flow import Flow
from lionagi.protocols.generic.pile import Pile

if TYPE_CHECKING:
    from uuid import UUID


__all__ = (
    "Executor",
    "Processor",
)


class Processor:
    """Priority queue processor with rate limiting and concurrency control.

    Processes events from a priority queue (min-heap) with configurable:
    - Batch capacity: max events per processing cycle
    - Concurrency: semaphore-limited parallel execution
    - Permission: extensible request_permission() for rate limits

    Attributes:
        event_type: Event subclass this processor handles (ClassVar).
        queue_capacity: Max events per batch before refresh.
        capacity_refresh_time: Seconds between capacity resets.
        concurrency_limit: Max concurrent event executions.
        pile: Shared event storage (reference to executor's Flow.items).
        executor: Parent executor for progression updates.

    Example:
        >>> processor = await Processor.create(
        ...     queue_capacity=10, capacity_refresh_time=1.0, pile=pile
        ... )
        >>> await processor.enqueue(event.id)
        >>> await processor.execute()  # Runs until stop() called
    """

    event_type: ClassVar[type[Event]]

    def __init__(
        self,
        queue_capacity: int,
        capacity_refresh_time: float,
        pile: Pile[Event],
        executor: Executor | None = None,
        concurrency_limit: int = 100,
        max_queue_size: int = 1000,
        max_denial_tracking: int = 10000,
    ) -> None:
        """Initialize processor with validated capacity constraints.

        Args:
            queue_capacity: Events per batch (1-10000).
            capacity_refresh_time: Seconds between refreshes (0.01-3600).
            pile: Event storage reference.
            executor: Parent executor (optional).
            concurrency_limit: Max parallel executions (default: 100).
            max_queue_size: Queue size limit (default: 1000).
            max_denial_tracking: Max tracked permission denials (default: 10000).

        Raises:
            ValidationError: If parameters out of valid ranges.
        """
        if queue_capacity < 1:
            raise ValidationError("Queue capacity must be greater than 0.")
        if queue_capacity > 10000:
            raise ValidationError(
                "Queue capacity must be <= 10000 (prevent unbounded batches)."
            )

        # Validate capacity_refresh_time (prevent hot loop or starvation)
        if capacity_refresh_time < 0.01:
            raise ValidationError(
                "Capacity refresh time must be >= 0.01s (prevent CPU hot loop)."
            )
        if capacity_refresh_time > 3600:
            raise ValidationError(
                "Capacity refresh time must be <= 3600s (prevent starvation)."
            )

        # Validate concurrency_limit
        if concurrency_limit < 1:
            raise ValidationError("Concurrency limit must be >= 1.")

        # Validate max_queue_size
        if max_queue_size < 1:
            raise ValidationError("Max queue size must be >= 1.")

        if max_denial_tracking < 1:
            raise ValidationError("Max denial tracking must be >= 1.")

        self.queue_capacity = queue_capacity
        self.capacity_refresh_time = capacity_refresh_time
        self.max_queue_size = max_queue_size
        self.max_denial_tracking = max_denial_tracking
        self.pile = pile
        self.executor = executor
        self.concurrency_limit = concurrency_limit

        # Priority queue: (priority, event_uuid) tuples, min-heap ordering
        import asyncio

        self.queue: asyncio.PriorityQueue[tuple[float, UUID]] = asyncio.PriorityQueue()

        self._available_capacity = queue_capacity
        self._execution_mode = False
        self._stop_event = concurrency.ConcurrencyEvent()
        self._denial_counts: dict[UUID, int] = {}
        self._concurrency_sem = concurrency.Semaphore(concurrency_limit)

    @property
    def available_capacity(self) -> int:
        """Remaining capacity in current batch."""
        return self._available_capacity

    @available_capacity.setter
    def available_capacity(self, value: int) -> None:
        self._available_capacity = value

    @property
    def execution_mode(self) -> bool:
        """True if execute() loop is running."""
        return self._execution_mode

    @execution_mode.setter
    def execution_mode(self, value: bool) -> None:
        self._execution_mode = value

    async def enqueue(self, event_id: UUID, priority: float | None = None) -> None:
        """Add event to priority queue. Lower priority = processed first.

        Args:
            event_id: UUID of event (must exist in pile).
            priority: Sort key (default: event.created_at timestamp).

        Raises:
            QueueFullError: If queue at max_queue_size.
            ValueError: If priority is NaN or infinite.
        """
        if self.queue.qsize() >= self.max_queue_size:
            raise QueueFullError(
                f"Queue size ({self.queue.qsize()}) exceeds max ({self.max_queue_size})",
                details={
                    "queue_size": self.queue.qsize(),
                    "max_size": self.max_queue_size,
                },
            )

        if priority is None:
            event = self.pile[event_id]
            priority = float(event.created_at)

        if not math.isfinite(priority) or math.isnan(priority):
            raise ValueError(
                f"Priority must be finite and not NaN, got {priority}",
            )

        await self.queue.put((priority, event_id))

    async def dequeue(self) -> Event:
        """Remove and return highest-priority event (lowest priority value)."""
        _, event_id = await self.queue.get()
        return self.pile[event_id]

    async def join(self) -> None:
        """Block until queue is empty (polling at 100ms intervals)."""
        while not self.queue.empty():
            await concurrency.sleep(0.1)

    async def stop(self) -> None:
        """Signal stop and clear denial tracking."""
        self._stop_event.set()
        self._denial_counts.clear()

    async def start(self) -> None:
        """Clear stop signal to allow processing."""
        if self._stop_event.is_set():
            self._stop_event = concurrency.ConcurrencyEvent()

    def is_stopped(self) -> bool:
        """True if stop() was called."""
        return self._stop_event.is_set()

    @classmethod
    async def create(
        cls,
        queue_capacity: int,
        capacity_refresh_time: float,
        pile: Pile[Event],
        executor: Executor | None = None,
        concurrency_limit: int = 100,
        max_queue_size: int = 1000,
        max_denial_tracking: int = 10000,
    ) -> Self:
        """Async factory. Same args as __init__."""
        return cls(
            queue_capacity=queue_capacity,
            capacity_refresh_time=capacity_refresh_time,
            pile=pile,
            executor=executor,
            concurrency_limit=concurrency_limit,
            max_queue_size=max_queue_size,
            max_denial_tracking=max_denial_tracking,
        )

    async def process(self) -> None:
        """Process events up to available capacity in parallel.

        Dequeues events, checks permissions, and executes with semaphore-limited
        concurrency. Permission denials trigger retry with backoff (3 strikes = abort).
        Resets capacity after processing if any events were handled.
        """
        events_processed = 0

        async with concurrency.create_task_group() as tg:
            while self.available_capacity > 0 and not self.queue.empty():
                priority, event_id = await self.queue.get()

                try:
                    next_event = self.pile[event_id]
                except NotFoundError:
                    self._denial_counts.pop(event_id, None)
                    continue

                if await self.request_permission(**next_event.request):
                    self._denial_counts.pop(event_id, None)

                    if self.executor:
                        await self.executor._update_progression(
                            next_event, EventStatus.PROCESSING
                        )

                    if next_event.streaming:

                        async def consume_stream(event: Event):
                            try:
                                async for _ in event.stream():  # type: ignore[attr-defined]
                                    pass
                                if self.executor:
                                    await self.executor._update_progression(event)
                            except Exception:
                                if self.executor:
                                    await self.executor._update_progression(event)

                        tg.start_soon(self._with_semaphore, consume_stream(next_event))
                    else:

                        async def invoke_and_update(event):
                            try:
                                await event.invoke()
                            finally:
                                if self.executor:
                                    await self.executor._update_progression(event)

                        tg.start_soon(
                            self._with_semaphore, invoke_and_update(next_event)
                        )

                    events_processed += 1
                    self._available_capacity -= 1
                else:
                    # Permission denied: track and retry with backoff, abort after 3 denials
                    if len(self._denial_counts) >= self.max_denial_tracking:
                        oldest_key = next(iter(self._denial_counts))
                        self._denial_counts.pop(oldest_key)

                    denial_count = self._denial_counts.get(event_id, 0) + 1
                    self._denial_counts[event_id] = denial_count

                    if denial_count >= 3:
                        if self.executor:
                            await self.executor._update_progression(
                                next_event, EventStatus.ABORTED
                            )
                        self._denial_counts.pop(event_id, None)
                    else:
                        backoff = denial_count * 1.0
                        await self.queue.put((priority + backoff, next_event.id))

                    break

        if events_processed > 0:
            self.available_capacity = self.queue_capacity

    async def request_permission(self, **kwargs: Any) -> bool:
        """Override for rate limits, auth, quotas. Returns True by default."""
        return True

    async def _with_semaphore(self, coro):
        """Execute coroutine under concurrency semaphore."""
        if self._concurrency_sem:
            async with self._concurrency_sem:
                return await coro
        return await coro

    async def execute(self) -> None:
        """Run process() loop until stop() called. Sleeps capacity_refresh_time between batches."""
        self.execution_mode = True
        await self.start()

        while not self.is_stopped():
            await self.process()
            await concurrency.sleep(self.capacity_refresh_time)

        self.execution_mode = False


class Executor:
    """Event lifecycle manager with Flow-based state tracking.

    Uses Flow progressions (1:1 with EventStatus) for O(1) status queries.
    Delegates processing to Processor instance for async execution.

    Attributes:
        processor_type: Processor subclass (ClassVar, set by subclasses).
        states: Flow with progressions per EventStatus.
        processor: Background processor (created on start()).

    Example:
        >>> class MyExecutor(Executor):
        ...     processor_type = MyProcessor
        >>> exec = MyExecutor(processor_config={"queue_capacity": 10})
        >>> await exec.append(event)
        >>> await exec.start()
    """

    processor_type: ClassVar[type[Processor]]

    def __init__(
        self,
        processor_config: dict[str, Any] | None = None,
        strict_event_type: bool = False,
        name: str | None = None,
    ) -> None:
        """Initialize executor with Flow state tracking.

        Args:
            processor_config: Kwargs for Processor.create().
            strict_event_type: Enforce exact event type (no subclasses).
            name: Flow name (default: "executor_states").
        """
        self.processor_config = processor_config or {}
        self.processor: Processor | None = None

        self.states = Flow[Event, Progression](
            name=name or "executor_states",
            item_type=self.processor_type.event_type,
            strict_type=strict_event_type,
        )

        for status in EventStatus:
            self.states.add_progression(Progression(name=status.value))

    @property
    def event_type(self) -> type[Event]:
        """Event subclass this executor handles."""
        return self.processor_type.event_type

    @property
    def strict_event_type(self) -> bool:
        """True if Flow rejects event subclasses."""
        return self.states.items.strict_type

    async def _update_progression(
        self, event: Event, force_status: EventStatus | None = None
    ) -> None:
        """Move event to progression matching its status. Thread-safe."""
        target_status = force_status if force_status else event.execution.status

        async with self.states.progressions:
            for prog in self.states.progressions:
                if event.id in prog:
                    prog.remove(event.id)

            try:
                status_prog = self.states.get_progression(target_status.value)
                status_prog.append(event.id)
            except KeyError as e:
                raise ConfigurationError(
                    f"Progression '{target_status.value}' not found in executor",
                    details={
                        "status": target_status.value,
                        "available": [p.name for p in self.states.progressions],
                    },
                ) from e

    async def forward(self) -> None:
        """Trigger immediate process() without waiting for capacity refresh."""
        if self.processor:
            await self.processor.process()

    async def start(self) -> None:
        """Create processor (if needed), backfill pending events, and start."""
        if not self.processor:
            await self._create_processor()
            if self.processor:
                for event in self.pending_events:
                    await self.processor.enqueue(event.id)
        if self.processor:
            await self.processor.start()

    async def stop(self) -> None:
        """Stop processor."""
        if self.processor:
            await self.processor.stop()

    async def _create_processor(self) -> None:
        """Instantiate processor with stored config."""
        self.processor = await self.processor_type.create(
            pile=self.states.items,
            executor=self,
            **self.processor_config,
        )

    async def append(self, event: Event, priority: float | None = None) -> None:
        """Add event to Flow (pending) and enqueue if processor exists.

        Args:
            event: Event to add.
            priority: Queue priority (default: event.created_at).
        """
        self.states.add_item(event, progressions="pending")

        if self.processor:
            await self.processor.enqueue(event.id, priority=priority)

    def get_events_by_status(self, status: EventStatus | str) -> list[Event]:
        """Get events in given status progression. O(n) where n = events in status."""
        status_str = status.value if isinstance(status, EventStatus) else status
        prog = self.states.get_progression(status_str)
        return [self.states.items[uid] for uid in prog]

    @property
    def completed_events(self) -> list[Event]:
        """Events with COMPLETED status."""
        return self.get_events_by_status(EventStatus.COMPLETED)

    @property
    def pending_events(self) -> list[Event]:
        """Events with PENDING status."""
        return self.get_events_by_status(EventStatus.PENDING)

    @property
    def failed_events(self) -> list[Event]:
        """Events with FAILED status."""
        return self.get_events_by_status(EventStatus.FAILED)

    @property
    def processing_events(self) -> list[Event]:
        """Events with PROCESSING status."""
        return self.get_events_by_status(EventStatus.PROCESSING)

    def status_counts(self) -> dict[str, int]:
        """Event count per status progression."""
        return {prog.name or "unnamed": len(prog) for prog in self.states.progressions}

    async def cleanup_events(self, statuses: list[EventStatus] | None = None) -> int:
        """Remove terminal events and clear denial tracking.

        Args:
            statuses: Statuses to clean (default: COMPLETED, FAILED, ABORTED).

        Returns:
            Number of events removed.
        """
        if statuses is None:
            statuses = [EventStatus.COMPLETED, EventStatus.FAILED, EventStatus.ABORTED]

        removed_count = 0
        async with self.states.items, self.states.progressions:
            for status in statuses:
                events = self.get_events_by_status(status)
                for event in events:
                    if self.processor:
                        self.processor._denial_counts.pop(event.id, None)
                    self.states.remove_item(event.id)
                    removed_count += 1

        return removed_count

    def inspect_state(self) -> str:
        """Debug helper: multiline status summary."""
        lines = [f"Executor State ({self.states.name}):"]
        for status in EventStatus:
            count = len(self.states.get_progression(status.value))
            lines.append(f"  {status.value}: {count} events")
        return "\n".join(lines)

    def __contains__(self, event: Event | UUID) -> bool:
        return event in self.states.items

    def __repr__(self) -> str:
        counts = self.status_counts()
        total = sum(counts.values())
        return f"Executor(total={total}, {', '.join(f'{k}={v}' for k, v in counts.items())})"
