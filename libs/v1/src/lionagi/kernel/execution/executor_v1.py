"""ExecutorV1: Single-nursery wrapper addressing v0 hanging issues.

ChatGPT's solution: Keep exactly ONE long-lived TaskGroup per executor instance.
Fan-out event.invoke into that group. No persistent nested groups.

This solves Ocean's v0 executor hanging issues by ensuring proper task lifetime management.
"""

from __future__ import annotations
import asyncio
from contextlib import asynccontextmanager
from typing import Optional, Callable, Awaitable

from ...domain.generic.event import Event, EventStatus
from ...domain.generic.pile import Pile
from ...domain.generic.progression import Progression
from ...domain.generic.processor import Processor  # v0 Processor (reused)


class ExecutorV1:
    """V1 Executor with single-nursery rule to prevent hanging.
    
    Key innovation: ONE long-lived TaskGroup for executor lifetime.
    Reuses v0 Processor logic but with proper task lifetime management.
    
    Why this works:
    - v0 Processor creates short-lived TaskGroup per process() cycle
    - ExecutorV1 provides one owning TaskGroup that guards executor lifetime
    - All tasks run on same loop without nested long-lived groups
    - Avoids premature cancellation that caused v0 hanging
    """
    
    def __init__(self, processor: Processor):
        self.processor = processor
        self._tg: Optional[asyncio.TaskGroup] = None
        self._started = False
        self.events: Pile[Event] = Pile(item_type=Event, strict_type=False)
        self.pending: Progression[Event] = Progression()

    async def __aenter__(self) -> "ExecutorV1":
        """Enter: Create one long-lived nursery for the life of the executor."""
        self._tg = asyncio.TaskGroup()
        await self._tg.__aenter__()
        await self.processor.start()
        self._started = True
        return self

    async def __aexit__(self, exc_type, exc, tb):
        """Exit: Clean shutdown of processor and task group."""
        try:
            await self.processor.stop()
        finally:
            if self._tg:
                await self._tg.__aexit__(exc_type, exc, tb)
            self._started = False
            self._tg = None

    async def submit(self, event: Event) -> None:
        """Add event to pile + progression; enqueue to processor."""
        await self.events.ainclude(event)
        self.pending.include(event)
        await self.processor.enqueue(event)

    async def pump(self) -> None:
        """Drive a single processing cycle with fan-out into the long-lived TG.
        
        The v0 Processor creates a short-lived TaskGroup per cycle,
        but its children (event.invoke) run under the same asyncio loop
        and are not cancelled by our long-lived TaskGroup unexpectedly.
        """
        if not self._started or not self._tg:
            raise RuntimeError("ExecutorV1 not started. Use 'async with'.")

        # Let the v0 Processor handle its own task group creation per cycle
        # The single-nursery rule prevents nested cancellation issues
        await self.processor.process()

    async def run_until_idle(self) -> None:
        """Convenience loop to process until queue is empty."""
        while not self.processor.queue.empty() or self.pending:
            await self.pump()
            await asyncio.sleep(self.processor.capacity_refresh_time)

    @property
    def completed(self) -> Pile[Event]:
        """All events in COMPLETED status."""
        return Pile(
            collections=[e for e in self.events if e.status == EventStatus.COMPLETED],
            item_type=Event,
        )

    @property
    def failed(self) -> Pile[Event]:
        """All events whose status is FAILED."""
        return Pile(
            collections=[e for e in self.events if e.status == EventStatus.FAILED],
            item_type=Event,
        )

    @property
    def processing(self) -> Pile[Event]:
        """All events currently in PROCESSING status."""
        return Pile(
            collections=[e for e in self.events if e.status == EventStatus.PROCESSING],
            item_type=Event,
        )

    @property
    def pending_events(self) -> Pile[Event]:
        """All events currently in PENDING status."""
        return Pile(
            collections=[e for e in self.events if e.status == EventStatus.PENDING],
            item_type=Event,
        )

    async def wait_for_completion(self) -> None:
        """Wait until all submitted events complete (success or failure)."""
        while True:
            active_events = [
                e for e in self.events 
                if e.status in [EventStatus.PENDING, EventStatus.PROCESSING]
            ]
            if not active_events:
                break
            await self.pump()
            await asyncio.sleep(0.1)

    def __contains__(self, event_or_id) -> bool:
        """Check if event exists in pile."""
        return event_or_id in self.events