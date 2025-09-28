"""EventBus for service communication and coordination.

Adapted from v1 using lionagi's native concurrency patterns.
Enables loose coupling between IPU, services, and capability providers.
"""

from typing import Dict, List, Callable, Any, Optional
from dataclasses import dataclass, field
from uuid import uuid4

from lionagi.ln.concurrency import (
    create_task_group,
    gather,
    fail_after,
    Queue,
    Event as ConcurrencyEvent,
)


@dataclass
class Event:
    """Event payload for the event bus."""
    id: str = field(default_factory=lambda: uuid4().hex)
    name: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)
    source: Optional[str] = None


class EventBus:
    """Async event bus for service coordination.

    The event bus enables:
    - IPU to request services
    - Services to respond to requests
    - Morphisms to emit events
    - Loose coupling between components
    """

    def __init__(self):
        # Event name -> list of async handlers
        self._subscribers: Dict[str, List[Callable]] = {}
        # Request handlers for synchronous request/response
        self._request_handlers: Dict[str, Callable] = {}
        # Event history for debugging
        self._history: List[Event] = []
        self._max_history = 1000

    async def emit(
        self,
        event_name: str,
        payload: Dict[str, Any],
        source: Optional[str] = None
    ) -> None:
        """Emit an event to all subscribers.

        Args:
            event_name: Name of the event
            payload: Event data
            source: Optional source identifier
        """
        event = Event(name=event_name, payload=payload, source=source)

        # Store in history
        self._history.append(event)
        if len(self._history) > self._max_history:
            self._history.pop(0)

        # Call all subscribers using lionagi's gather
        handlers = self._subscribers.get(event_name, [])
        if handlers:
            # Run all handlers concurrently with exception handling
            await gather(
                *[handler(event) for handler in handlers],
                return_exceptions=True
            )

    def subscribe(self, event_name: str, handler: Callable) -> None:
        """Subscribe to an event.

        Args:
            event_name: Event to subscribe to
            handler: Async function to handle the event
        """
        if not asyncio.iscoroutinefunction(handler):
            raise ValueError(f"Handler must be async: {handler}")

        if event_name not in self._subscribers:
            self._subscribers[event_name] = []
        self._subscribers[event_name].append(handler)

    def unsubscribe(self, event_name: str, handler: Callable) -> None:
        """Unsubscribe from an event."""
        if event_name in self._subscribers:
            try:
                self._subscribers[event_name].remove(handler)
            except ValueError:
                pass

    async def request(
        self,
        service: str,
        payload: Dict[str, Any],
        timeout: float = 5.0
    ) -> Optional[Any]:
        """Request a service and wait for response.

        This is for synchronous request/response patterns.

        Args:
            service: Service identifier
            payload: Request payload
            timeout: Timeout in seconds

        Returns:
            Service response or None if no handler/timeout
        """
        handler = self._request_handlers.get(service)
        if not handler:
            # Emit event for async handlers
            await self.emit(f"{service}.request", payload)
            return None

        # Call handler with timeout
        try:
            result = await asyncio.wait_for(
                handler(payload),
                timeout=timeout
            )
            return result
        except asyncio.TimeoutError:
            await self.emit(
                f"{service}.timeout",
                {"payload": payload, "timeout": timeout}
            )
            return None

    def register_service(self, service: str, handler: Callable) -> None:
        """Register a service handler for request/response.

        Args:
            service: Service identifier
            handler: Async function to handle requests
        """
        if not asyncio.iscoroutinefunction(handler):
            raise ValueError(f"Handler must be async: {handler}")

        self._request_handlers[service] = handler

    def get_history(self, event_name: Optional[str] = None) -> List[Event]:
        """Get event history, optionally filtered by name."""
        if event_name:
            return [e for e in self._history if e.name == event_name]
        return self._history.copy()


# Helper functions for common events
async def emit_morphism_start(
    bus: EventBus,
    branch,
    morphism
) -> None:
    """Emit morphism start event."""
    await bus.emit(
        "morphism.start",
        {
            "branch_id": str(branch.id),
            "morphism_name": morphism.name,
            "requires": list(morphism.requires) if morphism.requires else [],
        },
        source="runtime"
    )


async def emit_morphism_finish(
    bus: EventBus,
    branch,
    morphism,
    result: Dict[str, Any]
) -> None:
    """Emit morphism finish event."""
    await bus.emit(
        "morphism.finish",
        {
            "branch_id": str(branch.id),
            "morphism_name": morphism.name,
            "result_keys": list(result.keys()),
        },
        source="runtime"
    )


async def emit_validation_request(
    bus: EventBus,
    field_name: str,
    value: Any,
    field_type: type = None
) -> Optional[Any]:
    """Request validation from validation service."""
    return await bus.request(
        "validation.execute",
        {
            "field_name": field_name,
            "value": value,
            "field_type": str(field_type) if field_type else None,
        }
    )