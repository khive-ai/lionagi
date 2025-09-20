from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, ClassVar

from lionagi.ln.concurrency.utils import is_coro_func
from lionagi.protocols.generic.event import Event

logger = logging.getLogger(__name__)

__all__ = ("Broadcaster",)


class Broadcaster:
    """Real-time event broadcasting system for hook events. Should subclass to implement specific event types."""

    _instance: ClassVar[Broadcaster | None] = None
    _subscribers: ClassVar[list[Callable[[Any], None]]] = []
    _event_type: ClassVar[type[Event]]

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def subscribe(cls, callback: Callable[[Any], None]) -> None:
        """Subscribe to hook events with sync callback."""
        if callback not in cls._subscribers:
            cls._subscribers.append(callback)

    @classmethod
    def unsubscribe(cls, callback: Callable[[Any], None]) -> None:
        """Unsubscribe from hook events."""
        if callback in cls._subscribers:
            cls._subscribers.remove(callback)

    @classmethod
    async def broadcast(cls, event) -> None:
        """Broadcast event to all subscribers."""
        if not isinstance(event, cls._event_type):
            raise ValueError(
                f"Event must be of type {cls._event_type.__name__}"
            )

        for callback in cls._subscribers:
            try:
                if is_coro_func(callback):
                    await callback(event)
                else:
                    callback(event)
            except Exception as e:
                logger.error(
                    f"Error in subscriber callback: {e}", exc_info=True
                )

    @classmethod
    def get_subscriber_count(cls) -> int:
        """Get total number of subscribers."""
        return len(cls._subscribers)
