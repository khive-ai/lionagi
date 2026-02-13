# Copyright (c) 2023-2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import logging
import weakref
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

__all__ = ("Broadcaster",)

logger = logging.getLogger(__name__)


class Broadcaster:
    """Singleton pub/sub with weakref-based automatic subscriber cleanup.

    Subclass and set ``_event_type`` to define typed broadcasters.
    Subscribers are stored as weakrefs (WeakMethod for bound methods)
    so they are automatically cleaned up when the referenced object
    is garbage collected.

    Example::

        class OrderBroadcaster(Broadcaster):
            _event_type = OrderEvent

        OrderBroadcaster.subscribe(my_handler)
        await OrderBroadcaster.broadcast(OrderEvent(...))
    """

    _instance: ClassVar[Broadcaster | None] = None
    _subscribers: ClassVar[
        list[weakref.ref[Callable[[Any], None] | Callable[[Any], Awaitable[None]]]]
    ] = []
    _event_type: ClassVar[type]

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Each subclass gets its own subscriber list and singleton slot."""
        super().__init_subclass__(**kwargs)
        cls._instance = None
        cls._subscribers = []

    @classmethod
    def subscribe(cls, callback: Callable[[Any], None] | Callable[[Any], Awaitable[None]]) -> None:
        """Add subscriber callback (idempotent, stored as weakref).

        Args:
            callback: Sync or async callable receiving the event.
                      Bound methods use WeakMethod; functions use weakref.
        """
        for weak_ref in cls._subscribers:
            if weak_ref() is callback:
                return
        if hasattr(callback, "__self__"):
            weak_callback = weakref.WeakMethod(callback)
        else:
            weak_callback = weakref.ref(callback)
        cls._subscribers.append(weak_callback)

    @classmethod
    def unsubscribe(
        cls, callback: Callable[[Any], None] | Callable[[Any], Awaitable[None]]
    ) -> None:
        """Remove subscriber callback.

        Args:
            callback: Previously subscribed callback to remove.
        """
        for weak_ref in list(cls._subscribers):
            if weak_ref() is callback:
                cls._subscribers.remove(weak_ref)
                return

    @classmethod
    def _cleanup_dead_refs(
        cls,
    ) -> list[Callable[[Any], None] | Callable[[Any], Awaitable[None]]]:
        """Prune dead weakrefs, return live callbacks."""
        callbacks, alive_refs = [], []
        for weak_ref in cls._subscribers:
            if (cb := weak_ref()) is not None:
                callbacks.append(cb)
                alive_refs.append(weak_ref)
        cls._subscribers[:] = alive_refs
        return callbacks

    @classmethod
    async def broadcast(cls, event: Any) -> None:
        """Broadcast event to all subscribers sequentially.

        Args:
            event: Event instance (must match _event_type).

        Raises:
            ValueError: If event type doesn't match _event_type.

        Note:
            Callback exceptions are logged and suppressed to prevent
            one failing subscriber from blocking others.
        """
        if not isinstance(event, cls._event_type):
            raise ValueError(f"Event must be of type {cls._event_type.__name__}")
        for callback in cls._cleanup_dead_refs():
            try:
                result = callback(event)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error(f"Error in subscriber callback: {e}", exc_info=True)

    @classmethod
    def get_subscriber_count(cls) -> int:
        """Count live subscribers (triggers dead ref cleanup)."""
        return len(cls._cleanup_dead_refs())


# File: lionagi/protocols/generic/broadcaster.py
