from __future__ import annotations

import weakref
from collections import defaultdict
from collections.abc import Awaitable, Callable
from typing import Any

from lionagi.ln.concurrency import gather

__all__ = ("EventBus", "Handler")

Handler = Callable[..., Awaitable[None]]


class EventBus:
    def __init__(self) -> None:
        self._subs: dict[str, list[weakref.ref[Handler]]] = defaultdict(list)

    def subscribe(self, topic: str, handler: Handler) -> None:
        self._subs[topic].append(weakref.ref(handler))

    def unsubscribe(self, topic: str, handler: Handler) -> bool:
        if topic not in self._subs:
            return False
        for weak_ref in list(self._subs[topic]):
            if weak_ref() is handler:
                self._subs[topic].remove(weak_ref)
                return True
        return False

    def _cleanup_dead_refs(self, topic: str) -> list[Handler]:
        handlers, alive_refs = [], []
        for weak_ref in self._subs[topic]:
            if (handler := weak_ref()) is not None:
                handlers.append(handler)
                alive_refs.append(weak_ref)
        self._subs[topic] = alive_refs
        return handlers

    async def emit(self, topic: str, *args: Any, **kwargs: Any) -> None:
        if topic not in self._subs:
            return
        if handlers := self._cleanup_dead_refs(topic):
            await gather(*(h(*args, **kwargs) for h in handlers), return_exceptions=True)

    def clear(self, topic: str | None = None) -> None:
        if topic is None:
            self._subs.clear()
        else:
            self._subs.pop(topic, None)

    def topics(self) -> list[str]:
        return list(self._subs.keys())

    def handler_count(self, topic: str) -> int:
        if topic not in self._subs:
            return 0
        return len(self._cleanup_dead_refs(topic))
