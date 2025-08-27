from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import Awaitable, Callable
from typing import Any

from .graph import OpNode
from .types import Branch

Handler = Callable[..., Awaitable[None]]


class EventBus:
    """Simple in-proc pub/sub. Subscribe is sync; emit is async."""

    def __init__(self):
        self._subs: dict[str, list[Handler]] = defaultdict(list)

    def subscribe(self, topic: str, handler: Handler) -> None:
        self._subs[topic].append(handler)

    async def emit(self, topic: str, *args: Any, **kw: Any) -> None:
        handlers = self._subs.get(topic, [])
        if not handlers:
            return
        await asyncio.gather(*(h(*args, **kw) for h in handlers))


async def emit_node_start(bus: EventBus, br: Branch, node: OpNode) -> None:
    await bus.emit("node.start", br, node)


async def emit_node_finish(
    bus: EventBus, br: Branch, node: OpNode, result: dict
) -> None:
    await bus.emit("node.finish", br, node, result)
