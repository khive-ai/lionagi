from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Awaitable, Callable
from typing import Any

from lionagi.ln.concurrency import gather, move_on_after

from .graph import OpNode
from .types import Branch

Handler = Callable[..., Awaitable[None]]
logger = logging.getLogger(__name__)

# Configurable timeout for event handlers to prevent indefinite blocking
EVENT_HANDLER_TIMEOUT_S = 2.0


class EventBus:
    """Robust in-proc pub/sub with timeout protection and fault isolation.

    Features:
    - Handlers run concurrently with individual timeout protection
    - Handler failures are isolated and logged, not propagated
    - Slow handlers are cancelled after timeout to prevent blocking
    """

    def __init__(self, handler_timeout: float = EVENT_HANDLER_TIMEOUT_S):
        self._subs: dict[str, list[Handler]] = defaultdict(list)
        self.handler_timeout = handler_timeout

    def subscribe(self, topic: str, handler: Handler) -> None:
        """Register a handler for a topic."""
        self._subs[topic].append(handler)

    def unsubscribe(self, topic: str, handler: Handler) -> None:
        """Remove a handler from a topic."""
        if topic in self._subs and handler in self._subs[topic]:
            self._subs[topic].remove(handler)

    async def emit(self, topic: str, *args: Any, **kw: Any) -> None:
        """Emit an event to all handlers for a topic.

        Handlers run concurrently with timeout protection.
        Handler failures are logged but don't affect other handlers.
        """
        handlers = self._subs.get(topic, [])
        if not handlers:
            return

        async def safe_handler(h: Handler) -> None:
            """Wrap handler with timeout and error handling."""
            handler_name = getattr(h, "__name__", repr(h))
            try:
                # Enforce timeout to prevent hanging handlers
                with move_on_after(self.handler_timeout) as scope:
                    await h(*args, **kw)

                if scope.cancelled_caught:
                    logger.warning(
                        f"Event handler '{handler_name}' for topic '{topic}' "
                        f"exceeded timeout ({self.handler_timeout}s)"
                    )
            except Exception as e:
                # Log but don't propagate - isolated failure
                logger.error(
                    f"Handler '{handler_name}' failed for topic '{topic}': {e}",
                    exc_info=True,
                )

        # Execute all handlers concurrently with failure isolation
        await gather(
            *(safe_handler(h) for h in handlers),
            return_exceptions=True,  # Prevent one failure from killing others
        )


async def emit_node_start(bus: EventBus, br: Branch, node: OpNode) -> None:
    await bus.emit("node.start", br, node)


async def emit_node_finish(bus: EventBus, br: Branch, node: OpNode, result: dict) -> None:
    await bus.emit("node.finish", br, node, result)
