from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Awaitable, Callable
from typing import Any

from ..ln import gather, move_on_after
from .graph import OpNode
from .types import Branch

Handler = Callable[..., Awaitable[None]]
logger = logging.getLogger(__name__)

# Configurable timeout for event handlers to prevent indefinite blocking
EVENT_HANDLER_TIMEOUT_S = 2.0


class EventBus:
    """Robust in-proc pub/sub with timeout protection and fault isolation.

    Implements structured concurrency patterns for reliable event handling.
    Uses structured concurrency task groups to guarantee cleanup and prevent resource leaks.

    References:
    - Structured concurrency: https://en.wikipedia.org/wiki/Structured_concurrency
    - Python Trio/AnyIO patterns: https://trio.readthedocs.io/

    Features:
    - Handlers run concurrently with individual timeout protection
    - Handler failures are isolated and logged, not propagated
    - Slow handlers are cancelled after timeout to prevent blocking
    - Observability counters for monitoring and debugging
    """

    def __init__(self, handler_timeout: float = EVENT_HANDLER_TIMEOUT_S):
        self._subs: dict[str, list[Handler]] = defaultdict(list)
        self.handler_timeout = handler_timeout
        # Observability counters
        self._stats: dict[str, dict[str, int]] = defaultdict(
            lambda: {"emitted": 0, "handled": 0, "timed_out": 0, "failed": 0}
        )

    def subscribe(self, topic: str, handler: Handler) -> None:
        """Register a handler for a topic."""
        self._subs[topic].append(handler)

    def unsubscribe(self, topic: str, handler: Handler) -> None:
        """Remove a handler from a topic (idempotent)."""
        if topic in self._subs and handler in self._subs[topic]:
            self._subs[topic].remove(handler)
            # Clean up empty topic entries
            if not self._subs[topic]:
                del self._subs[topic]

    def statistics(self, topic: str) -> dict[str, int]:
        """Get statistics for a topic."""
        return dict(self._stats[topic])

    async def emit(self, topic: str, *args: Any, timeout_s: float | None = None, **kw: Any) -> None:
        """Emit an event to all handlers for a topic.

        Handlers run concurrently with timeout protection.
        Handler failures are logged but don't affect other handlers.

        Args:
            topic: Event topic
            *args: Positional arguments passed to handlers
            timeout_s: Optional timeout override for this emit (uses handler_timeout if None)
            **kw: Keyword arguments passed to handlers
        """
        # Copy handler list to avoid surprises if subscribe/unsubscribe happens during emit
        handlers = list(self._subs.get(topic, []))

        # Use provided timeout or fall back to default
        timeout = timeout_s if timeout_s is not None else self.handler_timeout

        # Update emit counter
        self._stats[topic]["emitted"] += 1

        if not handlers:
            logger.debug(f"Emitting event to topic '{topic}' with no subscribers")
            return

        async def safe_handler(h: Handler) -> None:
            """Wrap handler with timeout and error handling."""
            handler_name = getattr(h, "__name__", repr(h))
            try:
                # Enforce timeout to prevent hanging handlers
                with move_on_after(timeout) as scope:
                    await h(*args, **kw)

                if scope.cancelled_caught:
                    # Update timeout counter
                    self._stats[topic]["timed_out"] += 1
                    logger.warning(
                        f"Event handler '{handler_name}' for topic '{topic}' "
                        f"exceeded timeout ({timeout}s)"
                    )
                else:
                    # Update success counter
                    self._stats[topic]["handled"] += 1

            except Exception as e:
                # Update failure counter
                self._stats[topic]["failed"] += 1
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
