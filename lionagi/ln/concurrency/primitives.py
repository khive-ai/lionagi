"""Core async primitives (thin wrappers over anyio)"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generic, TypeVar

import anyio
import anyio.abc

T = TypeVar("T")


__all__ = (
    "Lock",
    "Semaphore",
    "CapacityLimiter",
    "Queue",
    "Event",
    "Condition",
)


class Lock:
    """Async mutex lock (anyio.Lock wrapper).

    Provides mutual exclusion for async code. Use with async context manager
    for automatic acquisition/release.
    """

    __slots__ = ("_lock",)

    def __init__(self) -> None:
        self._lock = anyio.Lock()

    async def acquire(self) -> None:
        """Acquire the lock, blocking if necessary."""
        await self._lock.acquire()

    def release(self) -> None:
        """Release the lock."""
        self._lock.release()

    async def __aenter__(self) -> Lock:
        await self.acquire()
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.release()


class Semaphore:
    """Async semaphore (anyio.Semaphore wrapper).

    Limits concurrent access to a resource. Initialized with a count,
    decremented on acquire, incremented on release.
    """

    __slots__ = ("_sem",)

    def __init__(self, initial_value: int) -> None:
        if initial_value < 0:
            raise ValueError("initial_value must be >= 0")
        self._sem = anyio.Semaphore(initial_value)

    async def acquire(self) -> None:
        """Acquire a semaphore slot, blocking if none available."""
        await self._sem.acquire()

    def release(self) -> None:
        """Release a semaphore slot."""
        self._sem.release()

    async def __aenter__(self) -> Semaphore:
        await self.acquire()
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.release()


class CapacityLimiter:
    """Capacity limiter for controlling resource usage (anyio.CapacityLimiter wrapper).

    Controls concurrent access to limited resources like threads or connections.
    Key advantages over Semaphore:
    - Supports fractional tokens for fine-grained control
    - Allows dynamic capacity adjustment at runtime
    - Provides delegation methods for resource pooling
    """

    __slots__ = ("_lim",)

    def __init__(self, total_tokens: float) -> None:
        """Initialize with given capacity.

        Args:
            total_tokens: Maximum capacity (must be > 0).
                         Can be fractional for fine-grained control.
        """
        if total_tokens <= 0:
            raise ValueError("total_tokens must be > 0")
        self._lim = anyio.CapacityLimiter(total_tokens)

    async def acquire(self) -> None:
        """Acquire capacity, blocking if none available."""
        await self._lim.acquire()

    def release(self) -> None:
        """Release capacity."""
        self._lim.release()

    @property
    def remaining_tokens(self) -> float:
        """Current available capacity (deprecated, use available_tokens)."""
        return self._lim.available_tokens

    @property
    def total_tokens(self) -> float:
        """Get the current capacity limit."""
        return self._lim.total_tokens

    @total_tokens.setter
    def total_tokens(self, value: float) -> None:
        """Dynamically adjust the capacity limit.

        Args:
            value: New capacity (must be > 0).
                  Can be adjusted at runtime to adapt to load.
        """
        if value <= 0:
            raise ValueError("total_tokens must be > 0")
        self._lim.total_tokens = value

    @property
    def borrowed_tokens(self) -> float:
        """Get the number of currently borrowed tokens."""
        return self._lim.borrowed_tokens

    @property
    def available_tokens(self) -> float:
        """Get the number of currently available tokens."""
        return self._lim.available_tokens

    def acquire_on_behalf_of(self, borrower: object) -> None:
        """Synchronously acquire capacity on behalf of another object.

        For resource pooling where the acquirer differs from the releaser.

        Args:
            borrower: Object that will be responsible for releasing.
        """
        self._lim.acquire_on_behalf_of(borrower)

    def release_on_behalf_of(self, borrower: object) -> None:
        """Release capacity that was acquired on behalf of an object.

        Args:
            borrower: Object that previously acquired the capacity.
        """
        self._lim.release_on_behalf_of(borrower)

    # Support idiomatic AnyIO usage: `async with limiter: ...`
    async def __aenter__(self) -> CapacityLimiter:
        await self.acquire()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        self.release()


@dataclass(slots=True)
class Queue(Generic[T]):
    """Async queue using anyio memory object streams.

    Provides FIFO queue semantics with optional maxsize for backpressure.
    Must call close() or use async context manager for proper cleanup.

    Usage:
        queue = Queue.with_maxsize(100)
        await queue.put(item)
        item = await queue.get()
        await queue.close()
    """

    _send: anyio.abc.ObjectSendStream[T]
    _recv: anyio.abc.ObjectReceiveStream[T]

    @classmethod
    def with_maxsize(cls, maxsize: int) -> Queue[T]:
        """Create queue with maximum buffer size."""
        send, recv = anyio.create_memory_object_stream(maxsize)
        return cls(send, recv)

    async def put(self, item: T) -> None:
        """Put item into queue. May block if queue is full."""
        await self._send.send(item)

    def put_nowait(self, item: T) -> None:
        """Put item into queue without blocking.

        Args:
            item: Item to put in the queue.

        Raises:
            anyio.WouldBlock: If the queue is full.
        """
        self._send.send_nowait(item)

    async def get(self) -> T:
        """Get item from queue. Blocks until item available."""
        return await self._recv.receive()

    def get_nowait(self) -> T:
        """Get item from queue without blocking.

        Returns:
            Next item from the queue.

        Raises:
            anyio.WouldBlock: If the queue is empty.
        """
        return self._recv.receive_nowait()

    async def close(self) -> None:
        """Close both send and receive streams. Call this for cleanup."""
        await self._send.aclose()
        await self._recv.aclose()

    async def __aenter__(self) -> Queue[T]:
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        await self.close()

    @property
    def sender(self) -> anyio.abc.ObjectSendStream[T]:
        """Direct access to send stream for advanced usage."""
        return self._send

    @property
    def receiver(self) -> anyio.abc.ObjectReceiveStream[T]:
        """Direct access to receive stream for advanced usage."""
        return self._recv


class Event:
    """Async event for signaling between tasks (anyio.Event wrapper).

    An event object manages an internal flag that can be set to true
    with set() and reset to false with clear(). The wait() method blocks
    until the flag is true.
    """

    __slots__ = ("_event",)

    def __init__(self) -> None:
        self._event = anyio.Event()

    def set(self) -> None:
        """Set the internal flag to true, waking up all waiting tasks."""
        self._event.set()

    def is_set(self) -> bool:
        """Return True if the internal flag is set."""
        return self._event.is_set()

    async def wait(self) -> None:
        """Block until the internal flag becomes true."""
        await self._event.wait()

    def statistics(self) -> anyio.EventStatistics:
        """Return statistics about waiting tasks."""
        return self._event.statistics()


class Condition:
    """Async condition variable (anyio.Condition wrapper).

    A condition variable allows one or more coroutines to wait until
    they are notified by another coroutine. Must be used with a Lock.
    """

    __slots__ = ("_condition",)

    def __init__(self, lock: Lock | None = None) -> None:
        """Initialize with an optional lock.

        Args:
            lock: Lock to use. If None, creates a new Lock.
        """
        _lock = lock._lock if lock else None
        self._condition = anyio.Condition(_lock)

    async def acquire(self) -> None:
        """Acquire the underlying lock."""
        await self._condition.acquire()

    def release(self) -> None:
        """Release the underlying lock."""
        self._condition.release()

    async def __aenter__(self) -> Condition:
        await self.acquire()
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.release()

    async def wait(self) -> None:
        """Wait until notified.

        Releases the lock, blocks until notified, then re-acquires the lock.
        """
        await self._condition.wait()

    def notify(self, n: int = 1) -> None:
        """Wake up at most n tasks waiting on this condition.

        Args:
            n: Maximum number of tasks to wake (default: 1)
        """
        self._condition.notify(n)

    def notify_all(self) -> None:
        """Wake up all tasks waiting on this condition."""
        self._condition.notify_all()

    def statistics(self) -> anyio.abc.ConditionStatistics:
        """Return statistics about waiting tasks."""
        return self._condition.statistics()
