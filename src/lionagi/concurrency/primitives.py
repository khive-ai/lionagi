"""Resource management primitives for structured concurrency."""

from types import TracebackType
from typing import Optional

import anyio


class Lock:
    """A mutex lock for controlling access to a shared resource.

    This lock is reentrant, meaning the same task can acquire it multiple times
    without deadlocking.
    """

    def __init__(self):
        """Initialize a new lock."""
        self._lock = anyio.Lock()

    async def __aenter__(self) -> None:
        """Acquire the lock.

        If the lock is already held by another task, this will wait until it's released.
        """
        await self.acquire()

    async def __aexit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        """Release the lock."""
        self.release()

    async def acquire(self) -> bool:
        """Acquire the lock.

        Returns:
            True if the lock was acquired, False otherwise.
        """
        await self._lock.acquire()
        return True

    def release(self) -> None:
        """Release the lock.

        Raises:
            RuntimeError: If the lock is not currently held by this task.
        """
        self._lock.release()


class Semaphore:
    """A semaphore for limiting concurrent access to a resource."""

    def __init__(self, initial_value: int):
        """Initialize a new semaphore.

        Args:
            initial_value: The initial value of the semaphore (must be >= 0)
        """
        if initial_value < 0:
            raise ValueError("The initial value must be >= 0")
        self._semaphore = anyio.Semaphore(initial_value)

    async def __aenter__(self) -> None:
        """Acquire the semaphore.

        If the semaphore value is zero, this will wait until it's released.
        """
        await self.acquire()

    async def __aexit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        """Release the semaphore."""
        self.release()

    async def acquire(self) -> None:
        """Acquire the semaphore.

        If the semaphore value is zero, this will wait until it's released.
        """
        await self._semaphore.acquire()

    def release(self) -> None:
        """Release the semaphore, incrementing its value."""
        self._semaphore.release()


class CapacityLimiter:
    """A context manager for limiting the number of concurrent operations."""

    def __init__(self, total_tokens: float):
        """Initialize a new capacity limiter.

        Args:
            total_tokens: The maximum number of tokens (>= 1)
        """
        if total_tokens < 1:
            raise ValueError("The total number of tokens must be >= 1")
        self._limiter = anyio.CapacityLimiter(total_tokens)

    async def __aenter__(self) -> None:
        """Acquire a token.

        If no tokens are available, this will wait until one is released.
        """
        await self.acquire()

    async def __aexit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        """Release the token."""
        self.release()

    async def acquire(self) -> None:
        """Acquire a token.

        If no tokens are available, this will wait until one is released.
        """
        await self._limiter.acquire()

    def release(self) -> None:
        """Release a token.

        Raises:
            RuntimeError: If the current task doesn't hold any tokens.
        """
        self._limiter.release()

    @property
    def total_tokens(self) -> float:
        """The total number of tokens."""
        return self._limiter.total_tokens

    @total_tokens.setter
    def total_tokens(self, value: float) -> None:
        """Set the total number of tokens.

        Args:
            value: The new total number of tokens (>= 1)
        """
        if value < 1:
            raise ValueError("The total number of tokens must be >= 1")
        self._limiter.total_tokens = value

    @property
    def borrowed_tokens(self) -> int:
        """The number of tokens currently borrowed."""
        return self._limiter.borrowed_tokens

    @property
    def available_tokens(self) -> float:
        """The number of tokens currently available."""
        return self._limiter.available_tokens


class Event:
    """An event object for task synchronization.

    An event can be in one of two states: set or unset. When set, tasks waiting
    on the event are allowed to proceed.
    """

    def __init__(self):
        """Initialize a new event in the unset state."""
        self._event = anyio.Event()

    def is_set(self) -> bool:
        """Check if the event is set.

        Returns:
            True if the event is set, False otherwise.
        """
        return self._event.is_set()

    def set(self) -> None:
        """Set the event, allowing all waiting tasks to proceed."""
        self._event.set()

    async def wait(self) -> None:
        """Wait until the event is set."""
        await self._event.wait()


class Condition:
    """A condition variable for task synchronization."""

    def __init__(self, lock: Optional[Lock] = None):
        """Initialize a new condition.

        Args:
            lock: The lock to use, or None to create a new one
        """
        self._lock = lock or Lock()
        self._condition = anyio.Condition(self._lock._lock)

    async def __aenter__(self) -> "Condition":
        """Acquire the underlying lock.

        Returns:
            The condition instance.
        """
        await self._lock.acquire()
        return self

    async def __aexit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        """Release the underlying lock."""
        self._lock.release()

    async def wait(self) -> None:
        """Wait for a notification.

        This releases the underlying lock, waits for a notification, and then
        reacquires the lock.
        """
        await self._condition.wait()

    async def notify(self, n: int = 1) -> None:
        """Notify waiting tasks.

        Args:
            n: The number of tasks to notify
        """
        await self._condition.notify(n)

    async def notify_all(self) -> None:
        """Notify all waiting tasks."""
        await self._condition.notify_all()
