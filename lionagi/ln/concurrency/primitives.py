"""Resource management primitives for structured concurrency.

Pure async primitives focused on correctness and simplicity.
"""

import math
from types import TracebackType

import anyio

from .resource_tracker import track_resource, untrack_resource


class Lock:
    """A mutex lock for controlling access to a shared resource."""

    def __init__(self):
        """Initialize a new lock."""
        self._lock = anyio.Lock()
        self._acquired = False
        track_resource(self, f"Lock-{id(self)}", "Lock")

    def __del__(self):
        """Clean up resource tracking when lock is destroyed."""
        try:
            untrack_resource(self)
        except Exception:
            pass

    async def __aenter__(self) -> None:
        """Acquire the lock."""
        await self._lock.acquire()
        self._acquired = True

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Release the lock."""
        self._lock.release()
        self._acquired = False

    async def acquire(self) -> None:
        """Acquire the lock directly."""
        await self._lock.acquire()
        self._acquired = True

    def release(self) -> None:
        """Release the lock directly."""
        if not self._acquired:
            raise RuntimeError(
                "Attempted to release lock that was not acquired by this task"
            )
        self._lock.release()
        self._acquired = False


class Semaphore:
    """A semaphore preventing excessive releases."""

    def __init__(self, initial_value: int):
        """Initialize a new semaphore."""
        if initial_value < 0:
            raise ValueError("The initial value must be >= 0")
        self._initial_value = initial_value
        self._current_acquisitions = 0
        self._semaphore = anyio.Semaphore(initial_value)
        track_resource(self, f"Semaphore-{id(self)}", "Semaphore")

    def __del__(self):
        """Clean up resource tracking when semaphore is destroyed."""
        try:
            untrack_resource(self)
        except Exception:
            pass

    async def __aenter__(self) -> None:
        """Acquire the semaphore."""
        await self.acquire()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Release the semaphore."""
        self.release()

    async def acquire(self) -> None:
        """Acquire the semaphore."""
        await self._semaphore.acquire()
        self._current_acquisitions += 1

    def release(self) -> None:
        """Release the semaphore."""
        if self._current_acquisitions <= 0:
            raise RuntimeError(
                "Cannot release semaphore: no outstanding acquisitions"
            )
        self._semaphore.release()
        self._current_acquisitions -= 1

    @property
    def current_acquisitions(self) -> int:
        """Get the current number of outstanding acquisitions."""
        return self._current_acquisitions

    @property
    def initial_value(self) -> int:
        """Get the initial semaphore value."""
        return self._initial_value


class CapacityLimiter:
    """A context manager for limiting the number of concurrent operations."""

    def __init__(self, total_tokens: int | float):
        """Initialize a new capacity limiter."""
        if total_tokens == math.inf:
            processed_tokens = math.inf
        elif isinstance(total_tokens, (int, float)) and total_tokens >= 1:
            processed_tokens = (
                int(total_tokens) if total_tokens != math.inf else math.inf
            )
        else:
            raise ValueError(
                "The total number of tokens must be >= 1 (int or math.inf)"
            )

        self._limiter = anyio.CapacityLimiter(processed_tokens)
        self._borrower_counter = 0
        self._active_borrowers = {}
        track_resource(self, f"CapacityLimiter-{id(self)}", "CapacityLimiter")

    def __del__(self):
        """Clean up resource tracking when limiter is destroyed."""
        try:
            untrack_resource(self)
        except Exception:
            pass

    async def __aenter__(self) -> None:
        """Acquire a token."""
        await self.acquire()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Release the token."""
        self.release()

    async def acquire(self) -> None:
        """Acquire a token."""
        # Create a unique borrower identity for each acquisition
        self._borrower_counter += 1
        borrower = f"borrower-{self._borrower_counter}"
        await self._limiter.acquire_on_behalf_of(borrower)
        self._active_borrowers[borrower] = True

    def release(self) -> None:
        """Release a token."""
        # Find and release the first active borrower
        if not self._active_borrowers:
            raise RuntimeError("No tokens to release")

        borrower = next(iter(self._active_borrowers))
        self._limiter.release_on_behalf_of(borrower)
        del self._active_borrowers[borrower]

    @property
    def total_tokens(self) -> int | float:
        """The total number of tokens."""
        return float(self._limiter.total_tokens)

    @total_tokens.setter
    def total_tokens(self, value: int | float) -> None:
        """Set the total number of tokens."""
        if value == math.inf:
            processed_value = math.inf
        elif isinstance(value, (int, float)) and value >= 1:
            processed_value = int(value) if value != math.inf else math.inf
        else:
            raise ValueError(
                "The total number of tokens must be >= 1 (int or math.inf)"
            )

        current_borrowed = self._limiter.borrowed_tokens
        if processed_value != math.inf and processed_value < current_borrowed:
            raise ValueError(
                f"Cannot set total_tokens to {processed_value}: {current_borrowed} tokens "
                f"are currently borrowed. Wait for tokens to be released or "
                f"set total_tokens to at least {current_borrowed}."
            )

        self._limiter.total_tokens = processed_value

    @property
    def borrowed_tokens(self) -> int:
        """The number of tokens currently borrowed."""
        return self._limiter.borrowed_tokens

    @property
    def available_tokens(self) -> int | float:
        """The number of tokens currently available."""
        return self._limiter.available_tokens


class Event:
    """An event object for task synchronization."""

    def __init__(self):
        """Initialize a new event in the unset state."""
        self._event = anyio.Event()
        track_resource(self, f"Event-{id(self)}", "Event")

    def __del__(self):
        """Clean up resource tracking when event is destroyed."""
        try:
            untrack_resource(self)
        except Exception:
            pass

    def is_set(self) -> bool:
        """Check if the event is set."""
        return self._event.is_set()

    def set(self) -> None:
        """Set the event, allowing all waiting tasks to proceed."""
        self._event.set()

    async def wait(self) -> None:
        """Wait until the event is set."""
        await self._event.wait()


class Condition:
    """A condition variable for task synchronization."""

    def __init__(self, lock: Lock | None = None):
        """Initialize a new condition."""
        self._lock = lock or Lock()
        self._condition = anyio.Condition(self._lock._lock)
        track_resource(self, f"Condition-{id(self)}", "Condition")

    def __del__(self):
        """Clean up resource tracking when condition is destroyed."""
        try:
            untrack_resource(self)
        except Exception:
            pass

    async def __aenter__(self) -> "Condition":
        """Acquire the underlying lock."""
        await self._lock.__aenter__()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Release the underlying lock."""
        await self._lock.__aexit__(exc_type, exc_val, exc_tb)

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
