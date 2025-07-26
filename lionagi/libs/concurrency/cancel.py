"""Cancellation scope implementation for structured concurrency."""

import time
from collections.abc import Iterator
from contextlib import contextmanager
from types import TracebackType
from typing import TypeVar

import anyio

T = TypeVar("T")


class CancelScope:
    """A context manager for controlling cancellation of tasks."""

    def __init__(self, deadline: float | None = None, shield: bool = False):
        """Initialize a new cancel scope.

        Args:
            deadline: The time (in seconds since the epoch) when this scope should be cancelled
            shield: If True, this scope shields its contents from external cancellation
        """
        self._scope = None
        self._deadline = deadline
        self._shield = shield
        self.cancel_called = False
        self.cancelled_caught = False

    def cancel(self) -> None:
        """Cancel this scope.

        This will cause all tasks within this scope to be cancelled.
        """
        self.cancel_called = True
        if self._scope is not None:
            self._scope.cancel()

    def __enter__(self) -> "CancelScope":
        """Enter the cancel scope context.

        Returns:
            The cancel scope instance.
        """
        # Use math.inf as the default deadline (no timeout)
        import math

        deadline = self._deadline if self._deadline is not None else math.inf
        self._scope = anyio.CancelScope(deadline=deadline, shield=self._shield)
        if self.cancel_called:
            self._scope.cancel()
        self._scope.__enter__()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool:
        """Exit the cancel scope context.

        Returns:
            True if the exception was handled, False otherwise.
        """
        if self._scope is None:
            return False

        try:
            result = self._scope.__exit__(exc_type, exc_val, exc_tb)
            self.cancelled_caught = self._scope.cancelled_caught
            return result
        finally:
            self._scope = None


@contextmanager
def move_on_after(seconds: float | None) -> Iterator[CancelScope]:
    """Return a context manager that cancels its contents after the given number of seconds.

    Args:
        seconds: The number of seconds to wait before cancelling, or None to disable the timeout

    Returns:
        A cancel scope that will be cancelled after the specified time

    Example:
        with move_on_after(5) as scope:
            await long_running_operation()
            if scope.cancelled_caught:
                print("Operation timed out")
    """
    deadline = None if seconds is None else time.time() + seconds
    scope = CancelScope(deadline=deadline)
    with scope:
        yield scope


@contextmanager
def fail_after(seconds: float | None) -> Iterator[CancelScope]:
    """Return a context manager that raises TimeoutError if its contents take longer than the given time.

    Args:
        seconds: The number of seconds to wait before raising TimeoutError, or None to disable the timeout

    Returns:
        A cancel scope that will raise TimeoutError after the specified time

    Raises:
        TimeoutError: If the operation takes longer than the specified time

    Example:
        try:
            with fail_after(5):
                await long_running_operation()
        except TimeoutError:
            print("Operation timed out")
    """
    if seconds is None:
        # No timeout
        scope = CancelScope(shield=True)
        with scope:
            yield scope
    else:
        deadline = time.time() + seconds
        scope = CancelScope(deadline=deadline)
        try:
            with scope:
                yield scope
        finally:
            if scope.cancelled_caught:
                raise TimeoutError(
                    f"Operation took longer than {seconds} seconds"
                )
