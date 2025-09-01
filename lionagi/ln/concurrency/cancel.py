"""Cancellation helpers for structured concurrency (anyio-backed)."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import anyio

CancelScope = anyio.CancelScope
_INF = float("inf")


__all__ = (
    "CancelScope",
    "fail_after",
    "move_on_after",
    "fail_at",
    "move_on_at",
    "effective_deadline",
)


@contextmanager
def fail_after(seconds: float | None) -> Iterator[CancelScope]:
    """Create a context with a timeout that raises TimeoutError on expiry.

    Args:
        seconds: Timeout duration in seconds. None means no timeout
            (but still cancellable by outer scopes).

    Yields:
        CancelScope that can be cancelled after the timeout.

    Raises:
        TimeoutError: If the timeout expires before the block completes.
    """
    if seconds is None:
        # No timeout, but still cancellable by outer scopes
        with CancelScope() as scope:
            yield scope
        return
    with anyio.fail_after(seconds) as scope:
        yield scope


@contextmanager
def move_on_after(seconds: float | None) -> Iterator[CancelScope]:
    """Create a context with a timeout that silently cancels on expiry.

    Args:
        seconds: Timeout duration in seconds. None means no timeout
            (but still cancellable by outer scopes).

    Yields:
        CancelScope with cancelled_caught attribute to check if timeout occurred.
    """
    if seconds is None:
        # No timeout, but still cancellable by outer scopes
        with CancelScope() as scope:
            yield scope
        return
    with anyio.move_on_after(seconds) as scope:
        yield scope


@contextmanager
def fail_at(deadline: float | None) -> Iterator[CancelScope]:
    """Create a context that raises TimeoutError at an absolute deadline.

    Args:
        deadline: Absolute monotonic timestamp for timeout.
            None means no timeout (but still cancellable).

    Yields:
        CancelScope that expires at the specified deadline.

    Raises:
        TimeoutError: If the deadline is reached before the block completes.
    """
    if deadline is None:
        # No timeout, but still cancellable by outer scopes
        with CancelScope() as scope:
            yield scope
        return
    now = anyio.current_time()
    seconds = max(0.0, deadline - now)
    with fail_after(seconds) as scope:
        yield scope


@contextmanager
def move_on_at(deadline: float | None) -> Iterator[CancelScope]:
    """Create a context that silently cancels at an absolute deadline.

    Args:
        deadline: Absolute monotonic timestamp for timeout.
            None means no timeout (but still cancellable).

    Yields:
        CancelScope with cancelled_caught attribute to check if deadline was reached.
    """
    if deadline is None:
        # No timeout, but still cancellable by outer scopes
        with CancelScope() as scope:
            yield scope
        return
    now = anyio.current_time()
    seconds = max(0.0, deadline - now)
    with anyio.move_on_after(seconds) as scope:
        yield scope


def effective_deadline() -> float | None:
    """Return the ambient effective deadline, or None if unlimited."""
    d = anyio.current_effective_deadline()
    return None if d == _INF else d
