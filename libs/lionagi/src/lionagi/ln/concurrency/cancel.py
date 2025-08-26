"""Cancellation helpers for structured concurrency (anyio-backed)."""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager

import anyio

# Style note: expose CancelScope for user code that prefers explicit naming
CancelScope = anyio.CancelScope


@contextmanager
def fail_after(seconds: float | None) -> Iterator[CancelScope]:
    """Raise TimeoutError if the block takes longer than ``seconds``.

    If ``seconds`` is None, no timeout is applied and the block is shielded from
    outer cancellation (matches prior project semantics).
    """
    if seconds is None:
        with CancelScope(shield=True) as scope:
            yield scope
        return

    # anyio.move_on_after gives us `cancelled_caught` semantics inside the block
    with anyio.move_on_after(seconds) as scope:
        try:
            yield scope
        finally:
            if scope.cancelled_caught:
                raise TimeoutError(f"Operation took longer than {seconds} seconds")


@contextmanager
def move_on_after(seconds: float | None) -> Iterator[CancelScope]:
    """Cancel the inner work after ``seconds`` but do not raise.

    When the timeout triggers, code continues and ``scope.cancelled_caught`` is True.
    If ``seconds`` is None, no timeout is applied and the block is shielded from
    outer cancellation (consistent with fail_after(None)).
    """
    if seconds is None:
        with CancelScope(shield=True) as scope:
            yield scope
        return

    with anyio.move_on_after(seconds) as scope:
        yield scope
