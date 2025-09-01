"""Lion Async Concurrency Patterns - Structured concurrency coordination utilities.

This module provides async coordination patterns built on AnyIO's structured
concurrency primitives. All patterns are backend-neutral (asyncio/trio).

Key Features:
- gather: Concurrent execution with fail-fast or exception collection
- race: First-to-complete coordination
- bounded_map: Concurrent mapping with rate limiting
- CompletionStream: Stream results as they become available
- retry: Deadline-aware exponential backoff

Note on Structural Concurrency:
These patterns follow structured concurrency principles where possible. In
particular, CompletionStream provides an explicit lifecycle to avoid the
pitfalls of unstructured as_completed-like patterns when breaking early.
See individual function docstrings for details.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterable, Sequence
from typing import TypeVar

import anyio

from ._compat import ExceptionGroup
from .cancel import effective_deadline
from .errors import is_cancelled
from .primitives import CapacityLimiter
from .task import create_task_group

T = TypeVar("T")
R = TypeVar("R")


__all__ = (
    "gather",
    "race",
    "bounded_map",
    "CompletionStream",
    "retry",
)


async def gather(
    *aws: Awaitable[T], return_exceptions: bool = False
) -> list[T | BaseException]:
    """Run awaitables concurrently, return list of results.

    Args:
        *aws: Awaitables to execute concurrently
        return_exceptions: If True, exceptions are returned as results
                           If False, first exception cancels all tasks and re-raises

    Returns:
        List of results in same order as input awaitables
    """
    if not aws:
        return []

    results: list[T | BaseException | None] = [None] * len(aws)

    async def _runner(idx: int, aw: Awaitable[T]) -> None:
        try:
            results[idx] = await aw
        except BaseException as exc:
            results[idx] = exc
            if not return_exceptions:
                raise  # Propagate to the TaskGroup

    try:
        async with create_task_group() as tg:
            for i, aw in enumerate(aws):
                tg.start_soon(_runner, i, aw)
    except ExceptionGroup as eg:
        if not return_exceptions:
            # Find the first "real" exception and raise it.
            non_cancel_excs = [e for e in eg.exceptions if not is_cancelled(e)]
            if non_cancel_excs:
                raise non_cancel_excs[0]
            raise  # Re-raise group if all were cancellations

    return results  # type: ignore


async def race(*aws: Awaitable[T]) -> T:
    """Run awaitables concurrently, return result of first completion.

    Returns the first result to complete, whether success or failure.
    All other tasks are cancelled when first task completes.
    If first completion is an exception, it's re-raised.

    Note: This returns first *completion*, not first *success*.
    For first-success semantics, consider implementing a first_success variant.
    """
    if not aws:
        raise ValueError("race() requires at least one awaitable")
    send, recv = anyio.create_memory_object_stream(0)

    async def _runner(aw: Awaitable[T]) -> None:
        try:
            res = await aw
            await send.send((True, res))
        except BaseException as exc:
            await send.send((False, exc))

    async with send, recv, create_task_group() as tg:
        for aw in aws:
            tg.start_soon(_runner, aw)
        ok, payload = await recv.receive()
        tg.cancel_scope.cancel()

    # Raise outside the TaskGroup context to avoid ExceptionGroup wrapping
    if ok:
        return payload  # type: ignore[return-value]
    raise payload  # type: ignore[misc]


async def bounded_map(
    func: Callable[[T], Awaitable[R]],
    items: Iterable[T],
    *,
    limit: int,
    return_exceptions: bool = False,
) -> list[R | BaseException]:
    """Apply async function to items with concurrency limit.

    Args:
        func: Async function to apply to each item
        items: Items to process
        limit: Maximum concurrent operations
        return_exceptions: If True, exceptions are returned as results.
                           If False, first exception cancels all tasks and re-raises.

    Returns:
        List of results in same order as input items.
        If return_exceptions is True, exceptions are included in results.
    """
    if limit <= 0:
        raise ValueError("limit must be >= 1")

    seq = list(items)
    if not seq:
        return []

    out: list[R | BaseException | None] = [None] * len(seq)
    limiter = CapacityLimiter(limit)

    async def _runner(i: int, x: T) -> None:
        async with limiter:
            try:
                out[i] = await func(x)
            except BaseException as exc:
                out[i] = exc
                if not return_exceptions:
                    raise  # Propagate to the TaskGroup

    try:
        async with create_task_group() as tg:
            for i, x in enumerate(seq):
                tg.start_soon(_runner, i, x)
    except ExceptionGroup as eg:
        if not return_exceptions:
            non_cancel_excs = [e for e in eg.exceptions if not is_cancelled(e)]
            if non_cancel_excs:
                raise non_cancel_excs[0]
            raise

    return out  # type: ignore


class CompletionStream:
    """Structured-concurrency-safe completion stream with explicit lifecycle management.

    This provides a safer alternative to as_completed() that allows explicit cancellation
    of remaining tasks when early termination is needed.

    Usage:
        async with CompletionStream(awaitables, limit=10) as stream:
            async for index, result in stream:
                if some_condition:
                    break  # Remaining tasks are automatically cancelled
    """

    def __init__(
        self, aws: Sequence[Awaitable[T]], *, limit: int | None = None
    ):
        self.aws = aws
        self.limit = limit
        self._task_group = None
        self._send = None
        self._recv = None
        self._completed_count = 0
        self._total_count = len(aws)

    async def __aenter__(self):
        n = len(self.aws)
        self._send, self._recv = anyio.create_memory_object_stream(n)
        self._task_group = anyio.create_task_group()
        await self._task_group.__aenter__()

        limiter = CapacityLimiter(self.limit) if self.limit else None

        async def _runner(i: int, aw: Awaitable[T]) -> None:
            if limiter:
                await limiter.acquire()
            try:
                res = await aw
                await self._send.send((i, res))
            finally:
                if limiter:
                    limiter.release()

        # Start all tasks
        for i, aw in enumerate(self.aws):
            self._task_group.start_soon(_runner, i, aw)

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # Cancel remaining tasks and clean up
        if self._task_group:
            await self._task_group.__aexit__(exc_type, exc_val, exc_tb)
        if self._send:
            await self._send.aclose()
        if self._recv:
            await self._recv.aclose()
        return False

    def __aiter__(self):
        if not self._recv:
            raise RuntimeError(
                "CompletionStream must be used as async context manager"
            )
        return self

    async def __anext__(self):
        if self._completed_count >= self._total_count:
            raise StopAsyncIteration

        try:
            result = await self._recv.receive()
            self._completed_count += 1
            return result
        except anyio.EndOfStream:
            raise StopAsyncIteration


async def retry(
    fn: Callable[[], Awaitable[T]],
    *,
    attempts: int = 3,
    base_delay: float = 0.1,
    max_delay: float = 2.0,
    retry_on: tuple[type[BaseException], ...] = (Exception,),
    jitter: float = 0.1,
) -> T:
    """Deadline-aware exponential backoff retry.

    If an ambient effective deadline exists, cap each sleep so the retry loop
    never outlives its parent scope.

    Args:
        fn: Async function to retry (takes no args)
        attempts: Maximum retry attempts
        base_delay: Initial delay between retries
        max_delay: Maximum delay between retries
        retry_on: Exception types that trigger retry
        jitter: Random jitter added to delay (0.0 to 1.0)

    Returns:
        Result of successful function call

    Raises:
        Last exception if all attempts fail
    """
    attempt = 0
    deadline = effective_deadline()
    while True:
        try:
            return await fn()
        except retry_on as exc:
            attempt += 1
            if attempt >= attempts:
                raise

            delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
            if jitter:
                import random

                delay *= 1 + random.random() * jitter

            # Cap by ambient deadline if one exists
            if deadline is not None:
                remaining = deadline - anyio.current_time()
                if remaining <= 0:
                    # Out of time; surface the last error
                    raise
                delay = min(delay, remaining)

            await anyio.sleep(delay)
