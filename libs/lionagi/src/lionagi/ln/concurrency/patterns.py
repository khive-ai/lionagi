"""Concurrency patterns built on anyio primitives."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterable, Sequence
from typing import AsyncIterator, TypeVar

import anyio

from .primitives import CapacityLimiter
from .task import create_task_group

T = TypeVar("T")
R = TypeVar("R")


async def gather(*aws: Awaitable[T], return_exceptions: bool = False) -> list[T | BaseException]:
    """Collect results in call order. Cancel peers on first error unless returning exceptions."""
    results: list[T | BaseException] = [None] * len(aws)  # type: ignore[list-item]
    first_exc: BaseException | None = None

    async def _runner(idx: int, aw: Awaitable[T]) -> None:
        nonlocal first_exc
        try:
            results[idx] = await aw
        except BaseException as exc:
            results[idx] = exc
            if not return_exceptions and first_exc is None:
                first_exc = exc

    async with create_task_group() as tg:
        for i, aw in enumerate(aws):
            tg.start_soon(_runner, i, aw)
        if not return_exceptions:

            async def _watch() -> None:
                nonlocal first_exc
                while first_exc is None:
                    await anyio.sleep(0)
                tg.cancel_scope.cancel()

            tg.start_soon(_watch)

    if first_exc is not None and not return_exceptions:
        raise first_exc
    return results


async def race(*aws: Awaitable[T]) -> T:
    """Return the first successful result; cancel the rest."""
    send, recv = anyio.create_memory_object_stream(0)

    async def _runner(aw: Awaitable[T]) -> None:
        res = await aw
        await send.send(res)

    async with send, recv, create_task_group() as tg:
        for aw in aws:
            tg.start_soon(_runner, aw)
        winner = await recv.receive()
        tg.cancel_scope.cancel()
        return winner


async def bounded_map(
    func: Callable[[T], Awaitable[R]], items: Iterable[T], *, limit: int
) -> list[R]:
    """Apply func over items with up to ``limit`` concurrent tasks. Preserve order."""
    if limit <= 0:
        raise ValueError("limit must be >= 1")

    seq = list(items)
    out: list[R] = [None] * len(seq)  # type: ignore[list-item]
    limiter = CapacityLimiter(limit)

    async def _runner(i: int, x: T) -> None:
        async with anyio.create_task_group() as inner:
            # Use limiter as async context by acquiring/releasing
            await limiter.acquire()
            try:
                out[i] = await func(x)
            finally:
                limiter.release()

    async with create_task_group() as tg:
        for i, x in enumerate(seq):
            tg.start_soon(_runner, i, x)

    return out


async def as_completed(
    aws: Sequence[Awaitable[T]], *, limit: int | None = None
) -> AsyncIterator[tuple[int, T]]:
    """Yield (index, result) pairs in completion order. Optional concurrency limit."""
    n = len(aws)
    send, recv = anyio.create_memory_object_stream(0)
    limiter = CapacityLimiter(limit) if limit else None

    async def _runner(i: int, aw: Awaitable[T]) -> None:
        if limiter:
            await limiter.acquire()
        try:
            res = await aw
            await send.send((i, res))
        finally:
            if limiter:
                limiter.release()

    async with send, recv, create_task_group() as tg:
        for i, aw in enumerate(aws):
            tg.start_soon(_runner, i, aw)
        for _ in range(n):
            yield await recv.receive()


async def retry(
    fn: Callable[[], Awaitable[T]],
    *,
    attempts: int = 3,
    base_delay: float = 0.1,
    max_delay: float = 2.0,
    retry_on: tuple[type[BaseException], ...] = (Exception,),
    jitter: float = 0.1,
) -> T:
    """Exponential backoff retry with jitter."""
    attempt = 0
    while True:
        try:
            return await fn()
        except retry_on:
            attempt += 1
            if attempt >= attempts:
                raise
            delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
            if jitter:
                import random

                delay += random.random() * jitter
            await anyio.sleep(delay)
