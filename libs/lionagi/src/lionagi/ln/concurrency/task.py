"""Task group wrapper (thin facade over anyio.create_task_group)."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, TypeVar

import anyio
import anyio.abc

T = TypeVar("T")
R = TypeVar("R")


class TaskGroup:
    """Minimal TaskGroup with stable surface."""

    __slots__ = ("_tg",)

    def __init__(self, tg: anyio.abc.TaskGroup) -> None:
        self._tg = tg

    @property
    def cancel_scope(self) -> anyio.CancelScope:
        return self._tg.cancel_scope

    def start_soon(
        self, func: Callable[..., Awaitable[Any]], *args: Any, name: str | None = None
    ) -> None:
        self._tg.start_soon(func, *args, name=name)

    async def start(
        self, func: Callable[..., Awaitable[R]], *args: Any, name: str | None = None
    ) -> R:
        return await self._tg.start(func, *args, name=name)

    async def cancel(self) -> None:
        self._tg.cancel_scope.cancel()

    async def __aenter__(self) -> "TaskGroup":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None


@asynccontextmanager
async def create_task_group() -> AsyncIterator[TaskGroup]:
    async with anyio.create_task_group() as tg:
        yield TaskGroup(tg)
