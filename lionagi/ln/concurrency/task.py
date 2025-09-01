"""Task group wrapper (thin facade over anyio.create_task_group)."""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any, TypeVar

import anyio

T = TypeVar("T")
R = TypeVar("R")

__all__ = (
    "TaskGroup",
    "create_task_group",
)


class TaskGroup:
    """Structured concurrency task group (anyio.abc.TaskGroup wrapper).

    Manages a group of concurrent tasks with structured lifecycle.
    If any task fails, all other tasks in the group are cancelled.

    Note: Lifecycle is managed by the create_task_group() context manager.
    Do not instantiate directly.

    Usage:
        async with create_task_group() as tg:
            tg.start_soon(worker_task, arg1)
            tg.start_soon(worker_task, arg2)
            # All tasks complete before exiting context
    """

    __slots__ = ("_tg",)

    def __init__(self, tg: anyio.abc.TaskGroup) -> None:
        self._tg = tg

    @property
    def cancel_scope(self) -> anyio.CancelScope:
        """Cancel scope controlling this task group's lifetime.

        Use this to cancel all tasks: tg.cancel_scope.cancel()
        """
        return self._tg.cancel_scope

    def start_soon(
        self,
        func: Callable[..., Awaitable[Any]],
        *args: Any,
        name: str | None = None,
    ) -> None:
        """Start a task without waiting for it to initialize.

        Args:
            func: Async function to run as a task
            *args: Arguments to pass to the function
            name: Optional name for the task (for debugging)
        """
        self._tg.start_soon(func, *args, name=name)

    async def start(
        self,
        func: Callable[..., Awaitable[R]],
        *args: Any,
        name: str | None = None,
    ) -> R:
        """Start a task and wait for it to initialize.

        The task function should use task_status.started() to signal initialization.

        Args:
            func: Async function to run as a task
            *args: Arguments to pass to the function
            name: Optional name for the task (for debugging)

        Returns:
            Value passed to task_status.started() by the task
        """
        return await self._tg.start(func, *args, name=name)


@asynccontextmanager
async def create_task_group() -> AsyncIterator[TaskGroup]:
    """Create a new task group for structured concurrency.

    Returns an async context manager that yields a TaskGroup.
    All tasks started within the group complete before the context exits.
    """
    async with anyio.create_task_group() as tg:
        yield TaskGroup(tg)
