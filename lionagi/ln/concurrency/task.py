"""Task group implementation for structured concurrency."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from types import TracebackType
from typing import Any, TypeVar

import anyio

T = TypeVar("T")
R = TypeVar("R")


class TaskGroup:
    """A group of tasks that are treated as a unit."""

    def __init__(self):
        """Initialize a new task group."""
        self._task_group = None

    async def start_soon(
        self,
        func: Callable[..., Awaitable[Any]],
        *args: Any,
        name: str | None = None,
    ) -> None:
        """Start a new task in this task group.

        Args:
            func: The coroutine function to call
            *args: Positional arguments to pass to the function
            name: Optional name for the task

        Note:
            This method does not wait for the task to initialize.
        """
        if self._task_group is None:
            raise RuntimeError("Task group is not active")
        self._task_group.start_soon(func, *args, name=name)

    async def start(
        self,
        func: Callable[..., Awaitable[R]],
        *args: Any,
        name: str | None = None,
    ) -> R:
        """Start a new task and wait for it to initialize.

        Args:
            func: The coroutine function to call
            *args: Positional arguments to pass to the function
            name: Optional name for the task

        Returns:
            The value passed to task_status.started()

        Note:
            The function must accept a task_status keyword argument and call
            task_status.started() once initialization is complete.
        """
        if self._task_group is None:
            raise RuntimeError("Task group is not active")
        return await self._task_group.start(func, *args, name=name)

    async def __aenter__(self) -> TaskGroup:
        """Enter the task group context.

        Returns:
            The task group instance.
        """
        task_group = anyio.create_task_group()
        self._task_group = await task_group.__aenter__()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool:
        """Exit the task group context.

        This will wait for all tasks in the group to complete.
        If any task raised an exception, it will be propagated.
        If multiple tasks raised exceptions, they will be combined into an ExceptionGroup.

        Returns:
            True if the exception was handled, False otherwise.
        """
        if self._task_group is None:
            return False

        try:
            return await self._task_group.__aexit__(exc_type, exc_val, exc_tb)
        finally:
            self._task_group = None


def create_task_group() -> TaskGroup:
    """Create a new task group.

    Returns:
        A new task group instance.

    Example:
        async with create_task_group() as tg:
            await tg.start_soon(task1)
            await tg.start_soon(task2)
    """
    return TaskGroup()
