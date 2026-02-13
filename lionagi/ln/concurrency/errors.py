# Copyright (c) 2025-2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Error/cancellation utilities with backend-agnostic behavior."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import ParamSpec, TypeVar

import anyio

from ._compat import BaseExceptionGroup

T = TypeVar("T")
P = ParamSpec("P")

__all__ = (
    "get_cancelled_exc_class",
    "is_cancelled",
    "non_cancel_subgroup",
    "shield",
    "split_cancellation",
)


def get_cancelled_exc_class() -> type[BaseException]:
    """Return backend-specific cancellation exception type.

    Returns:
        asyncio.CancelledError for asyncio, trio.Cancelled for trio.
    """
    return anyio.get_cancelled_exc_class()


def is_cancelled(exc: BaseException) -> bool:
    """Check if exception is a backend cancellation.

    Args:
        exc: Exception to check.

    Returns:
        True if exc is the backend's cancellation exception type.
    """
    return isinstance(exc, anyio.get_cancelled_exc_class())


async def shield(func: Callable[P, Awaitable[T]], *args: P.args, **kwargs: P.kwargs) -> T:
    """Execute async function protected from outer cancellation.

    Args:
        func: Async callable to shield.
        *args: Positional arguments for func.
        **kwargs: Keyword arguments for func.

    Returns:
        Result of func(*args, **kwargs).

    Note:
        Use sparingly. Shielded code cannot be cancelled, which may
        delay shutdown. Prefer short critical sections only.
    """
    with anyio.CancelScope(shield=True):
        result = await func(*args, **kwargs)
    return result  # type: ignore[return-value]


def split_cancellation(
    eg: BaseExceptionGroup,
) -> tuple[BaseExceptionGroup | None, BaseExceptionGroup | None]:
    """Partition ExceptionGroup into cancellations and other errors.

    Args:
        eg: ExceptionGroup to split.

    Returns:
        Tuple of (cancellation_group, other_errors_group).
        Either may be None if no matching exceptions.
    """
    return eg.split(anyio.get_cancelled_exc_class())


def non_cancel_subgroup(eg: BaseExceptionGroup) -> BaseExceptionGroup | None:
    """Extract non-cancellation exceptions from ExceptionGroup.

    Args:
        eg: ExceptionGroup to filter.

    Returns:
        ExceptionGroup of non-cancellation errors, or None if all were cancellations.
    """
    _, rest = eg.split(anyio.get_cancelled_exc_class())
    return rest
