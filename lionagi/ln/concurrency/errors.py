"""Error/cancellation utilities with backend-agnostic behavior."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

import anyio

T = TypeVar("T")


__all__ = (
    "get_cancelled_exc_class",
    "is_cancelled",
    "shield",
)


def get_cancelled_exc_class() -> type[BaseException]:
    """Return the backend-native cancellation exception class."""
    return anyio.get_cancelled_exc_class()


def is_cancelled(exc: BaseException) -> bool:
    """True if this is the backend-native cancellation exception."""
    return isinstance(exc, anyio.get_cancelled_exc_class())


async def shield(
    func: Callable[..., Awaitable[T]], *args: Any, **kwargs: Any
) -> T:
    """Run ``func`` immune to outer cancellation."""
    with anyio.CancelScope(shield=True):
        return await func(*args, **kwargs)
