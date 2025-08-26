"""Error/cancellation utilities with backend-agnostic behavior."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

import anyio

T = TypeVar("T")


def get_cancelled_exc_class() -> type[BaseException]:
    """Return the backend-native cancellation exception class."""
    return anyio.get_cancelled_exc_class()


async def shield(func: Callable[..., Awaitable[T]], *args: Any, **kwargs: Any) -> T:
    """Run ``func`` immune to outer cancellation."""
    with anyio.CancelScope(shield=True):
        return await func(*args, **kwargs)
