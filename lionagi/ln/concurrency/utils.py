import asyncio
from collections.abc import Callable
from functools import lru_cache
from typing import Any

__all__ = ("is_coro_func",)


@lru_cache(maxsize=None)
def _is_coro_func(func: Callable[..., Any]) -> bool:
    # Check if it's a native coroutine function
    if asyncio.iscoroutinefunction(func):
        return True

    # FIX: Check if it's a callable object with async __call__ method
    call = getattr(func, "__call__", None)
    return call is not None and asyncio.iscoroutinefunction(call)


def is_coro_func(func: Callable[..., Any]) -> bool:
    """Check if a function is a coroutine function, with caching for performance."""
    return _is_coro_func(func)
