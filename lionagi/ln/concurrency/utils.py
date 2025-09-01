import asyncio
from collections.abc import Callable
from functools import lru_cache
from typing import Any

__all__ = ("is_coro_func",)


@lru_cache(maxsize=None)
def _is_coro_func(func: Callable[..., Any]) -> bool:
    return asyncio.iscoroutinefunction(func)


def is_coro_func(func: Callable[..., Any]) -> bool:
    """Check if a function is a coroutine function, with caching for performance."""
    return _is_coro_func(func)
