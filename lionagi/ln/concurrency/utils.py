import asyncio
from functools import lru_cache
from typing import Any, Callable

__all__ = ("is_coro_func",)


@lru_cache(maxsize=None)
def _is_coro_func(func: Callable[..., Any]) -> bool:
    return asyncio.iscoroutinefunction(func)


def is_coro_func(func: Callable[..., Any]) -> bool:
    return _is_coro_func(func)
