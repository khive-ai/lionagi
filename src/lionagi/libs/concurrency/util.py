# Copyright (c) 2023 - 2025, HaiyangLi <quantocean.li at gmail dot com>
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import functools
from concurrent.futures import ThreadPoolExecutor
from functools import cache
from typing import Any, Callable, TypeVar # Added Any

T = TypeVar("T")

__all__ = ("is_coroutine_function", "force_async", "as_async_fn")

@cache
def is_coroutine_function(fn: Callable[..., Any], /) -> bool:
    """Check if a function is a coroutine function."""
    return asyncio.iscoroutinefunction(fn)

def force_async(fn: Callable[..., T], /) -> Callable[..., asyncio.Future[T]]:
    """Force a synchronous function to be awaitable by running it in a thread pool."""
    # Consider managing the lifecycle of this ThreadPoolExecutor if many such functions are created
    # or if the application has specific executor needs. For a general utility, a new one per call is simple.
    pool = ThreadPoolExecutor() 

    @functools.wraps(fn)
    def wrapper(*args, **kwargs) -> asyncio.Future[T]:
        future = pool.submit(fn, *args, **kwargs)
        return asyncio.wrap_future(future)

    return wrapper

@cache
def as_async_fn(fn: Callable[..., Any], /) -> Callable[..., Any]: # The return is an awaitable
    """
    Ensures the returned function is awaitable.
    If fn is already a coroutine function, it's returned directly.
    Otherwise, it's wrapped by force_async to run in a thread.
    """
    if is_coroutine_function(fn):
        return fn
    return force_async(fn)