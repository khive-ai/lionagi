# Copyright (c) 2023 - 2025, HaiyangLi <quantocean.li at gmail dot com>
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import functools
import time as std_time # Standard library time
from typing import Any, Callable, TypeVar

# Import from the util module in the same package
from .util import is_coro_func, force_async

T = TypeVar("T")

__all__ = ("Throttle", "throttle", "max_concurrent")

class Throttle:
    """
    Provides a throttling mechanism for function calls.
    Ensures that the decorated function can only be called once per specified period.
    """
    def __init__(self, period: float) -> None:
        self.period = period
        self.last_called_sync: float = 0.0
        self.last_called_async: float = 0.0

    def __call__(self, func: Callable[..., T]) -> Callable[..., T]: # For synchronous functions
        """Decorate a synchronous function with the throttling mechanism."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            current_time = std_time.time()
            elapsed = current_time - self.last_called_sync
            if elapsed < self.period:
                std_time.sleep(self.period - elapsed)
            self.last_called_sync = std_time.time()
            return func(*args, **kwargs)
        return wrapper

    async def call_async_throttled(self, func: Callable[..., Any], *args, **kwargs) -> Any:
        """Helper to call an async function with throttling."""
        try:
            current_time = asyncio.get_event_loop().time()
        except RuntimeError: 
            current_time = std_time.time()

        elapsed = current_time - self.last_called_async
        if elapsed < self.period:
            await asyncio.sleep(self.period - elapsed)
        
        try:
            self.last_called_async = asyncio.get_event_loop().time()
        except RuntimeError:
            self.last_called_async = std_time.time()
            
        return await func(*args, **kwargs)

def throttle(func: Callable[..., Any], period: float) -> Callable[..., Any]:
    """
    Throttle function execution to limit the rate of calls.
    Works for both synchronous and asynchronous functions.
    """
    throttle_instance = Throttle(period)
    
    if is_coro_func(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            return await throttle_instance.call_async_throttled(func, *args, **kwargs)
        return async_wrapper
    else:
        return throttle_instance(func)

def max_concurrent(func: Callable[..., Any], limit: int) -> Callable[..., Any]:
    """
    Limit the concurrency of async function execution using a semaphore.
    If the function is synchronous, it will be wrapped to run in a thread pool.
    """
    processed_func = func
    if not is_coro_func(processed_func):
        processed_func = force_async(processed_func)

    semaphore = asyncio.Semaphore(limit)
    @functools.wraps(processed_func) 
    async def wrapper(*args, **kwargs) -> Any:
        async with semaphore:
            return await processed_func(*args, **kwargs)
    return wrapper