from __future__ import annotations

import asyncio
import functools
from collections.abc import Callable
from time import sleep, time
from typing import Any, TypeVar

from typing_extensions import deprecated

T = TypeVar("T")


__all__ = ("Throttle",)


@deprecated("Throttle is deprecated and will be removed in a future release.")
class Throttle:
    """
    Provide a throttling mechanism for function calls.

    When used as a decorator, it ensures that the decorated function can only
    be called once per specified period. Subsequent calls within this period
    are delayed to enforce this constraint.

    Attributes:
        period: The minimum time period (in seconds) between successive calls.
    """

    def __init__(self, period: float) -> None:
        """
        Initialize a new instance of Throttle.

        Args:
            period: The minimum time period (in seconds) between
                successive calls.
        """
        self.period = period
        self.last_called = 0

    def __call__(self, func: Callable[..., T]) -> Callable[..., T]:
        """
        Decorate a synchronous function with the throttling mechanism.

        Args:
            func: The synchronous function to be throttled.

        Returns:
            The throttled synchronous function.
        """

        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            elapsed = time() - self.last_called
            if elapsed < self.period:
                sleep(self.period - elapsed)
            self.last_called = time()
            return func(*args, **kwargs)

        return wrapper

    def __call_async__(
        self, func: Callable[..., Callable[..., T]]
    ) -> Callable[..., Callable[..., T]]:
        """
        Decorate an asynchronous function with the throttling mechanism.

        Args:
            func: The asynchronous function to be throttled.

        Returns:
            The throttled asynchronous function.
        """

        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            elapsed = time() - self.last_called
            if elapsed < self.period:
                await asyncio.sleep(self.period - elapsed)
            self.last_called = time()
            return await func(*args, **kwargs)

        return wrapper
