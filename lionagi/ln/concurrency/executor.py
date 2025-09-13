"""AsyncExecutor - Clean async execution using existing concurrency patterns."""

from __future__ import annotations

import functools
import warnings
from collections.abc import Awaitable, Callable, Iterable, Sequence
from typing import Any, TypeVar

import anyio
import anyio.to_thread

from .patterns import CompletionStream, bounded_map, retry
from .utils import is_coro_func

T = TypeVar("T")
R = TypeVar("R")

__all__ = ("AsyncExecutor",)


class AsyncExecutor:
    """Clean async execution context with configurable concurrency and retry.

    Modern replacement for alcall/bcall patterns with clean separation of concerns.
    Supports single-function mapping, multi-function execution, streaming, and
    robust retry logic with exponential backoff and jitter.

    Args:
        max_concurrent: Maximum concurrent operations (None = unlimited)
        retry_attempts: Number of retry attempts (0 = no retries)
        retry_delay: Initial delay between retries (exponential backoff)
        retry_max_delay: Maximum delay between retries
        retry_jitter: Random jitter factor (0.0 to 1.0)
        retry_on: Exception types that trigger retry
        throttle_period: Optional delay between completions

    Usage:
        Basic single-function execution (alcall replacement):
        ```python
        async with AsyncExecutor(max_concurrent=10, retry_attempts=3) as executor:
            results = await executor(process_func, data)
        ```

        Multi-function execution (mcall replacement):
        ```python
        funcs = [func1, func2, func3]
        args_kwargs = [((arg1,), {}), ((arg2,), {"param": value}), ((arg3,), {})]
        async with AsyncExecutor(max_concurrent=5) as executor:
            results = await executor(funcs, args_kwargs)
        ```

        Streaming batches (bcall replacement):
        ```python
        async with AsyncExecutor(max_concurrent=20) as executor:
            async for batch in executor.stream(process_func, data, batch_size=50):
                handle_batch(batch)
        ```

        Advanced retry configuration:
        ```python
        async with AsyncExecutor(
            max_concurrent=10,
            retry_attempts=3,
            retry_delay=0.1,
            retry_max_delay=2.0,
            retry_jitter=0.3,
            retry_on=(ConnectionError, TimeoutError)
        ) as executor:
            results = await executor(network_call, urls)
        ```

    Note:
        Data preprocessing should be handled separately before execution:
        ```python
        # Explicit data sanitization (separated from execution)
        data = to_list(raw_data, flatten=True, dropna=True)
        async with AsyncExecutor(...) as executor:
            results = await executor(func, data)
        ```
    """

    def __init__(
        self,
        *,
        max_concurrent: int | None = None,
        retry_attempts: int = 0,
        retry_delay: float = 0.1,
        retry_max_delay: float = 2.0,
        retry_jitter: float = 0.1,
        retry_on: tuple[type[BaseException], ...] = (Exception,),
        throttle_period: float | None = None,
    ):
        self.max_concurrent = max_concurrent
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
        self.retry_max_delay = retry_max_delay
        self.retry_jitter = retry_jitter
        self.retry_on = retry_on
        self.throttle_period = throttle_period

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return False

    async def __call__(self, func_or_funcs, inputs_or_args=None) -> list[Any]:
        """
        Smart executor - detects pattern and executes appropriately.

        Patterns:
            await executor(func, inputs)                    # One function, many inputs
            await executor(funcs, args_kwargs)              # Many functions, many args
        """
        # Pattern detection
        if callable(func_or_funcs):
            # Single function case: await executor(func, inputs)
            return await self._execute_single(func_or_funcs, inputs_or_args)
        elif (
            isinstance(func_or_funcs, (list, tuple)) and len(func_or_funcs) > 0
        ):
            if callable(func_or_funcs[0]):
                # Multiple functions case: await executor(funcs, args_kwargs)
                return await self._execute_multiple(
                    func_or_funcs, inputs_or_args
                )

        raise ValueError("Expected callable or sequence of callables")

    async def _execute_single(
        self, func: Callable, inputs: Iterable[T]
    ) -> list[R]:
        """Execute one function across many inputs (alcall replacement)."""
        items = list(inputs) if inputs else []
        if not items:
            return []

        awaitables = self._create_awaitables_single(func, items)
        return await self._execute_awaitables(awaitables)

    async def _execute_multiple(
        self,
        funcs: Sequence[Callable],
        args_kwargs: Sequence[tuple[tuple, dict | None]],
    ) -> list[Any]:
        """Execute different functions with different arguments (mcall replacement)."""
        if len(funcs) != len(args_kwargs):
            raise ValueError("Functions and args must be same length")

        awaitables = self._create_awaitables_multiple(funcs, args_kwargs)
        return await self._execute_awaitables(awaitables)

    def _create_awaitables_single(
        self, func: Callable, items: list
    ) -> list[Awaitable]:
        """Create awaitables for single function execution."""
        awaitables = []
        for item in items:
            if self.retry_attempts > 0:
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        category=RuntimeWarning,
                        message=".*coroutine.*was never awaited",
                    )
                    awaitable = retry(
                        lambda i=item: self._call_func(func, i),
                        attempts=self.retry_attempts + 1,
                        base_delay=self.retry_delay,
                        max_delay=self.retry_max_delay,
                        jitter=self.retry_jitter,
                        retry_on=self.retry_on,
                    )
            else:
                awaitable = self._call_func(func, item)
            awaitables.append(awaitable)
        return awaitables

    def _create_awaitables_multiple(
        self, funcs: Sequence[Callable], args_kwargs: Sequence
    ) -> list[Awaitable]:
        """Create awaitables for multiple function execution."""
        awaitables = []
        for func, (args, kwargs) in zip(funcs, args_kwargs):
            if self.retry_attempts > 0:
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        category=RuntimeWarning,
                        message=".*coroutine.*was never awaited",
                    )
                    awaitable = retry(
                        lambda f=func, a=args, k=kwargs: self._call_func_with_args(
                            f, a, k
                        ),
                        attempts=self.retry_attempts + 1,
                        base_delay=self.retry_delay,
                        max_delay=self.retry_max_delay,
                        jitter=self.retry_jitter,
                        retry_on=self.retry_on,
                    )
            else:
                awaitable = self._call_func_with_args(func, args, kwargs)
            awaitables.append(awaitable)
        return awaitables

    async def _execute_awaitables(
        self, awaitables: list[Awaitable]
    ) -> list[Any]:
        """Execute awaitables using CompletionStream."""
        if not awaitables:
            return []

        # Use bounded_map for simple case (faster)
        if not self.throttle_period:
            return await bounded_map(
                lambda aw: aw,
                awaitables,
                limit=self.max_concurrent or len(awaitables),
            )

        # Use CompletionStream for throttling
        results = [None] * len(awaitables)
        async with CompletionStream(
            awaitables, limit=self.max_concurrent
        ) as stream:
            async for idx, result in stream:
                results[idx] = result
                if self.throttle_period and idx < len(awaitables) - 1:
                    await anyio.sleep(self.throttle_period)

        return results

    async def stream(
        self,
        func: Callable[[T], Awaitable[R]],
        inputs: Iterable[T],
        batch_size: int,
    ):
        """Stream results in batches (bcall replacement)."""
        items = list(inputs)
        for i in range(0, len(items), batch_size):
            batch = items[i : i + batch_size]
            # FIX: Call the correct internal method
            yield await self._execute_single(func, batch)

    async def _call_func(self, func: Callable, item: Any) -> Any:
        """Helper to call function (sync or async)."""
        if is_coro_func(func):
            return await func(item)
        else:
            # FIX: Use functools.partial instead of passing args directly
            partial_func = functools.partial(func, item)
            return await anyio.to_thread.run_sync(partial_func)

    async def _call_func_with_args(
        self, func: Callable, args: tuple, kwargs: dict | None
    ) -> Any:
        """Helper to call function with args/kwargs."""
        kw = kwargs or {}
        if is_coro_func(func):
            return await func(*args, **kw)
        else:
            # FIX: Use functools.partial instead of passing kwargs to anyio
            partial_func = functools.partial(func, *args, **kw)
            return await anyio.to_thread.run_sync(partial_func)
