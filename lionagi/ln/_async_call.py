import asyncio
from collections.abc import AsyncGenerator, Callable
from dataclasses import dataclass
from typing import Any, ClassVar

import anyio
import anyio.to_thread
from pydantic import BaseModel

from ._to_list import to_list
from .concurrency import Lock as ConcurrencyLock
from .concurrency import (
    Semaphore,
    create_task_group,
    get_cancelled_exc_class,
    is_coro_func,
    move_on_after,
)
from .types import Params, T, Unset, not_sentinel

__all__ = (
    "alcall",
    "bcall",
    "AlcallParams",
    "BcallParams",
)


async def alcall(
    input_: list[Any],
    func: Callable[..., T],
    /,
    *,
    input_flatten: bool = False,
    input_dropna: bool = False,
    input_unique: bool = False,
    input_flatten_tuple_set: bool = False,
    output_flatten: bool = False,
    output_dropna: bool = False,
    output_unique: bool = False,
    output_flatten_tuple_set: bool = False,
    delay_before_start: float = 0,
    retry_initial_deplay: float = 0,
    retry_backoff: float = 1,
    retry_default: Any = Unset,
    retry_timeout: float = None,
    retry_attempts: int = 0,
    max_concurrent: int | None = None,
    throttle_period: float | None = None,
    **kwargs: Any,
) -> list[T]:
    """
    Asynchronously apply a function to each element of a list, with optional input sanitization,
    retries, timeout, and output processing.
    """

    # Validate func is a single callable
    if not callable(func):
        # If func is not callable, maybe it's an iterable. Extract one callable if possible.
        try:
            func_list = list(func)  # Convert iterable to list
        except TypeError:
            raise ValueError(
                "func must be callable or an iterable containing one callable."
            )

        # Ensure exactly one callable is present
        if len(func_list) != 1 or not callable(func_list[0]):
            raise ValueError("Only one callable function is allowed.")

        func = func_list[0]

    # Process input if requested
    if any((input_flatten, input_dropna)):
        input_ = to_list(
            input_,
            flatten=input_flatten,
            dropna=input_dropna,
            unique=input_unique,
            flatten_tuple_set=input_flatten_tuple_set,
        )
    else:
        if not isinstance(input_, list):
            # Attempt to iterate
            if isinstance(input_, BaseModel):
                # Pydantic model, convert to list
                input_ = [input_]
            else:
                try:
                    iter(input_)
                    # It's iterable (tuple), convert to list of its contents
                    input_ = list(input_)
                except TypeError:
                    # Not iterable, just wrap in a list
                    input_ = [input_]

    # Optional initial delay before processing
    if delay_before_start:
        await anyio.sleep(delay_before_start)

    semaphore = Semaphore(max_concurrent) if max_concurrent else None
    throttle_delay = throttle_period or 0
    coro_func = is_coro_func(func)

    async def call_func(item: Any) -> T:
        if coro_func:
            # Async function
            if retry_timeout is not None:
                with move_on_after(retry_timeout) as cancel_scope:
                    result = await func(item, **kwargs)
                if cancel_scope.cancelled_caught:
                    raise asyncio.TimeoutError(
                        f"Function call timed out after {retry_timeout}s"
                    )
                return result
            else:
                return await func(item, **kwargs)
        else:
            # Sync function
            if retry_timeout is not None:
                with move_on_after(retry_timeout) as cancel_scope:
                    result = await anyio.to_thread.run_sync(
                        func, item, **kwargs
                    )
                if cancel_scope.cancelled_caught:
                    raise asyncio.TimeoutError(
                        f"Function call timed out after {retry_timeout}s"
                    )
                return result
            else:
                return await anyio.to_thread.run_sync(func, item, **kwargs)

    async def execute_task(i: Any, index: int) -> Any:
        attempts = 0
        current_delay = retry_initial_deplay
        while True:
            try:
                result = await call_func(i)
                return index, result

            # if cancelled, re-raise
            except get_cancelled_exc_class():
                raise

            # handle other exceptions
            except Exception:
                attempts += 1
                if attempts <= retry_attempts:
                    if current_delay:
                        await anyio.sleep(current_delay)
                        current_delay *= retry_backoff
                    # Retry loop continues
                else:
                    # Exhausted retries
                    if not_sentinel(retry_default):
                        return index, retry_default
                    # No default, re-raise
                    raise

    async def task_wrapper(item: Any, idx: int) -> Any:
        if semaphore:
            async with semaphore:
                result = await execute_task(item, idx)
        else:
            result = await execute_task(item, idx)

        return result

    # Use task group for structured concurrency
    results = []
    results_lock = ConcurrencyLock()  # Protect results list

    async def run_and_store(item: Any, idx: int):
        result = await task_wrapper(item, idx)
        async with results_lock:
            results.append(result)

    # Execute all tasks using task group
    async with create_task_group() as tg:
        for idx, item in enumerate(input_):
            tg.start_soon(run_and_store, item, idx)
            # Apply throttle delay between starting tasks
            if throttle_delay and idx < len(input_) - 1:
                await anyio.sleep(throttle_delay)

    # Sort by original index
    results.sort(key=lambda x: x[0])

    # (index, result)
    output_list = [r[1] for r in results]
    return to_list(
        output_list,
        flatten=output_flatten,
        dropna=output_dropna,
        unique=output_unique,
        flatten_tuple_set=output_flatten_tuple_set,
    )


async def bcall(
    input_: list[Any],
    func: Callable[..., T],
    /,
    batch_size: int,
    *,
    input_flatten: bool = False,
    input_dropna: bool = False,
    input_unique: bool = False,
    input_flatten_tuple_set: bool = False,
    output_flatten: bool = False,
    output_dropna: bool = False,
    output_unique: bool = False,
    output_flatten_tuple_set: bool = False,
    delay_before_start: float = 0,
    retry_initial_deplay: float = 0,
    retry_backoff: float = 1,
    retry_default: Any = Unset,
    retry_timeout: float = 0,
    retry_attempts: int = 0,
    max_concurrent: int | None = None,
    throttle_period: float | None = None,
    **kwargs: Any,
) -> AsyncGenerator[list[T | tuple[T, float]], None]:
    input_ = to_list(input_, flatten=True, dropna=True)

    for i in range(0, len(input_), batch_size):
        batch = input_[i : i + batch_size]  # noqa: E203
        yield await alcall(
            batch,
            func,
            input_flatten=input_flatten,
            input_dropna=input_dropna,
            input_unique=input_unique,
            input_flatten_tuple_set=input_flatten_tuple_set,
            output_flatten=output_flatten,
            output_dropna=output_dropna,
            output_unique=output_unique,
            output_flatten_tuple_set=output_flatten_tuple_set,
            delay_before_start=delay_before_start,
            retry_initial_deplay=retry_initial_deplay,
            retry_backoff=retry_backoff,
            retry_default=retry_default,
            retry_timeout=retry_timeout,
            retry_attempts=retry_attempts,
            max_concurrent=max_concurrent,
            throttle_period=throttle_period,
            **kwargs,
        )


@dataclass(slots=True, init=False, frozen=True)
class AlcallParams(Params):
    # ClassVar attributes
    _none_as_sentinel: ClassVar[bool] = True
    _func: ClassVar[Any] = alcall

    # input processing
    input_flatten: bool
    input_dropna: bool
    input_unique: bool
    input_flatten_tuple_set: bool

    # output processing
    output_flatten: bool
    output_dropna: bool
    output_unique: bool
    output_flatten_tuple_set: bool

    # retry and timeout
    delay_before_start: float
    retry_initial_deplay: float
    retry_backoff: float
    retry_default: Any
    retry_timeout: float
    retry_attempts: int

    # concurrency and throttling
    max_concurrent: int
    throttle_period: float

    kw: dict[str, Any] = Unset

    async def __call__(
        self, input_: list[Any], func: Callable[..., T], **kw
    ) -> list[T]:
        kwargs = {**self.default_kw(), **kw}
        return await alcall(input_, func, **kwargs)


@dataclass(slots=True, init=False, frozen=True)
class BcallParams(AlcallParams):
    _func: ClassVar[Any] = bcall

    batch_size: int

    async def __call__(
        self, input_: list[Any], func: Callable[..., T], **kw
    ) -> list[T]:
        kwargs = {**self.default_kw(), **kw}
        return await bcall(input_, func, self.batch_size, **kwargs)
