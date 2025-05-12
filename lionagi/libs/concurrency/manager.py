from collections.abc import Callable
from typing import Any, AsyncGenerator

from lionagi.utils import UNDEFINED, T


class ConcurrencyUtil:
    @staticmethod
    async def alcall(
        input_: list[Any],
        func: Callable[..., T],
        /,
        *,
        sanitize_input: bool = False,
        unique_input: bool = False,
        num_retries: int = 0,
        initial_delay: float = 0,
        retry_delay: float = 0,
        backoff_factor: float = 1,
        retry_default: Any = UNDEFINED,
        retry_timeout: float | None = None,
        retry_timing: bool = False,
        max_concurrent: int | None = None,
        throttle_period: float | None = None,
        flatten: bool = False,
        dropna: bool = False,
        unique_output: bool = False,
        flatten_tuple_set: bool = False,
        **kwargs: Any,
    ) -> list[T] | list[tuple[T, float]]:
        from .alcall import alcall

        return await alcall(
            input_,
            func,
            sanitize_input=sanitize_input,
            unique_input=unique_input,
            num_retries=num_retries,
            initial_delay=initial_delay,
            retry_delay=retry_delay,
            backoff_factor=backoff_factor,
            retry_default=retry_default,
            retry_timeout=retry_timeout,
            retry_timing=retry_timing,
            max_concurrent=max_concurrent,
            throttle_period=throttle_period,
            flatten=flatten,
            dropna=dropna,
            unique_output=unique_output,
            flatten_tuple_set=flatten_tuple_set,
            **kwargs,
        )

    @staticmethod
    async def bcall(
        input_: Any,
        func: Callable[..., T],
        /,
        batch_size: int,
        *,
        sanitize_input: bool = False,
        unique_input: bool = False,
        num_retries: int = 0,
        initial_delay: float = 0,
        retry_delay: float = 0,
        backoff_factor: float = 1,
        retry_default: Any = UNDEFINED,
        retry_timeout: float | None = None,
        retry_timing: bool = False,
        max_concurrent: int | None = None,
        throttle_period: float | None = None,
        flatten: bool = False,
        dropna: bool = False,
        unique_output: bool = False,
        flatten_tuple_set: bool = False,
        **kwargs: Any,
    ) -> AsyncGenerator[list[T | tuple[T, float]], None]:
        from .bcall import bcall

        return bcall(
            input_,
            func,
            batch_size=batch_size,
            sanitize_input=sanitize_input,
            unique_input=unique_input,
            num_retries=num_retries,
            initial_delay=initial_delay,
            retry_delay=retry_delay,
            backoff_factor=backoff_factor,
            retry_default=retry_default,
            retry_timeout=retry_timeout,
            retry_timing=retry_timing,
            max_concurrent=max_concurrent,
            throttle_period=throttle_period,
            flatten=flatten,
            dropna=dropna,
            unique_output=unique_output,
            flatten_tuple_set=flatten_tuple_set,
            **kwargs,
        )
