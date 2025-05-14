# Copyright (c) 2023 - 2025, HaiyangLi <quantocean.li at gmail dot com>
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import copy as _copy
import functools
import importlib.util
import json
import logging
import subprocess
import time as t_
from abc import ABC
from collections.abc import (
    AsyncGenerator,
    Callable,
    Iterable,
    Mapping,
    Sequence,
)
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, TypedDict, TypeVar, overload

from pydantic import BaseModel, model_validator
from pydantic_core import PydanticUndefinedType

from .settings import Settings

R = TypeVar("R")
T = TypeVar("T")
B = TypeVar("B", bound=BaseModel)

logger = logging.getLogger(__name__)


__all__ = (
    "UndefinedType",
    "KeysDict",
    "Params",
    "DataClass",
    "UNDEFINED",
    "copy",
    "is_same_dtype",
    "get_file_classes",
    "get_class_file_registry",
    "get_class_objects",
    "is_coro_func",
    "custom_error_handler",
    "to_list",
    "lcall",
    "alcall",
    "bcall",
    "create_path",
    "time",
    "fuzzy_parse_json",
    "fix_json_string",
    "ToListParams",
    "LCallParams",
    "ALCallParams",
    "BCallParams",
    "CreatePathParams",
    "get_bins",
    "EventStatus",
    "logger",
    "throttle",
    "max_concurrent",
    "force_async",
    "to_num",
    "breakdown_pydantic_annotation",
    "run_package_manager_command",
)


# --- General Global Utilities Types ---


class UndefinedType:
    def __init__(self) -> None:
        self.undefined = True

    def __bool__(self) -> Literal[False]:
        return False

    def __deepcopy__(self, memo):
        # Ensure UNDEFINED is universal
        return self

    def __repr__(self) -> Literal["UNDEFINED"]:
        return "UNDEFINED"

    __slots__ = ["undefined"]


class KeysDict(TypedDict, total=False):
    """TypedDict for keys dictionary."""

    key: Any  # Represents any key-type pair


def hash_dict(data) -> int:
    hashable_items = []
    if isinstance(data, BaseModel):
        data = data.model_dump()
    for k, v in data.items():
        if isinstance(v, (list, dict)):
            # Convert unhashable types to JSON string for hashing
            v = json.dumps(v, sort_keys=True)
        elif not isinstance(v, (str, int, float, bool, type(None))):
            # Convert other unhashable types to string representation
            v = str(v)
        hashable_items.append((k, v))
    return hash(frozenset(hashable_items))


class Params(BaseModel):

    def keys(self):
        return self.__class__.model_fields.keys()

    def __call__(self, *args, **kwargs):
        raise NotImplementedError(
            "This method should be implemented in a subclass"
        )


class DataClass(ABC):
    pass


# --- Create a global UNDEFINED object ---
UNDEFINED = UndefinedType()


# --- General Global Utilities Functions ---


def time(
    *,
    tz: timezone = Settings.Config.TIMEZONE,
    type_: Literal["timestamp", "datetime", "iso", "custom"] = "timestamp",
    sep: str | None = "T",
    timespec: str | None = "auto",
    custom_format: str | None = None,
    custom_sep: str | None = None,
) -> float | str | datetime:
    """
    Get current time in various formats.

    Args:
        tz: Timezone for the time (default: utc).
        type_: Type of time to return (default: "timestamp").
            Options: "timestamp", "datetime", "iso", "custom".
        sep: Separator for ISO format (default: "T").
        timespec: Timespec for ISO format (default: "auto").
        custom_format: Custom strftime format string for
            type_="custom".
        custom_sep: Custom separator for type_="custom",
            replaces "-", ":", ".".

    Returns:
        Current time in the specified format.

    Raises:
        ValueError: If an invalid type_ is provided or if custom_format
            is not provided when type_="custom".
    """
    now = datetime.now(tz=tz)

    if type_ == "iso":
        return now.isoformat(sep=sep, timespec=timespec)
    elif type_ == "timestamp":
        return now.timestamp()
    elif type_ == "datetime":
        return now
    elif type_ == "custom":
        if not custom_format:
            raise ValueError(
                "custom_format must be provided when type_='custom'"
            )
        formatted_time = now.strftime(custom_format)
        if custom_sep is not None:
            for old_sep in ("-", ":", "."):
                formatted_time = formatted_time.replace(old_sep, custom_sep)
        return formatted_time

    raise ValueError(
        f"Invalid value <{type_}> for `type_`, must be"
        " one of 'timestamp', 'datetime', 'iso', or 'custom'."
    )


def copy(obj: T, /, *, deep: bool = True, num: int = 1) -> T | list[T]:
    if num < 1:
        raise ValueError("Number of copies must be at least 1")

    copy_func = _copy.deepcopy if deep else _copy.copy
    return [copy_func(obj) for _ in range(num)] if num > 1 else copy_func(obj)


def is_same_dtype(
    input_: list[T] | dict[Any, T],
    dtype: type | None = None,
    return_dtype: bool = False,
) -> bool | tuple[bool, type | None]:
    if not input_:
        # If empty, trivially true. dtype is None since no elements exist.
        return (True, None) if return_dtype else True

    if isinstance(input_, Mapping):
        # For dictionaries, use values
        values_iter = iter(input_.values())
        first_val = next(values_iter, None)
        if dtype is None:
            dtype = type(first_val) if first_val is not None else None
        # Check the first element
        result = (dtype is None or isinstance(first_val, dtype)) and all(
            isinstance(v, dtype) for v in values_iter
        )
    else:
        # For lists (or list-like), directly access the first element
        first_val = input_[0]
        if dtype is None:
            dtype = type(first_val) if first_val is not None else None
        # Check all elements including the first
        result = all(isinstance(e, dtype) for e in input_)

    return (result, dtype) if return_dtype else result


@lru_cache(maxsize=None)
def is_coro_func(func: Callable[..., Any]) -> bool:
    return asyncio.iscoroutinefunction(func)


async def custom_error_handler(
    error: Exception, error_map: dict[type, Callable[[Exception], None]]
) -> None:
    if type(error) in error_map:
        if is_coro_func(error_map[type(error)]):
            return await error_map[type(error)](error)
        return error_map[type(error)](error)
    logging.error(f"Unhandled error: {error}")
    raise error


@overload
def to_list(
    input_: None | UndefinedType | PydanticUndefinedType,
    /,
) -> list: ...


@overload
def to_list(
    input_: str | bytes | bytearray,
    /,
    use_values: bool = False,
) -> list[str | bytes | bytearray]: ...


@overload
def to_list(
    input_: Mapping,
    /,
    use_values: bool = False,
) -> list[Any]: ...


@overload
def to_list(
    input_: Any,
    /,
    *,
    flatten: bool = False,
    dropna: bool = False,
    unique: bool = False,
    use_values: bool = False,
    flatten_tuple_set: bool = False,
) -> list: ...


def to_list(
    input_: Any,
    /,
    *,
    flatten: bool = False,
    dropna: bool = False,
    unique: bool = False,
    use_values: bool = False,
    flatten_tuple_set: bool = False,
) -> list:
    """Convert input to a list with optional transformations.

    Transforms various input types into a list with configurable processing
    options for flattening, filtering, and value extraction.

    Args:
        input_: Value to convert to list.
        flatten: If True, recursively flatten nested iterables.
        dropna: If True, remove None and undefined values.
        unique: If True, remove duplicates (requires flatten=True).
        use_values: If True, extract values from enums/mappings.
        flatten_tuple_items: If True, include tuples in flattening.
        flatten_set_items: If True, include sets in flattening.

    Returns:
        list: Processed list based on input and specified options.

    Raises:
        ValueError: If unique=True is used without flatten=True.

    Examples:
        >>> to_list([1, [2, 3], 4], flatten=True)
        [1, 2, 3, 4]
        >>> to_list([1, None, 2], dropna=True)
        [1, 2]
    """

    def _process_list(
        lst: list[Any],
        flatten: bool,
        dropna: bool,
    ) -> list[Any]:
        """Process list according to flatten and dropna options.

        Args:
            lst: Input list to process.
            flatten: Whether to flatten nested iterables.
            dropna: Whether to remove None/undefined values.

        Returns:
            list: Processed list based on specified options.
        """
        result = []
        skip_types = (str, bytes, bytearray, Mapping, BaseModel, Enum)

        if not flatten_tuple_set:
            skip_types += (tuple, set, frozenset)

        for item in lst:
            if dropna and (
                item is None
                or isinstance(item, (UndefinedType, PydanticUndefinedType))
            ):
                continue

            is_iterable = isinstance(item, Iterable)
            should_skip = isinstance(item, skip_types)

            if is_iterable and not should_skip:
                item_list = list(item)
                if flatten:
                    result.extend(
                        _process_list(
                            item_list, flatten=flatten, dropna=dropna
                        )
                    )
                else:
                    result.append(
                        _process_list(
                            item_list, flatten=flatten, dropna=dropna
                        )
                    )
            else:
                result.append(item)

        return result

    def _to_list_type(input_: Any, use_values: bool) -> list[Any]:
        """Convert input to initial list based on type.

        Args:
            input_: Value to convert to list.
            use_values: Whether to extract values from containers.

        Returns:
            list: Initial list conversion of input.
        """
        if input_ is None or isinstance(
            input_, (UndefinedType, PydanticUndefinedType)
        ):
            return []

        if isinstance(input_, list):
            return input_

        if isinstance(input_, type) and issubclass(input_, Enum):
            members = input_.__members__.values()
            return (
                [member.value for member in members]
                if use_values
                else list(members)
            )

        if isinstance(input_, (str, bytes, bytearray)):
            return list(input_) if use_values else [input_]

        if isinstance(input_, Mapping):
            return (
                list(input_.values())
                if use_values and hasattr(input_, "values")
                else [input_]
            )

        if isinstance(input_, BaseModel):
            return [input_]

        if isinstance(input_, Iterable) and not isinstance(
            input_, (str, bytes, bytearray)
        ):
            return list(input_)

        return [input_]

    if unique and not flatten:
        raise ValueError("unique=True requires flatten=True")

    initial_list = _to_list_type(input_, use_values=use_values)
    processed = _process_list(initial_list, flatten=flatten, dropna=dropna)

    if unique:
        seen = set()
        out = []
        try:
            return [x for x in processed if not (x in seen or seen.add(x))]
        except TypeError:
            for i in processed:
                hash_value = None
                try:
                    hash_value = hash(i)
                except TypeError:
                    if isinstance(i, (BaseModel, Mapping)):
                        hash_value = hash_dict(i)
                    else:
                        raise ValueError(
                            "Unhashable type encountered in list unique value processing."
                        )
                if hash_value not in seen:
                    seen.add(hash_value)
                    out.append(i)
            return out

    return processed


class ToListParams(Params):
    flatten: bool = False
    dropna: bool = False
    unique: bool = False
    use_values: bool = False
    flatten_tuple_set: bool = False

    def __call__(self, input_: Any):
        return to_list(
            input_,
            flatten=self.flatten,
            dropna=self.dropna,
            unique=self.unique,
            use_values=self.use_values,
            flatten_tuple_set=self.flatten_tuple_set,
        )


def lcall(
    input_: Iterable[T] | T,
    func: Callable[[T], R] | Iterable[Callable[[T], R]],
    /,
    *args: Any,
    sanitize_input: bool = False,
    unique_input: bool = False,
    flatten: bool = False,
    dropna: bool = False,
    unique_output: bool = False,
    flatten_tuple_set: bool = False,
    use_input_values: bool = False,
    **kwargs: Any,
) -> list[R]:
    """Apply function to each element in input list with optional processing.

    Maps a function over input elements and processes results. Can sanitize input
    and output using various filtering options.

    Args:
        input_: Single item or iterable to process.
        func: Function to apply or single-item iterable containing function.
        *args: Additional positional arguments passed to func.
        sanitize_input: If True, sanitize input using to_list.
        unique_input: If True with sanitize_input, remove input duplicates.
        flatten: If True, flatten output nested structures.
        dropna: If True, remove None values from output.
        unique_output: If True with flatten/dropna, remove output duplicates.
        flatten_tuple_set: If True, flatten tuples and sets.
        **kwargs: Additional keyword arguments passed to func.

    Returns:
        list: Results of applying func to each input element.

    Raises:
        ValueError: If func is not callable or unique_output used incorrectly.
        TypeError: If func or input processing fails.

    Examples:
        >>> lcall([1, 2, 3], str)
        ['1', '2', '3']
        >>> lcall([1, [2, 3]], str, flatten=True)
        ['1', '2', '3']
    """
    # Validate and extract callable function
    if not callable(func):
        try:
            func_list = list(func)
            if len(func_list) != 1 or not callable(func_list[0]):
                raise ValueError(
                    "func must contain exactly one callable function."
                )
            func = func_list[0]
        except TypeError as e:
            raise ValueError(
                "func must be callable or iterable with one callable."
            ) from e

    # Process input based on sanitization flag
    if sanitize_input:
        input_ = to_list(
            input_,
            flatten=True,
            dropna=True,
            unique=unique_input,
            flatten_tuple_set=flatten_tuple_set,
            use_values=use_input_values,
        )
    else:
        if not isinstance(input_, list):
            try:
                input_ = list(input_)
            except TypeError:
                input_ = [input_]

    # Validate output processing options
    if unique_output and not (flatten or dropna):
        raise ValueError(
            "unique_output requires flatten or dropna for post-processing."
        )

    # Process elements and collect results
    out = []
    append = out.append

    for item in input_:
        try:
            result = func(item, *args, **kwargs)
            append(result)
        except InterruptedError:
            return out
        except Exception:
            raise

    # Apply output processing if requested
    if flatten or dropna:
        out = to_list(
            out,
            flatten=flatten,
            dropna=dropna,
            unique=unique_output,
            flatten_tuple_set=flatten_tuple_set,
        )

    return out


class CallParams(Params):
    """params class for high order function with additional handling of lower order function parameters, can take arbitrary number of args and kwargs, args need to be in agrs=, kwargs can be passed as is"""

    args: list = []
    kwargs: dict = {}

    @model_validator(mode="before")
    def _validate_data(cls, data: dict):
        _d = {}
        for k in list(data.keys()):
            if k in cls.keys():
                _d[k] = data.pop(k)
        _d.setdefault("args", [])
        _d.setdefault("kwargs", {})
        _d["kwargs"].update(data)
        return _d

    def __call__(self, *args, **kwargs):
        raise NotImplementedError(
            "This method should be implemented in a subclass"
        )


class LCallParams(CallParams):
    func: Any = None
    sanitize_input: bool = False
    unique_input: bool = False
    flatten: bool = False
    dropna: bool = False
    unique_output: bool = False
    flatten_tuple_set: bool = False

    def __call__(self, input_: Any, func=None):
        if self.func is None and func is None:
            raise ValueError("a sync func must be provided")
        return lcall(
            input_,
            func or self.func,
            *self.args,
            sanitize_input=self.sanitize_input,
            unique_input=self.unique_input,
            flatten=self.flatten,
            dropna=self.dropna,
            unique_output=self.unique_output,
            flatten_tuple_set=self.flatten_tuple_set,
            **self.kwargs,
        )


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
    """
    Asynchronously apply a function to each element of a list, with optional input sanitization,
    retries, timeout, and output processing.

    Args:
        input_ (list[Any]): The list of inputs to process.
        func (Callable[..., T]): The function to apply (async or sync).
        sanitize_input (bool): If True, input is flattened, dropna applied, and made unique if unique_input.
        unique_input (bool): If True and sanitize_input is True, input is made unique.
        num_retries (int): Number of retry attempts on exception.
        initial_delay (float): Initial delay before starting executions.
        retry_delay (float): Delay between retries.
        backoff_factor (float): Multiplier for delay after each retry.
        retry_default (Any): Default value if all retries fail.
        retry_timeout (float | None): Timeout for each function call.
        retry_timing (bool): If True, return (result, duration) tuples.
        max_concurrent (int | None): Maximum number of concurrent operations.
        throttle_period (float | None): Delay after each completed operation.
        flatten (bool): Flatten the final result if True.
        dropna (bool): Remove None values from the final result if True.
        unique_output (bool): Deduplicate the output if True.
        flatten_tuple_set (bool): Tuples and sets will be flattened if True.
        **kwargs: Additional arguments passed to func.

    Returns:
        list[T] or list[tuple[T, float]]: The processed results, or results with timing if retry_timing is True.

    Raises:
        asyncio.TimeoutError: If a call times out and no default is provided.
        Exception: If retries are exhausted and no default is provided.
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
    if sanitize_input:
        input_ = to_list(
            input_,
            flatten=True,
            dropna=True,
            unique=unique_input,
            flatten_tuple_set=flatten_tuple_set,
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
    if initial_delay:
        await asyncio.sleep(initial_delay)

    semaphore = asyncio.Semaphore(max_concurrent) if max_concurrent else None
    throttle_delay = throttle_period or 0
    coro_func = is_coro_func(func)

    async def call_func(item: Any) -> T:
        if coro_func:
            # Async function
            if retry_timeout is not None:
                return await asyncio.wait_for(
                    func(item, **kwargs), timeout=retry_timeout
                )
            else:
                return await func(item, **kwargs)
        else:
            # Sync function
            if retry_timeout is not None:
                return await asyncio.wait_for(
                    asyncio.to_thread(func, item, **kwargs),
                    timeout=retry_timeout,
                )
            else:
                return func(item, **kwargs)

    async def execute_task(i: Any, index: int) -> Any:
        start_time = asyncio.get_running_loop().time()
        attempts = 0
        current_delay = retry_delay
        while True:
            try:
                result = await call_func(i)
                if retry_timing:
                    end_time = asyncio.get_running_loop().time()
                    return index, result, end_time - start_time
                else:
                    return index, result
            except asyncio.CancelledError as e:
                raise e

            except Exception:
                attempts += 1
                if attempts <= num_retries:
                    if current_delay:
                        await asyncio.sleep(current_delay)
                        current_delay *= backoff_factor
                    # Retry loop continues
                else:
                    # Exhausted retries
                    if retry_default is not UNDEFINED:
                        # Return default if provided
                        if retry_timing:
                            end_time = asyncio.get_running_loop().time()
                            duration = end_time - (start_time or end_time)
                            return index, retry_default, duration
                        else:
                            return index, retry_default
                    # No default, re-raise
                    raise

    async def task_wrapper(item: Any, idx: int) -> Any:
        if semaphore:
            async with semaphore:
                return await execute_task(item, idx)
        else:
            return await execute_task(item, idx)

    # Create tasks
    tasks = [task_wrapper(item, idx) for idx, item in enumerate(input_)]

    # Collect results as they complete
    results = []
    for coro in asyncio.as_completed(tasks):
        res = await coro
        results.append(res)
        if throttle_delay:
            await asyncio.sleep(throttle_delay)

    # Sort by original index
    results.sort(key=lambda x: x[0])

    if retry_timing:
        # (index, result, duration)
        filtered = [
            (r[1], r[2]) for r in results if not dropna or r[1] is not None
        ]
        return filtered
    else:
        # (index, result)
        output_list = [r[1] for r in results]
        return to_list(
            output_list,
            flatten=flatten,
            dropna=dropna,
            unique=unique_output,
            flatten_tuple_set=flatten_tuple_set,
        )


class ALCallParams(CallParams):
    func: Any = None
    sanitize_input: bool = False
    unique_input: bool = False
    num_retries: int = 0
    initial_delay: float = 0
    retry_delay: float = 0
    backoff_factor: float = 1
    retry_default: Any = UNDEFINED
    retry_timeout: float | None = None
    retry_timing: bool = False
    max_concurrent: int | None = None
    throttle_period: float | None = None
    flatten: bool = False
    dropna: bool = False
    unique_output: bool = False
    flatten_tuple_set: bool = False

    async def __call__(self, input_: Any, func=None):
        if self.func is None and func is None:
            raise ValueError("a sync/async func must be provided")
        return await alcall(
            input_,
            func or self.func,
            *self.args,
            sanitize_input=self.sanitize_input,
            unique_input=self.unique_input,
            num_retries=self.num_retries,
            initial_delay=self.initial_delay,
            retry_delay=self.retry_delay,
            backoff_factor=self.backoff_factor,
            retry_default=self.retry_default,
            retry_timeout=self.retry_timeout,
            retry_timing=self.retry_timing,
            max_concurrent=self.max_concurrent,
            throttle_period=self.throttle_period,
            flatten=self.flatten,
            dropna=self.dropna,
            unique_output=self.unique_output,
            flatten_tuple_set=self.flatten_tuple_set,
            **self.kwargs,
        )


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

    input_ = to_list(input_, flatten=True, dropna=True)

    for i in range(0, len(input_), batch_size):
        batch = input_[i : i + batch_size]  # noqa: E203
        yield await alcall(
            batch,
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


class BCallParams(CallParams):
    func: Any = None
    batch_size: int
    sanitize_input: bool = False
    unique_input: bool = False
    num_retries: int = 0
    initial_delay: float = 0
    retry_delay: float = 0
    backoff_factor: float = 1
    retry_default: Any = UNDEFINED
    retry_timeout: float | None = None
    retry_timing: bool = False
    max_concurrent: int | None = None
    throttle_period: float | None = None
    flatten: bool = False
    dropna: bool = False
    unique_output: bool = False
    flatten_tuple_set: bool = False

    async def __call__(self, input_, func=None):
        if self.func is None and func is None:
            raise ValueError("a sync/async func must be provided")
        return await bcall(
            input_,
            func or self.func,
            *self.args,
            batch_size=self.batch_size,
            sanitize_input=self.sanitize_input,
            unique_input=self.unique_input,
            num_retries=self.num_retries,
            initial_delay=self.initial_delay,
            retry_delay=self.retry_delay,
            backoff_factor=self.backoff_factor,
            retry_default=self.retry_default,
            retry_timeout=self.retry_timeout,
            retry_timing=self.retry_timing,
            max_concurrent=self.max_concurrent,
            throttle_period=self.throttle_period,
            flatten=self.flatten,
            dropna=self.dropna,
            unique_output=self.unique_output,
            flatten_tuple_set=self.flatten_tuple_set,
            **self.kwargs,
        )


def create_path(
    directory: Path | str,
    filename: str,
    extension: str = None,
    timestamp: bool = False,
    dir_exist_ok: bool = True,
    file_exist_ok: bool = False,
    time_prefix: bool = False,
    timestamp_format: str | None = None,
    random_hash_digits: int = 0,
) -> Path:
    """
    Generate a new file path with optional timestamp and a random suffix.

    Args:
        directory: The directory where the file will be created.
        filename: The base name of the file to create.
        extension: The file extension, if not part of filename.
        timestamp: If True, add a timestamp to the filename.
        dir_exist_ok: If True, don't error if directory exists.
        file_exist_ok: If True, allow overwriting existing files.
        time_prefix: If True, timestamp is prefixed instead of suffixed.
        timestamp_format: Custom format for timestamp (default: "%Y%m%d%H%M%S").
        random_hash_digits: Number of hex digits for a random suffix.

    Returns:
        The full Path to the new or existing file.

    Raises:
        ValueError: If filename is invalid.
        FileExistsError: If file exists and file_exist_ok=False.
    """
    from lionagi.libs.file.create_path import create_path

    return create_path(
        directory,
        filename,
        extension=extension,
        timestamp=timestamp,
        dir_exist_ok=dir_exist_ok,
        file_exist_ok=file_exist_ok,
        time_prefix=time_prefix,
        timestamp_format=timestamp_format,
        random_hash_digits=random_hash_digits,
    )


class CreatePathParams(Params):
    directory: Path | str
    filename: str
    extension: str = None
    timestamp: bool = False
    dir_exist_ok: bool = True
    file_exist_ok: bool = False
    time_prefix: bool = False
    timestamp_format: str | None = None
    random_hash_digits: int = 0

    def __call__(
        self, directory: Path | str = None, filename: str = None
    ) -> Path:
        return create_path(
            directory or self.directory,
            filename or self.filename,
            extension=self.extension,
            timestamp=self.timestamp,
            dir_exist_ok=self.dir_exist_ok,
            file_exist_ok=self.file_exist_ok,
            time_prefix=self.time_prefix,
            timestamp_format=self.timestamp_format,
            random_hash_digits=self.random_hash_digits,
        )


# --- JSON and XML Conversion ---


def to_xml(
    obj: dict | list | str | int | float | bool | None,
    root_name: str = "root",
) -> str:
    """
    Convert a dictionary into an XML formatted string.

    Rules:
    - A dictionary key becomes an XML tag.
    - If the dictionary value is:
      - A primitive type (str, int, float, bool, None): it becomes the text content of the tag.
      - A list: each element of the list will repeat the same tag.
      - Another dictionary: it is recursively converted to nested XML.
    - root_name sets the top-level XML element name.

    Args:
        obj: The Python object to convert (typically a dictionary).
        root_name: The name of the root XML element.

    Returns:
        A string representing the XML.

    Examples:
        >>> to_xml({"a": 1, "b": {"c": "hello", "d": [10, 20]}}, root_name="data")
        '<data><a>1</a><b><c>hello</c><d>10</d><d>20</d></b></data>'
    """
    from lionagi.libs.parse.to_xml import to_xml

    return to_xml(obj, root_name=root_name)


def fuzzy_parse_json(
    str_to_parse: str, /
) -> dict[str, Any] | list[dict[str, Any]]:
    """
    Attempt to parse a JSON string, trying a few minimal "fuzzy" fixes if needed.

    Steps:
    1. Parse directly with json.loads.
    2. Replace single quotes with double quotes, normalize spacing, and try again.
    3. Attempt to fix unmatched brackets using fix_json_string.
    4. If all fail, raise ValueError.

    Args:
        str_to_parse: The JSON string to parse

    Returns:
        Parsed JSON (dict or list of dicts)

    Raises:
        ValueError: If the string cannot be parsed as valid JSON
        TypeError: If the input is not a string
    """
    from lionagi.libs.parse.fuzzy_parse_json import fuzzy_parse_json

    return fuzzy_parse_json(str_to_parse)


def xml_to_dict(
    xml_string: str,
    /,
    suppress=False,
    remove_root: bool = True,
    root_tag: str = None,
) -> dict[str, Any]:
    """
    Parse an XML string into a nested dictionary structure.

    This function converts an XML string into a dictionary where:
    - Element tags become dictionary keys
    - Text content is assigned directly to the tag key if there are no children
    - Attributes are stored in a '@attributes' key
    - Multiple child elements with the same tag are stored as lists

    Args:
        xml_string: The XML string to parse.

    Returns:
        A dictionary representation of the XML structure.

    Raises:
        ValueError: If the XML is malformed or parsing fails.
    """
    from lionagi.libs.parse.xml_parser import xml_to_dict

    return xml_to_dict(
        xml_string,
        suppress=suppress,
        remove_root=remove_root,
        root_tag=root_tag,
    )


def dict_to_xml(data: dict, /, root_tag: str = "root") -> str:
    from lionagi.libs.parse.xml_parser import dict_to_xml

    return dict_to_xml(data, root_tag=root_tag)


def to_dict(
    input_: Any,
    /,
    *,
    use_model_dump: bool = True,
    fuzzy_parse: bool = False,
    suppress: bool = False,
    str_type: Literal["json", "xml"] | None = "json",
    parser: Callable[[str], Any] | None = None,
    recursive: bool = False,
    max_recursive_depth: int = None,
    recursive_python_only: bool = True,
    use_enum_values: bool = False,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Convert various input types to a dictionary, with optional recursive processing.

    Args:
        input_: The input to convert.
        use_model_dump: Use model_dump() for Pydantic models if available.
        fuzzy_parse: Use fuzzy parsing for string inputs.
        suppress: Return empty dict on errors if True.
        str_type: Input string type ("json" or "xml").
        parser: Custom parser function for string inputs.
        recursive: Enable recursive conversion of nested structures.
        max_recursive_depth: Maximum recursion depth (default 5, max 10).
        recursive_python_only: If False, attempts to convert custom types recursively.
        use_enum_values: Use enum values instead of names.
        **kwargs: Additional arguments for parsing functions.

    Returns:
        dict[str, Any]: A dictionary derived from the input.

    Raises:
        ValueError: If parsing fails and suppress is False.

    Examples:
        >>> to_dict({"a": 1, "b": [2, 3]})
        {'a': 1, 'b': [2, 3]}
        >>> to_dict('{"x": 10}', str_type="json")
        {'x': 10}
        >>> to_dict({"a": {"b": {"c": 1}}}, recursive=True, max_recursive_depth=2)
        {'a': {'b': {'c': 1}}}
    """
    from lionagi.libs.parse.to_dict import to_dict

    return to_dict(
        input_,
        use_model_dump=use_model_dump,
        fuzzy_parse=fuzzy_parse,
        suppress=suppress,
        str_type=str_type,
        parser=parser,
        recursive=recursive,
        max_recursive_depth=max_recursive_depth,
        recursive_python_only=recursive_python_only,
        use_enum_values=use_enum_values,
        **kwargs,
    )


def to_json(
    input_data: str | list[str], /, *, fuzzy_parse: bool = False
) -> dict[str, Any] | list[dict[str, Any]]:
    """
    Extract and parse JSON content from a string or markdown code blocks.

    Attempts direct JSON parsing first. If that fails, looks for JSON content
    within markdown code blocks denoted by ```json.

    Args:
        input_data (str | list[str]): The input string or list of strings to parse.
        fuzzy_parse (bool): If True, attempts fuzzy JSON parsing on failed attempts.

    Returns:
        dict or list of dicts:
            - If a single JSON object is found: returns a dict.
            - If multiple JSON objects are found: returns a list of dicts.
            - If no valid JSON found: returns an empty list.
    """
    from lionagi.libs.parse.to_json import to_json

    return to_json(
        input_data,
        fuzzy_parse=fuzzy_parse,
    )


def get_bins(input_: list[str], upper: int) -> list[list[int]]:
    """Organizes indices of strings into bins based on a cumulative upper limit.

    Args:
        input_ (List[str]): The list of strings to be binned.
        upper (int): The cumulative length upper limit for each bin.

    Returns:
        List[List[int]]: A list of bins, each bin is a list of indices from the input list.
    """
    current = 0
    bins = []
    current_bin = []
    for idx, item in enumerate(input_):
        if current + len(item) < upper:
            current_bin.append(idx)
            current += len(item)
        else:
            bins.append(current_bin)
            current_bin = [idx]
            current = len(item)
    if current_bin:
        bins.append(current_bin)
    return bins


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
                t_.sleep(self.period - elapsed)
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


def force_async(fn: Callable[..., T]) -> Callable[..., Callable[..., T]]:
    """
    Convert a synchronous function to an asynchronous function
    using a thread pool.

    Args:
        fn: The synchronous function to convert.

    Returns:
        The asynchronous version of the function.
    """
    pool = ThreadPoolExecutor()

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        future = pool.submit(fn, *args, **kwargs)
        return asyncio.wrap_future(future)  # Make it awaitable

    return wrapper


def throttle(
    func: Callable[..., T], period: float
) -> Callable[..., Callable[..., T]]:
    """
    Throttle function execution to limit the rate of calls.

    Args:
        func: The function to throttle.
        period: The minimum time interval between consecutive calls.

    Returns:
        The throttled function.
    """
    if not is_coro_func(func):
        func = force_async(func)
    throttle_instance = Throttle(period)

    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        await throttle_instance(func)(*args, **kwargs)
        return await func(*args, **kwargs)

    return wrapper


def max_concurrent(
    func: Callable[..., T], limit: int
) -> Callable[..., Callable[..., T]]:
    """
    Limit the concurrency of function execution using a semaphore.

    Args:
        func: The function to limit concurrency for.
        limit: The maximum number of concurrent executions.

    Returns:
        The function wrapped with concurrency control.
    """
    if not is_coro_func(func):
        func = force_async(func)
    semaphore = asyncio.Semaphore(limit)

    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        async with semaphore:
            return await func(*args, **kwargs)

    return wrapper


# Type definitions
NUM_TYPE_LITERAL = Literal["int", "float", "complex"]
NUM_TYPES = type[int] | type[float] | type[complex] | NUM_TYPE_LITERAL
NumericType = TypeVar("NumericType", int, float, complex)

# Type mapping
TYPE_MAP = {"int": int, "float": float, "complex": complex}

# Regex patterns for different numeric formats
PATTERNS = {
    "scientific": r"[-+]?(?:\d*\.)?\d+[eE][-+]?\d+",
    "complex_sci": r"[-+]?(?:\d*\.)?\d+(?:[eE][-+]?\d+)?[-+](?:\d*\.)?\d+(?:[eE][-+]?\d+)?[jJ]",
    "complex": r"[-+]?(?:\d*\.)?\d+[-+](?:\d*\.)?\d+[jJ]",
    "pure_imaginary": r"[-+]?(?:\d*\.)?\d*[jJ]",
    "percentage": r"[-+]?(?:\d*\.)?\d+%",
    "fraction": r"[-+]?\d+/\d+",
    "decimal": r"[-+]?(?:\d*\.)?\d+",
    "special": r"[-+]?(?:inf|infinity|nan)",
}


def to_num(
    input_: Any,
    /,
    *,
    upper_bound: int | float | None = None,
    lower_bound: int | float | None = None,
    num_type: NUM_TYPES = float,
    precision: int | None = None,
    num_count: int = 1,
) -> int | float | complex | list[int | float | complex]:
    """Convert input to numeric type(s) with validation and bounds checking.

    Args:
        input_value: The input to convert to number(s).
        upper_bound: Maximum allowed value (inclusive).
        lower_bound: Minimum allowed value (inclusive).
        num_type: Target numeric type ('int', 'float', 'complex' or type objects).
        precision: Number of decimal places for rounding (float only).
        num_count: Number of numeric values to extract.

    Returns:
        Converted number(s). Single value if num_count=1, else list.

    Raises:
        ValueError: For invalid input or out of bounds values.
        TypeError: For invalid input types or invalid type conversions.
    """
    from lionagi.libs.parse.to_num import to_num

    return to_num(
        input_,
        upper_bound=upper_bound,
        lower_bound=lower_bound,
        num_type=num_type,
        precision=precision,
        num_count=num_count,
    )


def extract_numbers(text: str) -> list[tuple[str, str]]:
    """Extract numeric values from text using ordered regex patterns.

    Args:
        text: The text to extract numbers from.

    Returns:
        List of tuples containing (pattern_type, matched_value).
    """
    from lionagi.libs.parse.to_num import extract_numbers

    return extract_numbers(text=text)


def validate_num_type(num_type: NUM_TYPES) -> type:
    """Validate and normalize numeric type specification.

    Args:
        num_type: The numeric type to validate.

    Returns:
        The normalized Python type object.

    Raises:
        ValueError: If the type specification is invalid.
    """
    from lionagi.libs.parse.to_num import validate_num_type

    return validate_num_type(num_type=num_type)


def infer_type(value: tuple[str, str]) -> type:
    """Infer appropriate numeric type from value.

    Args:
        value: Tuple of (pattern_type, matched_value).

    Returns:
        The inferred Python type.
    """
    from lionagi.libs.parse.to_num import infer_type

    return infer_type(value=value)


def convert_special(value: str) -> float:
    """Convert special float values (inf, -inf, nan).

    Args:
        value: The string value to convert.

    Returns:
        The converted float value.
    """
    from lionagi.libs.parse.to_num import convert_special

    return convert_special(value=value)


def convert_percentage(value: str) -> float:
    """Convert percentage string to float.

    Args:
        value: The percentage string to convert.

    Returns:
        The converted float value.

    Raises:
        ValueError: If the percentage value is invalid.
    """
    from lionagi.libs.parse.to_num import convert_percentage

    return convert_percentage(value=value)


def convert_complex(value: str) -> complex:
    """Convert complex number string to complex.

    Args:
        value: The complex number string to convert.

    Returns:
        The converted complex value.

    Raises:
        ValueError: If the complex number is invalid.
    """
    from lionagi.libs.parse.to_num import convert_complex

    return convert_complex(value=value)


def convert_type(
    value: float | complex,
    target_type: type,
    inferred_type: type,
) -> int | float | complex:
    """Convert value to target type if specified, otherwise use inferred type.

    Args:
        value: The value to convert.
        target_type: The requested target type.
        inferred_type: The inferred type from the value.

    Returns:
        The converted value.

    Raises:
        TypeError: If the conversion is not possible.
    """
    from lionagi.libs.parse.to_num import convert_type

    return convert_type(
        value=value,
        target_type=target_type,
        inferred_type=inferred_type,
    )


def apply_bounds(
    value: float | complex,
    upper_bound: float | None = None,
    lower_bound: float | None = None,
) -> float | complex:
    """Apply bounds checking to numeric value.

    Args:
        value: The value to check.
        upper_bound: Maximum allowed value (inclusive).
        lower_bound: Minimum allowed value (inclusive).

    Returns:
        The validated value.

    Raises:
        ValueError: If the value is outside bounds.
    """
    from lionagi.libs.parse.to_num import apply_bounds

    return apply_bounds(
        value=value,
        upper_bound=upper_bound,
        lower_bound=lower_bound,
    )


def apply_precision(
    value: float | complex,
    precision: int | None,
) -> float | complex:
    """Apply precision rounding to numeric value.

    Args:
        value: The value to round.
        precision: Number of decimal places.

    Returns:
        The rounded value.
    """
    from lionagi.libs.parse.to_num import apply_precision

    return apply_precision(value=value, precision=precision)


def parse_number(type_and_value: tuple[str, str]) -> float | complex:
    """Parse string to numeric value based on pattern type.

    Args:
        type_and_value: Tuple of (pattern_type, matched_value).

    Returns:
        The parsed numeric value.

    Raises:
        ValueError: If parsing fails.
    """
    from lionagi.libs.parse.to_num import parse_number

    return parse_number(type_and_value=type_and_value)


def breakdown_pydantic_annotation(
    model: type[B], max_depth: int | None = None, current_depth: int = 0
) -> dict[str, Any]:
    from lionagi.libs.schema.breakdown_pydantic_annotation import (
        breakdown_pydantic_annotation,
    )

    return breakdown_pydantic_annotation(
        model=model,
        max_depth=max_depth,
        current_depth=current_depth,
    )


def run_package_manager_command(
    args: Sequence[str],
) -> subprocess.CompletedProcess[bytes]:
    """Run a package manager command, using uv if available, otherwise falling back to pip."""
    from lionagi.libs.package.imports import run_package_manager_command

    return run_package_manager_command(args=args)


def check_import(
    package_name: str,
    module_name: str | None = None,
    import_name: str | None = None,
    pip_name: str | None = None,
    attempt_install: bool = True,
    error_message: str = "",
):
    """
    Check if a package is installed, attempt to install if not.

    Args:
        package_name: The name of the package to check.
        module_name: The specific module to import (if any).
        import_name: The specific name to import from the module (if any).
        pip_name: The name to use for pip installation (if different).
        attempt_install: Whether to attempt installation if not found.
        error_message: Custom error message to use if package not found.

    Raises:
        ImportError: If the package is not found and not installed.
        ValueError: If the import fails after installation attempt.
    """
    from lionagi.libs.package.imports import check_import

    return check_import(
        package_name=package_name,
        module_name=module_name,
        import_name=import_name,
        pip_name=pip_name,
        attempt_install=attempt_install,
        error_message=error_message,
    )


def import_module(
    package_name: str,
    module_name: str = None,
    import_name: str | list = None,
) -> Any:
    """
    Import a module by its path.

    Args:
        module_path: The path of the module to import.

    Returns:
        The imported module.

    Raises:
        ImportError: If the module cannot be imported.
    """
    from lionagi.libs.package.imports import import_module

    return import_module(
        package_name=package_name,
        module_name=module_name,
        import_name=import_name,
    )


def install_import(
    package_name: str,
    module_name: str | None = None,
    import_name: str | None = None,
    pip_name: str | None = None,
):
    """
    Attempt to import a package, installing it if not found.

    Args:
        package_name: The name of the package to import.
        module_name: The specific module to import (if any).
        import_name: The specific name to import from the module (if any).
        pip_name: The name to use for pip installation (if different).

    Raises:
        ImportError: If the package cannot be imported or installed.
        subprocess.CalledProcessError: If pip installation fails.
    """
    from lionagi.libs.package.imports import install_import

    install_import(
        package_name=package_name,
        module_name=module_name,
        import_name=import_name,
        pip_name=pip_name,
    )


def is_import_installed(package_name: str) -> bool:
    """
    Check if a package is installed.

    Args:
        package_name: The name of the package to check.

    Returns:
        bool: True if the package is installed, False otherwise.
    """
    return importlib.util.find_spec(package_name) is not None


def read_image_to_base64(image_path: str | Path) -> str:
    from lionagi.libs.file.file_util import FileUtil

    return FileUtil.read_image_to_base64(image_path=image_path)


def pdf_to_images(
    pdf_path: str, output_folder: str, dpi: int = 300, fmt: str = "jpeg"
) -> list:
    """
    Convert a PDF file into images, one image per page.

    Args:
        pdf_path (str): Path to the input PDF file.
        output_folder (str): Directory to save the output images.
        dpi (int): Dots per inch (resolution) for conversion (default: 300).
        fmt (str): Image format (default: 'jpeg'). Use 'png' if preferred.

    Returns:
        list: A list of file paths for the saved images.
    """
    from lionagi.libs.file.file_util import FileUtil

    return FileUtil.pdf_to_images(
        pdf_path=pdf_path,
        output_folder=output_folder,
        dpi=dpi,
        fmt=fmt,
    )
