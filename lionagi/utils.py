# Copyright (c) 2023 - 2025, HaiyangLi <quantocean.li at gmail dot com>
#
# SPDX-License-Identifier: Apache-2.0

import copy as _copy
import logging
import types
import uuid
from collections.abc import AsyncGenerator, Callable, Iterable, Mapping
from datetime import datetime, timezone
from inspect import isclass
from pathlib import Path
from typing import (
    Annotated,
    Any,
    Literal,
    TypeVar,
    Union,
    get_args,
    get_origin,
)

from pydantic import BaseModel
from typing_extensions import deprecated

from .ln import (
    extract_json,
    fuzzy_json,
    get_bins,
    hash_dict,
    import_module,
    is_coro_func,
    is_import_installed,
    to_dict,
    to_list,
)
from .ln.types import (
    DataClass,
    Enum,
    KeysDict,
    MaybeSentinel,
    MaybeUndefined,
    MaybeUnset,
    Params,
    Undefined,
    UndefinedType,
    Unset,
    UnsetType,
)
from .settings import Settings

R = TypeVar("R")
T = TypeVar("T")
B = TypeVar("B", bound=BaseModel)

logger = logging.getLogger(__name__)

UNDEFINED = Undefined

to_json = extract_json
fuzzy_parse_json = fuzzy_json

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
    "get_bins",
    "EventStatus",
    "logger",
    "max_concurrent",
    "force_async",
    "breakdown_pydantic_annotation",
    "run_package_manager_command",
    "StringEnum",
    "Enum",
    "hash_dict",
    "is_union_type",
    "union_members",
    "to_json",
    "Unset",
    "UnsetType",
    "Undefined",
    "MaybeSentinel",
    "MaybeUndefined",
    "MaybeUnset",
    "is_import_installed",
    "import_module",
    "to_dict",
)


# --- General Global Utilities Types ---
@deprecated("String Enum is deprecated, use `Enum` instead.")
class StringEnum(str, Enum):
    pass


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


def is_union_type(tp) -> bool:
    """True for typing.Union[...] and PEP 604 unions (A | B)."""
    origin = get_origin(tp)
    return origin is Union or origin is getattr(
        types, "UnionType", object()
    )  # Py3.10+


NoneType = type(None)
_UnionType = getattr(types, "UnionType", None)  # for A | B (PEP 604)


def _unwrap_annotated(tp):
    while get_origin(tp) is Annotated:
        tp = get_args(tp)[0]
    return tp


def union_members(
    tp, *, unwrap_annotated: bool = True, drop_none: bool = False
) -> tuple[type, ...]:
    """Return the member types of a Union (typing.Union or A|B). Empty tuple if not a Union."""
    tp = _unwrap_annotated(tp) if unwrap_annotated else tp
    origin = get_origin(tp)
    if origin is not Union and origin is not _UnionType:
        return ()
    members = get_args(tp)
    if unwrap_annotated:
        members = tuple(_unwrap_annotated(m) for m in members)
    if drop_none:
        members = tuple(m for m in members if m is not NoneType)
    return members


async def custom_error_handler(
    error: Exception, error_map: dict[type, Callable[[Exception], None]]
) -> None:
    if type(error) in error_map:
        if is_coro_func(error_map[type(error)]):
            return await error_map[type(error)](error)
        return error_map[type(error)](error)
    logging.error(f"Unhandled error: {error}")
    raise error


@deprecated(
    "Use `lionagi.ln.lcall` instead, function signature has changed, this will be removed in future versions."
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
    from lionagi.ln import lcall as _lcall

    return _lcall(
        input_,
        func,
        *args,
        input_flatten=sanitize_input,
        input_dropna=sanitize_input,
        input_flatten_tuple_set=flatten_tuple_set,
        input_unique=unique_input,
        input_use_values=use_input_values,
        output_flatten=flatten,
        output_dropna=dropna,
        output_flatten_tuple_set=flatten_tuple_set,
        output_unique=unique_output,
        **kwargs,
    )


@deprecated(
    "Use `lionagi.ln.alcall` instead, function signature has changed, this will be removed in future versions."
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
    max_concurrent: int | None = None,
    throttle_period: float | None = None,
    flatten: bool = False,
    dropna: bool = False,
    unique_output: bool = False,
    flatten_tuple_set: bool = False,
    **kwargs: Any,
) -> list[T]:
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
        max_concurrent (int | None): Maximum number of concurrent operations.
        throttle_period (float | None): Delay after each completed operation.
        flatten (bool): Flatten the final result if True.
        dropna (bool): Remove None values from the final result if True.
        unique_output (bool): Deduplicate the output if True.
        flatten_tuple_set (bool): Tuples and sets will be flattened if True.
        **kwargs: Additional arguments passed to func.

    Returns:
        list[T]: The processed results.

    Raises:
        asyncio.TimeoutError: If a call times out and no default is provided.
        Exception: If retries are exhausted and no default is provided.
    """
    from .ln._async_call import alcall as _alcall

    return await _alcall(
        input_,
        func,
        input_flatten=sanitize_input,
        input_dropna=sanitize_input,
        input_unique=unique_input,
        input_flatten_tuple_set=flatten_tuple_set,
        output_flatten=flatten,
        output_dropna=dropna,
        output_unique=unique_output,
        output_flatten_tuple_set=flatten_tuple_set,
        delay_before_start=initial_delay,
        retry_initial_deplay=retry_delay,
        retry_backoff=backoff_factor,
        retry_default=retry_default,
        retry_timeout=retry_timeout,
        retry_attempts=num_retries,
        max_concurrent=max_concurrent,
        throttle_period=throttle_period,
        **kwargs,
    )


@deprecated(
    "Use `lionagi.ln.alcall` instead, function signature has changed, this will be removed in future versions."
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
    max_concurrent: int | None = None,
    throttle_period: float | None = None,
    flatten: bool = False,
    dropna: bool = False,
    unique_output: bool = False,
    flatten_tuple_set: bool = False,
    **kwargs: Any,
) -> AsyncGenerator[list[T | tuple[T, float]], None]:
    from .ln._async_call import bcall as _bcall

    async for i in _bcall(
        input_,
        func,
        batch_size,
        input_flatten=sanitize_input,
        input_dropna=sanitize_input,
        input_unique=unique_input,
        input_flatten_tuple_set=flatten_tuple_set,
        output_flatten=flatten,
        output_dropna=dropna,
        output_unique=unique_output,
        output_flatten_tuple_set=flatten_tuple_set,
        delay_before_start=initial_delay,
        retry_initial_deplay=retry_delay,
        retry_backoff=backoff_factor,
        retry_default=retry_default,
        retry_timeout=retry_timeout,
        retry_attempts=num_retries,
        max_concurrent=max_concurrent,
        throttle_period=throttle_period,
        **kwargs,
    ):
        yield i


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
    if "/" in filename:
        sub_dir, filename = filename.split("/")[:-1], filename.split("/")[-1]
        directory = Path(directory) / "/".join(sub_dir)

    if "\\" in filename:
        raise ValueError("Filename cannot contain directory separators.")

    directory = Path(directory)

    # Extract name and extension from filename if present
    if "." in filename:
        name, ext = filename.rsplit(".", 1)
    else:
        name, ext = filename, extension

    # Ensure extension has a single leading dot
    ext = f".{ext.lstrip('.')}" if ext else ""

    # Add timestamp if requested
    if timestamp:
        ts_str = datetime.now().strftime(timestamp_format or "%Y%m%d%H%M%S")
        name = f"{ts_str}_{name}" if time_prefix else f"{name}_{ts_str}"

    # Add random suffix if requested
    if random_hash_digits > 0:
        # Use UUID4 and truncate its hex for random suffix
        random_suffix = uuid.uuid4().hex[:random_hash_digits]
        name = f"{name}-{random_suffix}"

    full_path = directory / f"{name}{ext}"

    # Check if file or directory existence
    full_path.parent.mkdir(parents=True, exist_ok=dir_exist_ok)
    if full_path.exists() and not file_exist_ok:
        raise FileExistsError(
            f"File {full_path} already exists and file_exist_ok is False."
        )

    return full_path


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


def breakdown_pydantic_annotation(
    model: type[B], max_depth: int | None = None, current_depth: int = 0
) -> dict[str, Any]:
    if not _is_pydantic_model(model):
        raise TypeError("Input must be a Pydantic model")

    if max_depth is not None and current_depth >= max_depth:
        raise RecursionError("Maximum recursion depth reached")

    out: dict[str, Any] = {}
    for k, v in model.__annotations__.items():
        origin = get_origin(v)
        if _is_pydantic_model(v):
            out[k] = breakdown_pydantic_annotation(
                v, max_depth, current_depth + 1
            )
        elif origin is list:
            args = get_args(v)
            if args and _is_pydantic_model(args[0]):
                out[k] = [
                    breakdown_pydantic_annotation(
                        args[0], max_depth, current_depth + 1
                    )
                ]
            else:
                out[k] = [args[0] if args else Any]
        else:
            out[k] = v

    return out


def _is_pydantic_model(x: Any) -> bool:
    try:
        return isclass(x) and issubclass(x, BaseModel)
    except TypeError:
        return False
