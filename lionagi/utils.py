# Copyright (c) 2023 - 2025, HaiyangLi <quantocean.li at gmail dot com>
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import contextlib
import copy as _copy
import dataclasses
import functools
import importlib.util
import json
import logging
import re
import shutil
import subprocess
import sys
import time as t_
import uuid
import xml.etree.ElementTree as ET
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
from decimal import Decimal
from enum import Enum
from functools import lru_cache, partial
from inspect import isclass
from pathlib import Path
from typing import (
    Any,
    Literal,
    TypedDict,
    TypeVar,
    get_args,
    get_origin,
    overload,
)

import anyio
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

from ._utils import (
    DataClass,
    KeysDict,
    Params,
    ToListParams,
    Undefined,
    UndefinedType,
    alcall,
    bcall,
    hash_dict,
    is_coro_func,
    lcall,
    to_list,
)

UNDEFINED = Undefined
"""Backwards compatibility alias for Undefined sentinel."""


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


async def custom_error_handler(
    error: Exception, error_map: dict[type, Callable[[Exception], None]]
) -> None:
    if type(error) in error_map:
        if is_coro_func(error_map[type(error)]):
            return await error_map[type(error)](error)
        return error_map[type(error)](error)
    logging.error(f"Unhandled error: {error}")
    raise error


# --- JSON and XML Conversion ---


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
    # Validate input
    if isinstance(input_, (list, tuple)):
        raise TypeError("Input cannot be a sequence")

    # Handle boolean input
    if isinstance(input_, bool):
        return validate_num_type(num_type)(input_)

    # Handle direct numeric input
    if isinstance(input_, (int, float, complex, Decimal)):
        inferred_type = type(input_)
        if isinstance(input_, Decimal):
            inferred_type = float
        value = float(input_) if not isinstance(input_, complex) else input_
        value = apply_bounds(value, upper_bound, lower_bound)
        value = apply_precision(value, precision)
        return convert_type(value, validate_num_type(num_type), inferred_type)

    # Convert input to string and extract numbers
    input_str = str(input_)
    number_matches = extract_numbers(input_str)

    if not number_matches:
        raise ValueError(f"No valid numbers found in: {input_str}")

    # Process numbers
    results = []
    target_type = validate_num_type(num_type)

    number_matches = (
        number_matches[:num_count]
        if num_count < len(number_matches)
        else number_matches
    )

    for type_and_value in number_matches:
        try:
            # Infer appropriate type
            inferred_type = infer_type(type_and_value)

            # Parse to numeric value
            value = parse_number(type_and_value)

            # Apply bounds if not complex
            value = apply_bounds(value, upper_bound, lower_bound)

            # Apply precision
            value = apply_precision(value, precision)

            # Convert to target type if different from inferred
            value = convert_type(value, target_type, inferred_type)

            results.append(value)

        except Exception as e:
            if len(type_and_value) == 2:
                raise type(e)(
                    f"Error processing {type_and_value[1]}: {str(e)}"
                )
            raise type(e)(f"Error processing {type_and_value}: {str(e)}")

    if results and num_count == 1:
        return results[0]
    return results


def extract_numbers(text: str) -> list[tuple[str, str]]:
    """Extract numeric values from text using ordered regex patterns.

    Args:
        text: The text to extract numbers from.

    Returns:
        List of tuples containing (pattern_type, matched_value).
    """
    combined_pattern = "|".join(PATTERNS.values())
    matches = re.finditer(combined_pattern, text, re.IGNORECASE)
    numbers = []

    for match in matches:
        value = match.group()
        # Check which pattern matched
        for pattern_name, pattern in PATTERNS.items():
            if re.fullmatch(pattern, value, re.IGNORECASE):
                numbers.append((pattern_name, value))
                break

    return numbers


def validate_num_type(num_type: NUM_TYPES) -> type:
    """Validate and normalize numeric type specification.

    Args:
        num_type: The numeric type to validate.

    Returns:
        The normalized Python type object.

    Raises:
        ValueError: If the type specification is invalid.
    """
    if isinstance(num_type, str):
        if num_type not in TYPE_MAP:
            raise ValueError(f"Invalid number type: {num_type}")
        return TYPE_MAP[num_type]

    if num_type not in (int, float, complex):
        raise ValueError(f"Invalid number type: {num_type}")
    return num_type


def infer_type(value: tuple[str, str]) -> type:
    """Infer appropriate numeric type from value.

    Args:
        value: Tuple of (pattern_type, matched_value).

    Returns:
        The inferred Python type.
    """
    pattern_type, _ = value
    if pattern_type in ("complex", "complex_sci", "pure_imaginary"):
        return complex
    return float


def convert_special(value: str) -> float:
    """Convert special float values (inf, -inf, nan).

    Args:
        value: The string value to convert.

    Returns:
        The converted float value.
    """
    value = value.lower()
    if "infinity" in value or "inf" in value:
        return float("-inf") if value.startswith("-") else float("inf")
    return float("nan")


def convert_percentage(value: str) -> float:
    """Convert percentage string to float.

    Args:
        value: The percentage string to convert.

    Returns:
        The converted float value.

    Raises:
        ValueError: If the percentage value is invalid.
    """
    try:
        return float(value.rstrip("%")) / 100
    except ValueError as e:
        raise ValueError(f"Invalid percentage value: {value}") from e


def convert_complex(value: str) -> complex:
    """Convert complex number string to complex.

    Args:
        value: The complex number string to convert.

    Returns:
        The converted complex value.

    Raises:
        ValueError: If the complex number is invalid.
    """
    try:
        # Handle pure imaginary numbers
        if value.endswith("j") or value.endswith("J"):
            if value in ("j", "J"):
                return complex(0, 1)
            if value in ("+j", "+J"):
                return complex(0, 1)
            if value in ("-j", "-J"):
                return complex(0, -1)
            if "+" not in value and "-" not in value[1:]:
                # Pure imaginary number
                imag = float(value[:-1] or "1")
                return complex(0, imag)

        return complex(value.replace(" ", ""))
    except ValueError as e:
        raise ValueError(f"Invalid complex number: {value}") from e


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
    try:
        # If no specific type requested, use inferred type
        if target_type is float and inferred_type is complex:
            return value

        # Handle explicit type conversions
        if target_type is int and isinstance(value, complex):
            raise TypeError("Cannot convert complex number to int")
        return target_type(value)
    except (ValueError, TypeError) as e:
        raise TypeError(
            f"Cannot convert {value} to {target_type.__name__}"
        ) from e


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
    if isinstance(value, complex):
        return value

    if upper_bound is not None and value > upper_bound:
        raise ValueError(f"Value {value} exceeds upper bound {upper_bound}")
    if lower_bound is not None and value < lower_bound:
        raise ValueError(f"Value {value} below lower bound {lower_bound}")
    return value


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
    if precision is None or isinstance(value, complex):
        return value
    if isinstance(value, float):
        return round(value, precision)
    return value


def parse_number(type_and_value: tuple[str, str]) -> float | complex:
    """Parse string to numeric value based on pattern type.

    Args:
        type_and_value: Tuple of (pattern_type, matched_value).

    Returns:
        The parsed numeric value.

    Raises:
        ValueError: If parsing fails.
    """
    num_type, value = type_and_value
    value = value.strip()

    try:
        if num_type == "special":
            return convert_special(value)

        if num_type == "percentage":
            return convert_percentage(value)

        if num_type == "fraction":
            if "/" not in value:
                raise ValueError(f"Invalid fraction: {value}")
            if value.count("/") > 1:
                raise ValueError(f"Invalid fraction: {value}")
            num, denom = value.split("/")
            if not (num.strip("-").isdigit() and denom.isdigit()):
                raise ValueError(f"Invalid fraction: {value}")
            denom_val = float(denom)
            if denom_val == 0:
                raise ValueError("Division by zero")
            return float(num) / denom_val
        if num_type in ("complex", "complex_sci", "pure_imaginary"):
            return convert_complex(value)
        if num_type == "scientific":
            if "e" not in value.lower():
                raise ValueError(f"Invalid scientific notation: {value}")
            parts = value.lower().split("e")
            if len(parts) != 2:
                raise ValueError(f"Invalid scientific notation: {value}")
            if not (parts[1].lstrip("+-").isdigit()):
                raise ValueError(f"Invalid scientific notation: {value}")
            return float(value)
        if num_type == "decimal":
            return float(value)

        raise ValueError(f"Unknown number type: {num_type}")
    except Exception as e:
        # Preserve the specific error type but wrap with more context
        raise type(e)(f"Failed to parse {value} as {num_type}: {str(e)}")


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


def run_package_manager_command(
    args: Sequence[str],
) -> subprocess.CompletedProcess[bytes]:
    """Run a package manager command, using uv if available, otherwise falling back to pip."""
    # Check if uv is available in PATH
    uv_path = shutil.which("uv")

    if uv_path:
        # Use uv if available
        try:
            return subprocess.run(
                [uv_path] + list(args),
                check=True,
                capture_output=True,
            )
        except subprocess.CalledProcessError:
            # If uv fails, fall back to pip
            print("uv command failed, falling back to pip...")

    # Fall back to pip
    return subprocess.run(
        [sys.executable, "-m", "pip"] + list(args),
        check=True,
        capture_output=True,
    )


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
    if not is_import_installed(package_name):
        if attempt_install:
            logging.info(
                f"Package {package_name} not found. Attempting to install.",
            )
            try:
                return install_import(
                    package_name=package_name,
                    module_name=module_name,
                    import_name=import_name,
                    pip_name=pip_name,
                )
            except ImportError as e:
                raise ValueError(
                    f"Failed to install {package_name}: {e}"
                ) from e
        else:
            logging.info(
                f"Package {package_name} not found. {error_message}",
            )
            raise ImportError(
                f"Package {package_name} not found. {error_message}",
            )

    return import_module(
        package_name=package_name,
        module_name=module_name,
        import_name=import_name,
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
    try:
        full_import_path = (
            f"{package_name}.{module_name}" if module_name else package_name
        )

        if import_name:
            import_name = (
                [import_name]
                if not isinstance(import_name, list)
                else import_name
            )
            a = __import__(
                full_import_path,
                fromlist=import_name,
            )
            if len(import_name) == 1:
                return getattr(a, import_name[0])
            return [getattr(a, name) for name in import_name]
        else:
            return __import__(full_import_path)

    except ImportError as e:
        raise ImportError(
            f"Failed to import module {full_import_path}: {e}"
        ) from e


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
    pip_name = pip_name or package_name

    try:
        return import_module(
            package_name=package_name,
            module_name=module_name,
            import_name=import_name,
        )
    except ImportError:
        logging.info(f"Installing {pip_name}...")
        try:
            run_package_manager_command(["install", pip_name])
            return import_module(
                package_name=package_name,
                module_name=module_name,
                import_name=import_name,
            )
        except subprocess.CalledProcessError as e:
            raise ImportError(f"Failed to install {pip_name}: {e}") from e
        except ImportError as e:
            raise ImportError(
                f"Failed to import {pip_name} after installation: {e}"
            ) from e


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
    import os

    convert_from_path = check_import(
        "pdf2image", import_name="convert_from_path"
    )

    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Convert PDF to a list of PIL Image objects
    images = convert_from_path(pdf_path, dpi=dpi)

    saved_paths = []
    for i, image in enumerate(images):
        # Construct the output file name
        image_file = os.path.join(output_folder, f"page_{i + 1}.{fmt}")
        image.save(image_file, fmt.upper())
        saved_paths.append(image_file)

    return saved_paths
