import contextlib
import copy as _copy
import importlib
import importlib.util
import uuid
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path as StdPath
from types import UnionType
from typing import Any, ParamSpec, TypeVar, Union, get_args, get_origin
from uuid import UUID

from anyio import Path as AsyncPath

P = ParamSpec("P")
R = TypeVar("R")
T = TypeVar("T")

__all__ = (
    "acreate_path",
    "async_synchronized",
    "coerce_created_at",
    "copy",
    "create_path",
    "extract_types",
    "get_bins",
    "import_module",
    "is_import_installed",
    "load_type_from_string",
    "now_utc",
    "register_type_prefix",
    "synchronized",
    "to_uuid",
)


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


async def acreate_path(
    directory: StdPath | AsyncPath | str,
    filename: str,
    extension: str | None = None,
    timestamp: bool = False,
    dir_exist_ok: bool = True,
    file_exist_ok: bool = False,
    time_prefix: bool = False,
    timestamp_format: str | None = None,
    random_hash_digits: int = 0,
    timeout: float | None = None,
) -> AsyncPath:
    """Async variant of create_path with optional timeout."""
    from .concurrency import move_on_after

    async def _impl() -> AsyncPath:
        nonlocal directory, filename

        if "/" in filename:
            sub_dir, filename = (
                filename.split("/")[:-1],
                filename.split("/")[-1],
            )
            directory = AsyncPath(directory) / "/".join(sub_dir)

        if "\\" in filename:
            raise ValueError("Filename cannot contain directory separators.")

        directory = AsyncPath(directory)
        if "." in filename:
            name, ext = filename.rsplit(".", 1)
        else:
            name = filename
            ext = extension or ""
        ext = f".{ext.lstrip('.')}" if ext else ""

        if timestamp:
            ts_str = datetime.now().strftime(timestamp_format or "%Y%m%d%H%M%S")
            name = f"{ts_str}_{name}" if time_prefix else f"{name}_{ts_str}"

        if random_hash_digits > 0:
            random_suffix = uuid.uuid4().hex[:random_hash_digits]
            name = f"{name}-{random_suffix}"

        full_path = directory / f"{name}{ext}"

        await full_path.parent.mkdir(parents=True, exist_ok=dir_exist_ok)

        if await full_path.exists() and not file_exist_ok:
            raise FileExistsError(
                f"File {full_path} already exists and file_exist_ok is False."
            )

        return full_path

    if timeout is None:
        return await _impl()

    with move_on_after(timeout) as cancel_scope:
        result = await _impl()
    if cancel_scope.cancelled_caught:
        raise TimeoutError(f"acreate_path timed out after {timeout}s")
    return result


def create_path(
    directory: StdPath | str,
    filename: str,
    extension: str | None = None,
    timestamp: bool = False,
    dir_exist_ok: bool = True,
    file_exist_ok: bool = False,
    time_prefix: bool = False,
    timestamp_format: str | None = None,
    random_hash_digits: int = 0,
) -> StdPath:
    if "/" in filename:
        parts = filename.split("/")
        directory = StdPath(directory).joinpath(*parts[:-1])
        filename = parts[-1]

    if "\\" in filename:
        raise ValueError("Filename cannot contain directory separators.")

    directory = StdPath(directory)
    if "." in filename:
        name, ext = filename.rsplit(".", 1)
    else:
        name = filename
        ext = extension or ""
    ext = f".{ext.lstrip('.')}" if ext else ""

    if timestamp:
        ts_str = datetime.now().strftime(timestamp_format or "%Y%m%d%H%M%S")
        name = f"{ts_str}_{name}" if time_prefix else f"{name}_{ts_str}"

    if random_hash_digits > 0:
        name = f"{name}-{uuid.uuid4().hex[:random_hash_digits]}"

    full_path = directory / f"{name}{ext}"
    full_path.parent.mkdir(parents=True, exist_ok=dir_exist_ok)

    if full_path.exists() and not file_exist_ok:
        raise FileExistsError(
            f"File {full_path} already exists and file_exist_ok is False."
        )

    return full_path


def get_bins(input_: list[str], upper: int) -> list[list[int]]:
    """Group string indices into bins whose cumulative char length stays below upper."""
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


def import_module(
    package_name: str,
    module_name: str = None,
    import_name: str | list = None,
) -> Any:
    try:
        full_import_path = (
            f"{package_name}.{module_name}" if module_name else package_name
        )

        if import_name:
            import_name = (
                [import_name] if not isinstance(import_name, list) else import_name
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
        raise ImportError(f"Failed to import module {full_import_path}: {e}") from e


def is_import_installed(package_name: str) -> bool:
    return importlib.util.find_spec(package_name) is not None


_TYPE_CACHE: dict[str, type] = {}

_DEFAULT_ALLOWED_PREFIXES: frozenset[str] = frozenset({"lionagi."})
_ALLOWED_MODULE_PREFIXES: set[str] = set(_DEFAULT_ALLOWED_PREFIXES)


def register_type_prefix(prefix: str) -> None:
    """Register a module prefix for dynamic type loading; prefix must end with '.' to prevent prefix-injection attacks."""
    if not prefix.endswith("."):
        raise ValueError(f"Prefix must end with '.': {prefix}")
    _ALLOWED_MODULE_PREFIXES.add(prefix)


def load_type_from_string(type_str: str) -> type:
    """Load a type from a fully-qualified string path; only allowlisted prefixes are permitted."""
    if type_str in _TYPE_CACHE:
        return _TYPE_CACHE[type_str]

    if not isinstance(type_str, str):
        raise ValueError(f"Expected string, got {type(type_str)}")

    if "." not in type_str:
        raise ValueError(f"Invalid type path (no module): {type_str}")

    if not any(type_str.startswith(prefix) for prefix in _ALLOWED_MODULE_PREFIXES):
        raise ValueError(
            f"Module '{type_str}' not in allowed prefixes: {sorted(_ALLOWED_MODULE_PREFIXES)}"
        )

    try:
        module_path, class_name = type_str.rsplit(".", 1)
        module = importlib.import_module(module_path)
        if module is None:
            raise ImportError(f"Module '{module_path}' not found")

        type_class = getattr(module, class_name)
        if not isinstance(type_class, type):
            raise ValueError(f"'{type_str}' is not a type")

        _TYPE_CACHE[type_str] = type_class
        return type_class

    except (ValueError, ImportError, AttributeError) as e:
        raise ValueError(f"Failed to load type '{type_str}': {e}") from e


def extract_types(item_type: Any) -> set[type]:
    """Flatten a type annotation (Union, list, set, or bare type) into a set of concrete types."""

    def is_union(t: Any) -> bool:
        origin = get_origin(t)
        return origin is Union or isinstance(t, UnionType)

    extracted: set[type] = set()

    if isinstance(item_type, set):
        for t in item_type:
            if is_union(t):
                extracted.update(get_args(t))
            else:
                extracted.add(t)
        return extracted

    if isinstance(item_type, list):
        for t in item_type:
            if is_union(t):
                extracted.update(get_args(t))
            else:
                extracted.add(t)
        return extracted

    if is_union(item_type):
        return set(get_args(item_type))

    return {item_type}


def to_uuid(value: Any) -> UUID:
    """Convert a UUID, UUID string, or object with .id to a UUID instance."""
    if isinstance(value, UUID):
        return value
    if isinstance(value, str):
        return UUID(value)
    if hasattr(value, "id"):
        v = value.id
        if isinstance(v, UUID):
            return v
        if isinstance(v, str):
            return UUID(v)
    raise ValueError("Cannot get ID from item.")


def coerce_created_at(v: Any) -> datetime:
    """Coerce a datetime, Unix timestamp, or ISO string to a UTC-aware datetime."""
    if isinstance(v, datetime):
        return v.replace(tzinfo=timezone.utc) if v.tzinfo is None else v

    if isinstance(v, (int, float)):
        return datetime.fromtimestamp(v, tz=timezone.utc)

    if isinstance(v, str):
        with contextlib.suppress(ValueError):
            return datetime.fromtimestamp(float(v), tz=timezone.utc)
        with contextlib.suppress(ValueError):
            return datetime.fromisoformat(v)
        raise ValueError(f"String '{v}' is neither timestamp nor ISO format")

    raise ValueError(f"Expected datetime/timestamp/string, got {type(v).__name__}")


def synchronized(func: Callable[P, R]) -> Callable[P, R]:
    """Wrap a method so it acquires self._lock before executing."""

    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        self = args[0]
        with self._lock:
            return func(*args, **kwargs)

    return wrapper


def async_synchronized(
    func: Callable[P, Awaitable[R]],
) -> Callable[P, Awaitable[R]]:
    """Wrap an async method so it acquires self._async_lock before executing."""

    @wraps(func)
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        self = args[0]
        async with self._async_lock:  # type: ignore[attr-defined]
            return await func(*args, **kwargs)

    return wrapper


def copy(obj: T, /, *, deep: bool = True, num: int = 1) -> T | list[T]:
    if num < 1:
        raise ValueError("Number of copies must be at least 1")
    copy_func = _copy.deepcopy if deep else _copy.copy
    return [copy_func(obj) for _ in range(num)] if num > 1 else copy_func(obj)
