from __future__ import annotations

from collections.abc import Callable
from dataclasses import MISSING
from typing import Any, Final, Literal, TypeAlias, TypeGuard, TypeVar

__all__ = (
    "Undefined",
    "Unset",
    "MaybeUndefined",
    "MaybeUnset",
    "MaybeSentinel",
    "AdditionalSentinels",
    "SingletonType",
    "UndefinedType",
    "UnsetType",
    "is_sentinel",
    "is_undefined",
    "is_unset",
    "not_sentinel",
    "T",
)

T = TypeVar("T")


class _SingletonMeta(type):
    """Metaclass ensuring exactly one instance per subclass for safe `is` identity checks."""

    _cache: dict[type, SingletonType] = {}

    def __call__(cls, *a, **kw):
        if cls not in cls._cache:
            cls._cache[cls] = super().__call__(*a, **kw)
        return cls._cache[cls]


class SingletonType(metaclass=_SingletonMeta):
    """Base class for singleton sentinel types."""

    __slots__: tuple[str, ...] = ()

    def __deepcopy__(self, memo):  # copy & deepcopy both noop
        return self

    def __copy__(self):
        return self

    # concrete classes *must* override the two methods below
    def __bool__(self) -> bool: ...
    def __repr__(self) -> str: ...


class UndefinedType(SingletonType):
    """Sentinel for a key or field entirely absent from a namespace."""

    __slots__ = ()

    def __bool__(self) -> Literal[False]:
        return False

    def __repr__(self) -> Literal["Undefined"]:
        return "Undefined"

    def __str__(self) -> Literal["Undefined"]:
        return "Undefined"

    def __reduce__(self):
        return "Undefined"  # preserves singleton identity across pickle


class UnsetType(SingletonType):
    """Sentinel distinguishing "not provided" from None."""

    __slots__ = ()

    def __bool__(self) -> Literal[False]:
        return False

    def __repr__(self) -> Literal["Unset"]:
        return "Unset"

    def __str__(self) -> Literal["Unset"]:
        return "Unset"

    def __reduce__(self):
        return "Unset"  # preserves singleton identity across pickle


Undefined: Final = UndefinedType()
Unset: Final = UnsetType()

MaybeUndefined: TypeAlias = T | UndefinedType
MaybeUnset: TypeAlias = T | UnsetType
MaybeSentinel: TypeAlias = T | UndefinedType | UnsetType

_EMPTY_TUPLE = (tuple(), set(), frozenset(), dict(), list(), "")
AdditionalSentinels = Literal["none", "empty", "pydantic", "dataclass"]


def is_undefined(value: Any) -> bool:
    return isinstance(value, UndefinedType)


def is_unset(value: Any) -> bool:
    return isinstance(value, UnsetType)


def _is_builtin_sentinel(value: Any) -> bool:
    return is_undefined(value) or is_unset(value)


def _is_pydantic_sentinel(value: Any) -> bool:
    from pydantic_core import PydanticUndefinedType

    return isinstance(value, PydanticUndefinedType)


SENTINEL_HANDLERS: dict[str, Callable[[Any], bool]] = {
    "none": lambda value: value is None,
    "empty": lambda value: value in _EMPTY_TUPLE,
    "pydantic": _is_pydantic_sentinel,
    "dataclass": lambda value: value is MISSING,
}


def is_sentinel(
    value: Any,
    *,
    additions: set[AdditionalSentinels] = frozenset(),
    none_as_sentinel: bool = False,
    empty_as_sentinel: bool = False,
) -> bool:
    active = set(additions)
    if none_as_sentinel:
        active.add("none")
    if empty_as_sentinel:
        active.add("empty")
    if _is_builtin_sentinel(value):
        return True
    return any(SENTINEL_HANDLERS[key](value) for key in active)


def not_sentinel(
    value: T | UndefinedType | UnsetType,
    none_as_sentinel: bool = False,
    empty_as_sentinel: bool = False,
    *,
    additions: set[AdditionalSentinels] = frozenset(),
) -> TypeGuard[T]:
    return not is_sentinel(
        value,
        additions=additions,
        none_as_sentinel=none_as_sentinel,
        empty_as_sentinel=empty_as_sentinel,
    )
