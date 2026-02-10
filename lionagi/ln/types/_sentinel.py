from __future__ import annotations

from collections.abc import Callable
from typing import Any, Final, Literal, TypeVar, Union

__all__ = (
    "AdditionalSentinels",
    "MaybeSentinel",
    "MaybeUndefined",
    "MaybeUnset",
    "SingletonType",
    "T",
    "Undefined",
    "UndefinedType",
    "Unset",
    "UnsetType",
    "is_sentinel",
    "is_undefined",
    "is_unset",
    "not_sentinel",
)

T = TypeVar("T")


class _SingletonMeta(type):
    """Metaclass that guarantees exactly one instance per subclass."""

    _cache: dict[type, SingletonType] = {}

    def __call__(cls, *a, **kw):
        if cls not in cls._cache:
            cls._cache[cls] = super().__call__(*a, **kw)
        return cls._cache[cls]


class SingletonType(metaclass=_SingletonMeta):
    """Base class for singleton sentinel types.

    Provides consistent interface for sentinel values with:
    - Identity preservation across deepcopy
    - Falsy boolean evaluation
    - Clear string representation
    """

    __slots__: tuple[str, ...] = ()

    def __deepcopy__(self, memo):
        return self

    def __copy__(self):
        return self

    # concrete classes *must* override the two methods below
    def __bool__(self) -> bool: ...
    def __repr__(self) -> str: ...


class UndefinedType(SingletonType):
    """Sentinel for a key or field entirely missing from a namespace.

    Use this when:
    - A field has never been set
    - A key doesn't exist in a mapping
    - A value is conceptually undefined (not just unset)

    Example:
        >>> d = {"a": 1}
        >>> d.get("b", Undefined) is Undefined
        True
    """

    __slots__ = ()

    def __bool__(self) -> Literal[False]:
        return False

    def __repr__(self) -> Literal["Undefined"]:
        return "Undefined"

    def __str__(self) -> Literal["Undefined"]:
        return "Undefined"

    def __reduce__(self):
        """Ensure pickle preservation of singleton identity."""
        return "Undefined"


class UnsetType(SingletonType):
    """Sentinel for a key present but value not yet provided.

    Use this when:
    - A parameter exists but hasn't been given a value
    - Distinguishing between None and "not provided"
    - API parameters that are optional but need explicit handling

    Example:
        >>> def func(param=Unset):
        ...     if param is not Unset:
        ...         # param was explicitly provided
        ...         process(param)
    """

    __slots__ = ()

    def __bool__(self) -> Literal[False]:
        return False

    def __repr__(self) -> Literal["Unset"]:
        return "Unset"

    def __str__(self) -> Literal["Unset"]:
        return "Unset"

    def __reduce__(self):
        """Ensure pickle preservation of singleton identity."""
        return "Unset"


Undefined: Final = UndefinedType()
"""A key or field entirely missing from a namespace"""
Unset: Final = UnsetType()
"""A key present but value not yet provided."""

MaybeUndefined = Union[T, UndefinedType]
MaybeUnset = Union[T, UnsetType]
MaybeSentinel = Union[T, UndefinedType, UnsetType]

AdditionalSentinels = Literal["none", "empty", "pydantic", "dataclass"]

_EMPTY_TUPLE = (tuple(), set(), frozenset(), dict(), list(), "")


def _is_builtin_sentinel(value: Any) -> bool:
    return isinstance(value, (UndefinedType, UnsetType))


def _is_none(value: Any) -> bool:
    return value is None


def _is_empty(value: Any) -> bool:
    return value in _EMPTY_TUPLE


def _is_pydantic_sentinel(value: Any) -> bool:
    from pydantic_core import PydanticUndefinedType

    return isinstance(value, PydanticUndefinedType)


def _is_dataclass_missing(value: Any) -> bool:
    from dataclasses import MISSING

    return value is MISSING


SENTINEL_HANDLERS: dict[str, Callable[[Any], bool]] = {
    "none": _is_none,
    "empty": _is_empty,
    "pydantic": _is_pydantic_sentinel,
    "dataclass": _is_dataclass_missing,
}

_HANDLE_SEQUENCE: tuple[str, ...] = ("none", "empty", "pydantic", "dataclass")


def is_sentinel(
    value: Any,
    additions: frozenset[str] | set[str] | bool = frozenset(),
    *,
    # backwards compat — will be removed in future
    none_as_sentinel: bool = False,
    empty_as_sentinel: bool = False,
) -> bool:
    """Check if a value is any sentinel (Undefined or Unset).

    Args:
        value: Any value to check.
        additions: Extra categories to treat as sentinel:
            "none" — treat None as sentinel
            "empty" — treat empty containers/strings as sentinel
            "pydantic" — treat PydanticUndefined as sentinel
            "dataclass" — treat dataclasses.MISSING as sentinel
        none_as_sentinel: Deprecated. Use additions={"none"}.
        empty_as_sentinel: Deprecated. Use additions={"empty"}.
    """
    if _is_builtin_sentinel(value):
        return True
    # backwards compat: bool positional was old none_as_sentinel
    if isinstance(additions, bool):
        none_as_sentinel = additions
        additions = frozenset()
    # backwards compat: convert bools to additions
    if none_as_sentinel or empty_as_sentinel:
        merged = set(additions) if additions else set()
        if none_as_sentinel:
            merged.add("none")
        if empty_as_sentinel:
            merged.add("empty")
        additions = frozenset(merged)
    for key in _HANDLE_SEQUENCE:
        if key in additions and SENTINEL_HANDLERS[key](value):
            return True
    return False


def is_undefined(value: Any) -> bool:
    """Check if value is the Undefined sentinel."""
    return isinstance(value, UndefinedType)


def is_unset(value: Any) -> bool:
    """Check if value is the Unset sentinel."""
    return isinstance(value, UnsetType)


def not_sentinel(
    value: Any,
    additions: frozenset[str] | set[str] | bool = frozenset(),
    *,
    none_as_sentinel: bool = False,
    empty_as_sentinel: bool = False,
) -> bool:
    """Check if a value is NOT a sentinel. Useful for filtering operations."""
    return not is_sentinel(
        value,
        additions,
        none_as_sentinel=none_as_sentinel,
        empty_as_sentinel=empty_as_sentinel,
    )
