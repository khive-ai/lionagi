from __future__ import annotations

from typing import Any, Final, Literal, TypeVar

T = TypeVar("T")


__all__ = (
    "Undefined",
    "Unset",
    "MaybeUndefined",
    "MaybeUnset",
    "MaybeSentinel",
    "SingletonType",
    "UndefinedType",
    "UnsetType",
    "is_sentinel",
    "not_sentinel",
    "T",
)


class _SingletonMeta(type):
    """Metaclass that guarantees exactly one instance per subclass.

    This ensures that sentinel values maintain identity across the entire application,
    allowing safe identity checks with 'is' operator.
    """

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

    def __deepcopy__(self, memo):  # copy & deepcopy both noop
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

MaybeUndefined = T | UndefinedType
"""A value that may be undefined."""

MaybeUnset = T | UnsetType
"""A value that may be unset."""

MaybeSentinel = T | UndefinedType | UnsetType
"""A value that may be any sentinel."""


def is_sentinel(value: Any, none_as_sentinel: bool = False) -> bool:
    """Check if a value is any sentinel (Undefined or Unset)."""
    if value is None and none_as_sentinel:
        return True
    return value is Undefined or value is Unset


def not_sentinel(value: Any, none_as_sentinel: bool = False) -> bool:
    """Check if a value is NOT a sentinel. Useful for filtering operations."""
    if value is None and none_as_sentinel:
        return False
    return value is not Undefined and value is not Unset
