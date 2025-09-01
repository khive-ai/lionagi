from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum as _Enum
from typing import Any, ClassVar, Final, Literal, TypeVar, Union

from typing_extensions import TypedDict, override

__all__ = (
    "Undefined",
    "Unset",
    "MaybeUndefined",
    "MaybeUnset",
    "MaybeSentinel",
    "SingletonType",
    "UndefinedType",
    "UnsetType",
    "KeysDict",
    "T",
    "Enum",
    "is_sentinel",
    "not_sentinel",
    "Params",
    "DataClass",
)

T = TypeVar("T")


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

MaybeUndefined = Union[T, UndefinedType]
MaybeUnset = Union[T, UnsetType]
MaybeSentinel = Union[T, UndefinedType, UnsetType]


def is_sentinel(value: Any) -> bool:
    """Check if a value is any sentinel (Undefined or Unset)."""
    return value is Undefined or value is Unset


def not_sentinel(value: Any) -> bool:
    """Check if a value is NOT a sentinel. Useful for filtering operations."""
    return value is not Undefined and value is not Unset


class Enum(_Enum):
    @classmethod
    def allowed(cls) -> tuple[str, ...]:
        return tuple(e.value for e in cls)


class KeysDict(TypedDict, total=False):
    """TypedDict for keys dictionary."""

    key: Any  # Represents any key-type pair


@dataclass(slots=True, frozen=True, init=False)
class Params:
    """Base class for parameters used in various functions."""

    _none_as_sentinel: ClassVar[bool] = False
    """If True, None is treated as a sentinel value."""

    _strict: ClassVar[bool] = False
    """No sentinels allowed if strict is True."""

    _prefill_unset: ClassVar[bool] = True
    """If True, unset fields are prefilled with Unset."""

    _allowed_keys: ClassVar[set[str]] = field(
        default=set(), init=False, repr=False
    )
    """Class variable cache to store allowed keys for parameters."""

    def __init__(self, **kwargs: Any):
        """Initialize the Params object with keyword arguments."""
        # Set all attributes from kwargs, allowing for sentinel values
        for k, v in kwargs.items():
            if k in self.allowed():
                object.__setattr__(self, k, v)
            else:
                raise ValueError(f"Invalid parameter: {k}")

        # Validate after setting all attributes
        self._validate()

    @classmethod
    def _is_sentinel(cls, value: Any) -> bool:
        """Check if a value is a sentinel (Undefined or Unset)."""
        if value is None and cls._none_as_sentinel:
            return True
        return is_sentinel(value)

    @classmethod
    def allowed(cls) -> set[str]:
        """Return the keys of the parameters."""
        if cls._allowed_keys:
            return cls._allowed_keys
        cls._allowed_keys = {
            i for i in cls.__dataclass_fields__.keys() if not i.startswith("_")
        }
        return cls._allowed_keys

    @override
    def _validate(self) -> None:
        def _validate_strict(k):
            if self._strict and self._is_sentinel(getattr(self, k, Unset)):
                raise ValueError(f"Missing required parameter: {k}")
            if (
                self._prefill_unset
                and getattr(self, k, Undefined) is Undefined
            ):
                object.__setattr__(self, k, Unset)

        for k in self.allowed():
            _validate_strict(k)

    def default_kw(self) -> Any:
        # create a partial function with the current parameters
        dict_ = self.to_dict()

        # handle kwargs if present, handle both 'kwargs' and 'kw'
        kw_ = {}
        kw_.update(dict_.pop("kwargs", {}))
        kw_.update(dict_.pop("kw", {}))
        dict_.update(kw_)
        return dict_

    def to_dict(self) -> dict[str, str]:
        data = {}
        for k in self.allowed():
            if not self._is_sentinel(v := getattr(self, k, Undefined)):
                data[k] = v
        return data


@dataclass(slots=True)
class DataClass:
    """A base class for data classes with strict parameter handling."""

    _none_as_sentinel: ClassVar[bool] = False
    """If True, None is treated as a sentinel value."""

    _strict: ClassVar[bool] = False
    """No sentinels allowed if strict is True."""

    _prefill_unset: ClassVar[bool] = True
    """If True, unset fields are prefilled with Unset."""

    _allowed_keys: ClassVar[set[str]] = field(
        default=set(), init=False, repr=False
    )
    """Class variable cache to store allowed keys for parameters."""

    def __post_init__(self):
        """Post-initialization to ensure all fields are set."""
        self._validate()

    @classmethod
    def allowed(cls) -> set[str]:
        """Return the keys of the parameters."""
        if cls._allowed_keys:
            return cls._allowed_keys
        cls._allowed_keys = {
            i for i in cls.__dataclass_fields__.keys() if not i.startswith("_")
        }
        return cls._allowed_keys

    @override
    def _validate(self) -> None:
        def _validate_strict(k):
            if self._strict and self._is_sentinel(getattr(self, k, Unset)):
                raise ValueError(f"Missing required parameter: {k}")
            if (
                self._prefill_unset
                and getattr(self, k, Undefined) is Undefined
            ):
                self.__setattr__(k, Unset)

        for k in self.allowed():
            _validate_strict(k)

    def to_dict(self) -> dict[str, str]:
        data = {}
        print(self.allowed())
        for k in type(self).allowed():
            if not self._is_sentinel(v := getattr(self, k)):
                data[k] = v
        return data

    @classmethod
    def _is_sentinel(cls, value: Any) -> bool:
        """Check if a value is a sentinel (Undefined or Unset)."""
        if value is None and cls._none_as_sentinel:
            return True
        return is_sentinel(value)
