from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar

from typing_extensions import TypedDict, override

from ._sentinel import Undefined, Unset, is_sentinel

__all__ = (
    "Params",
    "DataClass",
    "Meta",
    "KeysDict",
    "KeysLike",
)


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

    def to_dict(self, exclude: set[str] = None) -> dict[str, str]:
        data = {}
        exclude = exclude or set()
        for k in self.allowed():
            if k not in exclude and not self._is_sentinel(
                v := getattr(self, k, Undefined)
            ):
                data[k] = v
        return data

    def __hash__(self) -> int:
        from .._hash import hash_dict

        return hash_dict(self.to_dict())

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Params):
            return False
        return hash(self) == hash(other)

    def with_updates(self, **kwargs: Any) -> DataClass:
        """Return a new instance with updated fields."""
        dict_ = self.to_dict()
        dict_.update(kwargs)
        return type(self)(**dict_)


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

    def to_dict(self, exclude: set[str] = None) -> dict[str, str]:
        data = {}
        exclude = exclude or set()
        for k in type(self).allowed():
            if k not in exclude and not self._is_sentinel(
                v := getattr(self, k)
            ):
                data[k] = v
        return data

    @classmethod
    def _is_sentinel(cls, value: Any) -> bool:
        """Check if a value is a sentinel (Undefined or Unset)."""
        if value is None and cls._none_as_sentinel:
            return True
        return is_sentinel(value)

    def with_updates(self, **kwargs: Any) -> DataClass:
        """Return a new instance with updated fields."""
        dict_ = self.to_dict()
        dict_.update(kwargs)
        return type(self)(**dict_)

    def __hash__(self) -> int:
        from .._hash import hash_dict

        return hash_dict(self.to_dict())

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, DataClass):
            return False
        return hash(self) == hash(other)


KeysLike = Sequence[str] | KeysDict


@dataclass(slots=True, frozen=True)
class Meta:
    """Immutable metadata container for field templates and other configurations."""

    key: str
    value: Any

    @override
    def __hash__(self) -> int:
        """Make metadata hashable for caching.

        Note: For callables, we hash by id to maintain identity semantics.
        """
        # For callables, use their id
        if callable(self.value):
            return hash((self.key, id(self.value)))
        # For other values, try to hash directly
        try:
            return hash((self.key, self.value))
        except TypeError:
            # Fallback for unhashable types
            return hash((self.key, str(self.value)))

    @override
    def __eq__(self, other: object) -> bool:
        """Compare metadata for equality.

        For callables, compare by id to increase cache hits when the same
        validator instance is reused. For other values, use standard equality.
        """
        if not isinstance(other, Meta):
            return NotImplemented

        if self.key != other.key:
            return False

        # For callables, compare by identity
        if callable(self.value) and callable(other.value):
            return id(self.value) == id(other.value)

        # For other values, use standard equality
        return bool(self.value == other.value)
