"""Frozen parameter dataclass with sentinel value handling.

Design: Combines frozen immutability with custom __init__ for flexible
keyword-only construction. Used for function parameters, config objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar

from typing_extensions import override

from .._hash import hash_dict
from .sentinel import Undefined, Unset, is_sentinel

__all__ = ("Params",)


@dataclass(slots=True, frozen=True, init=False)  # Frozen + custom __init__
class Params:
    """Immutable parameter container with sentinel value handling.

    Design decisions:
    - frozen=True: Immutability for safe sharing across contexts
    - init=False: Custom __init__ enables flexible **kwargs construction
    - slots=True: Memory efficiency (~40% reduction vs __dict__)
    - Sentinel support: Distinguishes unset (Unset) from None values

    Class variables:
    - _none_as_sentinel: Treat None as sentinel (default: False)
    - _strict: Disallow sentinels, require all fields (default: False)
    - _prefill_unset: Auto-fill undefined fields with Unset (default: True)
    """

    _none_as_sentinel: ClassVar[bool] = False
    _strict: ClassVar[bool] = False
    _prefill_unset: ClassVar[bool] = True
    _allowed_keys: ClassVar[set[str]] = field(default=set(), init=False, repr=False)

    def __init__(self, **kwargs: Any):
        """Initialize with keyword arguments only.

        Validates: keys must be in allowed() set
        Then: calls _validate() for sentinel/strict checks
        """
        for k, v in kwargs.items():
            if k in self.allowed():
                object.__setattr__(self, k, v)  # Bypass frozen
            else:
                raise ValueError(f"Invalid parameter: {k}")

        self._validate()

    def is_sentinel(self, key: str) -> bool:
        """Check if field contains sentinel value."""
        if key not in self.allowed():
            raise ValueError(f"Invalid parameter: {key}")
        return self._is_sentinel(getattr(self, key, Unset))

    @classmethod
    def _is_sentinel(cls, value: Any) -> bool:
        """Check if value is Unset/Undefined (or None if _none_as_sentinel)."""
        return is_sentinel(value, cls._none_as_sentinel)

    @classmethod
    def allowed(cls) -> set[str]:
        """Return allowed field names (cached, excludes _ prefixed)."""
        if cls._allowed_keys:
            return cls._allowed_keys
        cls._allowed_keys = {
            i for i in cls.__dataclass_fields__.keys() if not i.startswith("_")
        }
        return cls._allowed_keys

    @override
    def _validate(self) -> None:
        """Enforce _strict and _prefill_unset policies."""

        def _validate_strict(k):
            if self._strict and self._is_sentinel(getattr(self, k, Unset)):
                raise ValueError(f"Missing required parameter: {k}")
            if self._prefill_unset and getattr(self, k, Undefined) is Undefined:
                object.__setattr__(self, k, Unset)  # Prefill missing

        for k in self.allowed():
            _validate_strict(k)

    def default_kw(self) -> dict[str, Any]:
        """Extract as dict, merging 'kwargs'/'kw' fields if present.

        Use case: Partial function application with stored parameters
        """
        dict_ = self.to_dict()
        kw_ = {}
        kw_.update(dict_.pop("kwargs", {}))
        kw_.update(dict_.pop("kw", {}))
        dict_.update(kw_)
        return dict_

    def to_dict(self, exclude: set[str] | None = None) -> dict[str, Any]:
        """Serialize to dict, excluding sentinels and specified keys."""
        data = {}
        exclude = exclude or set()
        for k in self.allowed():
            if k not in exclude and not self._is_sentinel(
                v := getattr(self, k, Undefined)
            ):
                data[k] = v
        return data

    def __hash__(self) -> int:
        """Content-based hash via to_dict() for deduplication."""
        return hash_dict(self.to_dict())

    def __eq__(self, other: Any) -> bool:
        """Equality via content hash (structural equality)."""
        if not isinstance(other, Params):
            return False
        return hash(self) == hash(other)

    def with_updates(self, **kwargs: Any) -> Params:
        """Return new instance with updated fields (copy-on-write)."""
        dict_ = self.to_dict()
        dict_.update(kwargs)
        return type(self)(**dict_)
