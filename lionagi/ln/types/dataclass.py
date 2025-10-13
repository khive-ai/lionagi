"""Mutable dataclass with sentinel value handling and post-init validation.

Design: Standard dataclass with __post_init__ for validation. Mutable alternative
to frozen Params, useful when mutability is acceptable (e.g., internal state).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar

from typing_extensions import override

from .._hash import hash_dict
from .sentinel import Undefined, Unset, is_sentinel

__all__ = ("DataClass",)


@dataclass(slots=True)  # Mutable, uses __post_init__
class DataClass:
    """Mutable dataclass with sentinel handling and validation.

    Design decisions:
    - slots=True: Memory efficiency (~40% reduction)
    - Mutable: Uses __post_init__ instead of custom __init__
    - __setattr__ available: Can modify after creation (unlike frozen Params)

    Class variables:
    - _none_as_sentinel: Treat None as sentinel (default: False)
    - _strict: Disallow sentinels, require all fields (default: False)
    - _prefill_unset: Auto-fill undefined fields with Unset (default: True)
    """

    _none_as_sentinel: ClassVar[bool] = False
    _strict: ClassVar[bool] = False
    _prefill_unset: ClassVar[bool] = True
    _allowed_keys: ClassVar[set[str]] = field(default=set(), init=False, repr=False)

    def __post_init__(self):
        """Run validation after standard dataclass init."""
        self._validate()

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
                self.__setattr__(k, Unset)  # Prefill missing

        for k in self.allowed():
            _validate_strict(k)

    def to_dict(self, exclude: set[str] | None = None) -> dict[str, Any]:
        """Serialize to dict, excluding sentinels and specified keys."""
        data = {}
        exclude = exclude or set()
        for k in type(self).allowed():
            if k not in exclude and not self._is_sentinel(v := getattr(self, k)):
                data[k] = v
        return data

    def is_sentinel(self, key: str) -> bool:
        """Check if field contains sentinel value."""
        if key not in self.allowed():
            raise ValueError(f"Invalid parameter: {key}")
        return self._is_sentinel(getattr(self, key, Unset))

    @classmethod
    def _is_sentinel(cls, value: Any) -> bool:
        """Check if value is Unset/Undefined (or None if _none_as_sentinel)."""
        if value is None and cls._none_as_sentinel:
            return True
        return is_sentinel(value)

    def with_updates(self, **kwargs: Any) -> DataClass:
        """Return new instance with updated fields (copy-on-write).

        Note: Despite being mutable, follows immutability pattern for consistency
        with Params and LionModel protocol.
        """
        dict_ = self.to_dict()
        dict_.update(kwargs)
        return type(self)(**dict_)

    def __hash__(self) -> int:
        """Content-based hash via to_dict() for deduplication."""
        return hash_dict(self.to_dict())

    def __eq__(self, other: Any) -> bool:
        """Equality via content hash (structural equality)."""
        if not isinstance(other, DataClass):
            return False
        return hash(self) == hash(other)
