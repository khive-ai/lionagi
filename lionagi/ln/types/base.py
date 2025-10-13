"""Core abstractions for Lion type system.

Design Philosophy:
- LionModel: Structural typing for data models with identity and transformation
- Meta: Ordered, immutable metadata for compositional field specifications
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

__all__ = (
    "LionModel",
    "Meta",
)


@runtime_checkable  # Enables isinstance() checks for structural compatibility
class LionModel(Protocol):
    """Foundation protocol for data models with identity, serialization, and immutability.

    Implementations: Params, DataClass, Spec
    Key invariants: Content-based equality, deterministic hashing, functional updates
    """

    def is_sentinel(self, key: str) -> bool:
        """Check if field contains sentinel (Unset/Undefined/None)."""
        ...

    @classmethod
    def allowed(cls) -> set[str]:
        """Return allowed field names (excludes private _ prefixed fields)."""
        ...

    def to_dict(self, exclude: set[str] | None = None) -> dict[str, Any]:
        """Serialize to dict, excluding sentinels and specified keys."""
        ...

    def with_updates(self, **kwargs: Any) -> Any:
        """Return new instance with updated fields (copy-on-write)."""
        ...

    def __hash__(self) -> int:
        """Content-based hash for identity and deduplication."""
        ...

    def __eq__(self, other: Any) -> bool:
        """Equality via content hash (structural equality)."""
        ...


@dataclass(slots=True, frozen=True)  # Memory-efficient, immutable
class Meta:
    """Immutable key-value metadata for field specifications.

    Design decisions:
    - Frozen: Enables use in tuple[Meta, ...] for hashable field specs
    - Slots: Reduces memory overhead (~40% vs dict-based attrs)
    - Custom hash/eq: Identity semantics for callables (validators, transforms)
    """

    key: str
    value: Any

    def __hash__(self) -> int:
        """Hash by identity for callables, by value otherwise.

        Rationale: Same validator instance should hash identically for cache hits.
        Fallback to str() for unhashable types (e.g., lists, dicts).
        """
        if callable(self.value):
            return hash((self.key, id(self.value)))  # Identity for functions
        try:
            return hash((self.key, self.value))
        except TypeError:  # Unhashable types
            return hash((self.key, str(self.value)))

    def __eq__(self, other: object) -> bool:
        """Compare by identity for callables, by value otherwise.

        Rationale: Enables cache hits when same validator instance is reused
        across multiple field specs (common pattern in field composition).
        """
        if not isinstance(other, Meta):
            return NotImplemented

        if self.key != other.key:
            return False

        # Callables: compare by id (same instance = same validator)
        if callable(self.value) and callable(other.value):
            return id(self.value) == id(other.value)

        return bool(self.value == other.value)
