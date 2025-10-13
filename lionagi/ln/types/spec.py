"""Field specification with ordered, immutable metadata for validation backends.

Design: tuple[Meta, ...] provides ordered, hashable metadata for compositional
field specifications. Frozen dataclass with SHA256 identity enables deduplication
and efficient caching across validation backends.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, ClassVar

from .._hash import _generate_hashable_representation
from .base import LionModel, Meta

__all__ = ("Spec",)


@dataclass(init=False, slots=True, frozen=True)
class Spec:
    """Immutable field specification with ordered metadata.

    Design decisions:
    - tuple[Meta, ...]: Ordered, immutable, hashable metadata sequence
    - SHA256 hash: Deterministic content-based identity for deduplication
    - _unique_meta: Enforces unique metadata keys (default: True)
    - Compositional: with_updates(), as_nullable(), as_listable() return new instances

    Example:
        >>> age_spec = Spec(int, min=0, max=120)
        >>> nullable_age = age_spec.as_nullable()
        >>> updated = age_spec.with_updates(max=150, description="Age")
    """

    metadata: tuple[Meta, ...]
    base_type: type[Any]
    sha256: str | None = None

    _unique_meta: ClassVar[bool] = True
    """Enforce unique metadata keys. Raises ValueError on duplicates."""

    def __init__(self, base_type: type[Any], **kw):
        """Initialize with base type and metadata kwargs.

        Args:
            base_type: Field type (int, str, list, etc.)
            **kw: Metadata as key-value pairs (min=0, max=100, etc.)

        Raises:
            ValueError: If duplicate metadata keys and _unique_meta=True
        """
        metas = tuple(Meta(k, v) for k, v in kw.items())

        object.__setattr__(self, "base_type", base_type)
        object.__setattr__(self, "metadata", metas)
        object.__setattr__(self, "sha256", self._compute_hash())

        self._validate_meta()

    def _validate_meta(self) -> None:
        """Validate metadata uniqueness if _unique_meta is True."""
        if not self._unique_meta:
            return

        keys = [m.key for m in self.metadata]
        if len(keys) != len(set(keys)):
            duplicates = [k for k in keys if keys.count(k) > 1]
            raise ValueError(f"Duplicate metadata keys: {set(duplicates)}")

    def _compute_hash(self) -> str:
        """Compute deterministic SHA256 hash of spec content.

        Uses _generate_hashable_representation for robust handling of:
        - Callables (validators, transforms) via identity semantics
        - Custom objects, mixed types, unhashable values
        - Consistent with Params/DataClass hash_dict pattern

        Hash includes: base_type + ordered metadata (key, value) pairs
        Enables: Deduplication, caching, content-addressed storage
        """
        content = {
            "type": self.base_type,
            "metadata": [(m.key, m.value) for m in self.metadata],
        }

        # Use robust representation generator from _hash module
        hashable_repr = _generate_hashable_representation(content)
        repr_str = str(hashable_repr)

        return hashlib.sha256(repr_str.encode()).hexdigest()

    @staticmethod
    def _type_to_str(typ: type[Any]) -> str:
        """Convert type to string for backend serialization (to_dict only)."""
        if hasattr(typ, "__origin__"):  # Generic types (list[int], etc.)
            origin = typ.__origin__.__name__
            args = ", ".join(Spec._type_to_str(arg) for arg in typ.__args__)
            return f"{origin}[{args}]"
        return typ.__name__

    def as_nullable(self) -> Spec:
        """Return new Spec with nullable=True constraint."""
        return self.with_updates(nullable=True)

    def as_listable(self) -> Spec:
        """Return new Spec with list[base_type] and listable=True.

        Transforms: int → list[int], str → list[str], etc.
        """
        # Get current metadata as dict, add listable constraint
        current = {m.key: m.value for m in self.metadata}
        current["listable"] = True
        return Spec(list[self.base_type], **current)

    def to_dict(self, exclude: set[str] | None = None) -> dict[str, Any]:
        """Serialize to dict for backend communication.

        Returns: {type: str, constraints: {key: value, ...}, sha256: str}
        """
        exclude = exclude or set()
        constraints = {m.key: m.value for m in self.metadata if m.key not in exclude}
        return {
            "type": self._type_to_str(self.base_type),
            "constraints": constraints,
            "sha256": self.sha256,
        }

    # LionModel protocol implementation

    def is_sentinel(self, key: str) -> bool:
        """Check if metadata key has sentinel value (always False for Spec)."""
        return False  # Spec doesn't use sentinels

    @classmethod
    def allowed(cls) -> set[str]:
        """Return allowed field names (metadata, base_type, sha256)."""
        return {"metadata", "base_type", "sha256"}

    def with_updates(self, **kwargs: Any) -> Spec:
        """Return new Spec with updated metadata (copy-on-write).

        Args:
            **kwargs: Metadata key-value pairs to add/update

        Returns: New Spec instance

        Example:
            >>> spec = Spec(int, min=0, max=100)
            >>> updated = spec.with_updates(max=200, description="Age")
        """
        # Convert current metadata to dict, merge with updates
        current = {m.key: m.value for m in self.metadata}
        current.update(kwargs)
        return Spec(self.base_type, **current)

    def __hash__(self) -> int:
        """Hash by SHA256 for content-based identity."""
        return int(self.sha256[:16], 16) if self.sha256 else 0

    def __eq__(self, other: Any) -> bool:
        """Equality via SHA256 content hash."""
        if not isinstance(other, Spec):
            return False
        return self.sha256 == other.sha256

    def __repr__(self) -> str:
        """Concise repr showing type and key constraints."""
        type_str = self._type_to_str(self.base_type)
        constraints = {m.key: m.value for m in self.metadata}
        return f"Spec({type_str}, {constraints})"
