from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, ClassVar, Protocol, Sequence, runtime_checkable

from typing_extensions import override

from .._hash import _generate_hashable_representation, hash_dict
from .sentinel import MaybeUnset, Undefined, Unset, is_sentinel

__all__ = (
    "LionModel",
    "Meta",
)


@runtime_checkable  # Enables isinstance() checks for structural compatibility
class LionModel(Protocol):
    """Foundation protocol for data models with identity, serialization, and immutability."""

    @classmethod
    def allowed(cls) -> set[str]:
        """Return allowed field names (excludes private _ prefixed fields)."""

    def to_dict(self, exclude: set[str] | None = None) -> dict[str, Any]:
        """Serialize to dict, excluding sentinels and specified keys."""

    def with_updates(self, **kwargs: Any) -> Any:
        """Return new instance with updated fields (copy-on-write)."""

    def __hash__(self) -> int:
        """Content-based hash for identity and deduplication."""

    def __eq__(self, other: Any) -> bool:
        """Equality via content hash (structural equality)."""


@dataclass(slots=True, frozen=True)
class Meta:
    """Immutable key-value metadata for field specifications."""

    key: str
    value: Any

    def __hash__(self) -> int:
        """Hash by identity for callables, by value otherwise."""
        if callable(self.value):
            return hash((self.key, id(self.value)))  # Identity for functions
        try:
            return hash((self.key, self.value))
        except TypeError:  # Unhashable types
            return hash((self.key, str(self.value)))

    def __eq__(self, other: object) -> bool:
        """Compare by identity for callables, by value otherwise."""
        if not isinstance(other, Meta):
            return NotImplemented

        if self.key != other.key:
            return False

        # Callables: compare by id (same instance = same validator)
        if callable(self.value) and callable(other.value):
            return id(self.value) == id(other.value)

        return bool(self.value == other.value)


@dataclass(slots=True)  # Mutable, uses __post_init__
class DataClass:
    """Mutable dataclass with sentinel handling and validation.

    Class variables:
    - _none_as_sentinel: Treat None as sentinel (default: False)
    - _strict: Disallow sentinels, require all fields (default: False)
    - _prefill_unset: Auto-fill undefined fields with Unset (default: True)
    """

    _none_as_sentinel: ClassVar[bool] = False
    _strict: ClassVar[bool] = False
    _prefill_unset: ClassVar[bool] = True
    _allowed_keys: ClassVar[set[str]] = field(
        default=set(), init=False, repr=False
    )

    @classmethod
    def _prep_params(cls, **kw) -> dict[str, MaybeUnset]:
        """Prepare parameters dict, enforcing allowed keys."""

        if any(k not in cls.allowed() for k in kw):
            invalid = [k for k in kw if k not in cls.allowed()]
            raise ValueError(f"Invalid parameters: {invalid}")
        if cls._prefill_unset:
            return {k: kw.get(k, Unset) for k in cls.allowed()}
        return kw.copy()

    def __init__(self, **kw):
        prep_params = type(self)._prep_params(**kw)
        super().__init__(**prep_params)
    
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
            if (
                self._prefill_unset
                and getattr(self, k, Undefined) is Undefined
            ):
                self.__setattr__(k, Unset)  # Prefill missing

        for k in self.allowed():
            _validate_strict(k)

    def to_dict(self, exclude: set[str] | None = None) -> dict[str, Any]:
        """Serialize to dict, excluding sentinels and specified keys."""
        data = {}
        exclude = exclude or set()
        for k in type(self).allowed():
            if k not in exclude and not self._is_sentinel(
                v := getattr(self, k)
            ):
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
        """Return new instance with updated fields (copy-on-write)."""
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


@dataclass(slots=True, frozen=True, init=False)  # Frozen + custom __init__
class Params:
    """Immutable parameter container with sentinel value handling.

    Class variables:
    - _none_as_sentinel: Treat None as sentinel (default: False)
    - _strict: Disallow sentinels, require all fields (default: False)
    - _prefill_unset: Auto-fill undefined fields with Unset (default: True)
    """

    _none_as_sentinel: ClassVar[bool] = False
    _strict: ClassVar[bool] = False
    _prefill_unset: ClassVar[bool] = True
    _allowed_keys: ClassVar[set[str]] = field(
        default=set(), init=False, repr=False
    )

    def __init__(self, **kwargs: Any):
        for k, v in kwargs.items():
            if k in self.allowed():
                object.__setattr__(self, k, v)
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
            if (
                self._prefill_unset
                and getattr(self, k, Undefined) is Undefined
            ):
                object.__setattr__(self, k, Unset)  # Prefill missing

        for k in self.allowed():
            _validate_strict(k)

    def default_kw(self) -> dict[str, Any]:
        """Extract as dict, merging 'kwargs'/'kw' fields if present."""
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


@dataclass(init=False, slots=True, frozen=True)
class Spec:
    """Immutable field specification with ordered metadata.

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

    _meta_is_unique: bool = None


    def __init__(self, base_type: type[Any], *args: Meta, **kw):
        """Initialize with base type and metadata kwargs."""
        
        
        
        
        
        
        
        
        
        metas = tuple(Meta(k, v) for k, v in kw.items())

        object.__setattr__(self, "base_type", base_type)
        object.__setattr__(self, "metadata", metas)
        object.__setattr__(self, "sha256", self._compute_hash())
        
        






        self._validate_meta()

    def has_unique_meta(self) -> bool:
        """Check if metadata keys are unique."""
        keys = [m.key for m in self.metadata]
        return len(keys) == len(set(keys))






    @property
    def meta_dict(self) -> dict[str, Any]:
        """Return metadata as a dictionary."""
        if self._unique_meta:
            return {m.key: m.value for m in self.metadata}

        keys = [m.key for m in self.metadata]
        if len(keys) != len(set(keys)):
            duplicates = [k for k in keys if keys.count(k) > 1]
            raise ValueError(f"Duplicate metadata keys: {set(duplicates)}")
        return {m.key: m.value for m in self.metadata}

    def _validate_meta(self) -> None:
        """Validate metadata uniqueness if _unique_meta is True."""
        if not self._unique_meta:
            return

        keys = [m.key for m in self.metadata]
        if len(keys) != len(set(keys)):
            duplicates = [k for k in keys if keys.count(k) > 1]
            raise ValueError(f"Duplicate metadata keys: {set(duplicates)}")

    def _compute_hash(self) -> str:
        """Compute deterministic SHA256 hash of spec content."""
        content = {
            "type": self.base_type,
            "metadata": [(m.key, m.value) for m in self.metadata],
        }

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
        """Return new Spec with list[base_type] and listable=True."""
        current = {m.key: m.value for m in self.metadata}
        current["listable"] = True
        return Spec(list[self.base_type], **current)

    def to_dict(self, exclude: set[str] | None = None) -> dict[str, Any]:
        exclude = exclude or set()
        constraints = {
            m.key: m.value for m in self.metadata if m.key not in exclude
        }
        return {
            "type": self._type_to_str(self.base_type),
            "constraints": constraints,
            "sha256": self.sha256,
        }

    




    def is_sentinel(self, key: str) -> bool:
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
        return type(self)(self.base_type, **current)

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

    def __getattr__(self, name: str) -> Any:
        """Handle access to custom attributes stored in metadata."""
        # Check if the attribute exists in metadata
        if not self.is_sentinel(self.metadata):
            for meta in self.metadata:
                if meta.key == name:
                    return meta.value

        # If not found, raise AttributeError as usual
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'"
        )






        
def _new_meta(metadata: tuple[Meta], *args, **kw) -> tuple[Meta, ...]:
    _meta: list = list(metadata) if metadata else []
    _meta.extend(args)
    for k, v in kw.items():
        _meta.append(Meta(k, v))
    return tuple(_meta)