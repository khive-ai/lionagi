"""Minimal FieldModel - framework-agnostic field configuration.

Lightweight immutable field definitions that materialize to any validation framework.
Dict metadata internally, Meta tuples for annotated() output, global cache for identity.
"""

from __future__ import annotations

import os
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Annotated, Any

from typing_extensions import Self

from lionagi.ln.types import Meta

__all__ = ("FieldModel",)

# Global cache for annotated types with bounded size
_MAX_CACHE_SIZE = int(os.environ.get("LIONAGI_FIELD_CACHE_SIZE", "10000"))
_annotated_cache: OrderedDict[int, type] = OrderedDict()
_cache_lock = threading.RLock()


@dataclass(slots=True, frozen=True, init=False)
class FieldModel:
    """Immutable field configuration for any validation framework.

    Metadata stored as dict internally, converted to Meta tuples for annotated() output.

    Attributes:
        base_type: Python type (int, str, list[str], etc.)
        metadata: Field metadata dict (default, title, description, validator, etc.)

    Special metadata keys: validator, serializer, nullable, listable, name, default,
    description, frozen, title. See individual method docs for details.
    """

    base_type: type[Any]
    metadata: dict[str, Any]

    def __init__(self, base_type: type[Any], **metadata):
        """Initialize field with base type and metadata.

        Args:
            base_type: Python type (int, str, list[str], etc.)
            **metadata: default, title, description, validator, etc.
        """
        # Handle legacy 'annotation' alias
        if "annotation" in metadata:
            base_type = metadata.pop("annotation")

        # Bypass frozen dataclass to set attributes
        object.__setattr__(self, "base_type", base_type)
        object.__setattr__(self, "metadata", dict(metadata))
        self._validate()

    @property
    def name(self) -> str:
        """Field name from metadata (defaults to "field")."""
        return self.metadata.get("name", "field")

    @property
    def annotation(self) -> type[Any]:
        """Transformed type with listable/nullable applied (listable first, nullable second)."""
        actual_type = self.base_type

        if self.metadata.get("listable"):
            actual_type = list[actual_type]  # type: ignore

        if self.metadata.get("nullable"):
            actual_type = actual_type | None  # type: ignore

        return actual_type

    @property
    def annotated(self) -> type[Any]:
        """Cached Annotated[Type, Meta(...)] with transformations applied.

        Transformations in fixed order: listable first (inner), nullable second (outer).
        """
        # Create hashable cache key from base_type and sorted metadata
        cache_key = hash(
            (
                self.base_type,
                tuple(sorted(self.metadata.items(), key=lambda x: x[0])),
            )
        )

        with _cache_lock:
            # Check cache first
            if cache_key in _annotated_cache:
                _annotated_cache.move_to_end(cache_key)  # LRU touch
                return _annotated_cache[cache_key]

            # Build type with fixed transformation order
            actual_type = self.base_type

            # 1. Apply listable first (inner transformation)
            if self.metadata.get("listable"):
                actual_type = list[actual_type]  # type: ignore

            # 2. Apply nullable second (outer transformation)
            if self.metadata.get("nullable"):
                actual_type = actual_type | None  # type: ignore

            # Convert metadata dict to Meta tuples (exclude internal markers)
            meta_tuples = tuple(
                Meta(k, v)
                for k, v in self.metadata.items()
                if k not in ("nullable", "listable")
            )

            # Build Annotated type
            if meta_tuples:
                result = Annotated.__class_getitem__((actual_type, *meta_tuples))  # type: ignore
            else:
                result = actual_type

            # Cache with LRU eviction
            _annotated_cache[cache_key] = result

            while len(_annotated_cache) > _MAX_CACHE_SIZE:
                try:
                    _annotated_cache.popitem(last=False)  # Remove oldest
                except KeyError:
                    # Race condition - cache became empty
                    break

            return result  # type: ignore

    def has_meta(self, key: str) -> bool:
        """Check if metadata key is present."""
        return key in self.metadata

    def with_updates(self, **kwargs) -> Self:
        """Return new instance with updated metadata (replaces existing keys)."""
        new_meta = {**self.metadata, **kwargs}
        return type(self)(self.base_type, **new_meta)

    def as_nullable(self) -> Self:
        """Mark field as nullable (Type | None applied in annotated())."""
        return self.with_updates(nullable=True)

    def as_listable(self) -> Self:
        """Mark field as list type (list[Type] applied in annotated()).

        Call order doesn't matter: both produce list[Type] | None.
        """
        return self.with_updates(listable=True)

    def is_valid(self, value: Any) -> bool:
        """Check if value passes validator (returns False on failure)."""
        if not self.has_meta("validator"):
            return True

        validator = self.metadata["validator"]
        try:
            # Try Pydantic-style validator (cls, value)
            validator(None, value)
            return True
        except TypeError:
            # Try simple validator (value) → bool
            result = validator(value)
            return result is not False
        except Exception:
            return False

    def validate(self, value: Any, field_name: str | None = None) -> None:
        """Validate value, raising ValidationError on failure."""
        if not self.has_meta("validator"):
            return

        from lionagi._errors import ValidationError

        validator = self.metadata["validator"]
        try:
            # Try Pydantic-style validator (cls, value)
            validator(None, value)
        except TypeError:
            # Try simple validator (value) → bool
            result = validator(value)
            if result is False:
                validator_name = getattr(validator, "__name__", "validator")
                raise ValidationError(
                    f"Validation failed for {validator_name}",
                    field_name=field_name or self.name,
                    value=value,
                    validator_name=validator_name,
                )
        except Exception:
            raise

    def serialize(self, value: Any) -> Any:
        """Apply serializer to value if present, otherwise return value unchanged."""
        if not self.has_meta("serializer"):
            return value

        serializer = self.metadata["serializer"]
        try:
            # Try Pydantic-style serializer (value, info) - pass None for info
            return serializer(value, None)
        except TypeError:
            # Try simple serializer (value)
            return serializer(value)

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict with base_type name and metadata."""
        return {
            "base_type": getattr(
                self.base_type, "__name__", str(self.base_type)
            ),
            "metadata": self.metadata,
        }

    def __getattr__(self, name: str) -> Any:
        """Allow attribute-style access to metadata keys."""
        if name in self.metadata:
            return self.metadata[name]
        raise AttributeError(f"FieldModel has no attribute '{name}'")

    def __repr__(self) -> str:
        """String representation showing type and metadata."""
        base_name = getattr(self.base_type, "__name__", str(self.base_type))
        meta_str = ", ".join(f"{k}={v!r}" for k, v in self.metadata.items())
        return f"FieldModel({base_name}, {meta_str})"

    def _validate(self) -> None:
        """Validate base_type is a valid type (type, GenericAlias, or UnionType)."""
        import types

        is_valid = (
            isinstance(self.base_type, type)
            or hasattr(self.base_type, "__origin__")  # GenericAlias: list[str]
            or isinstance(self.base_type, types.UnionType)  # Union: str | None
        )

        if not is_valid:
            raise ValueError(
                f"base_type must be a type or type annotation, got {self.base_type}"
            )


# File: lionagi/models/field_model2.py
