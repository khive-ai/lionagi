"""Base abstractions for compositional specifications.

This module defines the core protocol for Specs - immutable blueprints that can be:
- Composed with other specs
- Transformed systematically
- Materialized to different backends
- Cached for identity and performance
"""

from __future__ import annotations

import os
import threading
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Any, Generic, Protocol, TypeVar, runtime_checkable

__all__ = ("Spec", "Backend", "SpecCache")

T = TypeVar("T")  # What the spec describes (Field, Model, Operation, etc.)
B = TypeVar("B")  # Backend artifact type (FieldInfo, SQLAlchemy Column, etc.)


@runtime_checkable
class Spec(Protocol[T]):
    """Protocol for compositional specifications.

    A Spec is an immutable blueprint that describes something (a field, model, operation, etc.)
    and can be composed, transformed, and materialized to different backends.

    Key Properties:
    - Immutability: Specs never mutate, transformations create new specs
    - Composability: Specs can be combined to create richer specs
    - Multi-target: Same spec can materialize to different backends
    - Cacheable: Specs are hashable and can be cached for identity

    Type Parameters:
        T: The domain object this spec describes (FieldSpec, ModelSpec, etc.)
    """

    @abstractmethod
    def transform(self, **updates: Any) -> Spec[T]:
        """Apply transformations to create a new spec.

        This is the fundamental transformation operation - all other transformations
        (as_nullable, with_validator, etc.) should be implemented via this method.

        Args:
            **updates: Transformation parameters to apply

        Returns:
            New spec with transformations applied

        Example:
            >>> spec = FieldSpec(base_type=str)
            >>> nullable = spec.transform(nullable=True)
            >>> validated = spec.transform(validator=lambda x: len(x) > 0)
        """
        ...

    @abstractmethod
    def materialize(self, backend: Backend[T, B]) -> B:
        """Materialize this spec using a specific backend.

        Args:
            backend: Backend implementation to use for materialization

        Returns:
            Backend-specific artifact (FieldInfo, Column, GraphQL field, etc.)

        Example:
            >>> spec = FieldSpec(base_type=str, default="hello")
            >>> pydantic_field = spec.materialize(PydanticBackend())
            >>> sqlalchemy_col = spec.materialize(SQLAlchemyBackend())
        """
        ...

    @abstractmethod
    def __hash__(self) -> int:
        """Specs must be hashable for caching and identity checks.

        Returns:
            Hash value based on spec's immutable state
        """
        ...

    @abstractmethod
    def __eq__(self, other: object) -> bool:
        """Specs must define equality for caching and comparison.

        Args:
            other: Object to compare with

        Returns:
            True if specs are equal
        """
        ...


class Backend(ABC, Generic[T, B]):
    """Abstract backend for materializing specs into concrete artifacts.

    Backends convert abstract specs into framework-specific implementations
    (Pydantic fields, SQLAlchemy columns, Strawberry GraphQL fields, etc.).

    Type Parameters:
        T: Spec type this backend handles (FieldSpec, ModelSpec, etc.)
        B: Backend artifact type (FieldInfo, Column, etc.)
    """

    @abstractmethod
    def materialize(self, spec: Spec[T]) -> B:
        """Convert spec to backend-specific artifact.

        Args:
            spec: Spec to materialize

        Returns:
            Backend-specific artifact

        Raises:
            ValueError: If spec is incompatible with this backend
        """
        ...

    @abstractmethod
    def can_materialize(self, spec: Spec[T]) -> bool:
        """Check if this backend can materialize the given spec.

        Args:
            spec: Spec to check

        Returns:
            True if backend can handle this spec
        """
        ...


class SpecCache(Generic[T]):
    """Thread-safe LRU cache for materialized specs.

    This cache ensures that repeated materializations of the same spec
    return identical objects for performance and identity checks.

    Attributes:
        max_size: Maximum number of cached items (default: 10000)
    """

    def __init__(self, max_size: int | None = None):
        """Initialize cache with optional size limit.

        Args:
            max_size: Maximum cache size (default from LIONAGI_SPEC_CACHE_SIZE env var)
        """
        self._max_size = max_size or int(
            os.environ.get("LIONAGI_SPEC_CACHE_SIZE", "10000")
        )
        self._cache: OrderedDict[int, T] = OrderedDict()
        self._lock = threading.RLock()

    def get(self, key: int) -> T | None:
        """Get cached value and mark as recently used.

        Args:
            key: Cache key (typically hash of spec)

        Returns:
            Cached value if present, None otherwise
        """
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)  # LRU touch
                return self._cache[key]
            return None

    def set(self, key: int, value: T) -> None:
        """Set cached value with LRU eviction.

        Args:
            key: Cache key
            value: Value to cache
        """
        with self._lock:
            self._cache[key] = value
            # Evict oldest if cache is too large
            while len(self._cache) > self._max_size:
                try:
                    self._cache.popitem(last=False)  # Remove oldest
                except KeyError:
                    # Race condition - cache became empty
                    break

    def clear(self) -> None:
        """Clear all cached items."""
        with self._lock:
            self._cache.clear()

    def __len__(self) -> int:
        """Get current cache size."""
        return len(self._cache)
