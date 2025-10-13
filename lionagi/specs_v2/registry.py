"""Backend registry: Plugin system for validation backends.

This module provides the BackendRegistry for registering and routing
to different validation backends (Pydantic, Rust, Cloud, custom).
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from .spec import FieldSpec


@runtime_checkable
class Backend(Protocol):
    """Protocol that all validation backends must implement.

    Backends are responsible for:
    1. Validating values against FieldSpec constraints
    2. Creating backend-specific field objects (optional)

    Examples:
        >>> class CustomBackend:
        ...     def validate(self, spec: FieldSpec, value: Any) -> Any:
        ...         # Custom validation logic
        ...         return value
        ...
        ...     def create_field(self, spec: FieldSpec) -> Any:
        ...         # Create backend-specific field
        ...         return None
        ...
        >>> BackendRegistry.register("custom", CustomBackend())
    """

    def validate(self, spec: FieldSpec, value: Any) -> Any:
        """Validate a value against field specification.

        Args:
            spec: Field specification with constraints
            value: Value to validate

        Returns:
            Validated value (may be coerced/transformed)

        Raises:
            ValidationError: If value doesn't meet constraints
        """
        ...

    def create_field(self, spec: FieldSpec) -> Any:
        """Create backend-specific field object.

        Args:
            spec: Field specification

        Returns:
            Backend-specific field object (e.g., Pydantic FieldInfo)
        """
        ...


class BackendRegistry:
    """Global registry of validation backends.

    This class manages a registry of validation backends and provides
    methods to register, retrieve, and route validation requests.

    Backends are registered by name and can be used for validation
    by passing the backend name to the validate() method.

    Examples:
        >>> # Register a backend
        >>> BackendRegistry.register("pydantic", PydanticBackend())

        >>> # Set default backend
        >>> BackendRegistry.set_default("pydantic")

        >>> # Validate with default backend
        >>> spec = FieldSpec(int, {"min": 0, "max": 100})
        >>> value = BackendRegistry.validate(spec, 42)

        >>> # Validate with specific backend
        >>> value = BackendRegistry.validate(spec, 42, backend="rust")
    """

    _backends: dict[str, Backend] = {}
    _default: str = "pydantic"

    @classmethod
    def register(cls, name: str, backend: Backend) -> None:
        """Register a validation backend.

        Args:
            name: Backend identifier (e.g., "pydantic", "rust", "cloud")
            backend: Backend instance implementing Backend protocol

        Examples:
            >>> BackendRegistry.register("pydantic", PydanticBackend())
            >>> BackendRegistry.register("rust", RustBackend())
        """
        if not isinstance(backend, Backend):
            raise TypeError(
                f"Backend must implement Backend protocol, got {type(backend)}"
            )
        cls._backends[name] = backend

    @classmethod
    def get(cls, name: str | None = None) -> Backend:
        """Get a registered backend by name.

        Args:
            name: Backend name (uses default if None)

        Returns:
            Registered backend instance

        Raises:
            ValueError: If backend not found

        Examples:
            >>> backend = BackendRegistry.get("pydantic")
            >>> backend = BackendRegistry.get()  # Gets default
        """
        name = name or cls._default
        if name not in cls._backends:
            available = ", ".join(cls._backends.keys())
            raise ValueError(
                f"Backend '{name}' not registered. "
                f"Available backends: {available or 'none'}"
            )
        return cls._backends[name]

    @classmethod
    def set_default(cls, name: str) -> None:
        """Set the default backend.

        Args:
            name: Name of registered backend to use as default

        Raises:
            ValueError: If backend not registered

        Examples:
            >>> BackendRegistry.register("pydantic", PydanticBackend())
            >>> BackendRegistry.set_default("pydantic")
        """
        if name not in cls._backends:
            raise ValueError(f"Cannot set default to unregistered backend '{name}'")
        cls._default = name

    @classmethod
    def list_backends(cls) -> list[str]:
        """List all registered backend names.

        Returns:
            List of registered backend names

        Examples:
            >>> BackendRegistry.list_backends()
            ['pydantic', 'rust', 'cloud']
        """
        return list(cls._backends.keys())

    @classmethod
    def validate(
        cls,
        spec: FieldSpec,
        value: Any,
        backend: str | None = None
    ) -> Any:
        """Validate a value using specified backend.

        Args:
            spec: Field specification with constraints
            value: Value to validate
            backend: Backend name (uses default if None)

        Returns:
            Validated value

        Raises:
            ValueError: If backend not found
            ValidationError: If validation fails

        Examples:
            >>> spec = FieldSpec(int, {"min": 0, "max": 100})
            >>> value = BackendRegistry.validate(spec, 42)  # Default backend
            >>> value = BackendRegistry.validate(spec, 42, backend="rust")
        """
        return cls.get(backend).validate(spec, value)

    @classmethod
    def create_field(
        cls,
        spec: FieldSpec,
        backend: str | None = None
    ) -> Any:
        """Create backend-specific field object.

        Args:
            spec: Field specification
            backend: Backend name (uses default if None)

        Returns:
            Backend-specific field object

        Examples:
            >>> spec = FieldSpec(int, {"min": 0})
            >>> field = BackendRegistry.create_field(spec)  # Default backend
            >>> field = BackendRegistry.create_field(spec, backend="pydantic")
        """
        return cls.get(backend).create_field(spec)

    @classmethod
    def clear(cls) -> None:
        """Clear all registered backends (useful for testing).

        Examples:
            >>> BackendRegistry.clear()
            >>> BackendRegistry.list_backends()
            []
        """
        cls._backends.clear()
