"""Core FieldSpec: Lightweight field specification for backend validation.

This is a data structure, not a validator. Actual validation happens in backends.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, get_args, get_origin


@dataclass(frozen=True)
class FieldSpec:
    """Lightweight field specification for backend validation.

    This is a simple data structure that describes field constraints.
    It doesn't perform validation itself - that's delegated to backends
    (Pydantic, Rust, Cloud).

    Attributes:
        type: Python type annotation (int, str, list[int], etc.)
        constraints: Dictionary of validation constraints

    Examples:
        >>> # Simple field
        >>> age = FieldSpec(int, {"min": 0, "max": 120})

        >>> # Nullable field
        >>> email = FieldSpec(str, {"pattern": r"^[\w\.-]+@[\w\.-]+$"})
        >>> nullable_email = email.as_nullable()

        >>> # List field
        >>> tags = FieldSpec(str, {}).as_listable()

        >>> # Chaining transformations
        >>> optional_tags = FieldSpec(str, {}).as_listable().as_nullable()
    """

    type: type
    constraints: dict[str, Any]

    def __init__(self, type_: type, constraints: dict[str, Any] | None = None):
        """Create a field specification.

        Args:
            type_: Python type (int, str, list[int], etc.)
            constraints: Optional dict of validation constraints
        """
        # Use object.__setattr__ since dataclass is frozen
        object.__setattr__(self, "type", type_)
        object.__setattr__(self, "constraints", constraints or {})

    def with_constraint(self, key: str, value: Any) -> FieldSpec:
        """Add or update a constraint.

        If the key already exists, it will be replaced. This ensures
        no duplicate constraint keys (fixes the issue from ChatGPT report).

        Args:
            key: Constraint name ("min", "max", "pattern", etc.)
            value: Constraint value

        Returns:
            New FieldSpec with updated constraints

        Examples:
            >>> spec = FieldSpec(int, {})
            >>> spec = spec.with_constraint("min", 0)
            >>> spec = spec.with_constraint("max", 100)
        """
        new_constraints = {**self.constraints, key: value}
        return FieldSpec(self.type, new_constraints)

    def as_nullable(self) -> FieldSpec:
        """Mark field as nullable (allows None).

        Returns:
            New FieldSpec marked as nullable

        Examples:
            >>> email = FieldSpec(str, {"pattern": r"^[\w\.-]+@[\w\.-]+$"})
            >>> optional_email = email.as_nullable()
        """
        return self.with_constraint("nullable", True)

    def as_listable(self) -> FieldSpec:
        """Transform to list type.

        Returns:
            New FieldSpec with list[original_type]

        Examples:
            >>> tag_spec = FieldSpec(str, {"min_length": 1})
            >>> tags_spec = tag_spec.as_listable()  # list[str]
        """
        # Update type to list[original_type]
        new_type = list[self.type]
        new_constraints = {**self.constraints, "listable": True}
        return FieldSpec(new_type, new_constraints)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible dict for backend transmission.

        Returns:
            Dict representation suitable for JSON serialization

        Examples:
            >>> spec = FieldSpec(int, {"min": 0, "max": 100})
            >>> spec.to_dict()
            {'type': 'int', 'constraints': {'min': 0, 'max': 100}}
        """
        return {
            "type": _type_to_string(self.type),
            "constraints": self.constraints,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FieldSpec:
        """Deserialize from JSON-compatible dict.

        Args:
            data: Dict with 'type' and 'constraints' keys

        Returns:
            FieldSpec reconstructed from dict

        Examples:
            >>> data = {'type': 'int', 'constraints': {'min': 0, 'max': 100}}
            >>> spec = FieldSpec.from_dict(data)
        """
        return cls(
            type_=_string_to_type(data["type"]),
            constraints=data.get("constraints", {}),
        )

    def __repr__(self) -> str:
        """Human-readable representation."""
        type_name = _type_to_string(self.type)
        constraints_str = ", ".join(
            f"{k}={v!r}" for k, v in self.constraints.items()
        )
        return f"FieldSpec({type_name}, {{{constraints_str}}})"


# Type serialization utilities


def _type_to_string(type_: type) -> str:
    """Convert Python type to string representation.

    Examples:
        int → "int"
        str → "str"
        list[int] → "list[int]"
        dict[str, Any] → "dict[str, Any]"
    """
    # Handle basic types
    if hasattr(type_, "__name__"):
        name = type_.__name__
    else:
        name = str(type_)

    # Handle generic types (list[int], dict[str, Any], etc.)
    origin = get_origin(type_)
    args = get_args(type_)

    if origin is not None and args:
        args_str = ", ".join(_type_to_string(arg) for arg in args)
        origin_name = (
            origin.__name__ if hasattr(origin, "__name__") else str(origin)
        )
        return f"{origin_name}[{args_str}]"

    return name


def _string_to_type(type_str: str) -> type:
    """Convert string representation to Python type.

    This is a simplified version - for production, consider using
    a more robust type parser or storing type info differently.

    Examples:
        "int" → int
        "str" → str
        "list[int]" → list[int]
    """
    # Basic types
    basic_types = {
        "int": int,
        "str": str,
        "float": float,
        "bool": bool,
        "bytes": bytes,
        "dict": dict,
        "list": list,
        "tuple": tuple,
        "set": set,
        "Any": Any,
    }

    # Handle simple types
    if type_str in basic_types:
        return basic_types[type_str]

    # Handle generic types (simplified - just list[T] for now)
    if type_str.startswith("list[") and type_str.endswith("]"):
        inner = type_str[5:-1]
        inner_type = _string_to_type(inner)
        return list[inner_type]

    # Fallback: return as-is (might fail for complex types)
    # For production, use ast.literal_eval or typing.ForwardRef
    return eval(type_str, {"__builtins__": {}}, basic_types)
