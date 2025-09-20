# Copyright (c) 2023 - 2025, HaiyangLi <quantocean.li at gmail dot com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Operable: A frozen field container for form assembly.

This module provides the Operable class, which acts as an immutable registry
of field definitions. Unlike OperableModel, this is not a Pydantic model but
a pure field container used to assemble forms for structured output validation.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from lionagi.models.field_model import FieldModel
from lionagi.utils import UNDEFINED

T = TypeVar("T")


class Operable:
    """
    Immutable container of field definitions for form assembly.

    An Operable serves as a frozen registry of all possible field definitions
    that can be used in a workflow. Forms are assembled by selecting specific
    fields from this container based on flow definitions.

    Key features:
    - Not a Pydantic model - pure field container
    - Becomes immutable after freezing
    - Supports declarative field definitions
    - Provides field selection for form assembly
    - Maintains field metadata for validation

    Examples:
        >>> # Declarative definition
        >>> class CoderOperable:
        ...     context = FieldModel(str).with_description("Issue context")
        ...     plan = FieldModel(str).with_description("Multi-step plan")
        ...     code = FieldModel(str).with_description("Generated code")
        ...
        >>> # Create and freeze operable
        >>> operable = Operable.from_class(CoderOperable).freeze()
        >>>
        >>> # Select fields for form
        >>> fields = operable.select_fields("context", "plan")
    """

    def __init__(self, name: str = None, fields: dict[str, FieldModel] = None):
        """
        Initialize an Operable container.

        Args:
            name: Name for this operable (used for identification)
            fields: Initial field definitions as {name: FieldModel}
        """
        self._name = name or "Operable"
        self._fields: dict[str, FieldModel] = fields or {}
        self._frozen = False
        self._metadata: dict[str, Any] = {}

    @property
    def name(self) -> str:
        """Get the name of this operable."""
        return self._name

    @property
    def is_frozen(self) -> bool:
        """Check if the operable is frozen (immutable)."""
        return self._frozen

    @property
    def field_names(self) -> set[str]:
        """Get all field names in this operable."""
        return set(self._fields.keys())

    @property
    def fields(self) -> Mapping[str, FieldModel]:
        """Get read-only view of all fields."""
        return self._fields if self._frozen else dict(self._fields)

    def add_field(
        self, name: str, field: FieldModel | type = None, **kwargs
    ) -> Operable:
        """
        Add a field to the operable (only if not frozen).

        Args:
            name: Field name
            field: FieldModel instance or type to create FieldModel from
            **kwargs: Additional parameters for FieldModel creation

        Returns:
            Self for chaining

        Raises:
            RuntimeError: If operable is frozen
            ValueError: If field name already exists
        """
        if self._frozen:
            raise RuntimeError("Cannot add fields to frozen Operable")

        if name in self._fields:
            raise ValueError(f"Field '{name}' already exists in Operable")

        # Create FieldModel if needed
        if field is None:
            field = FieldModel(Any, name=name, **kwargs)
        elif not isinstance(field, FieldModel):
            # Assume it's a type, create FieldModel from it
            field = FieldModel(field, name=name, **kwargs)
        else:
            # Ensure field has correct name
            field = field.with_metadata("name", name)

        self._fields[name] = field
        return self

    def remove_field(self, name: str) -> Operable:
        """
        Remove a field from the operable (only if not frozen).

        Args:
            name: Field name to remove

        Returns:
            Self for chaining

        Raises:
            RuntimeError: If operable is frozen
            KeyError: If field doesn't exist
        """
        if self._frozen:
            raise RuntimeError("Cannot remove fields from frozen Operable")

        if name not in self._fields:
            raise KeyError(f"Field '{name}' not found in Operable")

        del self._fields[name]
        return self

    def freeze(self) -> Operable:
        """
        Freeze the operable, making it immutable.

        After freezing, no fields can be added, removed, or modified.
        This ensures the field definitions remain consistent during
        form assembly and validation.

        Returns:
            Self for chaining
        """
        self._frozen = True
        # Also mark individual FieldModels as frozen if they support it
        for field in self._fields.values():
            if hasattr(field, "_frozen"):
                object.__setattr__(field, "_frozen", True)
        return self

    def unfreeze(self) -> Operable:
        """
        Unfreeze the operable, allowing modifications again.

        Use with caution - unfreezing during active workflows can
        lead to inconsistent validation.

        Returns:
            Self for chaining
        """
        self._frozen = False
        for field in self._fields.values():
            if hasattr(field, "_frozen"):
                object.__setattr__(field, "_frozen", False)
        return self

    def select_fields(
        self, *names: str, strict: bool = True
    ) -> dict[str, FieldModel]:
        """
        Select specific fields for form assembly.

        This method is typically called during form creation to get
        the specific fields needed for a particular workflow step.

        Args:
            *names: Field names to select
            strict: If True, raise error for missing fields; if False, skip them

        Returns:
            Dictionary of selected fields {name: FieldModel}

        Raises:
            RuntimeError: If operable is not frozen (must be frozen for selection)
            KeyError: If strict=True and field not found
        """
        if not self._frozen:
            raise RuntimeError(
                "Operable must be frozen before field selection"
            )

        selected = {}
        for name in names:
            if name in self._fields:
                selected[name] = self._fields[name]
            elif strict:
                raise KeyError(
                    f"Field '{name}' not found in Operable '{self._name}'"
                )

        return selected

    def get_field(self, name: str) -> FieldModel | None:
        """
        Get a specific field by name.

        Args:
            name: Field name

        Returns:
            FieldModel if found, None otherwise
        """
        return self._fields.get(name)

    def has_field(self, name: str) -> bool:
        """
        Check if a field exists in the operable.

        Args:
            name: Field name to check

        Returns:
            True if field exists, False otherwise
        """
        return name in self._fields

    @classmethod
    def from_class(cls, model_class: type, freeze: bool = True) -> Operable:
        """
        Create an Operable from a declarative class definition.

        Scans the class for FieldModel attributes and adds them to
        the operable. Optionally freezes the result.

        Args:
            model_class: Class with FieldModel attributes
            freeze: Whether to freeze the operable after creation

        Returns:
            New Operable instance with fields from the class

        Examples:
            >>> class MyFields:
            ...     name = FieldModel(str).with_description("User name")
            ...     age = FieldModel(int).with_validator(lambda x: x >= 0)
            ...
            >>> operable = Operable.from_class(MyFields)
        """
        operable = cls(name=model_class.__name__)

        # Scan class attributes for FieldModels
        for attr_name in dir(model_class):
            if not attr_name.startswith("_"):
                attr_value = getattr(model_class, attr_name)
                if isinstance(attr_value, FieldModel):
                    # Clone the field to avoid modifying the original
                    field = attr_value.with_metadata("name", attr_name)
                    operable._fields[attr_name] = field

        if freeze:
            operable.freeze()

        return operable

    @classmethod
    def from_dict(
        cls, fields_dict: dict[str, Any], freeze: bool = True
    ) -> Operable:
        """
        Create an Operable from a dictionary of field definitions.

        Args:
            fields_dict: Dictionary mapping field names to types or FieldModels
            freeze: Whether to freeze the operable after creation

        Returns:
            New Operable instance

        Examples:
            >>> operable = Operable.from_dict({
            ...     "name": str,
            ...     "age": FieldModel(int).with_validator(lambda x: x >= 0),
            ...     "email": FieldModel(str).with_description("Email address")
            ... })
        """
        operable = cls(name="DictOperable")

        for name, field_def in fields_dict.items():
            if isinstance(field_def, FieldModel):
                operable.add_field(name, field_def)
            else:
                # Assume it's a type
                operable.add_field(name, FieldModel(field_def, name=name))

        if freeze:
            operable.freeze()

        return operable

    def to_model_params(self, field_names: list[str] = None) -> dict:
        """
        Convert selected fields to parameters for model creation.

        This is useful for creating Pydantic models from the operable's fields.

        Args:
            field_names: Specific fields to include (None for all)

        Returns:
            Dictionary with field_models and parameter_fields
        """
        from pydantic import Field as PydanticField

        field_names = field_names or list(self._fields.keys())
        field_models = []
        parameter_fields = {}

        for name in field_names:
            if name in self._fields:
                field = self._fields[name]

                # Add to field_models if it has complex configuration
                if field.has_validator() or field.metadata:
                    field_models.append(field)
                else:
                    # Simple field, add to parameter_fields
                    parameter_fields[name] = PydanticField(
                        default=field.extract_metadata("default") or UNDEFINED,
                        description=field.extract_metadata("description"),
                        title=field.extract_metadata("title"),
                        annotation=field.base_type,
                    )

        return {
            "field_models": field_models,
            "parameter_fields": parameter_fields,
        }

    def clone(self, name: str = None, freeze: bool = False) -> Operable:
        """
        Create a copy of this operable.

        Args:
            name: Name for the new operable (uses original if None)
            freeze: Whether to freeze the cloned operable

        Returns:
            New Operable instance with copied fields
        """
        # Create new operable with copied fields
        new_fields = {
            name: field.with_metadata("_cloned", True)
            for name, field in self._fields.items()
        }

        cloned = Operable(
            name=name or f"{self._name}_clone", fields=new_fields
        )

        # Copy metadata
        cloned._metadata = dict(self._metadata)

        if freeze:
            cloned.freeze()

        return cloned

    def merge(self, other: Operable, freeze: bool = False) -> Operable:
        """
        Merge another operable's fields into this one.

        Args:
            other: Another Operable to merge fields from
            freeze: Whether to freeze after merging

        Returns:
            Self for chaining

        Raises:
            RuntimeError: If this operable is frozen
            ValueError: If there are conflicting field names
        """
        if self._frozen:
            raise RuntimeError("Cannot merge into frozen Operable")

        # Check for conflicts
        conflicts = self.field_names & other.field_names
        if conflicts:
            raise ValueError(f"Conflicting field names: {conflicts}")

        # Add all fields from other
        for name, field in other._fields.items():
            self._fields[name] = field

        if freeze:
            self.freeze()

        return self

    def __contains__(self, field_name: str) -> bool:
        """Check if a field exists in the operable."""
        return field_name in self._fields

    def __len__(self) -> int:
        """Get the number of fields in the operable."""
        return len(self._fields)

    def __repr__(self) -> str:
        """String representation of the operable."""
        status = "frozen" if self._frozen else "mutable"
        return f"Operable(name='{self._name}', fields={len(self._fields)}, status={status})"

    def __str__(self) -> str:
        """Human-readable string representation."""
        field_list = (
            ", ".join(self.field_names) if self._fields else "no fields"
        )
        status = "frozen" if self._frozen else "mutable"
        return f"{self._name} ({status}): [{field_list}]"


# Example declarative operables for common use cases


class BaseOperable:
    """Base fields that many operables might share."""

    id = FieldModel(str).with_description("Unique identifier")
    timestamp = FieldModel(str).with_description("Creation timestamp")
    metadata = FieldModel(dict).with_description("Additional metadata")


class TextOperable:
    """Fields for text processing workflows."""

    input_text = FieldModel(str).with_description("Input text to process")
    processed_text = FieldModel(str).with_description("Processed output text")
    summary = FieldModel(str).with_description("Text summary")
    keywords = FieldModel(list[str]).with_description("Extracted keywords")
    sentiment = FieldModel(str).with_description("Sentiment analysis result")
