# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""SpecAdapter protocol: framework-agnostic field specification adapter.

Spec serves as the universal intermediate representation (IR) for typed structures.
Adapters implement bidirectional transformations:

    Spec/Operable --[Adapter]--> Framework-specific structures
    Framework structures --[Adapter]--> Spec/Operable

Concrete adapters:
    - PydanticSpecAdapter: Spec <-> Pydantic FieldInfo/BaseModel
    - DataClassSpecAdapter: Spec <-> dataclass fields/Params/DataClass
    - SQLSpecAdapter: Spec -> SQL DDL (one-way, no instance support)

Type parameters:
    F: Field representation (FieldInfo, dict, str)
    S: Structure type (BaseModel, DataClass, DDL string)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from lionagi.ln.types._sentinel import Unset, UnsetType

if TYPE_CHECKING:
    from ..operable import Operable
    from ..spec import Spec

__all__ = ("SpecAdapter",)

F = TypeVar("F")
S = TypeVar("S")


class SpecAdapter(ABC, Generic[F]):
    """Abstract adapter protocol for Spec <-> framework transformations.

    Required methods (abstract):
        create_field: Spec -> F (framework field)
        compose_structure: Operable -> structure (type or DDL string)
        extract_specs: structure -> tuple[Spec, ...] (reverse extraction)

    Optional methods (override as needed):
        create_field_validator: Spec -> validator (framework-specific)
        validate_instance: (structure, dict) -> instance
        dump_instance: instance -> dict
    """

    @classmethod
    @abstractmethod
    def create_field(cls, spec: Spec) -> F:
        """Convert Spec to framework-specific field representation.

        Args:
            spec: Field specification

        Returns:
            Framework field (FieldInfo for Pydantic, dict for DataClass, str for SQL)
        """
        ...

    @classmethod
    @abstractmethod
    def compose_structure(
        cls,
        op: Operable,
        name: str,
        /,
        *,
        include: set[str] | UnsetType = Unset,
        exclude: set[str] | UnsetType = Unset,
        frozen: bool | UnsetType = Unset,
        base_type: type | UnsetType = Unset,
        doc: str | UnsetType = Unset,
        **kwargs: Any,
    ) -> Any:
        """Compose a structure from Operable.

        Args:
            op: Operable containing Specs
            name: Structure name (class name, table name, etc.)
            include: Only include these field names
            exclude: Exclude these field names
            frozen: Whether the structure is frozen/immutable

        Returns:
            Composed structure (type for Pydantic/DataClass, str for SQL DDL)
        """
        ...

    @classmethod
    @abstractmethod
    def extract_specs(cls, structure: Any) -> tuple[Spec, ...]:
        """Extract Specs from an existing structure.

        Args:
            structure: Structure to extract from (BaseModel class, DataClass, etc.)

        Returns:
            Tuple of Specs representing the structure's fields
        """
        ...

    # Optional methods
    @classmethod
    def create_field_validator(cls, spec: Spec) -> Any | None:
        """Create framework-specific validator from Spec metadata.

        Override in adapters that support field validation.

        Returns:
            Validator object or None if not supported/not present
        """
        return None

    @classmethod
    def validate_instance(cls, structure: Any, data: dict, /) -> Any:
        """Validate dict data into a structure instance.

        Override in adapters that produce instantiable structures.

        Raises:
            NotImplementedError: If adapter doesn't support instance creation
        """
        raise NotImplementedError(
            f"{cls.__name__} does not support instance validation"
        )

    @classmethod
    def dump_instance(cls, instance: Any, **kwargs) -> dict[str, Any]:
        """Dump a structure instance to dict.

        Override in adapters that produce instantiable structures.

        Raises:
            NotImplementedError: If adapter doesn't support instance dumping
        """
        raise NotImplementedError(f"{cls.__name__} does not support instance dumping")
