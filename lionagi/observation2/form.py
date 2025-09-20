# Copyright (c) 2023 - 2025, HaiyangLi <quantocean.li at gmail dot com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Form: Capability carrier that travels through the system.

Forms hold fields and their values, track their journey, and carry capabilities.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field as dataclass_field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, Set
from uuid import UUID, uuid4

if TYPE_CHECKING:
    from .capability import Capability
    from .field import CapabilityField, FieldSpec


@dataclass
class Form:
    """
    A form that carries capabilities and records its journey.

    Forms are the primary data carriers in the system. They:
    - Hold field values
    - Carry capabilities
    - Record their journey through validations/transformations
    - Can only access fields they have capabilities for
    """

    actor: UUID  # Actor who created/owns this form
    capabilities: Set[Capability] = dataclass_field(default_factory=set)
    fields: Dict[str, Any] = dataclass_field(default_factory=dict)
    field_specs: Dict[str, FieldSpec] = dataclass_field(default_factory=dict)
    journey: list[dict] = dataclass_field(default_factory=list)
    id: UUID = dataclass_field(default_factory=uuid4)

    def add_field(self, spec: FieldSpec, value: Any = None) -> None:
        """
        Add a field if actor has capability to access it.

        Args:
            spec: Field specification with capability requirements
            value: Initial value for the field

        Raises:
            PermissionError: If actor lacks required capability
        """
        if not spec.can_access(self.capabilities):
            raise PermissionError(
                f"Actor {self.actor} lacks capability for field {spec.name}"
            )

        self.field_specs[spec.name] = spec
        self.fields[spec.name] = value if value is not None else spec.default
        self.record_event("field_added", {"field": spec.name})

    def get_field(self, name: str) -> Any:
        """Get field value with capability check."""
        if name not in self.field_specs:
            raise KeyError(f"Field {name} not in form")

        spec = self.field_specs[name]
        if not spec.can_access(self.capabilities):
            raise PermissionError(f"Cannot access field {name}")

        return self.fields.get(name)

    def set_field(self, name: str, value: Any) -> None:
        """Set field value with capability check."""
        if name not in self.field_specs:
            raise KeyError(f"Field {name} not in form")

        spec = self.field_specs[name]
        if not spec.can_modify(self.capabilities):
            raise PermissionError(f"Cannot modify field {name}")

        old_value = self.fields.get(name)
        self.fields[name] = value
        self.record_event(
            "field_modified", {"field": name, "old": old_value, "new": value}
        )

    def record_event(self, event_type: str, details: dict = None):
        """Record an event in the form's journey."""
        self.journey.append(
            {
                "timestamp": datetime.now(),
                "type": event_type,
                "details": details or {},
            }
        )

    def has_capability(self, resource: str, action: str = "read") -> bool:
        """Check if form has a specific capability."""
        from .capability import has_capability

        return has_capability(self.capabilities, resource, action)

    @property
    def action_fields(self) -> Dict[str, FieldSpec]:
        """Get fields that represent action capabilities."""
        return {
            name: spec
            for name, spec in self.field_specs.items()
            if spec.required_capability.startswith("action:")
        }

    def to_dict(self) -> dict:
        """Simple serialization for backends."""
        return {
            "id": str(self.id),
            "actor": str(self.actor),
            "fields": self.fields,
            "journey_length": len(self.journey),
        }
