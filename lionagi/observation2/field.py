# Copyright (c) 2023 - 2025, HaiyangLi <quantocean.li at gmail dot com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Field: Data container with capability requirements.

Fields represent both data and the permission needed to access/modify that data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Set

if TYPE_CHECKING:
    from .capability import Capability
    from .invariant import Invariant


@dataclass
class FieldSpec:
    """
    Specification for a field including its capability requirements.

    This is the "schema" - what a field needs and how it behaves.
    """

    name: str
    type: type
    required_capability: (
        str  # Simple string like "field:email", "action:execute"
    )
    invariants: list[Invariant] = field(default_factory=list)
    description: str = ""
    default: Any = None

    def can_access(self, capabilities: Set[Capability]) -> bool:
        """Check if given capabilities permit reading this field."""
        from .capability import has_capability

        return has_capability(capabilities, self.required_capability, "read")

    def can_modify(self, capabilities: Set[Capability]) -> bool:
        """Check if given capabilities permit writing this field."""
        from .capability import has_capability

        return has_capability(capabilities, self.required_capability, "write")


class CapabilityField:
    """
    A field instance with value and access tracking.

    This is a runtime field that tracks its value and access history.
    """

    def __init__(self, spec: FieldSpec, value: Any = None):
        self.spec = spec
        self._value = value if value is not None else spec.default
        self.travel_log: list[dict] = []

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def value(self) -> Any:
        return self._value

    @value.setter
    def value(self, new_value: Any):
        """Set value and record the change."""
        old_value = self._value
        self._value = new_value
        self.record_access(
            {"type": "modify", "old": old_value, "new": new_value}
        )

    def record_access(self, event: dict):
        """Record field access in travel log."""
        self.travel_log.append({"timestamp": datetime.now(), "event": event})

    def validate_type(self) -> bool:
        """Simple type check."""
        if self._value is None:
            return True  # None is allowed
        return isinstance(self._value, self.spec.type)
