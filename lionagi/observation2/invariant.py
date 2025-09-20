# Copyright (c) 2023 - 2025, HaiyangLi <quantocean.li at gmail dot com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Invariant: Mathematical properties that must always hold.

Simple invariants without complex dependency tracking or compilation.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from .form import Form


class Invariant(Protocol):
    """
    A mathematical property that must always be true.

    Invariants are checked before and after operations to ensure
    the system maintains its guarantees.
    """

    name: str

    @abstractmethod
    async def check_pre(self, form: Form) -> bool:
        """Check if invariant holds before operation."""
        ...

    @abstractmethod
    async def check_post(self, form: Form) -> bool:
        """Check if invariant holds after operation."""
        ...


class TypePreservation:
    """Values maintain their type unless explicitly transformed."""

    name = "TypePreservation"

    def __init__(self, field_name: str):
        self.field_name = field_name
        self._initial_type = None

    async def check_pre(self, form: Form) -> bool:
        """Record initial type."""
        if self.field_name in form.fields:
            self._initial_type = type(form.fields[self.field_name])
        return True

    async def check_post(self, form: Form) -> bool:
        """Verify type unchanged (unless transform capability exists)."""
        if self.field_name not in form.fields:
            return True

        current_type = type(form.fields[self.field_name])

        # Allow if transform capability exists
        from .capability import has_capability

        if has_capability(
            form.capabilities, f"field:{self.field_name}", "transform"
        ):
            return True

        return current_type == self._initial_type


class CapabilityMonotonicity:
    """Capabilities only decrease, never increase."""

    name = "CapabilityMonotonicity"

    def __init__(self):
        self._initial_caps = None

    async def check_pre(self, form: Form) -> bool:
        """Record initial capabilities."""
        self._initial_caps = form.capabilities.copy()
        return True

    async def check_post(self, form: Form) -> bool:
        """Verify capabilities didn't expand."""
        if self._initial_caps is None:
            return True
        return form.capabilities.issubset(self._initial_caps)


class FieldIntegrity:
    """Fields maintain their capability associations."""

    name = "FieldIntegrity"

    def __init__(self):
        self._field_caps = {}

    async def check_pre(self, form: Form) -> bool:
        """Record field-capability associations."""
        self._field_caps = {
            name: field.required_capability
            for name, field in form.field_specs.items()
        }
        return True

    async def check_post(self, form: Form) -> bool:
        """Verify associations unchanged."""
        for name, field in form.field_specs.items():
            if name in self._field_caps:
                if field.required_capability != self._field_caps[name]:
                    return False
        return True


class NonEmpty:
    """Field value cannot be empty."""

    name = "NonEmpty"

    def __init__(self, field_name: str):
        self.field_name = field_name

    async def check_pre(self, form: Form) -> bool:
        """No pre-check needed."""
        return True

    async def check_post(self, form: Form) -> bool:
        """Verify field is not empty."""
        if self.field_name not in form.fields:
            return True

        value = form.fields[self.field_name]
        if value is None:
            return False
        if isinstance(value, (str, list, dict, set, tuple)):
            return len(value) > 0
        return True


def default_invariants() -> list[Invariant]:
    """Get minimal set of default invariants."""
    return [
        CapabilityMonotonicity(),
        # Add others as needed
    ]
