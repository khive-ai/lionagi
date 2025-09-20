# Copyright (c) 2023 - 2025, HaiyangLi <quantocean.li at gmail dot com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Capability Expressions: Complex capability requirements with AND/OR/NOT logic.

Following opinion 2: Support capability composition in requirements, not in
the capability tokens themselves.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Set

if TYPE_CHECKING:
    from .capability import Capability


class CapabilityRequirement(ABC):
    """Abstract base for capability requirements."""

    @abstractmethod
    def is_satisfied(self, capabilities: Set[Capability]) -> bool:
        """Check if the requirement is satisfied by given capabilities."""
        ...


@dataclass
class SimpleRequirement(CapabilityRequirement):
    """
    Simple capability requirement.

    Just needs a single capability pattern like "field:email".
    """

    resource: str
    action: str = "read"

    def is_satisfied(self, capabilities: Set[Capability]) -> bool:
        """Check if any capability matches this requirement."""
        from .capability import has_capability

        return has_capability(capabilities, self.resource, self.action)


@dataclass
class AndRequirement(CapabilityRequirement):
    """
    AND composition of requirements.

    All sub-requirements must be satisfied.
    """

    requirements: list[CapabilityRequirement]

    def is_satisfied(self, capabilities: Set[Capability]) -> bool:
        """All requirements must be satisfied."""
        return all(req.is_satisfied(capabilities) for req in self.requirements)


@dataclass
class OrRequirement(CapabilityRequirement):
    """
    OR composition of requirements.

    At least one sub-requirement must be satisfied.
    """

    requirements: list[CapabilityRequirement]

    def is_satisfied(self, capabilities: Set[Capability]) -> bool:
        """At least one requirement must be satisfied."""
        return any(req.is_satisfied(capabilities) for req in self.requirements)


@dataclass
class NotRequirement(CapabilityRequirement):
    """
    NOT composition (negation).

    Use cautiously - complicates formal verification.
    """

    requirement: CapabilityRequirement

    def is_satisfied(self, capabilities: Set[Capability]) -> bool:
        """Requirement must NOT be satisfied."""
        return not self.requirement.is_satisfied(capabilities)


@dataclass
class ConditionalRequirement(CapabilityRequirement):
    """
    State-based conditional requirement.

    Following opinion 2: Use for runtime state checks, not workflow deps.
    """

    base: CapabilityRequirement
    condition: str  # e.g., "status == 'DRAFT'"

    def is_satisfied(
        self, capabilities: Set[Capability], context: dict = None
    ) -> bool:
        """Check base requirement and runtime condition."""
        if not self.base.is_satisfied(capabilities):
            return False

        # Evaluate condition against context
        if context is None:
            return True  # No context means no conditional check

        # Simple eval for demo - use safe expression evaluator in production
        try:
            return eval(self.condition, {"__builtins__": {}}, context)
        except Exception:
            return False


def build_requirement(spec: str | dict) -> CapabilityRequirement:
    """
    Build capability requirement from specification.

    Examples:
        "field:email"  # Simple
        {"and": ["field:email", "field:name"]}  # AND
        {"or": ["admin:*", "field:email:write"]}  # OR
        {"not": "field:sensitive"}  # NOT
    """
    if isinstance(spec, str):
        # Simple requirement
        parts = spec.split(":")
        if len(parts) == 2:
            return SimpleRequirement(spec, "read")
        elif len(parts) == 3:
            return SimpleRequirement(":".join(parts[:2]), parts[2])
        else:
            return SimpleRequirement(spec, "read")

    elif isinstance(spec, dict):
        if "and" in spec:
            reqs = [build_requirement(r) for r in spec["and"]]
            return AndRequirement(reqs)
        elif "or" in spec:
            reqs = [build_requirement(r) for r in spec["or"]]
            return OrRequirement(reqs)
        elif "not" in spec:
            return NotRequirement(build_requirement(spec["not"]))
        elif "condition" in spec:
            return ConditionalRequirement(
                build_requirement(spec.get("base", "*")), spec["condition"]
            )

    raise ValueError(f"Invalid capability requirement spec: {spec}")
