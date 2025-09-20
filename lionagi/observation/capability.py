# Copyright (c) 2023 - 2025, HaiyangLi <quantocean.li at gmail dot com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Capability: Permission tokens for field access and operations.

Simple capability model - no complex expressions, just resource:action patterns.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Set
from uuid import UUID


@dataclass(frozen=True)
class Capability:
    """
    A capability grants permission to perform operations on resources.

    This is kept simple - no DNF, no compilation, just pattern matching.
    """

    subject: UUID  # Actor who holds this capability
    resource: str  # Pattern like "field:email", "action:execute", "data:*"
    rights: Set[
        str
    ]  # Operations like {"read", "write", "validate", "transform"}

    def permits(self, action: str, resource: str) -> bool:
        """Check if this capability permits an action on a resource."""
        if action not in self.rights:
            return False

        # Simple pattern matching
        if self.resource == "*":
            return True
        if self.resource == resource:
            return True
        if self.resource.endswith("*"):
            prefix = self.resource[:-1]
            return resource.startswith(prefix)
        return False

    def attenuate(self, rights_subset: Set[str]) -> Capability:
        """Create a weaker capability (monotonic - can only reduce rights)."""
        if not rights_subset.issubset(self.rights):
            raise ValueError("Cannot expand rights during attenuation")
        return Capability(self.subject, self.resource, rights_subset)


def has_capability(
    capabilities: Set[Capability], resource: str, action: str
) -> bool:
    """Simple check if any capability permits the action."""
    return any(cap.permits(action, resource) for cap in capabilities)
