"""Capability system for access control and service provision.

Adapted from v1 with focus on capability-based security.
"""

from dataclasses import dataclass, field
from typing import Set, Optional, TYPE_CHECKING
from uuid import UUID

if TYPE_CHECKING:
    from lionagi.session.branch import Branch


@dataclass
class Capability:
    """Capability grants rights to a subject for certain resources.

    This follows the principle of least privilege - branches only
    get the capabilities they need.
    """

    subject: UUID  # Branch ID that has this capability
    rights: Set[str] = field(default_factory=set)
    resource: str = "*"  # Resource pattern (e.g., "fs.read:/data/*")

    def covers(self, required: str) -> bool:
        """Check if this capability covers a required right.

        Supports wildcard patterns:
        - "*" covers everything
        - "service.*" covers all service operations
        - "service.operation:resource" for specific resources
        """
        # Exact match
        if required in self.rights:
            return True

        # Wildcard in capability
        for right in self.rights:
            if self._matches_pattern(right, required):
                return True

        return False

    def _matches_pattern(self, pattern: str, required: str) -> bool:
        """Check if pattern matches required right."""
        # Full wildcard
        if pattern == "*":
            return True

        # Service wildcard (e.g., "validation.*")
        if pattern.endswith(".*"):
            prefix = pattern[:-2]
            if required.startswith(f"{prefix}."):
                return True

        # Resource wildcard (e.g., "fs.read:/data/*")
        if ":" in pattern and pattern.endswith("*"):
            pattern_base, pattern_resource = pattern.split(":", 1)
            if ":" in required:
                req_base, req_resource = required.split(":", 1)
                if pattern_base == req_base:
                    pattern_prefix = pattern_resource[:-1]
                    if req_resource.startswith(pattern_prefix):
                        return True

        return False


def grant_capability(
    branch: "Branch",
    rights: Set[str],
    resource: str = "*"
) -> Capability:
    """Grant capabilities to a branch.

    Args:
        branch: Branch to grant capabilities to
        rights: Set of rights to grant
        resource: Resource pattern

    Returns:
        The granted capability
    """
    cap = Capability(
        subject=branch.id,
        rights=rights,
        resource=resource
    )

    # Store capability on branch
    if not hasattr(branch, 'capabilities'):
        branch.capabilities = []
    branch.capabilities.append(cap)

    return cap


def revoke_capability(
    branch: "Branch",
    capability: Optional[Capability] = None,
    right: Optional[str] = None
) -> bool:
    """Revoke capabilities from a branch.

    Args:
        branch: Branch to revoke from
        capability: Specific capability to revoke
        right: Specific right to revoke

    Returns:
        True if something was revoked
    """
    if not hasattr(branch, 'capabilities'):
        return False

    if capability:
        try:
            branch.capabilities.remove(capability)
            return True
        except ValueError:
            return False

    if right:
        # Remove right from all capabilities
        revoked = False
        for cap in branch.capabilities:
            if right in cap.rights:
                cap.rights.discard(right)
                revoked = True
        return revoked

    return False


def has_capability(
    branch: "Branch",
    required: str
) -> bool:
    """Check if branch has a required capability.

    Args:
        branch: Branch to check
        required: Required right

    Returns:
        True if branch has the capability
    """
    if not hasattr(branch, 'capabilities'):
        return False

    for cap in branch.capabilities:
        if cap.subject == branch.id and cap.covers(required):
            return True

    return False


# Common capability sets
READONLY_CAPABILITIES = {
    "fs.read",
    "db.read",
    "cache.read",
}

VALIDATION_CAPABILITIES = {
    "validation.execute",
    "validation.autofix",
    "validation.report",
}

FULL_CAPABILITIES = {
    "*",  # All capabilities
}


def grant_readonly(branch: "Branch") -> Capability:
    """Grant read-only capabilities."""
    return grant_capability(branch, READONLY_CAPABILITIES)


def grant_validation(branch: "Branch") -> Capability:
    """Grant validation capabilities."""
    return grant_capability(branch, VALIDATION_CAPABILITIES)


def grant_full(branch: "Branch") -> Capability:
    """Grant full capabilities (use with caution)."""
    return grant_capability(branch, FULL_CAPABILITIES)