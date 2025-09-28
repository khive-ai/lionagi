"""Policy checking for capability-based access control.

Adapted from v1 policy.py with simplifications for current lionagi.
"""

from typing import Optional, Set, TYPE_CHECKING

if TYPE_CHECKING:
    from lionagi.session.branch import Branch
    from lionagi.morphism.base import Morphism


def policy_check(
    branch: "Branch",
    morphism: "Morphism",
    override_requirements: Optional[Set[str]] = None
) -> bool:
    """Check if branch has capabilities required by morphism.

    This is the core security check - morphisms declare what
    capabilities they need, and branches must have those capabilities.

    Args:
        branch: Branch attempting to execute morphism
        morphism: Morphism being executed
        override_requirements: Dynamic requirements (overrides morphism.requires)

    Returns:
        True if all requirements are satisfied
    """
    # Get requirements
    requirements = override_requirements
    if requirements is None:
        requirements = morphism.requires if morphism.requires else set()

    # No requirements means allowed
    if not requirements:
        return True

    # Check if branch has capabilities
    if not hasattr(branch, 'capabilities'):
        return False

    # Get all rights from branch capabilities
    branch_rights = set()
    for cap in branch.capabilities:
        if cap.subject == branch.id:
            branch_rights.update(cap.rights)

    # Check each requirement
    for req in requirements:
        if not _has_coverage(branch_rights, req):
            return False

    return True


def _has_coverage(rights: Set[str], required: str) -> bool:
    """Check if any right covers the requirement.

    Supports patterns:
    - Exact match: "validation.execute"
    - Wildcard: "*" or "validation.*"
    - Resource: "fs.read:/data/file.txt"
    """
    # Exact match
    if required in rights:
        return True

    # Check wildcards
    for right in rights:
        if _covers(right, required):
            return True

    return False


def _covers(have: str, need: str) -> bool:
    """Check if 'have' right covers 'need' requirement."""
    # Full wildcard
    if have == "*":
        return True

    # Service wildcard (e.g., "validation.*" covers "validation.execute")
    if have.endswith(".*"):
        prefix = have[:-2]
        if need.startswith(f"{prefix}."):
            return True

    # Resource patterns
    if ":" in have and ":" in need:
        have_base, have_resource = have.split(":", 1)
        need_base, need_resource = need.split(":", 1)

        # Base must match
        if have_base != need_base:
            return False

        # Check resource pattern
        if have_resource == "*":
            return True
        if have_resource.endswith("*"):
            prefix = have_resource[:-1]
            if need_resource.startswith(prefix):
                return True

    return False


def compute_effective_rights(branch: "Branch") -> Set[str]:
    """Compute all effective rights for a branch.

    This flattens all capabilities into a single set of rights.

    Args:
        branch: Branch to compute rights for

    Returns:
        Set of all effective rights
    """
    if not hasattr(branch, 'capabilities'):
        return set()

    rights = set()
    for cap in branch.capabilities:
        if cap.subject == branch.id:
            rights.update(cap.rights)

    return rights


def require_capability(required: str):
    """Decorator to require capability for morphism execution.

    Usage:
        @require_capability("validation.execute")
        class ValidationMorphism(Morphism):
            ...
    """
    def decorator(cls):
        # Add to requires set
        if not hasattr(cls, 'requires'):
            cls.requires = set()
        cls.requires.add(required)
        return cls
    return decorator


def require_any(*requirements: str):
    """Decorator to require any of the listed capabilities.

    Usage:
        @require_any("validation.execute", "validation.admin")
        class ValidationMorphism(Morphism):
            ...
    """
    def decorator(cls):
        # Store as custom attribute for special handling
        cls._require_any = set(requirements)

        # Override policy check
        original_init = cls.__init__

        def __init__(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            # Modify requires dynamically
            self.requires = self._require_any

        cls.__init__ = __init__
        return cls
    return decorator