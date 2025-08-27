from __future__ import annotations

import copy
from datetime import datetime
from typing import Any
from uuid import UUID, uuid4
from weakref import WeakValueDictionary

from msgspec import Struct, field

from lionagi.ln import now_utc

# Global, weakly-held cache for CapabilitySet instances (avoid leaks across tests/processes)
# Referenced in:
# - "Capability-based security" https://en.wikipedia.org/wiki/Capability-based_security
_capability_cache: WeakValueDictionary[UUID, "CapabilitySet"] = WeakValueDictionary()


class CapabilitySet(set):
    """A set that updates the parent Branch when modified.
    
    Based on capability-based security model where holding an unforgeable reference
    provides access to resources. See:
    https://en.wikipedia.org/wiki/Capability-based_security
    """
    __slots__ = ("_branch",)

    def __init__(self, branch: Branch):
        self._branch = branch
        # Initialize with current rights
        all_rights = set()
        for cap in branch.caps:
            all_rights.update(cap.rights)
        super().__init__(all_rights)

    def add(self, element: str) -> None:
        """Add capability and update the parent branch."""
        super().add(element)
        self._update_branch()

    def discard(self, element: str) -> None:
        """Remove capability and update the parent branch."""
        super().discard(element)
        self._update_branch()

    def remove(self, element: str) -> None:
        """Remove capability and update the parent branch."""
        super().remove(element)
        self._update_branch()

    def _update_branch(self) -> None:
        """Update the parent branch with current capabilities."""
        # msgspec.Struct is immutable; policy should read from this live view.
        # No-op here; consumers must call rights_view(branch) to honor runtime updates.
        return


def rights_view(branch: "Branch") -> set[str]:
    """Live capability view: if a CapabilitySet has been materialized for this branch,
    use it; otherwise derive from the frozen caps tuple.
    
    This ensures policy decisions use the current state including runtime capability updates.
    Based on capability-based security model: https://en.wikipedia.org/wiki/Capability-based_security
    """
    cs = _capability_cache.get(branch.id)  # may be None if never accessed
    if cs is not None:
        return set(cs)
    have: set[str] = set()
    for cap in branch.caps:
        have.update(cap.rights)
    return have


class Observable(Struct, kw_only=True):
    """Observable atom: stable id + timestamp + lineage."""

    id: UUID = field(default_factory=uuid4)
    ts: datetime = field(default_factory=now_utc)
    lineage: tuple[UUID, ...] = field(default_factory=tuple)
    tags: tuple[str, ...] = field(default_factory=tuple)


class Observation(Struct, kw_only=True):
    id: UUID = field(default_factory=uuid4)
    ts: datetime = field(default_factory=now_utc)
    lineage: tuple[UUID, ...] = field(default_factory=tuple)
    tags: tuple[str, ...] = field(default_factory=tuple)

    who: UUID = field(default_factory=uuid4)
    """emitter id, e.g. branch id"""
    what: str = ""
    """such as "node.start" | "node.finish" | ..."""
    payload: dict[str, Any] = field(default_factory=dict)


class Capability(Struct, kw_only=True):
    subject: UUID  # Branch id
    rights: set[str]  # {"net.out", "fs.read:/data/*", ...}
    object: str = "*"  # optional resource scoping


class Branch(Struct, kw_only=True):
    """Branch is a semantic 'space': isolated context + summary + capability view."""

    # Repeat Obj fields
    id: UUID = field(default_factory=uuid4)
    ts: datetime = field(default_factory=now_utc)
    lineage: tuple[UUID, ...] = field(default_factory=tuple)
    tags: tuple[str, ...] = field(default_factory=tuple)

    name: str = "default"
    ctx: dict[str, Any] = field(default_factory=dict)
    summary: dict[str, Any] = field(default_factory=dict)
    caps: tuple[Capability, ...] = field(default_factory=tuple)

    # For TDD compatibility - parent tracking
    _parent: Branch | None = field(default=None)
    _capability_set: CapabilitySet | None = field(default=None)

    @property
    def parent(self) -> Branch | None:
        """Get the parent branch for lineage tracking."""
        return self._parent

    @property
    def capabilities(self) -> CapabilitySet:
        """Get capabilities as a mutable set that updates the branch when modified."""
        cs = _capability_cache.get(self.id)
        if cs is None:
            cs = CapabilitySet(self)
            _capability_cache[self.id] = cs
        return cs

    def fork(self) -> Branch:
        """Create a child branch with isolated context and inherited capabilities.

        Returns:
            A new Branch with:
            - Deep-copied context (isolated from parent)
            - Inherited capabilities
            - Parent reference for lineage tracking
            - New unique ID
        """
        # Deep copy context for isolation
        new_ctx = copy.deepcopy(self.ctx)

        # Inherit the *live* rights view (includes runtime updates)
        inherited_rights = rights_view(self)
        inherited_caps = (
            Capability(subject=uuid4(), rights=set(inherited_rights), object="*"),
        )

        # Create child branch with parent reference
        child = Branch(
            id=uuid4(),
            ts=now_utc(),
            lineage=self.lineage + (self.id,),  # Extend lineage with parent ID
            tags=self.tags,
            name=f"{self.name}_child",
            ctx=new_ctx,
            summary=self.summary.copy(),
            caps=inherited_caps,
            _parent=self,
        )

        return child

    @classmethod
    def create(cls, **kwargs) -> Branch:
        """Create a Branch - use create_branch() function for capabilities support."""
        return cls(**kwargs)


def create_branch(**kwargs) -> Branch:
    """Create a Branch with optional capabilities parameter.

    Args:
        capabilities: Set of capability strings (optional)
        **kwargs: Other Branch parameters

    Returns:
        Branch instance with capabilities properly set
    """
    capabilities = kwargs.pop("capabilities", None)

    # Handle None context - convert to empty dict
    if kwargs.get("ctx") is None:
        kwargs["ctx"] = {}

    # Convert capabilities to caps if provided
    if capabilities is not None:
        # Create a Capability object from the set of strings
        branch_id = kwargs.get("id", uuid4())
        if capabilities:  # Non-empty set
            cap = Capability(subject=branch_id, rights=set(capabilities))  # Create a copy
            kwargs["caps"] = (cap,)
        else:  # Empty set
            kwargs["caps"] = tuple()

    branch = Branch(**kwargs)
    return branch
