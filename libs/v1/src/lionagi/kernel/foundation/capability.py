"""Capability system - fields as privileges.

Ocean: "a field is a privilege, it is an access"

This evolved from FieldModel - fields aren't just data, they're capabilities
that must be validated through invariants before granting access.

Implements a minimal, immutable capability algebra with:
- explicit operations (ops)
- constraints (predicates or fixed values)
- attenuation via composition (intersection of ops, tightened constraints)
"""

from dataclasses import dataclass, field
from typing import Any, Mapping, FrozenSet


@dataclass(frozen=True)
class CapabilityMeta:
    """Metadata about a capability (kept for compatibility)."""
    key: str
    value: Any


@dataclass(frozen=True)
class Capability:
    """Immutable privilege/access descriptor for a resource.

    - name: resource identifier (e.g., "filesystem:/tmp")
    - ops: finite set of allowed operations (e.g., {"read", "write"})
    - constraints: mapping of constraint-name -> predicate or fixed value
    """

    name: str
    ops: FrozenSet[str]
    constraints: Mapping[str, Any] = field(default_factory=dict)

    def allows(self, op: str, **ctx: Any) -> bool:
        """Return True if operation is allowed under constraints in given context."""
        if op not in self.ops:
            return False
        for k, v in self.constraints.items():
            if callable(v):
                if not v(**ctx):
                    return False
            else:
                if ctx.get(k) != v:
                    return False
        return True

    def compose(self, other: "Capability") -> "Capability":
        """Attenuate authority combining two capabilities on the same resource.

        - ops: intersection
        - constraints: both must pass (logical AND)
        """
        if self.name != other.name:
            raise ValueError("Can only compose capabilities for the same resource")

        def tighten(k: str, a: Any, b: Any) -> Any:
            if a is None:
                return b
            if b is None:
                return a
            if callable(a) and callable(b):
                return lambda **c: a(**c) and b(**c)
            if callable(a) and not callable(b):
                return lambda **c: a(**c) and (c.get(k) == b)
            if not callable(a) and callable(b):
                return lambda **c: (c.get(k) == a) and b(**c)
            # both are values: equal values keep, else impossible predicate
            return a if a == b else (lambda **c: False)

        keys = set(self.constraints) | set(other.constraints)
        merged = {k: tighten(k, self.constraints.get(k), other.constraints.get(k)) for k in keys}
        return Capability(self.name, self.ops & other.ops, merged)


class CapabilityTemplate:
    """Template for composing and evolving a base capability immutably."""

    def __init__(self, base: Capability):
        self.base = base

    def grant(self, *ops: str, **constraints: Any) -> "CapabilityTemplate":
        new = Capability(
            name=self.base.name,
            ops=frozenset(set(self.base.ops) | set(ops)),
            constraints={**self.base.constraints, **constraints},
        )
        return CapabilityTemplate(new)

    def revoke(self, *ops: str) -> "CapabilityTemplate":
        new = Capability(
            name=self.base.name,
            ops=frozenset(set(self.base.ops) - set(ops)),
            constraints=self.base.constraints,
        )
        return CapabilityTemplate(new)

    def compose(self, other: "CapabilityTemplate") -> "CapabilityTemplate":
        return CapabilityTemplate(self.base.compose(other.base))

