# Copyright (c) 2023-2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Core type definitions for lionagi.beta substrate.

Capability string format: "domain.action[:resource]"
  Examples: "net.out", "fs.read:/data/*", "llm.call", "tool.execute:search"

Design decisions:
  - Capability is a frozen msgspec.Struct. The `object` field was removed (#2)
    because it was never used by policy_check — resource scoping belongs IN the
    rights strings (e.g., "fs.read:/data/*"), not as a separate field.
  - Principal.rights() filters by subject (#1) so a Capability minted for a
    different Principal cannot escalate privileges.
  - Principal.caps is frozen (#10) — mutation requires grant() which returns a
    new Principal, enforcing capability monotonicity at the API level.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4

import msgspec
from pydantic import Field, field_serializer, field_validator

from lionagi.beta.core.base.element import Element

__all__ = (
    "Capability",
    "Observation",
    "Principal",
    "now_utc",
)


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


class Capability(msgspec.Struct, frozen=True, kw_only=True):
    """Immutable capability token granting rights to a subject.

    Attributes:
        subject: UUID of the Principal this capability is bound to.
        rights: Set of capability strings.
    """

    subject: UUID
    rights: frozenset[str]

    def __hash__(self) -> int:
        return hash((self.subject, self.rights))


class Principal(Element):
    """An identified process with isolated context and capability view.

    Principal is the execution authority object. Morphisms execute within a
    Principal context and have their capabilities checked against rights().

    Immutability:
        caps is frozen — use grant() to produce a new Principal with additional
        capabilities. This prevents mid-execution capability escalation.

    Subject binding:
        rights() only returns rights from Capabilities whose subject matches
        this Principal's id. Capabilities bound to other Principals are ignored.
    """

    name: str = Field(default="default")
    ctx: dict[str, Any] = Field(default_factory=dict)
    caps: tuple[Capability, ...] = Field(default_factory=tuple, frozen=True)
    lineage: tuple[UUID, ...] = Field(default_factory=tuple)
    tags: tuple[str, ...] = Field(default_factory=tuple)

    @field_validator("caps", mode="before")
    @classmethod
    def _validate_caps(cls, v: Any) -> tuple[Capability, ...]:
        if isinstance(v, (list, tuple)):
            out = []
            for item in v:
                if isinstance(item, Capability):
                    out.append(item)
                elif isinstance(item, dict):
                    subj = item["subject"]
                    if isinstance(subj, str):
                        subj = UUID(subj)
                    rights = item["rights"]
                    if isinstance(rights, (list, tuple)):
                        rights = frozenset(rights)
                    out.append(Capability(subject=subj, rights=rights))
                else:
                    out.append(item)
            return tuple(out)
        return v

    @field_serializer("caps")
    def _serialize_caps(self, caps: tuple[Capability, ...]) -> list[dict]:
        return [
            {"subject": str(c.subject), "rights": sorted(c.rights)}
            for c in caps
        ]

    def rights(self) -> frozenset[str]:
        """Flatten rights from capabilities bound to THIS principal only (#1)."""
        return frozenset(
            r for c in self.caps if c.subject == self.id for r in c.rights
        )

    def has_right(self, right: str) -> bool:
        return right in self.rights()

    def grant(self, *new_rights: str) -> Principal:
        """Return a new Principal with additional rights. Preserves identity."""
        new_cap = Capability(subject=self.id, rights=frozenset(new_rights))
        data = self.model_dump()
        data["caps"] = list(self.caps) + [new_cap]
        data["id"] = self.id
        return Principal(**data)

class Observation(msgspec.Struct, kw_only=True):
    """Structured event record emitted during execution."""

    id: UUID = msgspec.field(default_factory=uuid4)
    ts: datetime = msgspec.field(default_factory=now_utc)
    who: UUID = msgspec.field(default_factory=uuid4)
    what: str = ""
    payload: dict[str, Any] = msgspec.field(default_factory=dict)
    lineage: tuple[UUID, ...] = msgspec.field(default_factory=tuple)
    tags: tuple[str, ...] = msgspec.field(default_factory=tuple)
