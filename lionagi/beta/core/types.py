# Copyright (c) 2023-2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Core type definitions for lionagi.beta substrate."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4

import msgspec
from pydantic import Field, field_serializer, field_validator

from lionagi.protocols.generic.element import Element

__all__ = (
    "Capability",
    "Observation",
    "Principal",
    "now_utc",
)


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


class Capability(msgspec.Struct, frozen=True, kw_only=True):
    """Immutable capability token granting rights to a subject."""

    subject: UUID
    rights: frozenset[str]

    def __hash__(self) -> int:
        return hash((self.subject, self.rights))


class Principal(Element):
    """Execution authority with isolated context and capability-based access control.

    caps is frozen — use grant() to add rights without mid-execution escalation.
    rights() filters by subject, so capabilities minted for other Principals cannot escalate.
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
        return [{"subject": str(c.subject), "rights": sorted(c.rights)} for c in caps]

    def rights(self) -> frozenset[str]:
        """Return rights from capabilities bound to this principal only."""
        return frozenset(r for c in self.caps if c.subject == self.id for r in c.rights)

    def has_right(self, right: str) -> bool:
        return right in self.rights()

    def grant(self, *new_rights: str) -> Principal:
        """Return a new Principal with additional rights, preserving identity."""
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
