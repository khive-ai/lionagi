from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from msgspec import Struct, field

from lionagi.ln import now_utc


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
