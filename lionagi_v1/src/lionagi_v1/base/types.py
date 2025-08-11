from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4

import msgspec


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


class Obj(msgspec.Struct, kw_only=True):
    """Observable atom: stable id + timestamp + lineage."""

    id: UUID = msgspec.field(default_factory=uuid4)
    ts: datetime = msgspec.field(default_factory=now_utc)
    lineage: tuple[UUID, ...] = msgspec.field(default_factory=tuple)
    tags: tuple[str, ...] = msgspec.field(default_factory=tuple)


class Observation(msgspec.Struct, kw_only=True):
    # Repeat Obj fields (avoid inheritance ambiguity across msgspec versions)
    id: UUID = msgspec.field(default_factory=uuid4)
    ts: datetime = msgspec.field(default_factory=now_utc)
    lineage: tuple[UUID, ...] = msgspec.field(default_factory=tuple)
    tags: tuple[str, ...] = msgspec.field(default_factory=tuple)

    # emitter id (e.g., Branch id)
    who: UUID = msgspec.field(default_factory=uuid4)
    what: str = ""  # "node.start" | "node.finish" | ...
    payload: dict[str, Any] = msgspec.field(default_factory=dict)


class Capability(msgspec.Struct, kw_only=True):
    subject: UUID  # Branch id
    rights: set[str]  # {"net.out", "fs.read:/data/*", ...}
    object: str = "*"  # optional resource scoping


class Branch(msgspec.Struct, kw_only=True):
    """Branch is a semantic 'space': isolated context + summary + capability view."""

    # Repeat Obj fields
    id: UUID = msgspec.field(default_factory=uuid4)
    ts: datetime = msgspec.field(default_factory=now_utc)
    lineage: tuple[UUID, ...] = msgspec.field(default_factory=tuple)
    tags: tuple[str, ...] = msgspec.field(default_factory=tuple)

    name: str = "default"
    ctx: dict[str, Any] = msgspec.field(default_factory=dict)
    summary: dict[str, Any] = msgspec.field(default_factory=dict)
    caps: tuple[Capability, ...] = msgspec.field(default_factory=tuple)
