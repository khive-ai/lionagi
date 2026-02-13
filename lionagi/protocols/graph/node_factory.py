# Copyright (c) 2023-2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

from pydantic import Field

__all__ = (
    "NodeConfig",
    "create_node",
)


@dataclass(frozen=True, slots=True)
class NodeConfig:
    """Immutable configuration for Node persistence and lifecycle behavior.

    Controls DB schema mapping, content handling, embedding support, and
    audit trail. Set as ``node_config`` ClassVar on Node subclasses or
    pass to :func:`create_node`.

    When ``node_config`` is None (the default for base Node), all
    lifecycle methods are no-ops, preserving backwards compatibility.

    Attributes:
        table_name: DB table name. None means no persistence.
        schema: DB schema (default "public").
        soft_delete: Enable soft_delete()/restore() lifecycle.
        versioning: Track integer version on each touch().
        content_hashing: Compute SHA-256 hash of content on touch().
        track_updated_at: Track last modification timestamp.
    """

    table_name: str | None = None
    schema: str = "public"
    soft_delete: bool = False
    versioning: bool = False
    content_hashing: bool = False
    track_updated_at: bool = False

    @property
    def is_persisted(self) -> bool:
        """True if table_name is set (node has DB backing)."""
        return self.table_name is not None

    @property
    def has_audit_fields(self) -> bool:
        """True if any audit/lifecycle tracking is enabled."""
        return self.content_hashing or self.soft_delete or self.versioning or self.track_updated_at


def compute_content_hash(content: Any) -> str:
    """Compute a SHA-256 hash of content for change detection.

    Args:
        content: Any JSON-serializable content.

    Returns:
        Hex-encoded SHA-256 hash string.
    """
    import json

    if content is None:
        data = b"null"
    elif isinstance(content, (str, bytes)):
        data = content.encode() if isinstance(content, str) else content
    else:
        try:
            data = json.dumps(content, sort_keys=True, default=str).encode()
        except (TypeError, ValueError):
            data = str(content).encode()
    return hashlib.sha256(data).hexdigest()


def create_node(
    name: str,
    *,
    table_name: str | None = None,
    schema: str = "public",
    soft_delete: bool = False,
    versioning: bool = False,
    content_hashing: bool = False,
    track_updated_at: bool = False,
    extra_fields: dict[str, tuple[type, Any]] | None = None,
    doc: str | None = None,
) -> type:
    """Create a Node subclass with configured lifecycle behavior.

    Factory function for creating Node subclasses with declarative
    configuration. The returned class has a ``node_config`` ClassVar
    and any extra fields specified.

    Args:
        name: Class name for the new Node subclass.
        table_name: DB table name (None = no persistence).
        schema: DB schema (default "public").
        soft_delete: Enable soft_delete()/restore() methods.
        versioning: Track version number on touch().
        content_hashing: Compute content hash on touch().
        track_updated_at: Track updated_at timestamp.
        extra_fields: Additional Pydantic fields as ``{name: (type, default)}``.
        doc: Docstring for the created class.

    Returns:
        A new Node subclass with the specified configuration.

    Example::

        Job = create_node(
            "Job",
            table_name="jobs",
            soft_delete=True,
            versioning=True,
        )
        job = Job(content={"title": "Engineer"})
        job.touch()
        assert job.metadata.get("version") == 1
    """
    from .node import Node

    config = NodeConfig(
        table_name=table_name,
        schema=schema,
        soft_delete=soft_delete,
        versioning=versioning,
        content_hashing=content_hashing,
        track_updated_at=track_updated_at,
    )

    # Build field annotations and defaults
    annotations: dict[str, type] = {}
    namespace: dict[str, Any] = {
        "node_config": config,
        "__annotations__": {},
    }

    if extra_fields:
        for field_name, (field_type, default) in extra_fields.items():
            annotations[field_name] = field_type
            namespace[field_name] = Field(default=default)

    if annotations:
        namespace["__annotations__"] = annotations

    if doc:
        namespace["__doc__"] = doc

    # Create the subclass
    cls = type(name, (Node,), namespace)
    cls.node_config = config
    return cls


# File: lionagi/protocols/graph/node_factory.py
