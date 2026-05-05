# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Node: Persistable element with structured content and polymorphic serialization.

Provides Node (extends Element), NodeConfig, create_node factory, and DDL generation.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal
from uuid import UUID

from pydantic import BaseModel, field_serializer, field_validator

from lionagi.ln.types import ModelConfig, Params
from lionagi.ln.types._sentinel import (
    Unset,
    UnsetType,
    is_sentinel,
    is_unset,
    not_sentinel,
)
from lionagi.libs.db.types import VectorMeta, extract_db_meta
from lionagi._errors import NotAllowedError
from lionagi.beta.protocols import Deserializable, Serializable, implements
from lionagi.ln._hash import compute_hash
from lionagi.ln._json_dump import json_dumps
from lionagi.ln._utils import now_utc

from .element import Element

# --- Registries ---
# NODE_REGISTRY: Polymorphic lookup by class name (full or short)
# PERSISTABLE_NODE_REGISTRY: DB-bound nodes by table_name (DDL generation)

NODE_REGISTRY: dict[str, type[Node]] = {}
PERSISTABLE_NODE_REGISTRY: dict[str, type[Node]] = {}


def _register_persistable(table_name: str, cls: type[Node]) -> None:
    """Register Node class for DB persistence. Idempotent, detects collisions."""
    if table_name in PERSISTABLE_NODE_REGISTRY:
        existing = PERSISTABLE_NODE_REGISTRY[table_name]
        if existing is not cls:
            raise ValueError(
                f"Table '{table_name}' already registered by "
                f"{existing.__module__}.{existing.__name__}, "
                f"cannot register {cls.__module__}.{cls.__name__}"
            )
        return
    PERSISTABLE_NODE_REGISTRY[table_name] = cls


def _enable_embedding_requires_dim(config: NodeConfig) -> None:
    """Validate: embedding_enabled requires positive embedding_dim."""
    if config.embedding_enabled:
        if config.is_sentinel_field("embedding_dim"):
            raise ValueError("embedding_dim must be specified when embedding is enabled")
        if config.embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be positive, got {config.embedding_dim}")


def _only_typed_content_can_flatten(config: NodeConfig) -> None:
    """Validate: flatten_content requires explicit content_type."""
    if config.flatten_content and config.is_sentinel_field("content_type"):
        raise ValueError("content_type must be specified when flatten_content is True")


@dataclass(frozen=True, slots=True, init=False)
class NodeConfig(Params):
    """Immutable configuration for Node persistence and behavior.

    Controls DB schema mapping, content handling, embedding support, and audit trail.
    Pass to create_node() or set as class attribute on Node subclasses.

    Field Groups:
        DB Mapping: table_name, schema, meta_key
        Embedding: embedding_enabled, embedding_dim, embedding_format
        Time: time_format, timezone
        Polymorphism: polymorphic, registry_key
        Content: flatten_content, content_frozen, content_nullable, content_type
        Audit: content_hashing, integrity_hashing, soft_delete, versioning, track_*

    Validation Rules:
        - embedding_enabled=True requires positive embedding_dim
        - flatten_content=True requires explicit content_type

    Usage:
        # Via create_node (preferred)
        Job = create_node("Job", table_name="jobs", soft_delete=True)

        # Via class attribute (advanced)
        class Job(Node):
            node_config = NodeConfig(table_name="jobs", soft_delete=True)
    """

    _config: ClassVar[ModelConfig] = ModelConfig(
        sentinel_additions=frozenset({"none", "empty"}),
        prefill_unset=False,
    )

    # DB Mapping
    table_name: str | UnsetType = Unset
    schema: str = "public"
    meta_key: str = "node_metadata"

    # Embedding
    embedding_enabled: bool = False
    embedding_dim: int | UnsetType = Unset
    embedding_format: Literal["pgvector", "jsonb", "list"] = "pgvector"

    # Time
    time_format: Literal["datetime", "isoformat", "timestamp"] = "isoformat"
    timezone: str = "UTC"

    # Polymorphism
    polymorphic: bool = False
    registry_key: str | UnsetType = Unset

    # Content
    flatten_content: bool = False
    content_frozen: bool = False
    content_nullable: bool = False
    content_type: type | UnsetType = Unset

    # Audit & Lifecycle
    content_hashing: bool = False
    integrity_hashing: bool = False
    soft_delete: bool = False
    track_deleted_by: bool = False
    track_is_active: bool = False
    versioning: bool = False
    track_updated_at: bool = False
    track_updated_by: bool = False

    # Additional
    db_extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Run validation rules after initialization."""
        _enable_embedding_requires_dim(self)
        _only_typed_content_can_flatten(self)

    @property
    def is_persisted(self) -> bool:
        """True if table_name is set (node has DB backing)."""
        return not self.is_sentinel_field("table_name")

    @property
    def has_audit_fields(self) -> bool:
        """True if any audit/lifecycle tracking is enabled."""
        return (
            self.content_hashing
            or self.integrity_hashing
            or self.soft_delete
            or self.versioning
            or self.track_updated_at
        )


@implements(
    Deserializable,
    Serializable,
)
class Node(Element):
    """Persistable element with structured content and polymorphic serialization.

    Extends Element with:
        - NodeConfig: DB persistence, audit trail, embedding support
        - content: Typed field (BaseModel, Serializable, dict, or None)
        - Polymorphic from_dict/to_dict via NODE_REGISTRY lookup

    Class Attributes:
        node_config: NodeConfig instance (None = default config)
        content: Structured payload (validated, serializable)

    Lifecycle Methods (config-dependent):
        touch(by): Update timestamps, version, rehash
        soft_delete(by): Mark deleted (reversible)
        restore(by): Undelete
        activate(by): Mark active (requires track_is_active)
        deactivate(by): Mark inactive (requires track_is_active)
        rehash(): Recompute content_hash

    See Also:
        create_node(): Factory for Node subclasses with enforced config
        generate_ddl(): Generate CREATE TABLE from Node class

    """

    node_config: ClassVar[NodeConfig | None] = None
    content: dict[str, Any] | Serializable | BaseModel | UnsetType | None = None

    _resolved_content_type: ClassVar[type | None] = None

    @classmethod
    def get_config(cls) -> NodeConfig:
        """Return node_config or default NodeConfig if not set."""
        if cls.node_config is None:
            return NodeConfig()
        return cls.node_config

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: Any) -> None:
        """Auto-register in NODE_REGISTRY and PERSISTABLE_NODE_REGISTRY."""
        super().__pydantic_init_subclass__(**kwargs)

        config = cls.get_config()

        # Register in NODE_REGISTRY (polymorphic lookup)
        if config.polymorphic:
            registry_key = (
                cls.class_name(full=True)
                if config.is_sentinel_field("registry_key")
                else config.registry_key
            )
            NODE_REGISTRY[registry_key] = cls

        # Register in PERSISTABLE_NODE_REGISTRY (DB persistence)
        if config.is_persisted:
            _register_persistable(config.table_name, cls)

        # Store resolved content type from annotation if not explicit in config
        if config.is_sentinel_field("content_type") and "content" in cls.model_fields:
            content_field = cls.model_fields["content"]
            if content_field.annotation is not None:
                # Store for DDL generation (don't modify frozen config)
                cls._resolved_content_type = content_field.annotation
            else:
                cls._resolved_content_type = None
        else:
            cls._resolved_content_type = (
                None if config.is_sentinel_field("content_type") else config.content_type
            )

    @field_serializer("content")
    def _serialize_content(self, value: Any) -> Any:
        """Serialize content to JSON-compatible dict."""
        if value is None or is_sentinel(value):
            return None
        # DataClass (RoledContent etc.) — use to_dict() which strips sentinels
        if hasattr(value, "to_dict") and not isinstance(value, BaseModel):
            return value.to_dict(mode="json")
        return json_dumps(value, as_loaded=True)

    @field_validator("content", mode="before")
    @classmethod
    def _validate_content(cls, value: Any) -> Any:
        """Validate content type and handle polymorphic deserialization."""
        if is_sentinel(value):
            return value

        if value is not None and not isinstance(value, (Serializable, BaseModel, dict)):
            raise ValueError(
                f"content must be Serializable, BaseModel, dict, or None. "
                f"Got {type(value).__name__}. "
                f"Use dict for unstructured data: content={{'value': {value!r}}} "
                f"or Element.metadata for simple key-value pairs."
            )

        # Polymorphic: restore type from class metadata
        if isinstance(value, dict) and "metadata" in value:
            metadata = value.get("metadata", {})
            kron_class = metadata.get("kron_class")
            if kron_class:
                if kron_class in NODE_REGISTRY or kron_class.split(".")[-1] in NODE_REGISTRY:
                    return Node.from_dict(value)
                return Element.from_dict(value)
        return value

    def to_dict(
        self,
        mode: Literal["python", "json", "db"] = "python",
        created_at_format: (Literal["datetime", "isoformat", "timestamp"] | UnsetType) = Unset,
        meta_key: str | UnsetType = Unset,
        content_serializer: Callable[[Any], Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Serialize to dict with optional custom content handling.

        Args:
            mode: "python" (native types), "json" (JSON-safe), "db" (DB-ready)
            created_at_format: Override time format for created_at
            meta_key: Rename metadata field (e.g., "node_metadata" for DB)
            content_serializer: Custom serializer for content field
            **kwargs: Passed to model_dump()

        Returns:
            Serialized dict. If content_serializer provided, content is
            excluded from model_dump and replaced with serializer output.

        Notes:
            When mode="db" and config.flatten_content=True, content fields
            are spread into the result dict (no "content" key). This matches
            the flattened DDL schema.

        """
        config = self.get_config()

        # Resolve content type for flattening decision
        content_type = (
            config.content_type
            if not config.is_sentinel_field("content_type")
            else self._resolved_content_type
        )

        # DB mode with flatten_content: spread content fields into result
        # Only flatten when we have a typed BaseModel content that can be reconstructed
        can_flatten = (
            config.flatten_content
            and self.content is not None
            and content_type is not None
            and isinstance(content_type, type)
            and issubclass(content_type, BaseModel)
        )

        if mode == "db" and can_flatten:
            # Exclude content from base serialization
            exclude = kwargs.get("exclude", set())
            if isinstance(exclude, set):
                exclude = exclude | {"content"}
            elif isinstance(exclude, dict):
                exclude = exclude.copy()
                exclude["content"] = True
            else:
                exclude = {"content"}
            kwargs["exclude"] = exclude

            # Use config.meta_key for DB mode if not overridden
            effective_meta_key = meta_key if not is_unset(meta_key) else config.meta_key

            # Get base dict without content
            result = super().to_dict(
                mode=mode,
                created_at_format=created_at_format,
                meta_key=effective_meta_key,
                **kwargs,
            )

            # Flatten content fields into result (content is BaseModel per can_flatten check)
            content_dict = self.content.model_dump(mode="json")  # type: ignore[union-attr]
            result.update(content_dict)
            return result

        # Custom content serializer
        if content_serializer is not None:
            if not callable(content_serializer):
                typ = type(content_serializer).__name__
                raise TypeError(f"content_serializer must be callable, got {typ}")

            # Exclude content from model_dump
            exclude = kwargs.get("exclude", set())
            if isinstance(exclude, set):
                exclude = exclude | {"content"}
            elif isinstance(exclude, dict):
                exclude = exclude.copy()
                exclude["content"] = True
            else:
                exclude = {"content"}
            kwargs["exclude"] = exclude

            # Get dict without content
            result = super().to_dict(
                mode=mode,
                created_at_format=created_at_format,
                meta_key=meta_key,
                **kwargs,
            )

            # Add serialized content
            result["content"] = content_serializer(self.content)
            return result

        # Delegate to Element.to_dict
        return super().to_dict(
            mode=mode,
            created_at_format=created_at_format,
            meta_key=meta_key,
            **kwargs,
        )

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        meta_key: str | UnsetType = Unset,
        content_deserializer: Callable[[Any], Any] | None = None,
        from_row: bool = False,
        **kwargs: Any,
    ) -> Node:
        """Deserialize dict to Node with polymorphic type restoration.

        Looks up kron_class in metadata to restore original Node subclass.
        Handles legacy "node_metadata" key and custom meta_key mapping.

        Args:
            data: Dict from to_dict() or DB row
            meta_key: Custom metadata field name to restore
            content_deserializer: Transform content before validation
            from_row: If True and config.flatten_content, extract content fields
                from flattened row data (inverse of to_dict(mode="db"))
            **kwargs: Passed to model_validate()

        Returns:
            Node instance (or appropriate subclass via NODE_REGISTRY lookup)

        """
        data = data.copy()
        config = cls.get_config()

        # Handle flattened DB row: extract content fields and reconstruct content
        if from_row and config.flatten_content and "content" not in data:
            content_type = (
                config.content_type
                if not config.is_sentinel_field("content_type")
                else cls._resolved_content_type
            )
            if (
                content_type is not None
                and isinstance(content_type, type)
                and issubclass(content_type, BaseModel)
            ):
                content_field_names = set(content_type.model_fields.keys())
                content_data = {k: v for k, v in data.items() if k in content_field_names}
                for k in content_field_names:
                    data.pop(k, None)
                data["content"] = content_type(**content_data)

        # Handle meta_key for DB rows
        effective_meta_key = (
            meta_key if not is_unset(meta_key) else (config.meta_key if from_row else Unset)
        )

        if content_deserializer is not None:
            if not callable(content_deserializer):
                typ = type(content_deserializer).__name__
                raise TypeError(f"content_deserializer must be callable, got {typ}")
            if "content" in data:
                try:
                    data["content"] = content_deserializer(data["content"])
                except Exception as e:
                    raise ValueError(f"content_deserializer failed: {e}") from e

        # Restore metadata from custom key (meta_key or legacy "node_metadata")
        if not is_unset(effective_meta_key) and effective_meta_key in data:
            data["metadata"] = data.pop(effective_meta_key)
        elif "node_metadata" in data and "metadata" not in data:
            data["metadata"] = data.pop("node_metadata")
        data.pop("node_metadata", None)

        # Extract kron_class for polymorphic dispatch (remove from metadata)
        metadata = data.get("metadata", {})
        if isinstance(metadata, dict):
            metadata = metadata.copy()
            data["metadata"] = metadata
            kron_class = metadata.pop("kron_class", None)
        else:
            kron_class = None

        if kron_class and kron_class != cls.class_name(full=True):
            target_cls = NODE_REGISTRY.get(kron_class) or NODE_REGISTRY.get(
                kron_class.split(".")[-1]
            )
            if target_cls is not None and target_cls is not cls:
                return target_cls.from_dict(
                    data,
                    content_deserializer=content_deserializer,
                    from_row=from_row,
                    **kwargs,
                )

        return cls.model_validate(data, **kwargs)

    # --- Audit & Lifecycle ---

    def _has_field(self, name: str) -> bool:
        """Check if name is a declared model field (not property/method)."""
        return name in self.__class__.model_fields

    def rehash(self) -> str | None:
        """Recompute and store content_hash. Returns hash or None if disabled."""
        config = self.get_config()
        if not config.content_hashing:
            return None

        new_hash = compute_hash(self.content, none_as_valid=True)

        # Store in field if it exists, otherwise in metadata
        if self._has_field("content_hash"):
            self.content_hash = new_hash
        else:
            self.metadata["content_hash"] = new_hash

        return new_hash

    def update_integrity_hash(self, previous_hash: str | None = None) -> str | None:
        """Compute chain hash for tamper-evident audit trail.

        Args:
            previous_hash: Previous entry's hash (None for genesis/first entry)

        Returns:
            Computed integrity_hash, or None if integrity_hashing disabled

        """
        from lionagi.ln._hash import compute_chain_hash

        config = self.get_config()
        if not config.integrity_hashing:
            return None

        # Use existing content_hash or compute on-the-fly
        content_hash = None
        if self._has_field("content_hash"):
            content_hash = self.content_hash
        elif "content_hash" in self.metadata:
            content_hash = self.metadata.get("content_hash")
        if content_hash is None:
            content_hash = compute_hash(self.content, none_as_valid=True)

        new_integrity_hash = compute_chain_hash(content_hash, previous_hash)

        if self._has_field("integrity_hash"):
            self.integrity_hash = new_integrity_hash
        else:
            self.metadata["integrity_hash"] = new_integrity_hash

        return new_integrity_hash

    def touch(self, by: UUID | str | None = None) -> None:
        """Update timestamps, increment version, and rehash (per config).

        Args:
            by: Actor identifier for updated_by field

        """
        config = self.get_config()

        if config.track_updated_at and self._has_field("updated_at"):
            self.updated_at = now_utc()
        if by is not None and self._has_field("updated_by"):
            self.updated_by = str(by)
        if config.versioning and self._has_field("version"):
            self.version += 1
        if config.content_hashing:
            self.rehash()

    def soft_delete(self, by: UUID | str | None = None) -> None:
        """Mark as deleted (reversible). Requires soft_delete=True in config.

        Args:
            by: Actor identifier for deleted_by field

        Raises:
            NotAllowedError: If soft_delete not enabled

        """
        config = self.get_config()
        if not config.soft_delete:
            raise NotAllowedError(
                f"{self.__class__.__name__} does not support soft_delete. "
                f"Enable with create_node(..., soft_delete=True)"
            )

        if self._has_field("deleted_at"):
            self.deleted_at = now_utc()
        if self._has_field("is_deleted"):
            self.is_deleted = True
        if by is not None and self._has_field("deleted_by"):
            self.deleted_by = str(by)

        self.touch(by)

    def restore(self, by: UUID | str | None = None) -> None:
        """Undelete a soft-deleted node. Requires soft_delete=True in config.

        Args:
            by: Actor identifier for updated_by (deleted_by is cleared)

        Raises:
            NotAllowedError: If soft_delete not enabled

        """
        config = self.get_config()
        if not config.soft_delete:
            raise NotAllowedError(
                f"{self.__class__.__name__} does not support restore. "
                f"Enable with create_node(..., soft_delete=True)"
            )

        if self._has_field("deleted_at"):
            self.deleted_at = None
        if self._has_field("is_deleted"):
            self.is_deleted = False
        if self._has_field("deleted_by"):
            self.deleted_by = None  # Clear who deleted on restore

        self.touch(by)

    def activate(self, by: UUID | str | None = None) -> None:
        """Mark as active. Requires track_is_active=True in config.

        Args:
            by: Actor identifier for updated_by field

        Raises:
            NotAllowedError: If track_is_active not enabled

        """
        config = self.get_config()
        if not config.track_is_active:
            raise NotAllowedError(
                f"{self.__class__.__name__} does not support activate. "
                f"Enable with create_node(..., track_is_active=True)"
            )
        if self._has_field("is_active"):
            self.is_active = True
        self.touch(by)

    def deactivate(self, by: UUID | str | None = None) -> None:
        """Mark as inactive. Requires track_is_active=True in config.

        Args:
            by: Actor identifier for updated_by field

        Raises:
            NotAllowedError: If track_is_active not enabled

        """
        config = self.get_config()
        if not config.track_is_active:
            raise NotAllowedError(
                f"{self.__class__.__name__} does not support deactivate. "
                f"Enable with create_node(..., track_is_active=True)"
            )
        if self._has_field("is_active"):
            self.is_active = False
        self.touch(by)


NODE_REGISTRY[Node.__name__] = Node
NODE_REGISTRY[Node.class_name(full=True)] = Node


# --- Node Factory ---


def create_node(
    name: str,
    *,
    content: type[BaseModel] | None = None,
    embedding: Any | None = None,  # Vector[dim] annotation
    embedding_enabled: bool = False,  # Alternative: enable with dim
    embedding_dim: int | None = None,  # Alternative: specify dimension
    table_name: str | None = None,
    schema: str = "public",
    flatten_content: bool = True,
    immutable: bool = False,
    # Audit & lifecycle options
    content_hashing: bool = False,
    integrity_hashing: bool = False,
    soft_delete: bool = False,
    track_deleted_by: bool = False,
    track_is_active: bool = False,
    versioning: bool = False,
    track_updated_at: bool = True,
    track_updated_by: bool = True,
    doc: str | None = None,
    **config_kwargs: Any,
) -> type[Node]:
    """Create Node subclass with typed content, embedding, and audit fields.

    Factory ensures NodeConfig validation at class creation. Fields are
    generated from Spec catalog, not just configured.

    Args:
        name: Class name (e.g., "Job", "Evidence")
        content: BaseModel for typed content (FK[Model] preserved for DDL)
        embedding: Vector[dim] annotation (adds embedding: list[float] | None)
        embedding_enabled: Alternative to embedding - enable with explicit dim
        embedding_dim: Dimension when using embedding_enabled=True
        table_name: DB table name (registers in PERSISTABLE_NODE_REGISTRY)
        schema: DB schema (default: "public")
        flatten_content: Flatten content fields in DDL (default: True)
        immutable: Freeze content (append-only pattern)
        content_hashing: SHA-256 hash on content changes
        integrity_hashing: Chain hash for audit trail
        soft_delete: Enable soft_delete()/restore() methods
        track_deleted_by: Track deleted_by (requires soft_delete)
        track_is_active: Add is_active field with activate()/deactivate()
        versioning: Track version number
        track_updated_at: Add updated_at timestamp (default: True)
        track_updated_by: Track updated_by actor (default: True)
        **config_kwargs: Additional NodeConfig parameters

    Returns:
        Node subclass with configured fields and lifecycle methods.

    Example:
        >>> # Option 1: Vector annotation
        >>> Job = create_node("Job", embedding=Vector[1536])
        >>>
        >>> # Option 2: Explicit enable + dim (preferred for tests)
        >>> Job = create_node("Job", embedding_enabled=True, embedding_dim=1536)

    """
    from lionagi.ln.types.adapters._utils import AuditSpecs, ContentSpecs
    from lionagi.ln.types import Operable

    # Resolve embedding dimension
    resolved_embedding_dim: int | UnsetType = Unset
    has_embedding = False

    if embedding is not None:
        vec_meta = extract_db_meta(embedding, metas="Vector")
        if isinstance(vec_meta, VectorMeta):
            resolved_embedding_dim = vec_meta.dim
            has_embedding = True
        else:
            raise ValueError(
                f"embedding must be Vector[dim] annotation, got {embedding}. "
                f"Use: embedding=Vector[1536]"
            )
    elif embedding_enabled:
        if embedding_dim is None or embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive when embedding_enabled=True")
        resolved_embedding_dim = embedding_dim
        has_embedding = True

    # 1. Build all possible specs
    # Extract meta_key from config_kwargs if provided, else use default
    meta_key = config_kwargs.get("meta_key", "node_metadata")
    all_specs = ContentSpecs.get_specs(
        content_type=content if content else Unset,
        dim=resolved_embedding_dim,
        meta_key=meta_key,
    ) + AuditSpecs.get_specs(use_uuid=True)

    # 2. Track which fields to include
    include: list[str] = ["id", "created_at"]

    if content is not None:
        include.append("content")
    if has_embedding:
        include.append("embedding")

    needs_update_tracking = (
        track_updated_at or content_hashing or integrity_hashing or soft_delete or versioning
    )
    if needs_update_tracking:
        include.append("updated_at")
        if track_updated_by:
            include.append("updated_by")
    if content_hashing:
        include.append("content_hash")
    if integrity_hashing:
        include.append("integrity_hash")
    if soft_delete:
        include.extend(["is_deleted", "deleted_at"])
        if track_deleted_by:
            include.append("deleted_by")
    if versioning:
        include.append("version")
    if track_is_active:
        include.append("is_active")

    # 3. Build config
    node_config = NodeConfig(
        table_name=table_name if table_name else Unset,
        schema=schema,
        embedding_enabled=has_embedding,
        embedding_dim=resolved_embedding_dim,
        content_type=content if content else Unset,
        content_frozen=immutable,
        flatten_content=flatten_content,
        content_hashing=content_hashing,
        integrity_hashing=integrity_hashing,
        soft_delete=soft_delete,
        track_deleted_by=track_deleted_by,
        track_is_active=track_is_active,
        versioning=versioning,
        track_updated_at=track_updated_at,
        track_updated_by=track_updated_by,
        **config_kwargs,
    )

    # 4. Compose Node subclass
    op = Operable(all_specs, adapter="pydantic")
    node_cls: type[Node] = op.compose_structure(
        name,
        include=set(include),
        base_type=Node,
        doc=doc,
    )
    node_cls.node_config = node_config  # type: ignore[attr-defined]

    return node_cls


# --- DDL Generation ---


def _extract_base_type(annotation: Any) -> Any:
    """Extract non-None type from Union (e.g., T | None -> T)."""
    import types
    from typing import get_args, get_origin

    if annotation is None:
        return None

    if isinstance(annotation, types.UnionType) or get_origin(annotation) is type(int | str):
        args = get_args(annotation)
        non_none_args = [a for a in args if a is not type(None)]
        if non_none_args:
            return non_none_args[0]

    return annotation


def generate_ddl(node_cls: type[Node]) -> str:
    """Generate CREATE TABLE DDL from Node subclass.

    Flattens content fields (if configured), adds audit columns, and
    generates PostgreSQL DDL with pgvector support for embeddings.

    Audit column inclusion is driven by NodeConfig settings (track_updated_at,
    soft_delete, versioning, etc.).

    Args:
        node_cls: Persistable Node subclass (must have table_name)

    Returns:
        CREATE TABLE IF NOT EXISTS statement

    Raises:
        ValueError: If node_cls has no table_name configured

    """
    from lionagi.ln.types.adapters._utils import AuditSpecs, ContentSpecs
    from lionagi.ln.types import Operable

    config = node_cls.get_config()
    if not config.is_persisted:
        raise ValueError(f"{node_cls.__name__} is not persistable (no table_name configured)")

    # 1. Build all possible specs for this node
    content_type = (
        config.content_type
        if not config.is_sentinel_field("content_type")
        else _extract_base_type(node_cls._resolved_content_type)
    )

    all_specs = ContentSpecs.get_specs(
        dim=config.embedding_dim if config.embedding_enabled else Unset,
        meta_key=config.meta_key,
    ) + AuditSpecs.get_specs(use_uuid=True)

    # Flatten content: extract fields from BaseModel instead of generic JSONB
    if config.flatten_content and content_type is not None:
        from lionagi.ln.types.adapters._pydantic import PydanticSpecAdapter

        if isinstance(content_type, type) and issubclass(content_type, BaseModel):
            all_specs.extend(PydanticSpecAdapter.extract_specs(content_type))

    # 2. Track which field names to include
    include: set[str] = {"id", "created_at"}

    if config.embedding_enabled:
        include.add("embedding")

    # Content column (unless flattened into individual fields)
    if not (
        config.flatten_content
        and content_type is not None
        and isinstance(content_type, type)
        and issubclass(content_type, BaseModel)
    ):
        include.add("content")

    if not config.is_sentinel_field("meta_key") and config.meta_key != "metadata":
        include.add(config.meta_key)

    audit_cols = {
        "updated_at": config.track_updated_at,
        "updated_by": config.track_updated_by,
        "is_active": config.track_is_active,
        "is_deleted": config.soft_delete,
        "deleted_at": config.soft_delete,
        "deleted_by": config.soft_delete and config.track_deleted_by,
        "version": config.versioning,
        "content_hash": config.content_hashing,
        "integrity_hash": config.integrity_hashing,
    }

    for col, enabled in audit_cols.items():
        if enabled:
            include.add(col)

    # If flattened, include the extracted content field names
    if (
        config.flatten_content
        and content_type is not None
        and isinstance(content_type, type)
        and issubclass(content_type, BaseModel)
    ):
        include.update(content_type.model_fields.keys())

    # 3. Compose DDL via Operable
    op = Operable(all_specs, adapter="sql")
    return op.compose_structure(
        config.table_name,
        include=include,
        schema=config.schema,
        primary_key="id",
    )


def generate_all_ddl(*, schema: str | None = None) -> str:
    """Generate DDL for all registered persistable Node subclasses.

    Iterates PERSISTABLE_NODE_REGISTRY and generates CREATE TABLE for each.

    Args:
        schema: Filter to specific schema (None = all schemas)

    Returns:
        Combined DDL statements separated by blank lines

    """
    statements: list[str] = []

    for node_cls in PERSISTABLE_NODE_REGISTRY.values():
        config = node_cls.get_config()

        if schema is not None and config.schema != schema:
            continue

        ddl = generate_ddl(node_cls)
        statements.append(ddl)

    return "\n\n".join(statements)


def get_fk_dependencies(node_cls: type[Node]) -> set[str]:
    """Get table names that this node depends on via foreign keys.

    Used for topological sorting in migrations - ensures tables are
    created in dependency order.

    Args:
        node_cls: Node subclass to analyze

    Returns:
        Set of table names this node references via FK[Model]
    """

    config = node_cls.get_config()
    content_type = (
        config.content_type
        if not config.is_sentinel_field("content_type")
        else node_cls._resolved_content_type
    )

    if content_type is None or not hasattr(content_type, "model_fields"):
        return set()

    deps: set[str] = set()
    for field_info in content_type.model_fields.values():
        fk = extract_db_meta(field_info, metas="FK")
        if not_sentinel(fk):
            deps.add(fk.table_name)
    return deps


__all__ = (
    "NODE_REGISTRY",
    "PERSISTABLE_NODE_REGISTRY",
    "Node",
    "NodeConfig",
    "create_node",
    "generate_all_ddl",
    "generate_ddl",
    "get_fk_dependencies",
)
