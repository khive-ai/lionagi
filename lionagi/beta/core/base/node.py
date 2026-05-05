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
    """Immutable configuration for Node persistence, embedding, and audit lifecycle."""

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
    """Persistable element with typed content, DB persistence, and audit lifecycle."""

    node_config: ClassVar[NodeConfig | None] = None
    content: dict[str, Any] | Serializable | BaseModel | UnsetType | None = None

    _resolved_content_type: ClassVar[type | None] = None

    @classmethod
    def get_config(cls) -> NodeConfig:
        if cls.node_config is None:
            return NodeConfig()
        return cls.node_config

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: Any) -> None:
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
        if value is None or is_sentinel(value):
            return None
        # DataClass (RoledContent etc.) — use to_dict() which strips sentinels
        if hasattr(value, "to_dict") and not isinstance(value, BaseModel):
            return value.to_dict(mode="json")
        return json_dumps(value, as_loaded=True)

    @field_validator("content", mode="before")
    @classmethod
    def _validate_content(cls, value: Any) -> Any:
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
        """Serialize node; when mode='db' and flatten_content=True, spreads content fields into result dict."""
        config = self.get_config()

        content_type = (
            config.content_type
            if not config.is_sentinel_field("content_type")
            else self._resolved_content_type
        )

        # Only flatten typed BaseModel content — untyped content cannot be round-tripped from flat rows
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

            effective_meta_key = meta_key if not is_unset(meta_key) else config.meta_key

            result = super().to_dict(
                mode=mode,
                created_at_format=created_at_format,
                meta_key=effective_meta_key,
                **kwargs,
            )

            content_dict = self.content.model_dump(mode="json")  # type: ignore[union-attr]
            result.update(content_dict)
            return result

        if content_serializer is not None:
            if not callable(content_serializer):
                typ = type(content_serializer).__name__
                raise TypeError(f"content_serializer must be callable, got {typ}")

            exclude = kwargs.get("exclude", set())
            if isinstance(exclude, set):
                exclude = exclude | {"content"}
            elif isinstance(exclude, dict):
                exclude = exclude.copy()
                exclude["content"] = True
            else:
                exclude = {"content"}
            kwargs["exclude"] = exclude

            result = super().to_dict(
                mode=mode,
                created_at_format=created_at_format,
                meta_key=meta_key,
                **kwargs,
            )

            result["content"] = content_serializer(self.content)
            return result

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
        """Deserialize dict to Node; from_row=True reconstructs flattened DB rows."""
        data = data.copy()
        config = cls.get_config()

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

        if not is_unset(effective_meta_key) and effective_meta_key in data:
            data["metadata"] = data.pop(effective_meta_key)
        elif "node_metadata" in data and "metadata" not in data:
            data["metadata"] = data.pop("node_metadata")
        data.pop("node_metadata", None)

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
        return name in self.__class__.model_fields

    def rehash(self) -> str | None:
        config = self.get_config()
        if not config.content_hashing:
            return None

        new_hash = compute_hash(self.content, none_as_valid=True)

        if self._has_field("content_hash"):
            self.content_hash = new_hash
        else:
            self.metadata["content_hash"] = new_hash

        return new_hash

    def update_integrity_hash(self, previous_hash: str | None = None) -> str | None:
        """Compute chain hash for tamper-evident audit trail; None for genesis entry."""
        from lionagi.ln._hash import compute_chain_hash

        config = self.get_config()
        if not config.integrity_hashing:
            return None

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
            self.deleted_by = None

        self.touch(by)

    def activate(self, by: UUID | str | None = None) -> None:
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
    embedding_enabled: bool = False,
    embedding_dim: int | None = None,
    table_name: str | None = None,
    schema: str = "public",
    flatten_content: bool = True,
    immutable: bool = False,
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
    """Factory that creates a Node subclass with enforced NodeConfig validation at class creation time."""
    from lionagi.ln.types.adapters._utils import AuditSpecs, ContentSpecs
    from lionagi.ln.types import Operable

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

    meta_key = config_kwargs.get("meta_key", "node_metadata")
    all_specs = ContentSpecs.get_specs(
        content_type=content if content else Unset,
        dim=resolved_embedding_dim,
        meta_key=meta_key,
    ) + AuditSpecs.get_specs(use_uuid=True)

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
    """Generate CREATE TABLE DDL from a Node subclass."""
    from lionagi.ln.types.adapters._utils import AuditSpecs, ContentSpecs
    from lionagi.ln.types import Operable

    config = node_cls.get_config()
    if not config.is_persisted:
        raise ValueError(f"{node_cls.__name__} is not persistable (no table_name configured)")

    content_type = (
        config.content_type
        if not config.is_sentinel_field("content_type")
        else _extract_base_type(node_cls._resolved_content_type)
    )

    all_specs = ContentSpecs.get_specs(
        dim=config.embedding_dim if config.embedding_enabled else Unset,
        meta_key=config.meta_key,
    ) + AuditSpecs.get_specs(use_uuid=True)

    if config.flatten_content and content_type is not None:
        from lionagi.ln.types.adapters._pydantic import PydanticSpecAdapter

        if isinstance(content_type, type) and issubclass(content_type, BaseModel):
            all_specs.extend(PydanticSpecAdapter.extract_specs(content_type))

    include: set[str] = {"id", "created_at"}

    if config.embedding_enabled:
        include.add("embedding")

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

    if (
        config.flatten_content
        and content_type is not None
        and isinstance(content_type, type)
        and issubclass(content_type, BaseModel)
    ):
        include.update(content_type.model_fields.keys())

    op = Operable(all_specs, adapter="sql")
    return op.compose_structure(
        config.table_name,
        include=include,
        schema=config.schema,
        primary_key="id",
    )


def generate_all_ddl(*, schema: str | None = None) -> str:
    """Generate DDL for all registered persistable Node subclasses."""
    statements: list[str] = []

    for node_cls in PERSISTABLE_NODE_REGISTRY.values():
        config = node_cls.get_config()

        if schema is not None and config.schema != schema:
            continue

        ddl = generate_ddl(node_cls)
        statements.append(ddl)

    return "\n\n".join(statements)


def get_fk_dependencies(node_cls: type[Node]) -> set[str]:
    """Return table names this node references via FK[Model], for migration ordering."""

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
