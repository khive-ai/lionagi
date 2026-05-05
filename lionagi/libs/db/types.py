# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Type annotations for database fields.

Provides semantic typing for foreign keys and vector embeddings:
    FK[Model]   - Foreign key references to entity types
    Vector[dim] - pgvector embeddings with dimension

Extraction:
    extract_db_meta(source, metas="BOTH")
        Unified extraction from FieldInfo, annotations, or Spec objects.
"""

from __future__ import annotations

import re
import types
from typing import Annotated, Any, ForwardRef, Literal, Union, get_args, get_origin
from uuid import UUID

from lionagi.ln.types._sentinel import Unset, UnsetType, not_sentinel


def _is_field_info(obj: Any) -> bool:
    """Runtime check for Pydantic FieldInfo without hard import."""
    return type(obj).__name__ == "FieldInfo" and hasattr(obj, "metadata")


__all__ = [
    "FK",
    "FKMeta",
    "Vector",
    "VectorMeta",
    "extract_db_meta",
    "parse_forward_ref",
]


# =============================================================================
# Foreign Key
# =============================================================================


class FKMeta:
    """Metadata for foreign key fields.

    Carries:
    - model: Referenced Entity/Node class (or string for forward refs)
    - column: Referenced column (default "id")
    - on_delete/on_update: Referential actions
    - deferrable/initially_deferred: Constraint deferral

    Example:
        tenant_id: FK[Tenant]  # FKMeta(Tenant, "id", "CASCADE", "CASCADE")
    """

    __slots__ = (
        "column",
        "deferrable",
        "initially_deferred",
        "model",
        "on_delete",
        "on_update",
    )

    def __init__(
        self,
        model: type | str,
        column: str = "id",
        on_delete: str = "CASCADE",
        on_update: str = "CASCADE",
        deferrable: bool = False,
        initially_deferred: bool = False,
    ):
        self.model = model
        self.column = column
        self.on_delete = on_delete
        self.on_update = on_update
        self.deferrable = deferrable
        self.initially_deferred = initially_deferred

    @property
    def table_name(self) -> str:
        """Get referenced table name from model's config or convention."""
        if isinstance(self.model, str):
            return self.model.lower() + "s"
        if hasattr(self.model, "node_config"):
            config = self.model.node_config
            if config and hasattr(config, "table_name"):
                return config.table_name
        if hasattr(self.model, "_table_name"):
            return self.model._table_name
        return self.model.__name__.lower() + "s"

    @property
    def is_resolved(self) -> bool:
        """Check if FK reference has been resolved to a class."""
        return not isinstance(self.model, str)

    def resolve(self, model_cls: type) -> None:
        """Resolve string reference to actual class."""
        self.model = model_cls

    def __repr__(self) -> str:
        name = self.model if isinstance(self.model, str) else self.model.__name__
        return f"FK[{name}]"


class _FK:
    """Foreign key type annotation: FK[Model] -> Annotated[UUID, FKMeta(Model)]."""

    def __class_getitem__(cls, model: type | str) -> Any:
        return Annotated[UUID, FKMeta(model)]


FK = _FK


# =============================================================================
# Vector (pgvector)
# =============================================================================


class VectorMeta:
    """Metadata for vector embedding fields.

    Carries dimension for pgvector VECTOR(dim) type.

    Example:
        embedding: Vector[1536]  # VectorMeta(1536)
    """

    __slots__ = ("dim",)

    def __init__(self, dim: int):
        if dim <= 0:
            raise ValueError(f"Vector dimension must be positive, got {dim}")
        self.dim = dim

    def __repr__(self) -> str:
        return f"Vector[{self.dim}]"


class _Vector:
    """Vector type annotation: Vector[dim] -> Annotated[list[float], VectorMeta(dim)]."""

    def __class_getitem__(cls, dim: int) -> Any:
        return Annotated[list[float], VectorMeta(dim)]


Vector = _Vector


# =============================================================================
# Extraction
# =============================================================================

# Return type aliases for extract_db_meta
_MetaResult = FKMeta | VectorMeta | UnsetType
_BothResult = tuple[FKMeta | UnsetType, VectorMeta | UnsetType]


def _find_in_annotation(annotation: Any, meta_type: type) -> Any | None:
    """Find metadata of given type in an annotation (Annotated or Union)."""
    # Direct Annotated[T, Meta(...)]
    if get_origin(annotation) is Annotated:
        for arg in get_args(annotation):
            if isinstance(arg, meta_type):
                return arg

    # Union (T | None) with Annotated members
    origin = get_origin(annotation)
    if origin is Union or isinstance(annotation, types.UnionType):
        for member in get_args(annotation):
            if get_origin(member) is Annotated:
                for arg in get_args(member):
                    if isinstance(arg, meta_type):
                        return arg

    return None


def _find_in_field_info(field_info: Any, meta_type: type) -> Any | None:
    """Find metadata of given type in a Pydantic FieldInfo."""
    # Pydantic v2: metadata list
    if hasattr(field_info, "metadata"):
        for item in field_info.metadata:
            if isinstance(item, meta_type):
                return item
            # Pydantic may store Annotated types in metadata
            if get_origin(item) is Annotated:
                found = _find_in_annotation(item, meta_type)
                if found is not None:
                    return found

    # Fallback: check annotation
    annotation = getattr(field_info, "annotation", None)
    if annotation is not None:
        return _find_in_annotation(annotation, meta_type)

    return None


def _is_spec_like(obj: Any) -> bool:
    """Check if object looks like a Spec (duck typing to avoid hard import)."""
    return (
        hasattr(obj, "get")
        and hasattr(obj, "__class__")
        and "Spec" in type(obj).__name__
    )


def _detect_nullable(arg: str) -> bool:
    """Detect nullability from annotation string.

    Handles:
        - X | None, None | X (PEP 604 union)
        - Optional[X] (typing.Optional)
        - Union[X, None], Union[None, X] (typing.Union with nested types)
    """
    # PEP 604 style: X | None or None | X
    if re.search(r"\|\s*None\b", arg) or re.search(r"\bNone\s*\|", arg):
        return True
    # Optional[X]
    if re.search(r"\bOptional\s*\[", arg):
        return True
    # Union[..., None, ...] - if Union is present and None is anywhere in string
    # This handles nested types like Union[FK[User], None]
    return bool(re.search(r"\bUnion\s*\[", arg) and re.search(r"\bNone\b", arg))


def parse_forward_ref(
    fwd: ForwardRef,
) -> tuple[FKMeta | None, VectorMeta | None, bool]:
    """Parse FK/Vector metadata and nullability from a ForwardRef string.

    Canonical parser for ForwardRef annotations from 'from __future__ import annotations'.

    Handles:
        - FK[Model], FK["Model"], FK['Model'] (bare and string refs)
        - Vector[1536] (dimension as int literal)
        - Nullability: X | None, Optional[X], Union[X, None]

    Args:
        fwd: ForwardRef to parse

    Returns:
        Tuple of (fk_meta, vector_meta, is_nullable)
        - fk_meta: FKMeta if FK[...] found, else None
        - vector_meta: VectorMeta if Vector[dim] found, else None
        - is_nullable: True if nullable pattern detected
    """
    arg = fwd.__forward_arg__
    fk: FKMeta | None = None
    vec: VectorMeta | None = None
    nullable = _detect_nullable(arg)

    # Match FK[ModelName] or FK["ModelName"] or FK['ModelName']
    fk_match = re.search(r"FK\[(['\"]?)(\w+)\1\]", arg)
    if fk_match:
        model_name = fk_match.group(2)
        fk = FKMeta(model_name)

    # Match Vector[dim]
    vec_match = re.search(r"Vector\[(\d+)\]", arg)
    if vec_match:
        dim = int(vec_match.group(1))
        vec = VectorMeta(dim)

    return fk, vec, nullable


def _extract_from_forward_ref(
    fwd: ForwardRef,
) -> tuple[FKMeta | UnsetType, VectorMeta | UnsetType]:
    """Extract FK/Vector metadata from a ForwardRef (returns Unset for missing).

    Wrapper around parse_forward_ref for extract_db_meta compatibility.
    """
    fk, vec, _ = parse_forward_ref(fwd)
    return (fk if fk is not None else Unset, vec if vec is not None else Unset)


def extract_db_meta(
    from_: Any,
    metas: Literal["FK", "Vector", "BOTH"] = "BOTH",
) -> _MetaResult | _BothResult:
    """Extract FK and/or Vector metadata from a source.

    Unified extraction dispatching on source type:
    - FieldInfo: searches Pydantic metadata and annotation
    - type/annotation: searches Annotated/Union structure
    - Spec: reads spec metadata directly (if available)

    Args:
        from_: FieldInfo, type annotation, or Spec instance
        metas: What to extract - "FK", "Vector", or "BOTH"

    Returns:
        "FK" or "Vector": The meta object or Unset if not found
        "BOTH": Tuple of (fk_meta_or_Unset, vector_meta_or_Unset)
    """
    fk: FKMeta | UnsetType = Unset
    vec: VectorMeta | UnsetType = Unset

    if _is_field_info(from_):
        if metas in ("FK", "BOTH"):
            fk = _find_in_field_info(from_, FKMeta) or Unset
        if metas in ("Vector", "BOTH"):
            vec = _find_in_field_info(from_, VectorMeta) or Unset

    elif isinstance(from_, ForwardRef):
        # Handle ForwardRef from 'from __future__ import annotations'
        fk, vec = _extract_from_forward_ref(from_)

    elif get_origin(from_) is not None or isinstance(from_, type):
        # Raw type annotation
        if metas in ("FK", "BOTH"):
            fk = _find_in_annotation(from_, FKMeta) or Unset
        if metas in ("Vector", "BOTH"):
            vec = _find_in_annotation(from_, VectorMeta) or Unset

    elif _is_spec_like(from_):
        # Spec-like object (duck typed to avoid circular imports)
        if metas in ("FK", "BOTH"):
            fk_val = from_.get("as_fk", Unset)
            if not_sentinel(fk_val, additions={"none"}) and isinstance(fk_val, FKMeta):
                fk = fk_val
        if metas in ("Vector", "BOTH"):
            vec_val = from_.get("embedding", Unset)
            if not_sentinel(vec_val, additions={"none"}) and isinstance(vec_val, VectorMeta):
                vec = vec_val
    else:
        raise TypeError(
            f"from_ must be FieldInfo, type annotation, or Spec, got {type(from_).__name__}"
        )

    if metas == "FK":
        return fk
    if metas == "Vector":
        return vec
    return (fk, vec)
