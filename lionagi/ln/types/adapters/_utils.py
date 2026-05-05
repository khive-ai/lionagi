from __future__ import annotations

import datetime as dt
import types
from functools import reduce
from typing import Annotated, Any, ForwardRef, Union, get_args, get_origin
from uuid import UUID, uuid4

from lionagi.libs.db.types import Vector
from lionagi.ln._utils import now_utc
from lionagi.ln.types._sentinel import Unset, UnsetType, is_unset

from ..spec import Spec

__all__ = (
    "AuditSpecs",
    "ContentSpecs",
    "resolve_annotation_to_base_types",
)


class ContentSpecs:
    @staticmethod
    def get_specs(
        *,
        content_type: type | UnsetType = Unset,
        dim: int | UnsetType = Unset,
        meta_key: str = "node_metadata",
    ) -> list[Spec]:
        content_base = dict if is_unset(content_type) else content_type
        specs = [
            Spec(content_base, name="content", nullable=True),
            Spec(dict, name=meta_key, default_factory=dict),
        ]
        if not is_unset(dim):
            specs.append(Spec(Vector[dim], name="embedding", nullable=True))
        return specs


class AuditSpecs:
    @staticmethod
    def get_specs(*, use_uuid: bool = True) -> list[Spec]:
        id_type = UUID if use_uuid else str
        id_factory = uuid4 if use_uuid else lambda: str(uuid4())
        return [
            Spec(id_type, name="id", default_factory=id_factory, frozen=True),
            Spec(dt.datetime, name="created_at", default_factory=now_utc, frozen=True),
            Spec(dt.datetime, name="updated_at", default_factory=now_utc, nullable=True),
            Spec(str, name="updated_by", nullable=True),
            Spec(bool, name="is_active", default=True),
            Spec(bool, name="is_deleted", default=False),
            Spec(dt.datetime, name="deleted_at", nullable=True),
            Spec(str, name="deleted_by", nullable=True),
            Spec(int, name="version", default=0),
            Spec(str, name="content_hash", nullable=True),
            Spec(str, name="integrity_hash", nullable=True),
        ]


def _resolve_forward_ref(fwd: ForwardRef) -> dict[str, Any]:
    """Handle ForwardRef annotations (from 'from __future__ import annotations').

    Parses the string representation to extract type info for DDL generation.
    FK[Model] -> Annotated[UUID, FKMeta(model_name)], Vector[dim] -> list[float], etc.

    Uses parse_forward_ref from db_types as canonical parser.
    """
    from lionagi.libs.db.types import parse_forward_ref

    fk, vec, nullable = parse_forward_ref(fwd)

    # FK[Model] -> Annotated[UUID, FKMeta]
    if fk is not None:
        base_type = Annotated[UUID, fk]
        return {"base_type": base_type, "nullable": nullable, "listable": False}

    # Vector[dim] -> Annotated[list[float], VectorMeta]
    if vec is not None:
        base_type = Annotated[list[float], vec]
        return {"base_type": base_type, "nullable": nullable, "listable": False}

    # Default: treat as generic type (will map to TEXT in SQL)
    return {"base_type": str, "nullable": nullable, "listable": False}


def resolve_annotation_to_base_types(annotation: Any) -> dict[str, Any]:
    """Resolve an annotation to its base types, detecting nullable and listable.

    Args:
        annotation: Type annotation to resolve (may include Optional, list, etc.)

    Returns:
        Dict with keys:
            - base_type: The innermost type
            - nullable: Whether None is allowed
            - listable: Whether it's a list type
    """
    # Handle ForwardRef (from 'from __future__ import annotations')
    if isinstance(annotation, ForwardRef):
        return _resolve_forward_ref(annotation)

    def resolve_nullable_inner_type(_anno: Any) -> tuple[bool, Any]:
        origin = get_origin(_anno)

        if origin is type(None):
            return True, type(None)

        if origin in (type(int | str), types.UnionType) or origin is Union:
            args = get_args(_anno)
            non_none_args = [a for a in args if a is not type(None)]
            if len(args) != len(non_none_args):
                if len(non_none_args) == 1:
                    return True, non_none_args[0]
                if non_none_args:
                    return True, reduce(lambda a, b: a | b, non_none_args)
            return False, _anno

        return False, _anno

    def resolve_listable_element_type(_anno: Any) -> Any:
        origin = get_origin(_anno)

        if origin is list:
            args = get_args(_anno)
            if args:
                return True, args[0]
            return True, Any

        return False, _anno

    _null, _inner = resolve_nullable_inner_type(annotation)
    _list, _elem = resolve_listable_element_type(_inner)

    return {
        "base_type": _elem,
        "nullable": _null,
        "listable": _list,
    }
