# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Pydantic SpecAdapter: Spec <-> Pydantic FieldInfo/BaseModel.

Supports bidirectional transformation with:
    - Field validators (field_validator decorator)
    - Constraints (gt, ge, lt, le, min_length, max_length, pattern, etc.)
    - Rich metadata (aliases, descriptions, examples, json_schema_extra)
    - Type modifiers (nullable, listable)
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Annotated, Any

from pydantic import BaseModel, create_model, field_validator
from pydantic.fields import FieldInfo
from pydantic_core import PydanticUndefined
from pydantic_core._pydantic_core import PydanticUndefinedType

from lionagi.libs.db.types import FKMeta, VectorMeta
from lionagi.ln.types._sentinel import (
    Unset,
    UnsetType,
    is_sentinel,
    is_unset,
    not_sentinel,
)

from ..spec import Spec
from ._utils import resolve_annotation_to_base_types
from .protocol import SpecAdapter

if TYPE_CHECKING:
    from ..operable import Operable

__all__ = ("PydanticSpecAdapter",)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_FIELD_INFO_ATTRS = frozenset(
    {
        "alias",
        "validation_alias",
        "serialization_alias",
        "title",
        "description",
        "examples",
        "deprecated",
        "frozen",
        "json_schema_extra",
        "discriminator",
        "exclude",
        "repr",
        "init",
        "init_var",
        "kw_only",
        "validate_default",
    }
)

_CONSTRAINT_MAPPING = {
    "Gt": "gt",
    "Ge": "ge",
    "Lt": "lt",
    "Le": "le",
    "MultipleOf": "multiple_of",
    "MinLen": "min_length",
    "MaxLen": "max_length",
}


@functools.lru_cache(maxsize=1)
def _get_pydantic_field_params() -> set[str]:
    """Get valid Pydantic Field() parameter names (cached)."""
    import inspect

    from pydantic import Field as PydanticField

    params = set(inspect.signature(PydanticField).parameters.keys())
    params.discard("kwargs")
    return params


# ---------------------------------------------------------------------------
# FieldInfo -> Spec resolution (extract_specs direction)
# ---------------------------------------------------------------------------


def _is_valid_meta(v) -> bool:
    return not_sentinel(v) and not isinstance(v, PydanticUndefinedType)


def _ensure_annotation_from_field_info(fi: FieldInfo) -> Annotated | Any:
    annotation = fi.annotation
    if hasattr(fi, "metadata") and fi.metadata:
        for meta in fi.metadata:
            if isinstance(meta, (FKMeta, VectorMeta)):
                annotation = Annotated[annotation, meta]
                break
    return annotation


def _resolve_defaults_from_field_info(
    fi: FieldInfo,
) -> dict[str, Any]:
    if getattr(fi, "default", PydanticUndefined) is not PydanticUndefined:
        return {"default": fi.default}
    if (
        _df := getattr(fi, "default_factory", PydanticUndefined)
    ) is not PydanticUndefined and not_sentinel(_df, additions={"none", "empty"}):
        return {"default_factory": _df}
    return {}


def _resolve_constraint_metas(metadata: Any) -> dict[str, Any]:
    all_wanted = set(_CONSTRAINT_MAPPING.keys()) | {
        "Strict",
        "_PydanticGeneralMetadata",
    }
    out = {}

    for constraint in metadata:
        con_type = type(constraint).__name__
        if con_type not in all_wanted:
            continue

        _k, _v = Unset, Unset
        match con_type:
            case "Gt" | "Ge" | "Lt" | "Le" | "MultipleOf" | "MinLen" | "MaxLen":
                _k = _CONSTRAINT_MAPPING[con_type]
                _v = getattr(constraint, _k, Unset)
            case "Strict":
                _k, _v = "strict", getattr(constraint, "strict", True)
            case "_PydanticGeneralMetadata":
                pattern = getattr(constraint, "pattern", Unset)
                if is_sentinel(pattern, additions={"none", "empty"}):
                    _k, _v = "pattern", pattern
                strict = getattr(constraint, "strict", Unset)
                if is_sentinel(strict, additions={"none", "empty"}):
                    _k, _v = "strict", strict

        if not is_unset(_k) and _is_valid_meta(_v):
            out[_k] = _v

    return out


def _create_spec_metas_from_field_info(field_name: str, field_info: FieldInfo) -> dict:
    annotation = _ensure_annotation_from_field_info(field_info)
    base_metas = resolve_annotation_to_base_types(annotation)
    defaults = _resolve_defaults_from_field_info(field_info)
    updates = {
        attr: v
        for attr in _FIELD_INFO_ATTRS
        if _is_valid_meta(v := getattr(field_info, attr, Unset))
    }
    if not_sentinel(
        (meta := getattr(field_info, "metadata", Unset)),
        {"none", "empty"},
    ):
        updates.update(_resolve_constraint_metas(meta))

    return {"name": field_name, **defaults, **updates, **base_metas}


# ---------------------------------------------------------------------------
# PydanticSpecAdapter
# ---------------------------------------------------------------------------


class PydanticSpecAdapter(SpecAdapter[FieldInfo]):
    @classmethod
    def create_field(cls, spec: Spec) -> FieldInfo:
        """Convert Spec to Pydantic FieldInfo with annotation set."""
        from pydantic import Field as PydanticField

        pydantic_field_params = _get_pydantic_field_params()
        field_kwargs: dict[str, Any] = {}

        if not is_sentinel(spec.metadata, additions={"none"}):
            for meta in spec.metadata:
                if meta.key == "default":
                    if callable(meta.value):
                        field_kwargs["default_factory"] = meta.value
                    else:
                        field_kwargs["default"] = meta.value
                elif meta.key == "validator":
                    continue
                elif meta.key in pydantic_field_params:
                    if not_sentinel(meta.value):
                        field_kwargs[meta.key] = meta.value
                elif meta.key in {"nullable", "listable"}:
                    pass
                else:
                    if isinstance(meta.value, type):
                        continue
                    if "json_schema_extra" not in field_kwargs:
                        field_kwargs["json_schema_extra"] = {}
                    field_kwargs["json_schema_extra"][meta.key] = meta.value

        # Nullable fields default to None unless explicitly required
        is_required = any(m.key == "required" and m.value for m in spec.metadata)
        if (
            spec.is_nullable
            and "default" not in field_kwargs
            and "default_factory" not in field_kwargs
            and not is_required
        ):
            field_kwargs["default"] = None

        field_info = PydanticField(**field_kwargs)
        field_info.annotation = spec.annotation

        return field_info

    @classmethod
    def create_field_validator(cls, spec: Spec) -> dict[str, Any] | None:
        """Create Pydantic field_validator from Spec metadata. Returns None if no validator."""
        v = spec.get("validator")
        if is_sentinel(v):
            return None
        _func = field_validator(spec.name, check_fields=False)(v)
        return {f"_{spec.name}_validator": _func}

    @classmethod
    def compose_structure(
        cls,
        op: Operable,
        name: str,
        /,
        *,
        include: set[str] | UnsetType = Unset,
        exclude: set[str] | UnsetType = Unset,
        base_type: type[BaseModel] | UnsetType = Unset,
        doc: str | UnsetType = Unset,
    ) -> type[BaseModel]:
        """Generate Pydantic BaseModel subclass from Operable.

        Args:
            op: Operable containing Specs
            name: Model class name
            include/exclude: Field name filters
            base_type: Base class for the model
            doc: Docstring for the model

        Returns:
            Dynamically created BaseModel subclass with validators
        """
        use_specs = op.get_specs(include=include, exclude=exclude)
        use_fields = {i.name: cls.create_field(i) for i in use_specs if i.name}

        field_definitions = {
            field_name: (field_info.annotation, field_info)
            for field_name, field_info in use_fields.items()
        }

        validators = {}
        for spec in use_specs:
            if spec.name and (validator := cls.create_field_validator(spec)):
                validators.update(validator)

        if validators:
            base_with_validators = type(
                f"{name}Base",
                (base_type or BaseModel,),
                validators,
            )
            actual_base = base_with_validators
        else:
            actual_base = base_type or BaseModel

        model_cls: type[BaseModel] = create_model(
            name,
            __base__=actual_base,
            __doc__=doc,
            **field_definitions,
        )

        model_cls.model_rebuild()
        return model_cls

    @classmethod
    def validate_instance(cls, structure: type[BaseModel], data: dict, /) -> BaseModel:
        """Validate dict into BaseModel instance via model_validate."""
        if any(
            isinstance(v, BaseModel) or type(v).__name__ == "ActionCall"
            for v in data.values()
        ):
            return structure.model_construct(**data)
        return structure.model_validate(data)

    @classmethod
    def dump_instance(cls, instance: BaseModel) -> dict[str, Any]:
        """Dump BaseModel instance to dict via model_dump."""
        return instance.model_dump()

    @classmethod
    def extract_specs(cls, structure: type[BaseModel]) -> tuple[Spec, ...]:
        """Extract Specs from Pydantic model, preserving constraints and metadata.

        Raises:
            TypeError: If structure is not a BaseModel subclass
        """
        if not isinstance(structure, type) or not issubclass(structure, BaseModel):
            raise TypeError(
                f"structure must be a Pydantic BaseModel subclass, got {type(structure)}"
            )

        specs: list[Spec] = []
        for field_name, field_info in structure.model_fields.items():
            metas = _create_spec_metas_from_field_info(field_name, field_info)
            specs.append(Spec(**metas))

        return tuple(specs)
