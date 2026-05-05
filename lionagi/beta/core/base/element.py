# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib
import datetime as dt
from typing import Any, Literal
from uuid import UUID, uuid4

import orjson
from pydantic import BaseModel, ConfigDict, Field, field_validator

from lionagi.beta.protocols import (
    Deserializable,
    Hashable,
    Observable,
    Serializable,
    implements,
)
from lionagi.ln.types._sentinel import (
    MaybeSentinel,
    Unset,
    UnsetType,
    is_sentinel,
    is_unset,
)
from lionagi.ln._json_dump import json_dumps
from lionagi.ln._utils import (
    coerce_created_at,
    load_type_from_string,
    now_utc,
    to_uuid,
)

__all__ = ("LN_ELEMENT_FIELDS", "Element")


@implements(Observable, Serializable, Deserializable, Hashable)
class Element(BaseModel):
    """Base element with UUID identity, timestamps, polymorphic serialization.

    Attributes:
        id: UUID identifier (frozen, auto-generated)
        created_at: UTC datetime (frozen, auto-generated)
        metadata: Arbitrary metadata dict

    Serialization injects kron_class for polymorphic deserialization.
    """

    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True,
        validate_assignment=True,
        use_enum_values=True,
        extra="forbid",
    )

    id: UUID = Field(default_factory=uuid4, frozen=True)
    created_at: dt.datetime = Field(default_factory=now_utc, frozen=True)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("id", mode="before")
    @classmethod
    def _coerce_id(cls, v) -> UUID:
        """Coerce to UUID4."""
        return to_uuid(v)

    @field_validator("created_at", mode="before")
    @classmethod
    def _coerce_created_at(cls, v) -> dt.datetime:
        """Coerce to UTC datetime."""
        return coerce_created_at(v)

    @field_validator("metadata", mode="before")
    @classmethod
    def _validate_meta_integrity(
        cls, val: dict[str, Any] | MaybeSentinel
    ) -> dict[str, Any]:
        """Validate and coerce metadata to dict. Raises ValueError if conversion fails."""
        if is_sentinel(val, additions={"none"}):
            return {}

        with contextlib.suppress(orjson.JSONDecodeError, TypeError):
            val = json_dumps(val, as_loaded=True)

        if not isinstance(val, dict):
            raise ValueError("Invalid metadata: must be a dictionary")

        return val

    @classmethod
    def class_name(cls, full: bool = False) -> str:
        """Get class name, stripping generic parameters (Flow[E,P] -> Flow).

        Args:
            full: If True, returns module.Class; otherwise Class only.

        Returns:
            Class name string.
        """
        name = cls.__qualname__ if full else cls.__name__
        if "[" in name:
            name = name.split("[")[0]
        if full:
            return f"{cls.__module__}.{name}"
        return name

    def _to_dict(self, **kwargs: Any) -> dict[str, Any]:
        """Serialize to dict with kron_class injected in metadata."""
        data = self.model_dump(**kwargs)
        data.setdefault("metadata", {})
        data["metadata"]["kron_class"] = self.__class__.class_name(full=True)

        return data

    def to_dict(
        self,
        mode: Literal["python", "json", "db"] = "python",
        created_at_format: (
            Literal["datetime", "isoformat", "timestamp"] | UnsetType
        ) = Unset,
        meta_key: str | UnsetType = Unset,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Serialize to dict with kron_class metadata injected.

        Args:
            mode: python/json/db (db auto-renames metadata to node_metadata).
            created_at_format: datetime/isoformat/timestamp (auto-selected by mode).
            meta_key: Rename metadata field (overrides db default).
            **kwargs: Passed to model_dump().

        Returns:
            Serialized dict with kron_class in metadata for polymorphic restore.
        """
        if is_unset(created_at_format):
            created_at_format = "isoformat" if mode == "json" else "datetime"

        if is_unset(meta_key) and mode == "db":
            meta_key = "node_metadata"

        data = self._to_dict(**kwargs)
        if mode in ("json", "db"):
            data = json_dumps(data, as_loaded=True)

        if "created_at" in data:
            if created_at_format == "isoformat":
                if mode == "python":
                    data["created_at"] = self.created_at.isoformat()
            elif created_at_format == "timestamp":
                data["created_at"] = self.created_at.timestamp()
            elif created_at_format == "datetime":
                if mode == "json":
                    raise ValueError(
                        "created_at_format='datetime' not valid for mode='json'. "
                        "Use 'isoformat' or 'timestamp' for JSON serialization."
                    )
                if mode == "db" and isinstance(data["created_at"], str):
                    data["created_at"] = self.created_at

        if not is_unset(meta_key) and "metadata" in data:
            data[meta_key] = data.pop("metadata")

        return data

    @classmethod
    def from_dict(
        cls, data: dict[str, Any], meta_key: str | UnsetType = Unset, **kwargs: Any
    ) -> Element:
        """Deserialize from dict with polymorphic type restoration via kron_class.

        Args:
            data: Serialized element dict.
            meta_key: Restore metadata from this key (db mode compatibility).
            **kwargs: Passed to model_validate().

        Returns:
            Element instance (actual type determined by kron_class if present).

        Raises:
            ValueError: If kron_class invalid or not Element subclass.
        """
        data = data.copy()

        if not is_unset(meta_key) and meta_key in data:
            data["metadata"] = data.pop(meta_key)

        metadata = data.get("metadata", {})
        kron_class = Unset

        if isinstance(metadata, dict):
            metadata = metadata.copy()
            data["metadata"] = metadata
            kron_class = metadata.pop("kron_class", Unset)

        if not is_unset(kron_class) and kron_class != cls.class_name(full=True):
            try:
                target_cls = load_type_from_string(kron_class)
            except ValueError as e:
                raise ValueError(
                    f"Failed to deserialize class '{kron_class}': {e}"
                ) from e

            if not issubclass(target_cls, Element):
                raise ValueError(
                    f"'{kron_class}' is not an Element subclass. "
                    f"Cannot deserialize into {cls.__name__}"
                )

            target_func = getattr(
                target_cls.from_dict, "__func__", target_cls.from_dict
            )
            cls_func = getattr(cls.from_dict, "__func__", cls.from_dict)
            if target_func is cls_func:
                return target_cls.model_validate(data, **kwargs)

            return target_cls.from_dict(data, **kwargs)

        return cls.model_validate(data, **kwargs)

    def __eq__(self, other: Any) -> bool:
        """Equality by ID."""
        if not isinstance(other, Element):
            return NotImplemented
        return self.id == other.id

    def __hash__(self) -> int:
        """Hash by ID."""
        return hash(self.id)

    def __bool__(self) -> bool:
        """Always truthy."""
        return True

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.id})"


LN_ELEMENT_FIELDS: list[str] = list(Element.model_fields.keys())
