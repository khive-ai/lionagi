# Copyright (c) 2023-2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import datetime as dt
from collections.abc import Mapping, Sequence
from typing import Any, Generic, Literal, TypeAlias, TypeVar
from uuid import UUID, uuid4

import orjson
from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator

from lionagi._class_registry import get_class
from lionagi.ln._json_dump import get_orjson_default, json_dumps
from lionagi.ln._utils import now_utc
from lionagi.ln.types import not_sentinel
from lionagi.ln.types._sentinel import Unset, UnsetType, is_unset
from lionagi.utils import import_module, to_dict

from .._concepts import Collective, Observable, Ordering

__all__ = (
    "Element",
    "LN_ELEMENT_FIELDS",
    "validate_order",
)


class Element(BaseModel, Observable):
    """Basic identifiable, timestamped element.

    This Pydantic model provides a unique identifier (`id`), an automatically
    generated creation timestamp (`created_at`), and an optional metadata
    dictionary.

    Attributes:
        id (UUID):
            A unique ID based on UUIDv4 (defaults to a newly generated one).
        created_at (float):
            The creation timestamp as a float (Unix epoch). Defaults to
            the current time.
        metadata (dict):
            A dictionary for storing additional information about this Element.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        use_enum_values=True,
        populate_by_name=True,
        extra="forbid",
    )

    id: UUID = Field(
        default_factory=uuid4,
        title="ID",
        description="Unique identifier for this element.",
        frozen=True,
    )
    created_at: float = Field(
        default_factory=lambda: now_utc().timestamp(),
        title="Creation Timestamp",
        description="Timestamp of element creation.",
        frozen=True,
    )
    metadata: dict = Field(
        default_factory=dict,
        title="Metadata",
        description="Additional data for this element.",
    )

    @field_validator("metadata", mode="before")
    def _validate_meta_integrity(cls, val: dict) -> dict:
        """Validates that `metadata` is a dictionary and checks class naming.

        If a `lion_class` field is present in `metadata`, it must match the
        fully qualified name of this class. Converts `metadata` to a dict
        if needed.
        """
        if not val:
            return {}
        if not isinstance(val, dict):
            val = to_dict(val, recursive=True, suppress=True)
        if "lion_class" in val and val["lion_class"] != cls.class_name(full=True):
            raise ValueError("Metadata class mismatch.")
        if not isinstance(val, dict):
            raise ValueError("Invalid metadata.")
        return val

    @field_validator("created_at", mode="before")
    def _coerce_created_at(cls, val: float | dt.datetime | str | None) -> float:
        """Coerces `created_at` to a float-based timestamp."""
        if val is None:
            return now_utc().timestamp()
        if isinstance(val, float):
            return val
        if isinstance(val, dt.datetime):
            return val.timestamp()
        if isinstance(val, str):
            # Parse datetime string from database
            try:
                # Handle datetime strings like "2025-08-30 10:54:59.310329"
                # Convert space to T for ISO format, but handle timezone properly
                iso_string = val.replace(" ", "T")
                parsed_dt = dt.datetime.fromisoformat(iso_string)

                # If parsed as naive datetime (no timezone), treat as UTC to avoid local timezone issues
                if parsed_dt.tzinfo is None:
                    parsed_dt = parsed_dt.replace(tzinfo=dt.timezone.utc)

                return parsed_dt.timestamp()
            except ValueError:
                # Try parsing as float string as fallback
                try:
                    return float(val)
                except ValueError:
                    raise ValueError(f"Invalid datetime string: {val}") from None
        try:
            return float(val)  # type: ignore
        except Exception:
            raise ValueError(f"Invalid created_at: {val}") from None

    @field_validator("id", mode="before")
    def _ensure_UUID(cls, val: UUID | str) -> UUID:
        """Ensures `id` is validated as an UUID."""
        if isinstance(val, UUID):
            return val
        return UUID(str(val))

    @field_serializer("id")
    def _serialize_id_type(self, val: UUID) -> str:
        """Serializes the `id` field to a string."""
        return str(val)

    @property
    def created_datetime(self) -> dt.datetime:
        """Returns the creation time as a datetime object."""
        return dt.datetime.fromtimestamp(self.created_at, tz=dt.timezone.utc)

    def __eq__(self, other: Any) -> bool:
        """Compares two Element instances by their ID."""
        if not isinstance(other, Element):
            return NotImplemented
        return self.id == other.id

    def __hash__(self) -> int:
        """Returns a hash of this element's ID."""
        return hash(self.id)

    def __bool__(self) -> bool:
        """Elements are always considered truthy."""
        return True

    @classmethod
    def _coerce_id(cls, value: Any) -> UUID:
        """Resolve various ID forms to UUID."""
        return ID.get_id(value)

    @classmethod
    def class_name(cls, full: bool = False) -> str:
        """Returns this class's name.

        full (bool): If True, returns the fully qualified class name; otherwise,
            returns only the class name.
        """
        if full:
            return f"{cls.__module__}.{cls.__qualname__}"
        return cls.__name__

    def _to_dict(self, **kw) -> dict:
        """kw for model_dump."""
        dict_ = self.model_dump(**kw)
        cls_name = self.class_name(full=True)
        if "metadata" in dict_:
            dict_["metadata"]["lion_class"] = cls_name
        return {k: v for k, v in dict_.items() if not_sentinel(v)}

    def to_dict(
        self,
        mode: Literal["python", "json", "db"] = "python",
        created_at_format: (
            Literal["datetime", "isoformat", "timestamp"] | UnsetType
        ) = Unset,
        meta_key: str | UnsetType = Unset,
        **kw,
    ) -> dict:
        """Converts this Element to a dictionary.

        Args:
            mode: Serialization mode: 'python' (default), 'json', or 'db'.
            created_at_format: How to serialize created_at. Accepts
                'timestamp' (float), 'isoformat' (ISO 8601 string), or
                'datetime' (Python datetime object, python/db modes only).
                Defaults to leaving created_at as the stored float.
            meta_key: When set, renames the 'metadata' key in the output dict.
                Useful for DB serialization (e.g. meta_key='node_metadata').
        """
        if mode == "python":
            data = self._to_dict(**kw)
        elif mode == "json":
            data = orjson.loads(self.to_json(decode=False, **kw))
        elif mode == "db":
            data = orjson.loads(self.to_json(decode=False, **kw))
            if is_unset(meta_key):
                meta_key = "node_metadata"
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        # Apply created_at transformation when explicitly requested
        if not is_unset(created_at_format) and "created_at" in data:
            raw = data["created_at"]
            # raw is a float (from model_dump) or a float-string (after JSON round-trip)
            ts: float = float(raw) if not isinstance(raw, float) else raw
            if created_at_format == "timestamp":
                data["created_at"] = ts
            elif created_at_format == "isoformat":
                data["created_at"] = dt.datetime.fromtimestamp(
                    ts, tz=dt.timezone.utc
                ).isoformat()
            elif created_at_format == "datetime":
                if mode == "json":
                    raise ValueError(
                        "created_at_format='datetime' not valid for mode='json'. "
                        "Use 'isoformat' or 'timestamp' for JSON serialization."
                    )
                data["created_at"] = dt.datetime.fromtimestamp(ts, tz=dt.timezone.utc)

        # Apply metadata key renaming when requested
        if not is_unset(meta_key) and "metadata" in data:
            data[meta_key] = data.pop("metadata")

        return data

    @classmethod
    def from_dict(
        cls,
        data: dict,
        meta_key: str | UnsetType = Unset,
        **kwargs,
    ) -> Element:
        """Deserializes a dictionary into an Element or subclass of Element.

        If `lion_class` in `metadata` refers to a subclass,
        this method is polymorphic and will attempt to create an instance of
        that subclass.

        Args:
            data: Dictionary representation of the Element.
            meta_key: When set, reads metadata from this key instead of
                'metadata' (e.g. meta_key='node_metadata' for DB rows).
        """
        # Shallow copy to avoid mutating the caller's dict. The nested metadata
        # dict is also copied because we pop from it (lion_class extraction).
        data = dict(data)

        # Support alternate metadata key (e.g. 'node_metadata' from DB rows)
        if not is_unset(meta_key) and meta_key in data:
            data["metadata"] = data.pop(meta_key)

        # Preprocess database format if needed
        metadata = {}

        if "node_metadata" in data and "metadata" not in data:
            metadata = dict(data.pop("node_metadata"))
        elif "metadata" in data:
            metadata = dict(data.pop("metadata"))

        subcls_name: str | None = metadata.pop("lion_class", None)

        if subcls_name and subcls_name != cls.class_name(full=True):
            try:
                # Try full qualified name first (LION_CLASS_REGISTRY is
                # populated with full-qualified names at import time).
                # Fall back to short name for classes registered before the
                # full-name convention was adopted.
                try:
                    subcls_type: type[Element] = get_class(subcls_name)
                except (KeyError, ValueError):
                    subcls_type = get_class(subcls_name.split(".")[-1])
                # Delegate when there is a custom from_dict OR when the
                # concrete type differs from cls (so model_validate uses the
                # right schema). Restore metadata before the recursive call
                # so the delegate sees a self-consistent dict.
                if hasattr(subcls_type, "from_dict") and (
                    subcls_type.from_dict.__func__ != cls.from_dict.__func__
                    or subcls_type is not cls
                ):
                    data["metadata"] = metadata
                    return subcls_type.from_dict(data, **kwargs)

            except (KeyError, ValueError, ImportError, AttributeError, TypeError):
                import logging as _logging

                _logger = _logging.getLogger(__name__)
                try:
                    mod, imp = subcls_name.rsplit(".", 1)
                    subcls_type = import_module(mod, import_name=imp)
                    data["metadata"] = metadata
                    if hasattr(subcls_type, "from_dict") and (subcls_type is not cls):
                        return subcls_type.from_dict(data, **kwargs)
                except Exception as _err:
                    _logger.debug(
                        "Element.from_dict: fallback import of '%s' failed: %s",
                        subcls_name,
                        _err,
                    )

        data["metadata"] = metadata
        return cls.model_validate(data)

    def to_json(self, decode: bool = True, **kw) -> str:
        """Converts this Element to a JSON string."""
        kw.pop("mode", None)
        dict_ = self._to_dict(**kw)
        return json_dumps(dict_, default=DEFAULT_ELEMENT_SERIALIZER, decode=decode)

    @classmethod
    def from_json(cls, json_str: str) -> Element:
        """Deserializes a JSON string into an Element or subclass of Element."""
        return cls.from_dict(orjson.loads(json_str))

    # Alias for beta compatibility: beta code uses cls.lion_class(full=True)
    # whereas production uses cls.class_name(full=True). Both return the same value.
    lion_class = class_name


DEFAULT_ELEMENT_SERIALIZER = get_orjson_default(
    order=[Element, BaseModel],
    additional={
        Element: lambda o: o.to_dict(),
        BaseModel: lambda o: o.model_dump(mode="json"),
    },
)

# Field list exported for beta compatibility (event.py uses LN_ELEMENT_FIELDS to
# exclude base fields when cloning events via as_fresh_event).
LN_ELEMENT_FIELDS: list[str] = list(Element.model_fields.keys())


def validate_order(order: Any) -> list[UUID]:
    """Validates and flattens an ordering into a list of UUID objects.

    This function accepts a variety of possible representations for ordering
    (e.g., a single Element, a list of Elements, a dictionary with ID keys,
    or a nested structure) and returns a flat list of UUID objects.

    Returns:
        list[UUID]: A flat list of validated UUID objects.

    Raises:
        ValueError: If an invalid item is encountered or if there's a mixture
            of types not all convertible to UUID.
    """
    if isinstance(order, Element):
        return [order.id]
    if isinstance(order, Mapping):
        order = list(order.keys())

    stack = [order]
    out: list[UUID] = []
    while stack:
        cur = stack.pop()
        if cur is None:
            continue
        if isinstance(cur, Element):
            out.append(cur.id)
        elif isinstance(cur, UUID):
            out.append(cur)
        elif isinstance(cur, str):
            out.append(UUID(cur))
        elif isinstance(cur, (list, tuple, set)):
            stack.extend(reversed(cur))
        else:
            raise ValueError("Invalid item in order.")

    return [] if not out else out


E = TypeVar("E", bound=Element)


class ID(Generic[E]):
    """Utility class for working with UUID objects and Elements.

    This class provides helper methods to extract IDs from Elements, strings,
    or UUIDs, and to test whether a given object can be interpreted as
    an ID.
    """

    ID: TypeAlias = UUID
    Item: TypeAlias = E | Element  # type: ignore
    Ref: TypeAlias = UUID | E | str  # type: ignore
    IDSeq: TypeAlias = Sequence[UUID] | Ordering[E]  # type: ignore
    ItemSeq: TypeAlias = Sequence[E] | Collective[E]  # type: ignore
    RefSeq: TypeAlias = ItemSeq | Sequence[Ref] | Ordering[E]  # type: ignore

    @staticmethod
    def get_id(item: E) -> UUID:
        """Retrieves an UUID from multiple possible item forms.

        Acceptable item types include:
        - Element: Uses its `id` attribute.
        - UUID: Returns it directly.
        - UUID: Validates and wraps it.
        - str: Interpreted as a UUID if possible.

        Returns:
            UUID: The validated ID.

        Raises:
            ValueError: If the item cannot be converted to an UUID.
        """
        if isinstance(item, UUID):
            return item
        if isinstance(item, Element):
            return item.id
        if isinstance(item, str):
            return UUID(item)
        raise ValueError("Cannot get ID from item.")

    @staticmethod
    def is_id(item: Any) -> bool:
        """Checks if an item can be validated as an UUID.

        Returns:
            bool: True if `item` is or can be validated as an UUID;
                otherwise, False.
        """
        try:
            ID.get_id(item)  # type: ignore
            return True
        except ValueError:
            return False


# File: lionagi/protocols/generic/element.py
