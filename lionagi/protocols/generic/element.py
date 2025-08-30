# Copyright (c) 2023 - 2025, HaiyangLi <quantocean.li at gmail dot com>
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import datetime as dt
from collections.abc import Mapping, Sequence
from typing import Any, Generic, Literal, TypeAlias, TypeVar
from uuid import UUID, uuid4

import orjson
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
)

from lionagi import ln
from lionagi._class_registry import get_class
from lionagi._errors import IDError
from lionagi.settings import Settings
from lionagi.utils import import_module, time, to_dict

from .._concepts import Collective, Observable, Ordering

__all__ = (
    "IDType",
    "Element",
    "ID",
    "validate_order",
    "DEFAULT_ELEMENT_SERIALIZER",
)


class IDType:
    """Represents a UUIDv4-based identifier.

    This class wraps a UUID object and provides helper methods for
    validating and creating UUID version 4. It also implements equality
    and hashing so that it can be used as dictionary keys or in sets.

    Attributes:
        _id (UUID): The wrapped UUID object.
    """

    __slots__ = ("_id",)

    def __init__(self, id: UUID) -> None:
        """Initializes an IDType instance.

        Args:
            id (UUID): A UUID object (version 4 preferred).
        """
        self._id = id

    @classmethod
    def validate(cls, value: str | UUID | IDType) -> IDType:
        """Validates and converts a value into an IDType.

        Args:
            value (str | UUID | IDType):
                A string representing a UUID, a UUID instance, or another
                IDType instance.

        Returns:
            IDType: The validated IDType object.

        Raises:
            IDError: If the provided value is not a valid UUIDv4.
        """
        if isinstance(value, IDType):
            return value
        try:
            return cls(UUID(str(value), version=4))
        except ValueError:
            raise IDError(f"Invalid ID: {value}") from None

    @classmethod
    def create(cls) -> IDType:
        """Creates a new IDType with a randomly generated UUIDv4.

        Returns:
            IDType: A new IDType instance with a random UUIDv4.
        """
        return cls(uuid4())

    def __str__(self) -> str:
        """Returns the string representation of the underlying UUID.

        Returns:
            str: The string form of this IDType's UUID.
        """
        return str(self._id)

    def __repr__(self) -> str:
        """Returns the unambiguous string representation of this IDType.

        Returns:
            str: A developer-friendly string for debugging.
        """
        return f"IDType({self._id})"

    def __eq__(self, other: Any) -> bool:
        """Checks equality with another IDType based on UUID value.

        Args:
            other (Any): Another object for equality comparison.

        Returns:
            bool: True if both have the same underlying UUID; False otherwise.
        """
        if not isinstance(other, IDType):
            return NotImplemented
        return self._id == other._id

    def __hash__(self) -> int:
        """Returns a hash based on the underlying UUID.

        Returns:
            int: The hash of this object, allowing IDType to be dictionary keys.
        """
        return hash(self._id)


class Element(BaseModel, Observable):
    """Basic identifiable, timestamped element.

    This Pydantic model provides a unique identifier (`id`), an automatically
    generated creation timestamp (`created_at`), and an optional metadata
    dictionary.

    Attributes:
        id (IDType):
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

    id: IDType = Field(
        default_factory=IDType.create,
        title="ID",
        description="Unique identifier for this element.",
        frozen=True,
    )
    created_at: float = Field(
        default_factory=lambda: time(
            tz=Settings.Config.TIMEZONE, type_="timestamp"
        ),
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
        if "lion_class" in val and val["lion_class"] != cls.class_name(
            full=True
        ):
            raise ValueError("Metadata class mismatch.")
        if not isinstance(val, dict):
            raise ValueError("Invalid metadata.")
        return val

    @field_validator("created_at", mode="before")
    def _coerce_created_at(
        cls, val: float | dt.datetime | str | None
    ) -> float:
        """Coerces `created_at` to a float-based timestamp.

        Args:
            val (float | datetime | str | None): The initial creation time value.

        Returns:
            float: A float representing Unix epoch time in seconds.

        Raises:
            ValueError: If `val` cannot be converted to a float timestamp.
        """
        if val is None:
            return time(tz=Settings.Config.TIMEZONE, type_="timestamp")
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
                    raise ValueError(
                        f"Invalid datetime string: {val}"
                    ) from None
        try:
            return float(val)  # type: ignore
        except Exception:
            raise ValueError(f"Invalid created_at: {val}") from None

    @field_validator("id", mode="before")
    def _ensure_idtype(cls, val: IDType | UUID | str) -> IDType:
        """Ensures `id` is validated as an IDType.

        Args:
            val (IDType | UUID | str):
                The incoming value for the `id` field.

        Returns:
            IDType: A validated IDType object.
        """
        return IDType.validate(val)

    @field_serializer("id")
    def _serialize_id_type(self, val: IDType) -> str:
        """Serializes the `id` field to a string.

        Args:
            val (IDType): The IDType object to be serialized.

        Returns:
            str: The string representation of the UUID.
        """
        return str(val)

    @property
    def created_datetime(self) -> dt.datetime:
        """Returns the creation time as a datetime object.

        Returns:
            datetime: The creation time in UTC.
        """
        return dt.datetime.fromtimestamp(self.created_at, tz=dt.timezone.utc)

    def __eq__(self, other: Any) -> bool:
        """Compares two Element instances by their ID."""
        if not isinstance(other, Element):
            raise NotImplementedError(
                f"Cannot compare Element with {type(other)}"
            )
        return self.id == other.id

    def __hash__(self) -> int:
        """Returns a hash of this element's ID."""
        return hash(self.id)

    def __bool__(self) -> bool:
        """Elements are always considered truthy."""
        return True

    @classmethod
    def class_name(cls, full: bool = False) -> str:
        """Returns this class's name.

        full (bool): If True, returns the fully qualified class name; otherwise,
            returns only the class name.
        """
        if full:
            return str(cls).split("'")[1]
        return cls.__name__

    def _to_dict(self) -> dict:
        dict_ = self.model_dump()
        dict_["metadata"].update({"lion_class": self.class_name(full=True)})
        return {k: v for k, v in dict_.items() if ln.not_sentinel(v)}

    def to_dict(
        self, mode: Literal["python", "json", "db"] = "python"
    ) -> dict:
        """Converts this Element to a dictionary."""
        if mode == "python":
            return self._to_dict()
        if mode == "json":
            return orjson.loads(self.to_json(decode=False))
        if mode == "db":
            dict_ = orjson.loads(self.to_json(decode=False))
            dict_["node_metadata"] = dict_.pop("metadata", {})
            dict_["created_at"] = self.created_datetime.isoformat(sep=" ")
            return dict_

    def as_jsonable(self) -> dict:
        """Converts this Element to a JSON-serializable dictionary."""
        return self.to_dict(mode="json")

    @classmethod
    def from_dict(cls, data: dict, /, mode: str = "python") -> Element:
        """Deserializes a dictionary into an Element or subclass of Element.

        If `lion_class` in `metadata` refers to a subclass, this method
        is polymorphic, it will attempt to create an instance of that subclass.

        Args:
            data (dict): A dictionary of field data.
            mode (str): Format mode - "python" for normal dicts, "db" for database format.
        """
        # Preprocess database format if needed
        if mode == "db":
            data = cls._preprocess_db_data(data.copy())
        metadata = {}

        if "node_metadata" in data:
            metadata = data.pop("node_metadata")
        elif "metadata" in data:
            metadata = data.pop("metadata")
        if "lion_class" in metadata:
            subcls: str = metadata.pop("lion_class")
            if subcls != Element.class_name(full=True):
                try:
                    # Attempt dynamic lookup by registry
                    subcls_type: type[Element] = get_class(
                        subcls.split(".")[-1]
                    )
                    # If there's a custom from_dict, delegate to it
                    if (
                        hasattr(subcls_type, "from_dict")
                        and subcls_type.from_dict.__func__
                        != cls.from_dict.__func__
                    ):
                        return subcls_type.from_dict(data)

                except Exception:
                    mod, imp = subcls.rsplit(".", 1)
                    subcls_type = import_module(mod, import_name=imp)
                    data["metadata"] = metadata
                    if hasattr(subcls_type, "from_dict") and (
                        subcls_type is not cls
                    ):
                        return subcls_type.from_dict(data)
        data["metadata"] = metadata
        return cls.model_validate(data)

    @classmethod
    def _preprocess_db_data(cls, data: dict) -> dict:
        """Preprocess raw database data for Element compatibility."""
        import datetime as dt
        import json

        # Handle created_at field - convert datetime string to timestamp
        if "created_at" in data and isinstance(data["created_at"], str):
            try:
                # Parse datetime string and convert to timestamp
                dt_obj = dt.datetime.fromisoformat(
                    data["created_at"].replace(" ", "T")
                )
                # Treat as UTC if naive
                if dt_obj.tzinfo is None:
                    dt_obj = dt_obj.replace(tzinfo=dt.timezone.utc)
                data["created_at"] = dt_obj.timestamp()
            except (ValueError, TypeError):
                # Keep as string if parsing fails
                pass

        # Handle JSON string fields - parse to dict/list
        json_fields = ["content", "node_metadata", "embedding"]
        for field in json_fields:
            if field in data and isinstance(data[field], str):
                if data[field] in ("null", ""):
                    data[field] = None if field == "embedding" else {}
                else:
                    try:
                        data[field] = json.loads(data[field])
                    except (json.JSONDecodeError, TypeError):
                        # Keep as empty dict for metadata fields, None for embedding
                        data[field] = {} if field != "embedding" else None

        # Handle node_metadata -> metadata mapping
        if "node_metadata" in data:
            if (
                data["node_metadata"] == "null"
                or data["node_metadata"] is None
            ):
                data["metadata"] = {}
            else:
                data["metadata"] = (
                    data["node_metadata"] if data["node_metadata"] else {}
                )
            # Remove node_metadata to avoid Pydantic validation error
            data.pop("node_metadata", None)

        return data

    def to_json(self, decode: bool = True) -> str:
        """Converts this Element to a JSON string."""
        dict_ = self._to_dict()
        if decode:
            return orjson.dumps(
                dict_,
                default=DEFAULT_ELEMENT_SERIALIZER,
                option=ln.DEFAULT_SERIALIZER_OPTION,
            ).decode()
        return orjson.dumps(dict_, default=DEFAULT_ELEMENT_SERIALIZER)

    @classmethod
    def from_json(cls, json_str: str, mode: str = "python") -> Element:
        """Deserializes a JSON string into an Element or subclass of Element."""
        data = orjson.loads(json_str)
        return cls.from_dict(data, mode=mode)


DEFAULT_ELEMENT_SERIALIZER = ln.get_orjson_default(
    order=[IDType, Element, BaseModel],
    additional={
        IDType: lambda o: str(o),
        Element: lambda o: o.to_dict(),
        BaseModel: lambda o: o.model_dump(mode="json"),
    },
)


def validate_order(order: Any) -> list[IDType]:
    """Validates and flattens an ordering into a list of IDType objects.

    This function accepts a variety of possible representations for ordering
    (e.g., a single Element, a list of Elements, a dictionary with ID keys,
    or a nested structure) and returns a flat list of IDType objects.

    Args:
        order (Any): A potentially nested structure of items to be ordered.

    Returns:
        list[IDType]: A flat list of validated IDType objects.

    Raises:
        ValueError: If an invalid item is encountered or if there's a mixture
            of types not all convertible to IDType.
    """
    if isinstance(order, Element):
        return [order.id]
    if isinstance(order, Mapping):
        order = list(order.keys())

    stack = [order]
    out: list[IDType] = []
    while stack:
        cur = stack.pop()
        if cur is None:
            continue
        if isinstance(cur, Element):
            out.append(cur.id)
        elif isinstance(cur, IDType):
            out.append(cur)
        elif isinstance(cur, UUID):
            out.append(IDType.validate(cur))
        elif isinstance(cur, str):
            out.append(IDType.validate(cur))
        elif isinstance(cur, (list, tuple, set)):
            stack.extend(reversed(cur))
        else:
            raise ValueError("Invalid item in order.")

    if not out:
        return []

    # Check for consistent IDType usage
    first_type = type(out[0])
    if first_type is IDType:
        for item in out:
            if not isinstance(item, IDType):
                raise ValueError("Mixed types in order.")
        return out
    raise ValueError("Unrecognized type(s) in order.")


E = TypeVar("E", bound=Element)


class ID(Generic[E]):
    """Utility class for working with IDType objects and Elements.

    This class provides helper methods to extract IDs from Elements, strings,
    or UUIDs, and to test whether a given object can be interpreted as
    an ID.
    """

    ID: TypeAlias = IDType
    Item: TypeAlias = E | Element  # type: ignore
    Ref: TypeAlias = IDType | E | str  # type: ignore
    IDSeq: TypeAlias = Sequence[IDType] | Ordering[E]  # type: ignore
    ItemSeq: TypeAlias = Sequence[E] | Collective[E]  # type: ignore
    RefSeq: TypeAlias = ItemSeq | Sequence[Ref] | Ordering[E]  # type: ignore

    @staticmethod
    def get_id(item: E) -> IDType:
        """Retrieves an IDType from multiple possible item forms.

        Acceptable item types include:
        - Element: Uses its `id` attribute.
        - IDType: Returns it directly.
        - UUID: Validates and wraps it.
        - str: Interpreted as a UUID if possible.

        Args:
            item (E): The item to convert to an ID.

        Returns:
            IDType: The validated ID.

        Raises:
            ValueError: If the item cannot be converted to an IDType.
        """
        if isinstance(item, Element):
            return item.id
        if isinstance(item, (IDType, UUID, str)):
            return IDType.validate(item)
        raise ValueError("Cannot get ID from item.")

    @staticmethod
    def is_id(item: Any) -> bool:
        """Checks if an item can be validated as an IDType.

        Args:
            item (Any): The object to check.

        Returns:
            bool: True if `item` is or can be validated as an IDType;
                otherwise, False.
        """
        try:
            IDType.validate(item)
            return True
        except IDError:
            return False


# File: lionagi/protocols/generic/element.py
