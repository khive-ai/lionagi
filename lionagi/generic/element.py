import contextlib
from datetime import datetime, timezone
from typing import Any, Literal
from uuid import UUID, uuid4

import orjson
from pydantic import BaseModel, ConfigDict, Field, field_validator

from lionagi.ln.types import not_sentinel

from ..ln import import_module, json_dumpb, now_utc, to_dict
from ._concepts import Observable

__all__ = ("Element",)


class Element(BaseModel, Observable):

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        use_enum_values=True,
        populate_by_name=True,
        extra="forbid",
        use_attribute_docstrings=True,
        validate_assignment=True,
        validate_by_alias=True,
    )

    id: UUID = Field(default_factory=uuid4, frozen=True)
    """A unique identifier for the element."""

    created_at: datetime = Field(default_factory=now_utc, frozen=True)
    """The timestamp when the element was created."""

    metadata: dict = Field(default_factory=dict, alias="node_metadata")
    """Additional data for this element."""

    @classmethod
    def class_name(cls, full: bool = False) -> str:
        """Returns this class's name.

        full (bool): If True, returns the fully qualified class name; otherwise,
            returns only the class name.
        """
        if full:
            return str(cls).split("'")[1]
        return cls.__name__

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
        if (lc := val.get("lion_class")) and lc != cls.class_name(full=True):
            raise ValueError(
                f"Metadata lion_class '{lc}' does not match Element class '{cls.class_name(full=True)}'."
            )
        return val

    @field_validator("created_at", mode="before")
    def _coerce_created_at(cls, value: Any) -> datetime:
        if value is None:
            return now_utc()
        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            try:
                dt = datetime.fromisoformat(value)
                return dt.astimezone(tz=timezone.utc)
            except Exception as e:
                try:
                    value = float(value)
                except Exception:
                    raise ValueError(
                        f"Invalid datetime string: {value}"
                    ) from e
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(value, tz=timezone.utc)
        raise TypeError(f"Invalid type for created_at: {type(value)}")

    @field_validator("id", mode="before")
    def _validate_id(cls, value: Any) -> UUID:
        if isinstance(value, UUID):
            return value
        if isinstance(value, str):
            try:
                return UUID(value)
            except Exception as e:
                raise ValueError(f"Invalid UUID string: {value}") from e
        raise TypeError(f"Invalid type for id: {type(value)}")

    def __bool__(self) -> bool:
        """Elements are always considered truthy."""
        return True

    def __hash__(self) -> int:
        """Returns a hash of this element's ID."""
        return hash(self.id)

    def _serialize_model(
        self,
        *,
        mode: Literal["python", "json", "sql"],
        handle_sentinels: Literal["as_none", "remove"],
        include_lion_class: bool,
        sql_datetime_format: Literal["iso", "datetime", "timestamp"] = "iso",
        **kw,
    ):
        d_ = None
        aggressive_json = True

        if mode == "sql":
            kw["by_alias"] = True

        # first try json mode if requested
        if mode == "json":
            with contextlib.suppress(Exception):
                d_ = self.model_dump(mode="json", **kw)
                if isinstance(d_.get("metadata"), str):
                    d_["metadata"] = orjson.loads(d_["metadata"])
                aggressive_json = False

        # we will let it fail here if something is wrong
        if d_ is None:
            d_ = self.model_dump(mode="python", **kw)

        # add lion_class if needed
        if include_lion_class:
            d_["metadata"].update({"lion_class": self.class_name(full=True)})

        handle_sentinels = (
            "remove" if kw.get("exclude_none") else handle_sentinels
        )
        if handle_sentinels == "as_none":
            d_ = {k: (v if not_sentinel(v) else None) for k, v in d_.items()}
        if handle_sentinels == "remove":
            d_ = {k: v for k, v in d_.items() if not_sentinel(v)}

        # handle sql datetime format
        if mode == "sql":
            match sql_datetime_format:
                case "iso":
                    d_["created_at"] = self.created_at.isoformat()
                case "datetime":
                    d_["created_at"] = self.created_at
                case "timestamp":
                    d_["created_at"] = self.created_at.timestamp()
                case _:
                    raise ValueError(
                        f"Invalid sql_datetime_format: {sql_datetime_format}"
                    )

        if not aggressive_json or mode != "json":
            return d_

        # we will try a more aggressive json serialization
        return json_dumpb(
            d_,
            sort_keys=True,
            deterministic_sets=True,
        )

    @classmethod
    def from_dict(cls, data: dict) -> "Element":
        """Deserializes a dictionary into an Element or subclass of Element.

        If `lion_class` in `metadata` refers to a subclass, this method
        is polymorphic, it will attempt to create an instance of that subclass.
        """
        # Preprocess database format if needed
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
                    from .._class_registry import get_class

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

    def to_dict(
        self,
        *,
        mode: Literal["python", "json", "sql"] = "python",
        handle_sentinels: Literal["as_none", "remove"] = "remove",
        include_lion_class: bool = True,
        include: set[str] | None = None,
        exclude: set[str] | None = None,
        sql_datetime_format: Literal["iso", "datetime", "timestamp"] = "iso",
        **kw,
    ):
        """Serializes this Element
        - if mode is "python", default pydantic model_dump
        - if mode is "json", tries to return a JSON-serializable dict
        - if mode is "sql", uses by_alias to match SQL column names

        handle_sentinels:
        - if "as_none", converts sentinel values to None
        - if "remove", removes sentinel values from the output

        include_lion_class:
        - if True, adds full qualified "lion_class" to metadata
        """
        kw["handle_sentinels"] = handle_sentinels
        kw["include_lion_class"] = include_lion_class
        kw["include"] = include
        kw["exclude"] = exclude
        kw["sql_datetime_format"] = sql_datetime_format

        if mode == "json":
            return orjson.loads(self.to_json(decode=False, **kw))
        return self._serialize_model(mode=mode, **kw)

    def to_json(self, decode: bool = True, **kw) -> str:
        """Converts this Element to a JSON string. Check to_dict for kw details."""
        _b = self._serialize_model(mode="json", **kw)
        return _b.decode("utf-8") if decode is True else _b
