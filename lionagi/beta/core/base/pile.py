# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib
import functools
import threading
from collections.abc import Callable, Iterator
from typing import Any, Generic, Literal, TypeVar, overload
from uuid import UUID

from pydantic import Field, PrivateAttr, field_serializer, field_validator
from typing_extensions import override

from lionagi._errors import ExistsError, NotAllowedError, NotFoundError
from lionagi.beta.protocols import (
    Containable,
    Deserializable,
    Observable,
    Serializable,
    implements,
)
from lionagi.ln._utils import extract_types, load_type_from_string, synchronized
from lionagi.ln.concurrency import Lock as AsyncLock
from lionagi.ln.types._sentinel import Unset, UnsetType, is_unset

from lionagi.protocols.generic.element import Element
from lionagi.protocols.generic.progression import Progression

__all__ = ("Pile",)

T = TypeVar("T", bound=Element)


# Invariant Assertions
def kron_class_must_be_allowed(strict: bool, actual: type, allowed: set[type]) -> None:
    if strict:
        if actual not in allowed:
            raise NotAllowedError(
                f"Item type {actual} not in allowed types {allowed} (strict_type=True)",
                details={
                    "actual": str(actual),
                    "allowed": [str(t) for t in allowed],
                    "strict": True,
                },
            )
    else:
        if not any(issubclass(actual, t) for t in allowed):
            raise NotAllowedError(
                f"Item type {actual} is not a subclass of any allowed type {allowed}",
                details={
                    "actual": str(actual),
                    "allowed": [str(t) for t in allowed],
                    "strict": False,
                },
            )


def _must_be_allowed(p: Pile[T], item: T | type) -> None:
    """Validate item against pile's type constraints. Raises NotAllowedError on failure."""
    if isinstance(item, type):
        item_type_actual = item
    else:
        if not isinstance(item, Observable):
            raise NotAllowedError(
                f"Item must expose an observable UUID id, got {type(item)}",
                details={"actual": str(type(item))},
            )
        item_type_actual = type(item)

    if p.item_type is not None:
        kron_class_must_be_allowed(p.strict_type, item_type_actual, p.item_type)


def must_be_allowed(func: Callable) -> Callable:
    """Decorator: validate first positional arg against pile's item_type before calling func."""

    @functools.wraps(func)
    def wrapper(self: Pile, item: Any, *args: Any, **kwargs: Any) -> Any:
        _must_be_allowed(self, item)
        return func(self, item, *args, **kwargs)

    return wrapper


def _must_exist(p: Pile[T], item: UUID | str | T, *, strict: bool = True):
    try:
        uid = p._coerce_id(item)
    except Exception:
        if not strict:
            return False
        raise NotFoundError(f"Item {item} not found in pile") from None

    if uid not in p._items:
        if not strict:
            return False
        raise NotFoundError(f"Item {uid} not found in pile")

    return True


def must_exist(func: Callable) -> Callable:
    """Decorator: validate first positional arg exists in pile by ID before calling func."""

    @functools.wraps(func)
    def wrapper(self: Pile, item: Any, *args: Any, **kwargs: Any) -> Any:
        _must_exist(self, item)
        return func(self, item, *args, **kwargs)

    return wrapper


@implements(
    Containable,
    Serializable,
    Deserializable,
)
class Pile(Element, Generic[T]):
    """Thread-safe, ordered, type-validated container for Element subclasses."""

    _items: dict[UUID, T] = PrivateAttr(default_factory=dict)
    _progression: Progression = PrivateAttr(default_factory=Progression)
    _lock: threading.RLock = PrivateAttr(default_factory=threading.RLock)
    _async_lock: AsyncLock = PrivateAttr(default_factory=AsyncLock)

    @property
    def progression(self) -> Progression:
        return Progression(order=list(self._progression.order), name=self._progression.name)

    item_type: set[type] | None = Field(
        default=None,
        frozen=True,
        description="Allowed types for validation (None = any Element subclass)",
    )
    strict_type: bool = Field(
        default=False,
        frozen=True,
        description="Enforce exact type match (disallow subclasses)",
    )

    @field_validator("item_type", mode="before")
    @classmethod
    def _normalize_item_type(cls, v: Any) -> set[type] | None:
        if v is None:
            return None
        if isinstance(v, list) and v and isinstance(v[0], str):
            return {load_type_from_string(type_str) for type_str in v}
        return extract_types(v)

    @override
    def __init__(
        self,
        items: list[T] | None = None,
        item_type: type[T] | set[type] | list[type] | None = None,
        order: list[UUID] | Progression | None = None,
        strict_type: bool = False,
        **kwargs,
    ):
        super().__init__(**{"item_type": item_type, "strict_type": strict_type, **kwargs})

        if items:
            for item in items:
                self.add(item)

        if order:
            order_list = list(order.order) if isinstance(order, Progression) else order
            for uid in order_list:
                if uid not in self._items:
                    raise NotFoundError(f"UUID {uid} in order not found in items")
            self._progression = Progression(order=order_list)

    @field_serializer("item_type")
    def _serialize_item_type(self, v: set[type] | None) -> list[str] | None:
        if v is None:
            return None
        return [f"{t.__module__}.{t.__name__}" for t in v]

    @override
    def to_dict(
        self,
        mode: Literal["python", "json", "db"] = "python",
        created_at_format: (Literal["datetime", "isoformat", "timestamp"] | UnsetType) = Unset,
        meta_key: str | UnsetType = Unset,
        item_meta_key: str | UnsetType = Unset,
        item_created_at_format: (Literal["datetime", "isoformat", "timestamp"] | UnsetType) = Unset,
        **kwargs: Any,
    ) -> dict[str, Any]:
        data = super().to_dict(
            mode=mode, created_at_format=created_at_format, meta_key=meta_key, **kwargs
        )

        actual_meta_key = (
            meta_key
            if not is_unset(meta_key)
            else ("node_metadata" if mode == "db" else "metadata")
        )

        if self._progression.name and actual_meta_key in data:
            data[actual_meta_key]["progression_name"] = self._progression.name

        item_mode = "python" if mode == "python" else "json"
        items = []
        for item in self:
            if isinstance(item, Element):
                items.append(
                    item.to_dict(
                        mode=item_mode,
                        meta_key=item_meta_key,
                        created_at_format=item_created_at_format,
                    )
                )
            else:
                items.append(item.to_dict(mode=item_mode))
        data["items"] = items

        return data

    # ==================== Core Operations ====================

    @synchronized
    @must_be_allowed
    def add(self, item: T) -> None:
        if item.id in self._items:
            raise ExistsError(f"Item {item.id} already exists in pile")

        self._items[item.id] = item
        self._progression.append(item.id)

    @synchronized
    @must_exist
    def remove(self, item_id: UUID | str | Element) -> T:
        uid = self._coerce_id(item_id)
        item = self._items.pop(uid)
        self._progression.remove(uid)
        return item

    @synchronized
    def pop(self, item_id: UUID | str | Element, default: Any = ...) -> T | Any:
        uid = self._coerce_id(item_id)

        try:
            item = self._items.pop(uid)
            self._progression.remove(uid)
            return item
        except KeyError:
            if default is ...:
                raise NotFoundError(f"Item {uid} not found in pile") from None
            return default

    @synchronized
    def get(self, item_id: UUID | str | Element, default: Any = ...) -> T | Any:
        uid = self._coerce_id(item_id)

        try:
            return self._items[uid]
        except KeyError:
            if default is ...:
                raise NotFoundError(f"Item {uid} not found in pile") from None
            return default

    @synchronized
    @must_be_allowed
    @must_exist
    def update(self, item: T) -> None:
        self._items[item.id] = item

    @synchronized
    def clear(self) -> None:
        self._items.clear()
        self._progression.clear()

    # ==================== Set-like Operations ====================

    @synchronized
    def include(self, item: T) -> bool:
        if item.id in self._items:
            return True
        try:
            _must_be_allowed(self, item)
            self._items[item.id] = item
            self._progression.append(item.id)
            return True
        except Exception:
            return False

    @synchronized
    def exclude(self, item: UUID | str | Element) -> bool:
        try:
            uid = self._coerce_id(item)
        except Exception:
            return False
        if uid not in self._items:
            return True
        self._items.pop(uid, None)
        with contextlib.suppress(ValueError):
            self._progression.remove(uid)
        return True

    # ==================== Rich __getitem__ (Type Dispatch) ====================

    @overload
    def __getitem__(self, key: UUID | str) -> T: ...
    @overload
    def __getitem__(self, key: Progression) -> Pile[T]: ...
    @overload
    def __getitem__(self, key: int) -> T: ...
    @overload
    def __getitem__(self, key: slice) -> Pile[T]: ...
    @overload
    def __getitem__(self, key: list[int] | tuple[int, ...]) -> Pile[T]: ...
    @overload
    def __getitem__(self, key: list[UUID] | tuple[UUID, ...]) -> Pile[T]: ...
    @overload
    def __getitem__(self, key: Callable[[T], bool]) -> Pile[T]: ...

    def __getitem__(self, key: Any) -> T | Pile[T]:
        if isinstance(key, (UUID, str)):
            return self.get(key)
        elif isinstance(key, int):
            return self._get_by_index(key)
        elif isinstance(key, Progression):
            return self._filter_by_progression(key)
        elif isinstance(key, slice):
            return self._get_by_slice(key)
        elif isinstance(key, (list, tuple)):
            return self._get_by_list(key)
        elif callable(key):
            return self._filter_by_function(key)
        else:
            raise TypeError(
                f"Invalid key type: {type(key)}. Expected UUID, str, int, slice, list, tuple, Progression, or callable"
            )

    def _filter_by_progression(self, prog: Progression) -> Pile[T]:
        if any(uid not in self._items for uid in prog):
            raise NotFoundError("Some items from progression not found in pile")

        return Pile(
            items=[self._items[uid] for uid in prog],
            item_type=self.item_type,
            strict_type=self.strict_type,
        )

    @synchronized
    def _get_by_index(self, index: int) -> T:
        uid: UUID = self._progression[index]
        return self._items[uid]

    @synchronized
    def _get_by_slice(self, s: slice) -> Pile[T]:
        uids: list[UUID] = self._progression[s]
        return Pile(
            items=[self._items[uid] for uid in uids],
            item_type=self.item_type,
            strict_type=self.strict_type,
        )

    @synchronized
    def _get_by_list(self, keys: list | tuple) -> Pile[T]:
        if not keys:
            raise ValueError("Cannot get items with empty list/tuple")

        first = keys[0]
        if isinstance(first, int):
            if not all(isinstance(k, int) for k in keys):
                raise TypeError("Cannot mix int and UUID in list/tuple indexing")
            items = [self._get_by_index(idx) for idx in keys]
        elif isinstance(first, (UUID, str)):
            if not all(isinstance(k, (UUID, str)) for k in keys):
                raise TypeError("Cannot mix int and UUID in list/tuple indexing")
            items = [self.get(uid) for uid in keys]
        else:
            raise TypeError(f"list/tuple must contain only int or UUID, got {type(first)}")

        return Pile(
            items=items,
            item_type=self.item_type,
            strict_type=self.strict_type,
        )

    def _filter_by_function(self, func: Callable[[T], bool]) -> Pile[T]:
        return Pile(
            items=[item for item in self if func(item)],
            item_type=self.item_type,
            strict_type=self.strict_type,
        )

    def filter_by_type(self, item_type: type[T] | set[type] | list[type]) -> Pile[T]:
        types_to_filter = extract_types(item_type)

        if self.item_type is not None:
            for t in types_to_filter:
                _must_be_allowed(self, t)

        filtered_items = [
            item for item in self if any(isinstance(item, t) for t in types_to_filter)
        ]

        if not filtered_items:
            raise NotFoundError(f"No items of type(s) {types_to_filter} found in pile")

        return Pile(
            items=filtered_items,
            item_type=self.item_type,
            strict_type=self.strict_type,
        )

    # ==================== Context Managers ====================

    async def __aenter__(self) -> Pile[T]:
        await self._async_lock.acquire()
        return self

    async def __aexit__(self, _exc_type, _exc_val, _exc_tb) -> None:
        self._async_lock.release()

    # ==================== Query Operations ====================

    @synchronized
    def __contains__(self, item: UUID | str | Element) -> bool:
        with contextlib.suppress(Exception):
            uid = self._coerce_id(item)
            return uid in self._items
        return False

    @synchronized
    def __len__(self) -> int:
        return len(self._items)

    def __bool__(self) -> bool:
        return len(self._items) > 0

    @synchronized
    def __iter__(self) -> Iterator[T]:  # type: ignore[override]
        """Iterate in progression order. LSP override: yields T, not field tuples."""
        for uid in self._progression:
            yield self._items[uid]

    def keys(self) -> Iterator[UUID]:
        return iter(self._progression)

    def items(self) -> Iterator[tuple[UUID, T]]:
        for i in self:
            yield (i.id, i)

    def __list__(self) -> list[T]:
        return [i for i in self]

    def is_empty(self) -> bool:
        return len(self._items) == 0

    # ==================== Deserialization ====================

    @classmethod
    @override
    def from_dict(
        cls,
        data: dict[str, Any],
        meta_key: str | UnsetType = Unset,
        item_meta_key: str | UnsetType = Unset,
        **kwargs: Any,
    ) -> Pile[T]:
        from lionagi.protocols.generic.element import Element

        data = data.copy()

        if not is_unset(meta_key) and meta_key in data:
            data["metadata"] = data.pop(meta_key)
        elif "node_metadata" in data and "metadata" not in data:
            data["metadata"] = data.pop("node_metadata")
        data.pop("node_metadata", None)

        item_type_data = data.get("item_type") or kwargs.get("item_type")
        strict_type = data.get("strict_type", False)

        items_data = data.get("items", [])
        if item_type_data is not None and items_data:
            if (
                isinstance(item_type_data, list)
                and item_type_data
                and isinstance(item_type_data[0], str)
            ):
                allowed_types = {load_type_from_string(type_str) for type_str in item_type_data}
            else:
                allowed_types = extract_types(item_type_data)

            for item_dict in items_data:
                metadata = item_dict.get("metadata", {})
                kron_class = metadata.get("kron_class") or metadata.get("lion_class")
                if kron_class:
                    try:
                        item_type_actual = load_type_from_string(kron_class)
                    except ValueError:
                        continue
                    kron_class_must_be_allowed(strict_type, item_type_actual, allowed_types)

        pile_data = data.copy()
        pile_data.pop("items", None)
        pile_data.pop("item_type", None)
        pile_data.pop("strict_type", None)
        pile = cls(item_type=item_type_data, strict_type=strict_type, **pile_data)

        metadata = data.get("metadata", {})
        progression_name = metadata.get("progression_name")
        if progression_name:
            pile._progression.name = progression_name

        for item_dict in items_data:
            metadata = item_dict.get("metadata", {})
            if "lion_class" in metadata and "kron_class" not in metadata:
                from lionagi.protocols.generic import Element as ProductionElement

                item = ProductionElement.from_dict(item_dict)
            else:
                item = Element.from_dict(item_dict, meta_key=item_meta_key)
            pile.add(item)  # type: ignore[arg-type]

        return pile

    def __repr__(self) -> str:
        return f"Pile(len={len(self)})"
