from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any, Generic, TypeVar

from lionfuncs.concurrency import Lock
from pydantic import Field, field_serializer, field_validator, model_validator
from pydantic.fields import FieldInfo
from pydapter.protocols import Identifiable
from typing_extensions import Self

from lionagi.core.errors import ItemExistsError, ItemNotFoundError
from lionagi.core.progression import Progression, validate_order

T = TypeVar("T", bound=Identifiable)
D = TypeVar("D")


class Pile(Identifiable, Generic[T]):
    """A collection of items with a specific type and progression.

    if item_type is not specified, it defaults to Identifiable.
    if strict_type is set to True, the item_type must be exactly the same as the type of items in the collection. Not even subclasses are allowed.
    if strict_type is set to False, the item_type can be a superclass of the items in the collection.
    """

    collections: dict[T.id, T] = Field(default_factory=dict)
    progression: Progression[T] = Field(default_factory=Progression)
    item_type: type[T] = Identifiable
    strict_type: bool = False

    @field_serializer("collections")
    def _serialize_collections(self, v: dict[T.id, T]) -> list[dict[str, Any]]:
        return [item.model_dump() for item in v.values()]

    @field_validator("collections", mode="before")
    def _validate_collections(cls, v):
        if isinstance(v, dict):
            v = list(v.values())
        v = [v] if not isinstance(v, list) else v
        if not all(isinstance(item, Identifiable) for item in v):
            raise TypeError("All items must be Identifiable.")
        dict_ = {item.id: item for item in v}
        if len(dict_) != len(v):
            raise ValueError("Duplicate IDs found in collections.")
        return dict_

    @model_validator(mode="after")
    def _validate_collections_type_length(self) -> Self:
        if self.strict_type and self.item_type is not Identifiable:
            if not all(
                item.__class__ is self.item_type
                for item in self.collections.values()
            ):
                raise TypeError(
                    f"All items must be of type {self.item_type.__name__}"
                )
        if len(self.order) > 0 and len(self.collections) != len(self.order):
            raise ValueError(
                "The number of items in collections and order must match."
            )
        if len(self.order) == 0 and len(self.collections) > 0:
            self.progression.order = list(self.collections.keys())
        return self

    @property
    def order(self) -> list[T.id]:
        return self.progression.order

    def __pydantic_extra__(self) -> dict[str, FieldInfo]:
        return {
            "_async_lock": Field(default_factory=Lock),
        }

    def __pydantic_private__(self) -> dict[str, FieldInfo]:
        return self.__pydantic_extra__()

    def __getstate__(self):
        """Prepare for pickling."""
        state = self.__dict__.copy()
        state["_async_lock"] = None
        return state

    def __setstate__(self, state):
        """Restore after unpickling."""
        self.__dict__.update(state)
        self._async_lock = Lock()

    async def to_list(self) -> list[T]:
        async with self._async_lock:
            return [
                self.collections[item_id] for item_id in self.progression.order
            ]

    async def __aenter__(self) -> Pile[T]:
        """Enters the asynchronous context, acquiring the internal lock."""
        await self._async_lock.acquire()
        return self

    async def __aexit__(
        self, exc_type: Any, exc_val: Any, exc_tb: Any
    ) -> None:
        """Exits the asynchronous context, releasing the internal lock."""
        self._async_lock.release()

    def _validate_item_type(self, item: T) -> None:
        if not isinstance(item, self.item_type):
            raise TypeError(
                f"Item must be of type {self.item_type.__name__}, "
                f"not {item.__class__.__name__}."
            )
        if self.strict_type and item.__class__ is not self.item_type:
            raise TypeError(
                f"Item must be of type {self.item_type.__name__}, "
                f"not {item.__class__.__name__}."
            )

    async def add(self, item: T) -> None:
        async with self._async_lock:
            self._validate_item_type(item)
            if item.id in self.collections:
                raise ValueError(f"Item with ID '{item.id}' already exists.")
            self.collections[item.id] = item
            self.progression.order.append(item.id)

    async def get(self, item: T | T.id | int, default: D = ...) -> T | D:
        async with self._async_lock:
            if isinstance(item, int):
                item = self.progression.order[item]

            item = item.id if isinstance(item, Identifiable) else item
            if item not in self.collections:
                if default is not ...:
                    return default
                raise ItemNotFoundError(
                    "Item not found in collections.", items=item
                )
            return self.collections[item]

    async def pop(self, item: T | T.id | int, default: D = ...) -> T | D:
        async with self._async_lock:
            if isinstance(item, int):
                item = self.progression.order[item]
            item = item.id if isinstance(item, Identifiable) else item
            if item not in self.collections:
                if default is not ...:
                    return default
                raise ItemNotFoundError(
                    "Item not found in collections.", items=item
                )
            item = self.collections.pop(item)
            self.progression.exclude(item)
            return item

    async def include(self, item: list[T] | T, /):
        async with self._async_lock:
            item = [item] if not isinstance(item, list) else item
            for i in item:
                try:
                    self._validate_item_type(i)
                except TypeError:
                    return False
            self.collections.update(
                {i.id: i for i in item if i.id not in self.collections}
            )
            self.progression.include(item)
            return True

    async def update(self, item: list[T] | T, /):
        async with self._async_lock:
            item = [item] if not isinstance(item, list) else item
            for i in item:
                self._validate_item_type(i)
            self.collections.update({i.id: i for i in item})
            self.progression.include(item)
            return True

    async def insert(self, index: int, item: T, /) -> None:
        async with self._async_lock:
            self._validate_item_type(item)
            if item in self:
                raise ItemExistsError(
                    f"Item already exists in the collection.", items=item.id
                )
            self.collections.update({i.id: i for i in item})
            self.progression.insert(index, item)

    async def exclude(self, item: list[T] | T | T.id | list[T.id], /):
        async with self._async_lock:
            item = [item] if not isinstance(item, list) else item
            try:
                item_ids = validate_order(item)
            except TypeError:
                return False

            for i in item_ids:
                self.collections.pop(i, None)
            self.progression.exclude(item_ids)
            return True

    async def clear(self) -> None:
        async with self._async_lock:
            self.collections.clear()
            self.progression.order.clear()

    async def length(self) -> int:
        """
        Purpose: Asynchronously returns the number of items in the collection.
        """
        async with self._async_lock:
            return len(self.progression.order)

    async def __aiter__(self) -> AsyncIterator[T]:
        """
        Purpose: Provides an asynchronous iterator over the items in order.
        Note: Creates a snapshot of the order under lock for safe iteration.
        """
        async with self._async_lock:
            current_order_snapshot = list(self.order)
            elements_snapshot = {
                id_: self.collections[id_]
                for id_ in current_order_snapshot
                if id_ in self.collections
            }

        for item_id in current_order_snapshot:
            if item_id in elements_snapshot:
                yield elements_snapshot[item_id]
            await asyncio.sleep(0)

    async def aitems(self) -> AsyncIterator[tuple[T.id, T]]:
        """asynchronous iterator over (ID, item) pairs in order."""
        async with self._async_lock:
            current_order_snapshot = list(self.order)
            elements_snapshot = {
                id_: self.collections[id_]
                for id_ in current_order_snapshot
                if id_ in self.collections
            }

        for item_id in current_order_snapshot:
            if item_id in elements_snapshot:
                yield (item_id, elements_snapshot[item_id])
            await asyncio.sleep(0)

    async def avalues(self) -> AsyncIterator[T]:
        """asynchronous iterator over item values in order"""
        async for _, value in self.aitems():
            yield value

    async def akeys(self) -> AsyncIterator[T.id]:
        """asynchronous iterator over item IDs in order."""
        async with self._async_lock:
            current_order_snapshot = list(self.order)

        for item_id in current_order_snapshot:
            yield item_id
            await asyncio.sleep(0)
