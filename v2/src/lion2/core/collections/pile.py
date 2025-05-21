from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any, Generic, TypeVar
from uuid import UUID

from lionfuncs.concurrency import Lock
from pydantic import (
    Field,
    field_serializer,
)
from pydantic.fields import FieldInfo
from pydapter.protocols import Identifiable

from lion2.errors import ItemExistsError, ItemNotFoundError

from .progression import Progression, validate_order

T = TypeVar("T", bound=Identifiable)
D = TypeVar("D")


class Pile(Identifiable, Generic[T]):
    """A collection of items with a specific type and progression.

    if item_type is not specified, it defaults to None.
    if strict_type is set to True, the item_type must be exactly the same as the type of items in the collection. Not even subclasses are allowed.
    if strict_type is set to False, the item_type can be a superclass of the items in the collection.
    """

    collections: dict[UUID, T] = Field(default_factory=dict)
    progression: Progression[T] = Field(default_factory=Progression)
    item_type: type[T] | None = Field(default=None)
    strict_type: bool = Field(default=False, frozen=True)

    def __pydantic_extra__(self) -> dict[str, FieldInfo]:
        return {
            "_async_lock": Field(default_factory=Lock),
        }

    def __pydantic_private__(self) -> dict[str, FieldInfo]:
        return self.__pydantic_extra__()

    @property
    def order(self) -> list[UUID]:
        """Returns the order of items in the collection."""
        return self.progression.order

    @property
    def async_lock(self):
        """Async lock."""
        if not hasattr(self, "_async_lock") or self._async_lock is None:
            self._async_lock = Lock()
        return self._async_lock

    @field_serializer("collections")
    def _serialize_collections(self, v: dict[UUID, T]) -> list[dict[str, Any]]:
        return [item.model_dump() for item in v.values()]

    def __init__(
        self,
        collections: dict[UUID, T] | None = None,
        progression: Progression[T] | None = None,
        item_type: type[T] | None = None,
        strict_type: bool = False,
        **kwargs: Any,
    ):
        data = {
            "collections": collections,
            "progression": progression,
            "item_type": item_type,
            "strict_type": strict_type,
            **kwargs,
        }
        data = _prepare_pile_data(data)
        return super().__init__(**data)

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
        async with self.async_lock:
            return [
                self.collections[item_id] for item_id in self.progression.order
            ]

    async def __aenter__(self) -> Pile[T]:
        """Enters the asynchronous context, acquiring the internal lock."""
        await self.async_lock.acquire()
        return self

    async def __aexit__(
        self, exc_type: Any, exc_val: Any, exc_tb: Any
    ) -> None:
        """Exits the asynchronous context, releasing the internal lock."""
        self.async_lock.release()

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
        async with self.async_lock:
            self._validate_item_type(item)
            if item.id in self.collections:
                raise ValueError(f"Item with ID '{item.id}' already exists.")
            self.collections[item.id] = item
            self.progression.order.append(item.id)

    async def get(self, item: T | UUID | int, default: D = ...) -> T | D:
        async with self.async_lock:
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

    async def pop(self, item: T | UUID | int, default: D = ...) -> T | D:
        async with self.async_lock:
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
        async with self.async_lock:
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
        async with self.async_lock:
            item = [item] if not isinstance(item, list) else item
            for i in item:
                self._validate_item_type(i)
            self.collections.update({i.id: i for i in item})
            self.progression.include(item)
            return True

    async def insert(self, index: int, item: T, /) -> None:
        async with self.async_lock:
            self._validate_item_type(item)
            if item.id in self.collections:
                raise ItemExistsError(
                    "Item already exists in the collection.", items=item.id
                )
            self.collections[item.id] = item
            self.progression.insert(index, item)

    async def exclude(self, item: list[T] | T | UUID | list[UUID], /):
        async with self.async_lock:
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
        async with self.async_lock:
            self.collections.clear()
            self.progression.order.clear()

    async def length(self) -> int:
        """
        Purpose: Asynchronously returns the number of items in the collection.
        """
        async with self.async_lock:
            return len(self.progression.order)

    async def __aiter__(self) -> AsyncIterator[T]:
        """
        Purpose: Provides an asynchronous iterator over the items in order.
        Note: Creates a snapshot of the order under lock for safe iteration.
        """
        async with self.async_lock:
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

    async def aitems(self) -> AsyncIterator[tuple[UUID, T]]:
        """asynchronous iterator over (ID, item) pairs in order."""
        async with self.async_lock:
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

    async def akeys(self) -> AsyncIterator[UUID]:
        """asynchronous iterator over item IDs in order."""
        async with self.async_lock:
            current_order_snapshot = list(self.order)

        for item_id in current_order_snapshot:
            yield item_id
            await asyncio.sleep(0)

    def __len__(self) -> int:
        """Returns the number of items in the collection."""
        return len(self.progression.order)

    def __getitem__(self, item, /) -> T | list[T]:
        if isinstance(item, int | slice):
            item = self.progression.order[item]
            item = [item] if not isinstance(item, list) else item

        else:
            item = validate_order(item)

        for i in item:
            if i not in self.collections:
                raise ItemNotFoundError(
                    "Item not found in collections.", items=i
                )

        if isinstance(item, list | tuple) and len(item) == 1:
            return self.collections[item[0]]

        return [self.collections[i] for i in item]


def _prepare_pile_data(data: Any) -> Any:
    if not isinstance(data, dict):
        return data

    item_type_to_use = data.get("item_type") or Identifiable
    if not (
        isinstance(item_type_to_use, type)
        and issubclass(item_type_to_use, Identifiable)
    ):
        raise ValueError(
            f"Invalid item_type provided for Pile: {item_type_to_use}"
        )

    collections_input = data.get("collections")
    processed_collections: dict[UUID, Identifiable] = {}

    if collections_input:
        input_list = []
        if isinstance(collections_input, list):
            input_list = collections_input
        elif isinstance(collections_input, dict):
            # Check if it's already a dict of {UUID: Identifiable instances}
            if all(
                isinstance(k, UUID) and isinstance(v, item_type_to_use)
                for k, v in collections_input.items()
            ):
                processed_collections = collections_input  # Use as is
                input_list = (
                    None  # Skip further processing loop for collections
                )
            else:  # Assume it's a single item dict if not {uuid:item_obj} that needs validation
                input_list = [collections_input]
        elif isinstance(collections_input, Identifiable):
            input_list = [collections_input]
        else:
            raise TypeError(
                "Input for collections must be a list, dict, or Identifiable instance."
            )

        if input_list is not None:
            for item_data in input_list:
                current_item_id = None
                try:
                    if isinstance(item_data, dict):
                        current_item_id = item_data.get("id", "Unknown ID")
                        instance = item_type_to_use.model_validate(item_data)
                        if instance.id in processed_collections:
                            raise ValueError(
                                f"Duplicate ID '{instance.id}' found in input collections."
                            )
                        processed_collections[instance.id] = instance
                    elif isinstance(item_data, Identifiable):
                        current_item_id = item_data.id
                        if item_data.id in processed_collections:
                            raise ValueError(
                                f"Duplicate ID '{item_data.id}' found in input collections."
                            )
                        processed_collections[item_data.id] = item_data
                    else:
                        raise TypeError(
                            "Items in collections must be Identifiable instances or dictionaries."
                        )
                except Exception as e:
                    id_info = (
                        f" (Item ID: {current_item_id})"
                        if current_item_id
                        else ""
                    )
                    raise TypeError(
                        f"Failed to process item{id_info} for collections as type "
                        f"{item_type_to_use.__name__}: {item_data}. Original Error: {e}"
                    )

    data["collections"] = processed_collections

    # Handle Progression
    progression_input = data.get("progression")
    progression_order: list[UUID] = []

    if isinstance(progression_input, Progression):
        progression_order = list(progression_input.order)
    elif isinstance(progression_input, dict) and "order" in progression_input:
        raw_order = progression_input.get("order", [])
        try:
            validated_id_list = [UUID(str(oid)) for oid in raw_order]
            progression_order = [
                oid
                for oid in validated_id_list
                if oid in processed_collections
            ]
        except Exception as e:
            raise ValueError(
                f"Failed to process 'order' from progression data: {raw_order}. Error: {e}"
            )
    elif not progression_input and processed_collections:
        progression_order = list(processed_collections.keys())

    # Ensure progression order length matches collections if order is not empty
    # If order was specified, it must now be a subset of collection keys.
    # If it was inferred, it will match.
    if progression_order:
        if len(progression_order) != len(processed_collections):
            raise ValueError("The items in collections and order must match.")
        if set(progression_order) != set(processed_collections.keys()):
            raise ValueError("The items in collections and order must match.")

    data["progression"] = Progression(order=progression_order)

    # Strict type checking
    strict_type = data.get("strict_type", None)
    if strict_type and item_type_to_use is not Identifiable:
        for item_id, item_instance in processed_collections.items():
            if item_instance.__class__ is not item_type_to_use:
                raise TypeError(
                    f"All items must be of type {item_type_to_use.__name__}"
                )
    # Non-strict check (already handled by item_type_to_use.model_validate for dicts,
    # and isinstance(item_data, Identifiable) for instances,
    elif not strict_type and item_type_to_use is not Identifiable:
        for item_id, item_instance in processed_collections.items():
            if not isinstance(item_instance, item_type_to_use):
                raise TypeError(
                    f"Item {item_id} is not an instance of {item_type_to_use.__name__} "
                    f"(or its subclass), found {item_instance.__class__.__name__}."
                )
    return data
