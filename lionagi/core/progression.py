from __future__ import annotations

from typing import Any, Generic, Mapping, TypeVar
from uuid import UUID

from pydantic import Field, field_serializer, field_validator
from pydapter.protocols import Identifiable
from pydapter.protocols.utils import validate_uuid

from .errors import ItemNotFoundError

T = TypeVar("T", bound=Identifiable)


class Progression(Identifiable, Generic[T]):
    """Tracks an ordered sequence of item IDs, with optional naming."""

    name: str | None = None
    order: list[T.id] = Field(default_factory=list)

    @field_validator("order", mode="before")
    def _validate_ordering(cls, v: Any) -> list[T.id]:
        return validate_order(v)

    @field_serializer("order")
    def _serialize_order(self, v: list[T.id]) -> list[str]:
        return [str(x) for x in v]

    def __len__(self) -> int:
        """Returns the number of items in this progression."""
        return len(self.order)

    def __bool__(self) -> bool:
        """Indicates if this progression has any items."""
        return bool(self.order)

    def __contains__(self, item: Any) -> bool:
        """Checks if one or more IDs exist in the progression.

        Args:
            item (Any): Could be an `Element`, `IDType`, `UUID`, string,
                or a sequence of these.
        """
        try:
            if isinstance(item, UUID):
                return item in self.order
            if isinstance(item, Identifiable):
                return item.id in self.order
            refs = self._validate_ordering(item)
            return all(ref in self.order for ref in refs)
        except Exception:
            return False

    def __getitem__(self, key: int | slice) -> T.id | list[T.id]:
        """Gets one or more items by index or slice."""
        if not isinstance(key, (int, slice)):
            raise TypeError(
                f"indices must be integers or slices, not {key.__class__.__name__}"
            )
        try:
            if not (a := self.order[key]):
                raise ItemNotFoundError(
                    "Item not found in progression.", items=key
                )
            return a
        except IndexError as e:
            raise ItemNotFoundError(
                "Item not found in progression.", items=key
            ) from e

    def __setitem__(self, key: int | slice, value: Any) -> None:
        """Sets items by index or slice."""
        refs = self._validate_ordering(value)
        if isinstance(key, slice):
            self.order[key] = refs
        else:
            try:
                self.order[key] = refs[0]
            except IndexError:
                # If key is out of range, insertion occurs
                self.order.insert(key, refs[0])

    def __delitem__(self, key: int | slice) -> None:
        """Deletes item(s) by index or slice."""
        del self.order[key]

    def __iter__(self):
        """Iterates over the IDs in this progression."""
        return iter(self.order)

    def __next__(self) -> T.id:
        """Returns the next item if used as an iterator."""
        try:
            return next(iter(self.order))
        except StopIteration:
            raise StopIteration("No more items in the progression")

    def clear(self) -> None:
        """Removes all items from the progression."""
        self.order.clear()

    def append(self, item: Any, /) -> None:
        """Appends one or more IDs at the end of the progression."""
        if isinstance(item, Identifiable):
            self.order.append(item.id)
            return
        refs = validate_order(item)
        self.order.extend(refs)

    def pop(self, index: int = -1) -> T.id:
        """Removes and returns one ID by index."""
        try:
            return self.order.pop(index)
        except Exception as e:
            raise ItemNotFoundError(f"Index {index} not found in progression.")

    def popleft(self) -> T.id:
        """Removes and returns the first ID."""
        if not self.order:
            raise ItemNotFoundError("No items in progression.")
        return self.order.pop(0)

    def remove(self, item: Any, /) -> None:
        """Removes the first occurrence of each specified ID."""
        refs = validate_order(item)
        if not refs:
            raise ItemNotFoundError("No items to remove.")

        missing = [r for r in refs if r not in self.order]
        if missing:
            raise ItemNotFoundError(f"Items not found", items=missing)
        self.order = [x for x in self.order if x not in refs]

    def count(self, item: Any, /) -> int:
        """Counts the number of occurrences of an ID."""
        if not isinstance(item, (Identifiable, UUID)):
            raise TypeError(
                f"Item '{item}' is not a valid ID or Identifiable object."
            )
        id = item.id if isinstance(item, Identifiable) else item
        return self.order.count(id)

    def index(self, item: Any, start: int = 0, end: int | None = None) -> int:
        """Finds the index of the first occurrence of an ID."""
        if not isinstance(item, (Identifiable, UUID)):
            raise TypeError(
                f"Item '{item}' is not a valid ID or Identifiable object."
            )
        id = item.id if isinstance(item, Identifiable) else item
        if end is not None:
            return self.order.index(id, start, end)
        return self.order.index(id, start)

    def extend(self, other: Progression) -> None:
        """Appends all IDs from another Progression to this one."""
        if not isinstance(other, Progression):
            raise ValueError("Can only extend with another Progression.")
        self.order.extend(other.order)

    def insert(self, index: int, item: Any, /) -> None:
        """Inserts one or more IDs at a specified index."""
        item_ = validate_order(item)
        for i in reversed(item_):
            self.order.insert(index, i)

    def __reversed__(self) -> Progression[T]:
        """Returns a new reversed Progression.

        Returns:
            Progression[E]: A reversed copy of the current progression.
        """
        return Progression(order=reversed(self.order[:]))

    def __eq__(self, other: object) -> bool:
        """Checks equality with another Progression."""
        if not isinstance(other, Progression):
            return NotImplemented
        return (self.order == other.order) and (self.name == other.name)

    def __repr__(self) -> str:
        """Returns a string representation of the progression.

        Returns:
            str: A formatted string showing name and order contents.
        """
        return f"Progression(name={self.name}, order={self.order})"

    def include(self, item: Any, /) -> bool:
        """Guarantees the inclusion of membership.

        Args:
            item (Any): Could be a single ID/Element or a list/tuple
                of them.
        """
        try:
            refs = validate_order(item)
        except ValueError:
            return False
        if not refs:
            return True

        existing = set(self.order)
        appended = False
        for ref in refs:
            if ref not in existing:
                self.order.append(ref)
                existing.add(ref)
                appended = True
        return appended

    def exclude(self, item: Any, /) -> bool:
        """Guarantees the exclusion of membership.

        Args:
            item (Any):
                Could be a single ID/Element or a list/tuple of them.

        Returns:
            bool: False if error, else True.
        """
        try:
            refs = validate_order(item)
        except ValueError:
            return False
        if not refs:
            return True

        rset = set(refs)
        self.order = [x for x in self.order if x not in rset]
        return True

    def to_list(self) -> list[T.id]:
        """Returns a copy of the order as a list."""
        return self.order[:]


def validate_order(order: Any) -> list[T.id]:
    """Validates and flattens an ordering into a list of IDType objects.

    This function accepts a variety of possible representations for ordering
    (e.g., a single Element, a list of Elements, a dictionary with ID keys,
    or a nested structure) and returns a flat list of IDType objects.
    """
    if isinstance(order, Identifiable):
        return [order.id]
    if isinstance(order, Mapping):
        order = list(order.keys())

    stack = [order]
    out: list[UUID] = []
    while stack:
        cur = stack.pop()
        if cur is None:
            continue
        if isinstance(cur, Identifiable):
            out.append(cur.id)
        elif isinstance(cur, UUID):
            out.append(cur)
        elif isinstance(cur, str):
            out.append(validate_uuid(cur))
        elif isinstance(cur, (list, tuple, set)):
            stack.extend(reversed(cur))
        else:
            raise ValueError("Invalid item in order.")

    if not out:
        return []
    return out
