from __future__ import annotations

from typing import Any, Generic, TypeVar
from uuid import UUID

from pydantic import Field, field_validator

from ._concepts import Ordering
from .element import Element
from .ids import get_ids, validate_ids

T = TypeVar("T")
E = TypeVar("E", bound=Element)


IDAble = UUID | E
IDLike = UUID | str | E


class Progression(Element, Ordering[E], Generic[E]):

    order: list[UUID] = Field(default_factory=list)
    """A sequence of IDs representing the progression."""

    name: str | None = None
    """An optional name for this progression."""

    @field_validator("order", mode="before")
    def _validate_order(cls, value: Any) -> list[UUID]:
        return validate_ids(value)

    @field_validator("name", mode="before")
    def _validate_name(cls, value: Any) -> str | None:
        if not value:
            return None
        return str(value).strip() or None

    def __len__(self) -> int:
        return len(self._order)

    def __list__(self) -> list[UUID]:
        return self._order[:]

    def __getitem__(self, key: int | slice | list[int]) -> UUID | list[UUID]:
        """Gets one or more items by index or slice."""

        if isinstance(key, list):
            if not all(isinstance(i, int) for i in key):
                raise TypeError("All indices in the list must be integers.")
            return [self.order[i] for i in key]

        if not isinstance(key, (int, slice)):
            raise TypeError(
                f"indices must be integers or slices, not {type(key).__name__} type"
            )

        if not (a := self.order[key]):
            raise IndexError("Index out of range")
        if isinstance(key, slice):
            return list(a)
        return a

    def __setitem__(self, key: int | slice, value: Any) -> None:
        """Sets items by index or slice."""
        refs = validate_ids(value)
        if isinstance(key, slice):
            self._order[key] = refs
        else:
            try:
                self._order[key] = refs[0]
            except IndexError:
                # If key is out of range, insertion occurs
                self._order.insert(key, refs[0])

    def __delitem__(self, key: int | slice) -> None:
        """Deletes item(s) by index or slice."""
        del self._order[key]

    def __iter__(self):
        """Iterates over the IDs in this progression."""
        return iter(self._order)

    def __next__(self) -> UUID:
        """Returns the next item if used as an iterator."""
        try:
            return next(iter(self._order))
        except StopIteration:
            raise StopIteration("No more items in the progression")

    def __eq__(self, other: object) -> bool:
        """Checks equality with another Progression."""
        if not isinstance(other, Progression):
            return NotImplemented
        return (self._order == other._order) and (self.name == other.name)

    def __str__(self) -> str:
        return f"Progression(name={self.name}, order={self._order})"

    def __repr__(self) -> str:
        return self.__str__()

    def include(self, item: Any, /, *, as_single: bool = True) -> bool:
        """Adds new IDs at the end if they are not already present.
        - always True if all items are present after operation
        - False if errors occur
        """
        try:
            refs = get_ids(item, as_single)
        except ValueError:
            return False
        if not refs:
            return True

        non_existing = [r for r in refs if r not in set(self.order)]
        if non_existing:
            self.order.extend(non_existing)
        return True

    def exclude(self, item: Any, /, *, as_single: bool = True) -> bool:
        """Removes occurrences of the specified IDs.
        - always True if no items are present after operation
        - False if errors occur
        """
        try:
            refs = get_ids(item, as_single)
        except ValueError:
            return False
        if not refs:
            return True

        self.order = [x for x in self.order if x not in set(refs)]
        return True

    def append(self, item: Any, /) -> None:
        """Appends one or more IDs at the end of the progression.

        - treats collective and ordering as single items
        """
        if isinstance(item, Element):
            self._order.append(item.id)
            return
        refs = validate_ids(item)
        self._order.extend(refs)

    def extend(self, item: Any, /) -> None:
        """Extends the progression by appending one or more IDs at the end.
        Treats collective and ordering as multiple items.
        """
        refs = validate_ids(item)
        self._order.extend(refs)

    def insert(self, index: int, item: Any, /, as_items: bool = True) -> None:
        """Inserts one or more IDs at a specified index."""
        item_ = get_ids(item, as_items=as_items)
        for i in reversed(item_):
            self.order.insert(index, i)

    def __contains__(self, item) -> bool:
        try:
            refs = get_ids(item)
        except ValueError:
            return False
        return all(r in self.order for r in refs)