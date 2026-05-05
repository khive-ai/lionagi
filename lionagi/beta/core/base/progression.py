# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib
from typing import Any, overload
from uuid import UUID

from pydantic import Field, PrivateAttr, field_validator

from lionagi._errors import NotFoundError
from lionagi.beta.protocols import Containable, implements

from .element import Element

__all__ = ("Progression",)


@implements(Containable)
class Progression(Element):
    """Ordered UUID sequence with O(1) membership via auxiliary set.

    Uses list for ordered storage + set for O(1) `in` checks.
    Allows duplicates in order (set tracks presence, not count).

    Warning:
        Do NOT mutate `order` directly - use provided methods to keep
        internal `_members` set synchronized.

    Attributes:
        name: Optional identifier for this progression.
        order: Ordered UUID sequence (allows duplicates).

    Example:
        >>> prog = Progression(name="queue")
        >>> prog.append(some_element)
        >>> some_element.id in prog  # O(1)
        True
    """

    name: str | None = Field(
        default=None,
        description="Optional name for this progression (e.g., 'execution_order')",
    )
    order: list[UUID] = Field(
        default_factory=list,
        description="Ordered sequence of UUIDs",
    )
    # Auxiliary set for O(1) membership checks (not serialized)
    _members: set[UUID] = PrivateAttr(default_factory=set)

    @field_validator("order", mode="before")
    @classmethod
    def _validate_order(cls, value: Any) -> list[UUID]:
        """Coerce input to list[UUID]. Accepts None, single item, or iterable."""
        if value is None:
            return []

        # Normalize single values to list
        if isinstance(value, (UUID, str, Element)):
            value = [value]
        elif not isinstance(value, list):
            value = list(value)

        # Coerce all items to UUIDs (let coercion errors raise)
        return [cls._coerce_id(item) for item in value]

    def model_post_init(self, __context: Any) -> None:
        """Initialize _members set from order."""
        super().model_post_init(__context)
        self._members = set(self.order)

    def _rebuild_members(self) -> None:
        """Rebuild _members from order (after slice assignment)."""
        self._members = set(self.order)

    # ==================== Core Operations ====================

    def append(self, item_id: UUID | Element) -> None:
        """Append item to end. O(1)."""
        uid = self._coerce_id(item_id)
        self.order.append(uid)
        self._members.add(uid)

    def insert(self, index: int, item_id: UUID | Element) -> None:
        """Insert item at index. O(n)."""
        uid = self._coerce_id(item_id)
        self.order.insert(index, uid)
        self._members.add(uid)

    def remove(self, item_id: UUID | Element) -> None:
        """Remove first occurrence. O(n). Raises ValueError if not found."""
        uid = self._coerce_id(item_id)
        self.order.remove(uid)
        if uid not in self.order:
            self._members.discard(uid)

    def pop(self, index: int = -1, default: Any = ...) -> UUID | Any:
        """Remove and return item at index.

        Args:
            index: Position to pop (default: -1, last item).
            default: Return value if index invalid (default: raise NotFoundError).

        Returns:
            UUID at index, or default if provided and index invalid.

        Raises:
            NotFoundError: If index out of bounds and no default.
        """
        try:
            uid = self.order.pop(index)
            if uid not in self.order:
                self._members.discard(uid)
            return uid
        except IndexError as e:
            if default is ...:
                raise NotFoundError(
                    f"Index {index} not found in progression of length {len(self)}",
                    details={"index": index, "length": len(self)},
                ) from e
            return default

    def popleft(self) -> UUID:
        """Remove and return first item. O(n) due to list shift.

        Raises:
            NotFoundError: If empty. Use deque for frequent popleft.
        """
        if not self.order:
            raise NotFoundError("Cannot pop from empty progression")
        uid = self.order.pop(0)
        if uid not in self.order:
            self._members.discard(uid)
        return uid

    def clear(self) -> None:
        """Remove all items."""
        self.order.clear()
        self._members.clear()

    def extend(self, items: list[UUID | Element]) -> None:
        """Append multiple items. O(k) where k = len(items)."""
        uids = [self._coerce_id(item) for item in items]
        self.order.extend(uids)
        self._members.update(uids)

    # ==================== Query Operations ====================

    def __contains__(self, item: UUID | Element) -> bool:
        """O(1) membership check via auxiliary set."""
        with contextlib.suppress(Exception):
            uid = self._coerce_id(item)
            return uid in self._members
        return False

    def __len__(self) -> int:
        return len(self.order)

    def __bool__(self) -> bool:
        return len(self.order) > 0

    def __iter__(self):
        return iter(self.order)

    @overload
    def __getitem__(self, index: int) -> UUID: ...

    @overload
    def __getitem__(self, index: slice) -> list[UUID]: ...

    def __getitem__(self, index: int | slice) -> UUID | list[UUID]:
        return self.order[index]

    def __setitem__(self, index: int | slice, value: UUID | Element | list) -> None:
        """Set item(s) at index. Slice assignment requires list value."""
        if isinstance(index, slice):
            if not isinstance(value, list):
                raise TypeError(f"Cannot assign {type(value).__name__} to slice, expected list")
            new_uids = [self._coerce_id(v) for v in value]
            self.order[index] = new_uids
            self._rebuild_members()
        else:
            old_uid = self.order[index]
            new_uid = self._coerce_id(value)
            self.order[index] = new_uid
            if old_uid not in self.order:
                self._members.discard(old_uid)
            self._members.add(new_uid)

    def index(self, item_id: UUID | Element) -> int:
        """Return first index of item. O(n). Raises ValueError if not found."""
        uid = self._coerce_id(item_id)
        return self.order.index(uid)

    def __reversed__(self):
        return reversed(self.order)

    def __list__(self) -> list[UUID]:
        return list(self.order)

    def _validate_index(self, index: int, allow_end: bool = False) -> int:
        """Normalize and validate index (supports negative indexing)."""
        length = len(self.order)
        if length == 0 and not allow_end:
            raise NotFoundError("Progression is empty")

        if index < 0:
            index = length + index

        max_index = length if allow_end else length - 1
        if index < 0 or index > max_index:
            raise NotFoundError(
                f"Index {index} out of range for progression of length {length}",
                details={"index": index, "length": length, "allow_end": allow_end},
            )

        return index

    # ==================== Workflow Operations ====================

    def move(self, from_index: int, to_index: int) -> None:
        """Move item from one position to another. O(n)."""
        from_index = self._validate_index(from_index)
        to_index = self._validate_index(to_index, allow_end=True)

        item = self.order.pop(from_index)
        if from_index < to_index:
            to_index -= 1
        self.order.insert(to_index, item)

    def swap(self, index1: int, index2: int) -> None:
        """Swap items at two positions. O(1)."""
        index1 = self._validate_index(index1)
        index2 = self._validate_index(index2)
        self.order[index1], self.order[index2] = self.order[index2], self.order[index1]

    def reverse(self) -> None:
        """Reverse order in-place. O(n)."""
        self.order.reverse()

    # ==================== Set-like Operations ====================

    def include(self, item: UUID | Element) -> bool:
        """Add item if not present (idempotent). Returns True if added."""
        uid = self._coerce_id(item)
        if uid not in self._members:
            self.order.append(uid)
            self._members.add(uid)
            return True
        return False

    def exclude(self, item: UUID | Element) -> bool:
        """Remove item if present (idempotent). Returns True if removed."""
        uid = self._coerce_id(item)
        if uid in self._members:
            self.order.remove(uid)
            if uid not in self.order:
                self._members.discard(uid)
            return True
        return False

    def __repr__(self) -> str:
        name_str = f" name='{self.name}'" if self.name else ""
        return f"Progression(len={len(self)}{name_str})"
