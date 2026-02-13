# Copyright (c) 2023-2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import threading
from typing import Any, Generic, TypeVar
from uuid import UUID

from pydantic import Field, PrivateAttr, model_validator
from typing_extensions import Self  # noqa: UP035

from lionagi._errors import ItemExistsError, ItemNotFoundError

from .element import ID, Element
from .pile import Pile
from .progression import Progression

__all__ = ("Flow",)

E = TypeVar("E", bound=Element)
P = TypeVar("P", bound=Progression)


class Flow(Element, Generic[E, P]):
    """Workflow state container pairing items (Pile) with named progressions.

    Progressions reference items by UUID. Referential integrity is
    validated: every UUID in a progression must exist in the items pile.

    Attributes:
        name: Optional flow identifier.
        items: Element storage (Pile[E]).
        progressions: Named UUID sequences (Pile[P]).
    """

    name: str | None = Field(
        default=None,
        description="Optional name for this flow.",
    )
    items: Pile[E] = Field(
        default_factory=Pile,
        description="Items that progressions reference.",
    )
    progressions: Pile[P] = Field(
        default_factory=Pile,
        description="Workflow stages as named progressions.",
    )
    _progression_names: dict[str, UUID] = PrivateAttr(default_factory=dict)
    _lock: threading.RLock = PrivateAttr(default_factory=threading.RLock)

    @model_validator(mode="after")
    def _validate_referential_integrity(self) -> Self:
        """Validate all progression UUIDs exist in items pile."""
        item_ids = set(self.items.keys())
        for prog in self.progressions:
            missing = set(prog) - item_ids
            if missing:
                raise ItemNotFoundError(
                    f"Progression '{prog.name}' references missing items: {missing}"
                )
        return self

    def model_post_init(self, __context: Any) -> None:
        """Rebuild _progression_names index from progressions."""
        super().model_post_init(__context)
        for progression in self.progressions:
            if progression.name:
                self._progression_names[progression.name] = progression.id

    # ==================== Progression Management ====================

    def add_progression(self, progression: P) -> None:
        """Add progression with name uniqueness and referential integrity.

        Args:
            progression: Progression to add.

        Raises:
            ItemExistsError: If name already registered.
            ItemNotFoundError: If progression contains UUIDs not in items.
        """
        with self._lock:
            if progression.name and progression.name in self._progression_names:
                raise ItemExistsError(f"Progression with name '{progression.name}' already exists.")

            item_ids = set(self.items.keys())
            missing = set(progression) - item_ids
            if missing:
                raise ItemNotFoundError(f"Progression references missing items: {missing}")

            self.progressions.include(progression)
            if progression.name:
                self._progression_names[progression.name] = progression.id

    def remove_progression(self, key: UUID | str | P) -> None:
        """Remove progression by UUID or name.

        Args:
            key: UUID, name string, or Progression instance.

        Raises:
            ItemNotFoundError: If progression not found.
        """
        with self._lock:
            if isinstance(key, str) and key in self._progression_names:
                uid = self._progression_names.pop(key)
                self.progressions.pop(uid)
                return

            uid = ID.get_id(key)
            prog = self.progressions[uid]
            if prog.name and prog.name in self._progression_names:
                del self._progression_names[prog.name]
            self.progressions.pop(uid)

    def get_progression(self, key: UUID | str | P) -> P:
        """Get progression by UUID or name.

        Args:
            key: UUID, name string, or Progression instance.

        Returns:
            Matching progression.

        Raises:
            ItemNotFoundError: If not found.
        """
        with self._lock:
            if isinstance(key, str) and key in self._progression_names:
                uid = self._progression_names[key]
                return self.progressions[uid]

            if isinstance(key, str):
                try:
                    uid = ID.get_id(key)
                    return self.progressions[uid]
                except Exception as exc:
                    raise ItemNotFoundError(
                        f"Progression '{key}' not found in flow"
                    ) from exc

            uid = key.id if isinstance(key, Progression) else key
            return self.progressions[uid]

    # ==================== Item Management ====================

    def add_item(
        self,
        item: E,
        progressions: list[UUID | str | P] | UUID | str | P | None = None,
    ) -> None:
        """Add item and optionally append to progressions.

        Args:
            item: Element to add.
            progressions: Progression(s) to append item to (by instance, UUID, or name).

        Raises:
            ItemExistsError: If item UUID already in pile.
            KeyError: If any progression not found.
        """
        with self._lock:
            resolved: list[P] = []
            if progressions is not None:
                if isinstance(progressions, (str, UUID, Progression)):
                    progs_list = [progressions]
                else:
                    progs_list = list(progressions)

                for p in progs_list:
                    if isinstance(p, Progression):
                        resolved.append(p)
                    else:
                        resolved.append(self.get_progression(p))

            self.items.include(item)

            for prog in resolved:
                prog.append(item)

    def remove_item(self, item_id: UUID | str | Element) -> None:
        """Remove item from storage and all progressions.

        Args:
            item_id: UUID, UUID string, or Element instance.

        Raises:
            ItemNotFoundError: If item not in pile.
        """
        with self._lock:
            uid = ID.get_id(item_id)

            for progression in self.progressions:
                if uid in progression:
                    progression.exclude(uid)

            self.items.pop(uid)

    def clear(self) -> None:
        """Clear all items and progressions."""
        with self._lock:
            self.items.clear()
            self.progressions.clear()
            self._progression_names.clear()

    def __repr__(self) -> str:
        name_str = f", name='{self.name}'" if self.name else ""
        return f"Flow(items={len(self.items)}, progressions={len(self.progressions)}{name_str})"

    def __len__(self) -> int:
        return len(self.items)


# File: lionagi/protocols/generic/flow.py
