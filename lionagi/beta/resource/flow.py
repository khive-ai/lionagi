# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import threading
from typing import Any, Generic, Literal, TypeVar, cast
from uuid import UUID

from pydantic import Field, PrivateAttr, field_validator, model_validator

from lionagi._errors import ExistsError, NotFoundError
from lionagi.beta.protocols import Serializable, implements
from lionagi.beta.resource.pile import Pile
from lionagi.ln._utils import extract_types, synchronized
from lionagi.ln.types._sentinel import Unset, UnsetType
from lionagi.protocols.generic.element import Element
from lionagi.protocols.generic.progression import Progression

__all__ = ("Flow",)

E = TypeVar("E", bound=Element)  # Element type for items
P = TypeVar("P", bound=Progression)  # Progression type


@implements(Serializable)
class Flow(Element, Generic[E, P]):
    """Workflow state container with items and named progressions.

    Composition: items Pile (storage) + progressions Pile (ordered UUID sequences).
    Progressions reference items by UUID; referential integrity is validated.

    Thread Safety:
        Flow methods are RLock-synchronized. Direct pile access bypasses lock.

    Attributes:
        name: Optional flow identifier.
        items: Element storage (Pile[E]).
        progressions: Named UUID sequences (Pile[P]).

    Generic Parameters:
        E: Element type for items.
        P: Progression type.

    Example:
        >>> flow = Flow[Node, Progression](item_type=Node)
        >>> flow.add_item(node, progressions="ready")
    """

    name: str | None = Field(
        default=None,
        description="Optional name for this flow (e.g., 'task_workflow')",
    )
    progressions: Pile[P] = Field(
        default_factory=Pile,
        description="Workflow stages as named progressions",
    )
    items: Pile[E] = Field(
        default_factory=Pile,
        description="Items that progressions reference",
    )
    _progression_names: dict[str, UUID] = PrivateAttr(default_factory=dict)
    _lock: threading.RLock = PrivateAttr(default_factory=threading.RLock)

    def __init__(
        self,
        items: list[E] | Pile[E] | Element | None = None,
        progressions: list[P] | Pile[P] | None = None,
        name: str | None = None,
        item_type: type[E] | set[type] | list[type] | None = None,
        strict_type: bool = False,
        **data,
    ):
        """Initialize Flow with items, progressions, and type validation.

        Args:
            items: Initial items (Element, list, Pile, or list[dict]).
            progressions: Initial progressions (list, Pile, or list[dict]).
            name: Flow identifier.
            item_type: Allowed item type(s) for validation.
            strict_type: If True, reject subclasses.
            **data: Additional Element fields.
        """
        item_type = extract_types(item_type) if item_type else None

        if isinstance(items, Pile):
            data["items"] = items
        elif isinstance(items, dict):
            # Dict from deserialization - let field validator handle it
            data["items"] = items
        elif isinstance(items, list) and items and isinstance(items[0], dict):
            # List of dicts from deserialization - let field validator handle it
            data["items"] = items
        elif items is not None or item_type is not None or strict_type:
            # Normalize to list
            if isinstance(items, Element):
                items = cast("list[E]", [items])

            # Create Pile with items and type validation (item_type/strict_type are frozen)
            # Even if items=None, create Pile if item_type/strict_type specified
            data["items"] = Pile(
                items=items, item_type=item_type, strict_type=strict_type
            )

        # Handle progressions - let field validator convert dict/list to Pile
        if progressions is not None:
            data["progressions"] = progressions

        if name is not None:
            data["name"] = name

        super().__init__(**data)

    @field_validator("items", "progressions", mode="wrap")
    @classmethod
    def _validate_piles(cls, v: Any, handler: Any, info) -> Any:
        """Coerce Pile, dict, or list inputs to Pile."""
        if isinstance(v, Pile):
            return v
        if isinstance(v, dict):
            return Pile.from_dict(v)
        if isinstance(v, list):
            pile: Pile[Any] = Pile()
            for item in v:
                if isinstance(item, dict):
                    pile.add(Element.from_dict(item))
                else:
                    pile.add(item)
            return pile
        return handler(v)

    @model_validator(mode="after")
    def _validate_referential_integrity(self) -> Flow:
        """Validate all progression UUIDs exist in items pile."""
        item_ids = set(self.items.keys())

        for prog in self.progressions:
            missing_ids = set(list(prog)) - item_ids
            if missing_ids:
                raise NotFoundError(
                    f"Progression '{prog.name}' contains UUIDs not in items pile: {missing_ids}"
                )

        return self

    def model_post_init(self, __context: Any) -> None:
        """Rebuild _progression_names index from progressions."""
        super().model_post_init(__context)
        for progression in self.progressions:
            if progression.name:
                self._progression_names[progression.name] = progression.id

    def _check_item_exists(self, item_id: UUID) -> E:
        """Get item or raise NotFoundError with flow context."""
        try:
            return self.items[item_id]
        except NotFoundError as e:
            raise NotFoundError(
                f"Item {item_id} not found in flow",
                details=e.details,
                retryable=e.retryable,
                cause=e,
            ) from e

    def _check_progression_exists(self, progression_id: UUID) -> P:
        """Get progression or raise NotFoundError with flow context."""
        try:
            return self.progressions[progression_id]
        except NotFoundError as e:
            raise NotFoundError(
                f"Progression {progression_id} not found in flow",
                details=e.details,
                retryable=e.retryable,
                cause=e,
            ) from e

    # ==================== Progression Management ====================

    @synchronized
    def add_progression(self, progression: P) -> None:
        """Add progression with name uniqueness and referential integrity checks.

        Args:
            progression: Progression to add.

        Raises:
            ExistsError: If name already registered.
            NotFoundError: If progression contains UUIDs not in items.
        """
        if progression.name and progression.name in self._progression_names:
            raise ExistsError(
                f"Progression with name '{progression.name}' already exists. Names must be unique."
            )

        item_ids = set(self.items.keys())
        missing_ids = set(list(progression)) - item_ids
        if missing_ids:
            raise NotFoundError(
                f"Progression '{progression.name or progression.id}' contains UUIDs not in items pile: {missing_ids}"
            )

        self.progressions.add(progression)

        if progression.name:
            self._progression_names[progression.name] = progression.id

    @synchronized
    def remove_progression(self, progression_id: UUID | str | P) -> P:
        """Remove progression by UUID or name.

        Args:
            progression_id: UUID, name string, or Progression instance.

        Returns:
            Removed progression.

        Raises:
            NotFoundError: If progression not found.
        """
        name_to_delete: str | None
        if (
            isinstance(progression_id, str)
            and progression_id in self._progression_names
        ):
            uid = self._progression_names[progression_id]
            name_to_delete = progression_id
        else:
            uid = self._coerce_id(progression_id)
            prog = self._check_progression_exists(uid)
            name_to_delete = prog.name if prog.name in self._progression_names else None

        removed = self.progressions.remove(uid)

        if name_to_delete and name_to_delete in self._progression_names:
            del self._progression_names[name_to_delete]

        return removed

    @synchronized
    def get_progression(self, key: UUID | str | P) -> P:
        """Get progression by UUID or name.

        Args:
            key: UUID, name string, or Progression instance.

        Returns:
            Matching progression.

        Raises:
            KeyError: If not found.
        """
        if isinstance(key, str):
            if key in self._progression_names:
                uid = self._progression_names[key]
                return self.progressions[uid]

            try:
                uid = self._coerce_id(key)
                return self.progressions[uid]
            except (ValueError, TypeError) as e:
                raise KeyError(f"Progression '{key}' not found in flow") from e

        uid = key.id if isinstance(key, Progression) else key
        return self.progressions[uid]

    # ==================== Item Management ====================

    @synchronized
    def add_item(
        self,
        item: E,
        progressions: list[UUID | str | P] | UUID | str | P | None = None,
    ) -> None:
        """Add item to storage and optionally to progressions.

        Args:
            item: Element to add.
            progressions: Progression(s) to append item to (by instance, UUID, or name).

        Raises:
            ExistsError: If item UUID already in pile.
            KeyError: If any progression not found (no side effects on failure).
        """
        resolved_progs: list[P] = []
        if progressions is not None:
            if isinstance(progressions, (str, UUID, Progression)):
                progs = [progressions]
            else:
                progs = list(progressions)

            for prog in progs:
                if isinstance(prog, Progression):
                    resolved_progs.append(prog)
                else:
                    resolved_progs.append(self.get_progression(prog))

        self.items.add(item)

        for prog in resolved_progs:
            prog.append(item)

    @synchronized
    def remove_item(self, item_id: UUID | str | Element) -> E:
        """Remove item from storage and all progressions.

        Args:
            item_id: UUID, UUID string, or Element instance.

        Returns:
            Removed item.

        Raises:
            NotFoundError: If item not in pile.
        """
        uid = self._coerce_id(item_id)

        for progression in self.progressions:
            if uid in progression:
                progression.remove(uid)

        return self.items.remove(uid)

    @synchronized
    def clear(self) -> None:
        """Clear all items and progressions."""
        self.items.clear()
        self.progressions.clear()
        self._progression_names.clear()

    def __repr__(self) -> str:
        name_str = f", name='{self.name}'" if self.name else ""
        return f"Flow(items={len(self.items)}, progressions={len(self.progressions)}{name_str})"

    def to_dict(
        self,
        mode: Literal["python", "json", "db"] = "python",
        created_at_format: (
            Literal["datetime", "isoformat", "timestamp"] | UnsetType
        ) = Unset,
        meta_key: str | UnsetType = Unset,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Serialize Flow including nested Pile contents.

        Overrides Element.to_dict() to properly serialize items and progressions
        with their contents (not just pile metadata).
        """
        exclude = kwargs.pop("exclude", set())
        if isinstance(exclude, set):
            exclude = exclude | {"items", "progressions"}
        else:
            exclude = set(exclude) | {"items", "progressions"}

        data = super().to_dict(
            mode=mode,
            created_at_format=created_at_format,
            meta_key=meta_key,
            exclude=exclude,
            **kwargs,
        )

        data["items"] = self.items.to_dict(
            mode=mode, created_at_format=created_at_format, meta_key=meta_key
        )
        data["progressions"] = self.progressions.to_dict(
            mode=mode, created_at_format=created_at_format, meta_key=meta_key
        )

        return data
