# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any
from uuid import UUID

from lionagi.beta.core.base.pile import Pile
from lionagi.ln.types._sentinel import Undefined, UndefinedType, is_sentinel

from .imodel import iModel

__all__ = ("ResourceRegistry",)


class ResourceRegistry:
    """O(1) name-indexed store for iModel instances; duplicate names raise unless update=True."""

    def __init__(self):
        from .imodel import iModel

        self._pile: Pile[iModel] = Pile(item_type=iModel)
        self._name_index: dict[str, UUID] = {}

    def register(self, model: iModel, update: bool = False) -> UUID:
        if model.name in self._name_index:
            if not update:
                raise ValueError(f"Resource '{model.name}' already registered")
            old_uid = self._name_index[model.name]
            self._pile.remove(old_uid)

        self._pile.add(model)
        self._name_index[model.name] = model.id

        return model.id

    def unregister(self, name: str) -> iModel:
        if name not in self._name_index:
            raise KeyError(f"Resource '{name}' not found")

        uid = self._name_index.pop(name)
        return self._pile.remove(uid)

    def get(self, name: str | UUID | iModel, default: Any | UndefinedType = Undefined) -> iModel:
        if isinstance(name, UUID):
            return self._pile[name]
        if isinstance(name, iModel):
            return name
        if name not in self._name_index:
            if not is_sentinel(default):
                return default
            raise KeyError(f"Resource '{name}' not found")

        uid = self._name_index[name]
        return self._pile[uid]

    def has(self, name: str) -> bool:
        return name in self._name_index

    def list_names(self) -> list[str]:
        return list(self._name_index.keys())

    def list_by_tag(self, tag: str) -> Pile[iModel]:
        return self._pile[lambda m: tag in m.tags]

    def count(self) -> int:
        return len(self._pile)

    def clear(self) -> None:
        self._pile.clear()
        self._name_index.clear()

    def __len__(self) -> int:
        return len(self._pile)

    def __contains__(self, name: str) -> bool:
        return name in self._name_index

    def __repr__(self) -> str:
        return f"ResourceRegistry(count={len(self)})"
