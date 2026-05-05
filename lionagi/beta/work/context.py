# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Request context for service operations."""

from __future__ import annotations

from collections.abc import Awaitable, Mapping
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from lionagi.beta.session.context import RequestContext

__all__ = ("CrudOperation", "CrudPattern", "QueryFn")


def validate_identifier(name: str, label: str = "identifier") -> str:
    """Validate a SQL identifier used by declarative CRUD patterns."""
    if not name.isidentifier():
        raise ValueError(f"Invalid {label}: {name!r}")
    return name


class CrudOperation(str, Enum):
    """CRUD operation types for declarative phrases."""

    READ = "read"
    INSERT = "insert"
    UPDATE = "update"
    SOFT_DELETE = "soft_delete"


_EMPTY_MAP: MappingProxyType = MappingProxyType({})


@dataclass(frozen=True, slots=True)
class CrudPattern:
    """Declarative CRUD pattern for auto-generating phrase handlers."""

    table: str
    operation: CrudOperation | str = CrudOperation.READ
    lookup: frozenset[str] = frozenset()
    filters: Mapping[str, Any] = None  # type: ignore[assignment]
    set_fields: Mapping[str, Any] = None  # type: ignore[assignment]
    defaults: Mapping[str, Any] = None  # type: ignore[assignment]

    def __post_init__(self):
        validate_identifier(self.table, "table")
        if isinstance(self.operation, str):
            object.__setattr__(self, "operation", CrudOperation(self.operation))
        if not isinstance(self.lookup, frozenset):
            object.__setattr__(self, "lookup", frozenset(self.lookup))
        # Immutable empty maps prevent accidental mutation across instances
        object.__setattr__(
            self,
            "filters",
            (
                _EMPTY_MAP
                if self.filters is None
                else MappingProxyType(dict(self.filters))
            ),
        )
        object.__setattr__(
            self,
            "set_fields",
            (
                _EMPTY_MAP
                if self.set_fields is None
                else MappingProxyType(dict(self.set_fields))
            ),
        )
        object.__setattr__(
            self,
            "defaults",
            (
                _EMPTY_MAP
                if self.defaults is None
                else MappingProxyType(dict(self.defaults))
            ),
        )


class QueryFn(Protocol):
    """Callable protocol bridging CrudPattern phrases to a DB backend."""

    def __call__(
        self,
        table: str,
        operation: str,
        where: dict[str, Any] | None,
        data: dict[str, Any] | None,
        ctx: RequestContext,
    ) -> Awaitable[dict[str, Any] | None]: ...
