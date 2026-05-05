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
    """Declarative CRUD pattern for auto-generating phrase handlers.

    Attributes:
        table: Validated database table name (alphanumeric + underscores).
        operation: CRUD operation type (read, insert, update, soft_delete).
        lookup: Fields from options used in WHERE clause (for read/update/delete).
        filters: Static key-value pairs added to WHERE clause. Use for
            hardcoded filters like {"status": "active"}.
        set_fields: Explicit field mappings for update. Values can be:
            - Field name (str): copy from options
            - "ctx.{attr}": read from context (e.g., "ctx.now", "ctx.user_id")
            - Literal value: use directly
        defaults: Static default values for insert.

    The auto-handler resolves output fields in order:
        1. ctx metadata attribute (e.g., tenant_id -- only if explicitly set)
        2. options pass-through (if field in inputs)
        3. row column (direct from query result)
        4. result_parser (for computed fields)
    """

    table: str
    operation: CrudOperation | str = CrudOperation.READ
    lookup: frozenset[str] = frozenset()
    filters: Mapping[str, Any] = None  # type: ignore[assignment]
    set_fields: Mapping[str, Any] = None  # type: ignore[assignment]
    defaults: Mapping[str, Any] = None  # type: ignore[assignment]

    def __post_init__(self):
        # Validate table name against SQL injection
        validate_identifier(self.table, "table")
        # Normalize operation to enum
        if isinstance(self.operation, str):
            object.__setattr__(self, "operation", CrudOperation(self.operation))
        # Normalize lookup to frozenset
        if not isinstance(self.lookup, frozenset):
            object.__setattr__(self, "lookup", frozenset(self.lookup))
        # Normalize None mappings to immutable empty maps; freeze mutable dicts
        object.__setattr__(
            self,
            "filters",
            (_EMPTY_MAP if self.filters is None else MappingProxyType(dict(self.filters))),
        )
        object.__setattr__(
            self,
            "set_fields",
            (_EMPTY_MAP if self.set_fields is None else MappingProxyType(dict(self.set_fields))),
        )
        object.__setattr__(
            self,
            "defaults",
            (_EMPTY_MAP if self.defaults is None else MappingProxyType(dict(self.defaults))),
        )

class QueryFn(Protocol):
    """Protocol for CRUD query functions used by declarative phrases.

    The query_fn is the bridge between declarative CrudPattern phrases
    and the actual database backend. Implementations MUST use parameterized
    queries to prevent SQL injection.

    Args:
        table: Validated database table name (alphanumeric + underscores only).
        operation: CRUD operation enum value.
        where: WHERE clause as dict of column -> value. None for insert.
        data: Data dict for insert/update. None for select/delete.
        ctx: The RequestContext (for connection, tenant isolation, etc).

    Returns:
        Row as dict if found/affected, None otherwise.
    """

    def __call__(
        self,
        table: str,
        operation: str,
        where: dict[str, Any] | None,
        data: dict[str, Any] | None,
        ctx: RequestContext,
    ) -> Awaitable[dict[str, Any] | None]: ...
