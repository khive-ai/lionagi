# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Core service interfaces - Service protocol, CallContext, and base types."""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Mapping
from typing import Any, Generic, Protocol, TypeVar
from types import MappingProxyType
from uuid import UUID, uuid4

import anyio
import msgspec

# Type variables for generic service interface
Req = TypeVar("Req")
Res = TypeVar("Res")
Chunk = TypeVar("Chunk")


class CallContext(msgspec.Struct, kw_only=True, frozen=True):
    """Lifeline for time & identity - carries call metadata through the service pipeline.

    This replaces ad-hoc kwargs; middleware can rely on it; transport reads the deadline.
    All per-call data flows through CallContext for deterministic behavior.

    Using msgspec for v1 performance - orders of magnitude faster serialization.
    """

    call_id: UUID
    branch_id: UUID
    deadline_s: float | None = None  # monotonic absolute deadline
    capabilities: frozenset[str] = msgspec.field(default_factory=frozenset)  # for policy gate
    attrs: Mapping[str, Any] = msgspec.field(
        default_factory=lambda: MappingProxyType({})
    )  # user-defined (trace/span, request_id, ...)

    @staticmethod
    def _current_time() -> float:
        """Get current time, preferring anyio.current_time() if in async context."""
        try:
            # Try anyio first for consistency with async operations
            return anyio.current_time()
        except RuntimeError:
            # Fall back to monotonic time for sync contexts
            return time.monotonic()

    @classmethod
    def new(
        cls,
        branch_id: UUID,
        *,
        deadline_s: float | None = None,
        capabilities: set[str] | frozenset[str] | None = None,
        attrs: Mapping[str, Any] | None = None,
        **extra_attrs: Any,
    ) -> CallContext:
        """Create new call context with generated call_id."""
        # Handle both explicit attrs dict and **extra_attrs
        final_attrs = dict(extra_attrs)
        if attrs is not None:
            final_attrs.update(attrs)

        return cls(
            call_id=uuid4(),
            branch_id=branch_id,
            deadline_s=deadline_s,
            capabilities=frozenset(capabilities or set()),
            attrs=MappingProxyType(final_attrs),
        )

    @classmethod
    def with_timeout(
        cls,
        branch_id: UUID,
        timeout_s: float,
        *,
        capabilities: set[str] | frozenset[str] | None = None,
        attrs: Mapping[str, Any] | None = None,
        **extra_attrs: Any,
    ) -> CallContext:
        """Create call context with relative timeout."""
        deadline = cls._current_time() + timeout_s
        return cls.new(
            branch_id=branch_id,
            deadline_s=deadline,
            capabilities=capabilities,
            attrs=attrs,
            **extra_attrs,
        )

    @property
    def remaining_time(self) -> float | None:
        """Get remaining time until deadline, or None if no deadline set."""
        if self.deadline_s is None:
            return None
        return max(0.0, self.deadline_s - self._current_time())

    @property
    def is_expired(self) -> bool:
        """Check if the deadline has passed."""
        if self.deadline_s is None:
            return False
        return self._current_time() >= self.deadline_s


class Service(Generic[Req, Res, Chunk], Protocol):
    """Core service interface - the main "port" for all service implementations.

    Keep this tiny. Everything else composes around it. Services implement
    both call() for single responses and stream() for streaming responses.
    """

    name: str

    async def call(self, req: Req, *, ctx: CallContext) -> Res:
        """Execute single call and return response."""
        ...

    async def stream(self, req: Req, *, ctx: CallContext) -> AsyncIterator[Chunk]:
        """Execute streaming call and yield response chunks."""
        ...
