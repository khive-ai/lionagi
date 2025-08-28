# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Core service interfaces - Service protocol, CallContext, and base types."""

from __future__ import annotations

from typing import Any, AsyncIterator, Generic, Mapping, Protocol, TypeVar
from uuid import UUID, uuid4

import anyio
import msgspec

# Type variables for generic service interface
Req = TypeVar("Req")
Res = TypeVar("Res")
Chunk = TypeVar("Chunk")


class CallContext(msgspec.Struct, kw_only=True):
    """Lifeline for time & identity - carries call metadata through the service pipeline.

    This replaces ad-hoc kwargs; middleware can rely on it; transport reads the deadline.
    All per-call data flows through CallContext for deterministic behavior.
    
    Using msgspec for v1 performance - orders of magnitude faster serialization.
    """

    call_id: UUID
    branch_id: UUID
    deadline_s: float | None = None  # monotonic absolute deadline
    capabilities: set[str] = msgspec.field(default_factory=set)  # for policy gate
    attrs: Mapping[str, Any] = msgspec.field(default_factory=dict)  # user-defined (trace/span, request_id, ...)

    @classmethod
    def new(
        cls,
        branch_id: UUID,
        *,
        deadline_s: float | None = None,
        capabilities: set[str] | None = None,
        **attrs: Any,
    ) -> CallContext:
        """Create new call context with generated call_id."""
        return cls(
            call_id=uuid4(),
            branch_id=branch_id,
            deadline_s=deadline_s,
            capabilities=capabilities or set(),
            attrs=attrs,
        )

    @classmethod
    def with_timeout(
        cls,
        branch_id: UUID,
        timeout_s: float,
        *,
        capabilities: set[str] | None = None,
        **attrs: Any,
    ) -> CallContext:
        """Create call context with relative timeout."""
        deadline = anyio.current_time() + timeout_s
        return cls.new(branch_id=branch_id, deadline_s=deadline, capabilities=capabilities, **attrs)

    @property
    def remaining_time(self) -> float | None:
        """Get remaining time until deadline, or None if no deadline set."""
        if self.deadline_s is None:
            return None
        return max(0.0, self.deadline_s - anyio.current_time())

    @property
    def is_expired(self) -> bool:
        """Check if the deadline has passed."""
        if self.deadline_s is None:
            return False
        return anyio.current_time() >= self.deadline_s


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

