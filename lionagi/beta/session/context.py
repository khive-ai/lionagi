from __future__ import annotations

from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

from lionagi.ln.types import DataClass

if TYPE_CHECKING:
    from lionagi.beta.resource.service import Service
    from lionagi.beta.session.session import Branch, Session

__all__ = ("RequestContext",)


@dataclass(slots=True)
class RequestContext(DataClass):
    name: str
    id: UUID = field(default_factory=uuid4)
    session_id: UUID | None = None
    branch: UUID | str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    conn: Any | None = None
    query_fn: Any | None = None
    now: datetime | None = None
    service: UUID | str | None = None

    def __init__(
        self,
        name: str,
        session_id: UUID | None = None,
        branch: UUID | str | None = None,
        id: UUID | None = None,
        conn: Any | None = None,
        query_fn: Any | None = None,
        now: datetime | None = None,
        service: UUID | str | None = None,
        **kwargs: Any,
    ):
        self.name = name
        self.id = id or uuid4()
        self.session_id = session_id
        self.branch = branch
        self.conn = conn
        self.query_fn = query_fn
        self.now = now
        self.metadata = kwargs
        self.service = service

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        metadata = object.__getattribute__(self, "metadata")
        try:
            return metadata[name]
        except KeyError:
            raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'") from None

    async def get_session(self) -> Session | None:
        if "_bound_session" in self.metadata:
            return self.metadata["_bound_session"]
        if self.session_id is None:
            return None
        from lionagi.beta.session.registry import get_session

        return await get_session(self.session_id)

    async def get_branch(self) -> Branch | None:
        if "_bound_branch" in self.metadata:
            return self.metadata["_bound_branch"]
        session = await self.get_session()
        if session is None or self.branch is None:
            return None
        return session.get_branch(self.branch)

    async def get_service(self) -> Service | None:
        if "_bound_service" in self.metadata:
            return self.metadata["_bound_service"]
        service = self.service
        if service is None and ":" in self.name:
            service = self.name.split(":", 1)[0]
        elif service is None and "." in self.name:
            service = self.name.split(".", 1)[0]
        if service is None:
            return None
        from lionagi.beta.resource.service import get_service

        return await get_service(service)

    async def conduct(self, operation_type: str, params: Any | None = None) -> Any:
        session = await self.get_session()
        branch = await self.get_branch()
        if session is None or branch is None:
            raise RuntimeError("RequestContext is not bound to a session and branch")
        op = await session.conduct(
            operation_type,
            branch=branch,
            params=params,
            verbose=bool(self.metadata.get("_verbose", False)),
        )
        if op.execution.error is not None:
            raise op.execution.error
        return op.response

    async def stream_conduct(
        self,
        operation_type: str,
        params: Any | None = None,
    ) -> AsyncGenerator[Any, None]:
        session = await self.get_session()
        branch = await self.get_branch()
        if session is None or branch is None:
            raise RuntimeError("RequestContext is not bound to a session and branch")
        async for item in session.stream_conduct(
            operation_type,
            branch=branch,
            params=params,
            verbose=bool(self.metadata.get("_verbose", False)),
        ):
            yield item
