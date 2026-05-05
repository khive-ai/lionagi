# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Operation: executable DAG node that delegates invocation through the session's Runner path."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from pydantic import Field, PrivateAttr

from lionagi.protocols.graph.node import Node
from lionagi.beta.session.context import RequestContext
from lionagi.protocols.generic.event import Event

if TYPE_CHECKING:
    from lionagi.beta.session.session import Branch, Session

__all__ = ("Operation",)


class Operation(Node, Event):
    """Executable DAG node that delegates invocation to the bound session's Runner path."""

    operation_type: str
    parameters: dict[str, Any] | Any = Field(
        default_factory=dict,
        description="Operation parameters (Params dataclass, dict, or model)",
    )
    control_type: Literal["halt", "abort", "retry", "skip", "route"] | str | None = (
        Field(
            default=None,
            description=(
                "Runner control semantic: halt/abort, retry, skip, or route. Custom "
                "types execute as control checkpoints but do not mutate the graph "
                "unless their morphism returns a supported action."
            ),
        )
    )
    control_policy: dict[str, Any] | None = Field(
        default=None,
        description="Type-specific config (e.g. quorum threshold, halt reason)",
    )
    morphism: Any | None = Field(
        default=None,
        exclude=True,
        description="Optional Runner-native morphism injected directly into OpGraph.",
    )

    _session: Any = PrivateAttr(default=None)
    _branch: Any = PrivateAttr(default=None)
    _verbose: bool = PrivateAttr(default=False)

    @property
    def is_control(self) -> bool:
        return self.control_type is not None

    @property
    def response(self) -> Any:
        return self.execution.response

    def bind(self, session: Session, branch: Branch) -> Operation:
        """Bind session and branch before direct invoke() calls outside Session.conduct()."""
        self._session = session
        self._branch = branch
        return self

    def _require_binding(self) -> tuple[Session, Branch]:
        if self._session is None or self._branch is None:
            raise RuntimeError(
                "Operation not bound to session/branch. "
                "Use operation.bind(session, branch) or session.conduct(...)"
            )
        return self._session, self._branch

    def make_context(
        self,
        session: Session,
        branch: Branch,
        *,
        verbose: bool | None = None,
        principal: Any | None = None,
    ) -> RequestContext:
        metadata: dict[str, Any] = {
            "_bound_session": session,
            "_bound_branch": branch,
            "_verbose": self._verbose if verbose is None else verbose,
        }
        if principal is not None:
            metadata["principal"] = principal

        return RequestContext(
            name=self.operation_type,
            session_id=session.id,
            branch=branch.name or str(branch.id),
            **metadata,
        )

    async def _invoke(self) -> Any:
        session, branch = self._require_binding()
        return await session._execute_operation_handler(self, branch)

    def __repr__(self) -> str:
        bound = "bound" if self._session is not None else "unbound"
        return f"Operation(type={self.operation_type}, status={self.execution.status.value}, {bound})"
