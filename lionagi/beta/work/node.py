# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Operation: executable graph node bridging session operations to Runner.

Operation.invoke() delegates to the bound session, which wraps the registered
handler as a Morphism and executes it through Runner. Handlers still receive
the canonical (params, ctx) call shape.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from pydantic import Field, PrivateAttr

from lionagi.beta.core.base.event import Event
from lionagi.beta.core.base.node import Node
from lionagi.beta.session.context import RequestContext

if TYPE_CHECKING:
    from lionagi.beta.session.session import Branch, Session

__all__ = ("Operation",)


class Operation(Node, Event):
    """Executable operation node.

    Bridges session.conduct() to Runner by:
    1. Storing bound session/branch references
    2. Letting the session compile the operation declaration to an OpGraph
    3. Preserving Event.invoke() lifecycle state around Runner execution

    The result is stored in execution.response (via Event.invoke).
    """

    operation_type: str
    parameters: dict[str, Any] | Any = Field(
        default_factory=dict,
        description="Operation parameters (Params dataclass, dict, or model)",
    )
    control_type: Literal["halt", "abort", "retry", "skip", "route"] | str | None = Field(
        default=None,
        description=(
            "Runner control semantic: halt/abort, retry, skip, or route. Custom "
            "types execute as control checkpoints but do not mutate the graph "
            "unless their morphism returns a supported action."
        ),
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
        """True if this operation is a control node (conditions flow progression)."""
        return self.control_type is not None

    @property
    def response(self) -> Any:
        return self.execution.response

    def bind(self, session: Session, branch: Branch) -> Operation:
        """Bind session and branch for execution.

        Must be called before invoke() if not using Session.conduct().

        Args:
            session: Session with operations registry and services.
            branch: Branch for message context.

        Returns:
            Self for chaining.
        """
        self._session = session
        self._branch = branch
        return self

    def _require_binding(self) -> tuple[Session, Branch]:
        """Return bound (session, branch) tuple or raise RuntimeError if unbound."""
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
        """Build the canonical RequestContext for this operation."""
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
        """Execute handler via the bound session's Runner path.

        The session compiles the operation declaration into a single-node
        OpGraph and calls the handler through MorphismAdapter.from_operation().

        Returns:
            Handler result (stored in execution.response).

        Raises:
            RuntimeError: If not bound.
            KeyError: If operation_type not registered.
        """
        session, branch = self._require_binding()
        return await session._execute_operation_handler(self, branch)

    def __repr__(self) -> str:
        bound = "bound" if self._session is not None else "unbound"
        return (
            f"Operation(type={self.operation_type}, status={self.execution.status.value}, {bound})"
        )
