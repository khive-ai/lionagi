import logging
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field, PrivateAttr

from lionagi.protocols.types import ID, Event, Node

BranchOperations = Literal[
    "chat",
    "operate",
    "communicate",
    "parse",
    "ReAct",
    "select",
    "interpret",
    "act",
    "ReActStream",
]

logger = logging.getLogger("operation")


class Operation(Node, Event):
    """Operation node for flow graphs.

    Does NOT override invoke(). The state machine (idempotency, status
    transitions, error handling) lives in Event.invoke(). Subclasses
    only implement _invoke().

    Set ``_branch`` before calling ``invoke()``.

    Control operations (``control_type`` is set) condition flow
    progression rather than executing LLM work. The executor evaluates
    them based on ``control_type`` and ``control_policy`` and applies
    the resulting ``ControlDecision`` to the DAG.
    """

    operation: BranchOperations | str
    parameters: dict[str, Any] | BaseModel = Field(
        default_factory=dict,
        description="Parameters for the operation",
        exclude=True,
    )
    control_type: Literal["iterate", "quorum", "halt"] | str | None = Field(
        default=None,
        description="Control semantic: iterate (LLM-backed re-plan), quorum (completion threshold), halt (flow pause). Custom types resolved via executor.register_control_handler().",
    )
    control_policy: dict[str, Any] | None = Field(
        default=None,
        description="Type-specific config (e.g. gate artifacts, quorum threshold)",
    )
    _branch: Any = PrivateAttr(default=None)

    @property
    def is_control(self) -> bool:
        return self.control_type is not None

    @property
    def branch_id(self) -> UUID | None:
        if a := self.metadata.get("branch_id"):
            return ID.get_id(a)

    @branch_id.setter
    def branch_id(self, value: str | UUID | None):
        if value is None:
            self.metadata.pop("branch_id", None)
        else:
            self.metadata["branch_id"] = str(value)

    @property
    def graph_id(self) -> str | None:
        if a := self.metadata.get("graph_id"):
            return ID.get_id(a)

    @graph_id.setter
    def graph_id(self, value: str | UUID | None):
        if value is None:
            self.metadata.pop("graph_id", None)
        else:
            self.metadata["graph_id"] = str(value)

    @property
    def request(self) -> dict:
        # Convert parameters to dict if it's a BaseModel
        params = self.parameters
        if hasattr(params, "model_dump"):
            params = params.model_dump()
        elif hasattr(params, "dict"):
            params = params.dict()

        return params if isinstance(params, dict) else {}

    @property
    def response(self):
        """Get the response from the execution."""
        return self.execution.response if self.execution else None

    async def _invoke(self):
        """Execute the operation on the pre-set branch.

        Called by Event.invoke() which handles all state transitions.
        """
        branch = self._branch
        if branch is None:
            raise RuntimeError(
                "Operation._branch must be set before invoke(). "
                "Use operation._branch = branch before calling invoke()."
            )

        meth = branch.get_operation(self.operation)
        if meth is None:
            raise ValueError(f"Unsupported operation type: {self.operation}")

        self.branch_id = branch.id

        if self.operation == "ReActStream":
            res = []
            async for i in meth(**self.request):
                res.append(i)
            return res
        return await meth(**self.request)


def create_operation(
    operation: BranchOperations | str,
    parameters: dict[str, Any] | BaseModel = None,
    **kwargs,
):
    """Create an Operation node."""
    return Operation(operation=operation, parameters=parameters, **kwargs)
