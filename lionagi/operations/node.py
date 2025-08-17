import asyncio
import logging
from typing import Any, Literal
from uuid import UUID

from anyio import get_cancelled_exc_class
from pydantic import BaseModel, Field

from lionagi.protocols.types import ID, Event, EventStatus, IDType, Node
from lionagi.session.branch import Branch

BranchOperations = Literal[
    "chat",
    "operate",
    "communicate",
    "parse",
    "ReAct",
    "select",
    "translate",
    "interpret",
    "act",
    "ReActStream",
    "instruct",
]

logger = logging.getLogger("operation")


class Operation(Node, Event):
    operation: BranchOperations | str
    parameters: dict[str, Any] | BaseModel = Field(
        default_factory=dict,
        description="Parameters for the operation",
        exclude=True,
    )

    @property
    def branch_id(self) -> IDType | None:
        if a := self.metadata.get("branch_id"):
            return ID.get_id(a)

    @branch_id.setter
    def branch_id(self, value: str | UUID | IDType | None):
        if value is None:
            self.metadata.pop("branch_id", None)
        else:
            self.metadata["branch_id"] = str(value)

    @property
    def graph_id(self) -> str | None:
        if a := self.metadata.get("graph_id"):
            return ID.get_id(a)

    @graph_id.setter
    def graph_id(self, value: str | UUID | IDType | None):
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

    async def invoke(self, branch: Branch):
        meth = branch.get_operation(self.operation)
        if meth is None:
            raise ValueError(f"Unsupported operation type: {self.operation}")

        start = asyncio.get_event_loop().time()
        try:
            self.execution.status = EventStatus.PROCESSING
            self.branch_id = branch.id
            response = await self._invoke(meth)

            self.execution.response = response
            self.execution.status = EventStatus.COMPLETED

        except get_cancelled_exc_class():
            self.execution.error = "Operation cancelled"
            self.execution.status = EventStatus.CANCELLED
            raise

        except Exception as e:
            self.execution.error = str(e)
            self.execution.status = EventStatus.FAILED
            logger.error(f"Operation failed: {e}")

        finally:
            self.execution.duration = asyncio.get_event_loop().time() - start

    async def _invoke(self, meth):
        if self.operation == "ReActStream":
            res = []
            async for i in meth(**self.request):
                res.append(i)
            return res
        return await meth(**self.request)
