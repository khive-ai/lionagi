import asyncio
import logging
from typing import Any, Literal
from uuid import UUID

from pydantic import Field
from typing_extensions import deprecated

from lionagi.ln import get_cancelled_exc_class
from lionagi.operations.morph import Morphism
from lionagi.protocols.types import ID, Event, EventStatus, IDType, Node

BranchOperations = Literal[
    "chat",
    "operate",
    "communicate",
    "parse",
    "ReAct",
    "select",
    "translate",
    "interpret",
]

logger = logging.getLogger("operation")


class Operation(Node, Event):
    operation: BranchOperations
    morph: Morphism = Field(exclude=True)
    """The morphism that defines the operation."""

    @deprecated("Use `morph.request` instead", version="1.0.0")
    @property
    def parameters(self) -> dict[str, Any]:
        return self.morph.request

    @property
    def branch_id(self) -> IDType | None:
        if a := self.metadata.get("branch_id"):
            return ID.get_id(a)

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
        return self.morph.request

    @property
    def response(self):
        """Get the response from the execution."""
        return self.execution.response if self.execution else None

    async def invoke(self):
        start = asyncio.get_event_loop().time()

        try:
            self.execution.status = EventStatus.PROCESSING
            response = await self.morph.apply()
            self.execution.response = response
            self.execution.status = EventStatus.COMPLETED

        except get_cancelled_exc_class() as e:
            self.execution.error = str(e)
            self.execution.status = EventStatus.ABORTED
            logger.warning(f"Operation aborted: {e}")
            raise e

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
