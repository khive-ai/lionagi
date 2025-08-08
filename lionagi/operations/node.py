from __future__ import annotations

import logging
from typing import Any

from anyio import current_time, get_cancelled_exc_class
from pydantic import BaseModel, Field

from lionagi.protocols.types import Event, EventStatus, Node
from lionagi.session.branch import Branch

from .morphism import Morphism, MorphismContext

logger = logging.getLogger("operation")


class Operation(Node, Event):
    morphism: Morphism = Field(..., exclude=True)
    branch: Branch | None = Field(..., exclude=True)

    @property
    def name(self) -> str:
        return self.morphism.meta["name"] if self.morphism else "Operation"

    @classmethod
    def create(
        cls,
        morphism: type[Morphism],
        branch: Branch,
        params: dict | BaseModel,
        stream_morphism: bool = False,
    ):
        ctx = MorphismContext(params=params, stream_morphism=stream_morphism)
        return cls(
            morphism=morphism(ctx=ctx),
            branch=branch,
            metadata={
                "morphism_meta": morphism.meta,
                "branch_id": str(branch.id),
            },
        )

    @property
    def request(self) -> dict:
        return self.morphism.ctx.to_dict()

    async def invoke(self):
        """Invoke the operation asynchronously."""
        if not self.branch:
            raise RuntimeError("Branch is not set for the operation.")

        meth = (
            self._consume_stream
            if self.morphism.ctx.stream_morphism
            else self.morphism.apply
        )
        start = current_time()

        try:
            self.execution.status = EventStatus.PROCESSING
            response = await meth(self.branch)
            self.execution.response = response
            self.execution.status = EventStatus.COMPLETED

        except get_cancelled_exc_class() as e:
            self.execution.error = "Operation cancelled"
            self.execution.status = EventStatus.CANCELLED
            raise

        except Exception as e:
            self.execution.error = str(e)
            self.execution.status = EventStatus.FAILED
            logger.error(f"Operation failed: {e}")

        finally:
            self.execution.duration = current_time() - start

    async def _consume_stream(self, _branch, /) -> list[Any]:
        """Consume the stream and return a list of results, will list aggregated result"""
        responses = []
        try:
            self.streaming = True
            async for i in self.morphism.stream(self.branch):
                responses.append(i)
        finally:
            self.streaming = False
            return responses