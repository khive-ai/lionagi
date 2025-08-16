from abc import abstractmethod
from dataclasses import dataclass
from typing import TypedDict

from lionagi.ln import DataClass, Params
from lionagi.protocols.types import Invariant
from lionagi.session.branch import Branch

__all__ = (
    "Morphism",
    "MorphMeta",
)


class MorphMeta(TypedDict, total=False):
    name: str
    description: str
    version: str


@dataclass(slots=True, frozen=True, init=False)
class Morphism(Invariant):

    meta: MorphMeta
    params: Params
    branch: Branch
    ctx: DataClass | None = None

    async def apply(self, **kw):
        _dict = self.params.to_dict()
        _dict.update(self.ctx.to_dict() if self.ctx else {})
        _dict.update(kw)
        return await self._apply(**_dict)

    @abstractmethod
    async def _apply(self, /, **kw):
        """override in subclass"""
