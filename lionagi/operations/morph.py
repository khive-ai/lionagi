from abc import abstractmethod
from dataclasses import dataclass
from typing import ClassVar, TypedDict

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
    ctx_cls: ClassVar[type[DataClass]]
    """The context class for this morphism."""

    meta: MorphMeta
    params: Params
    ctx: DataClass

    @property
    def name(self) -> str:
        return self.meta.get("name")

    @property
    def request(self) -> dict:
        _dict = self.params.to_dict()
        _dict.update(self.ctx.to_dict() if self.ctx else {})
        return _dict

    async def apply(self, branch: Branch, /) -> dict:
        return await self._apply(branch, **self.request)

    @abstractmethod
    async def _apply(self, branch: Branch, /, **kw):
        """override in subclass"""
