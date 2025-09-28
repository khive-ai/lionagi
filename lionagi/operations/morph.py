from abc import abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, TypedDict

from lionagi.ln.types import DataClass, Params
from lionagi.protocols._concepts import Invariant

if TYPE_CHECKING:
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

    async def invoke(self, branch: "Branch", /) -> dict:
        return await self._invoke(branch, **self.request)

    @abstractmethod
    async def _invoke(self, branch: "Branch", /, **kw):
        """override in subclass"""

    async def pre(self, branch: "Branch", /, **kw) -> bool:
        return True

    async def post(self, branch: "Branch", /, result: dict) -> bool:
        return True
