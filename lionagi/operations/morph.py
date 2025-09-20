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
    _request: dict | None

    def __init__(self, *, meta: MorphMeta, params: Params, ctx: DataClass):
        object.__setattr__(self, "meta", meta)
        object.__setattr__(self, "params", params)
        object.__setattr__(self, "ctx", ctx)
        object.__setattr__(self, "_request", None)

    @property
    def name(self) -> str:
        return self.meta.get("name")

    @property
    def request(self) -> dict:
        if self._request is not None:
            return self._request

        _dict = self.params.to_dict()
        _dict.update(self.ctx.to_dict() if self.ctx else {})

        self._request = _dict
        return _dict

    async def apply(self, branch: "Branch", /) -> dict:
        return await self._apply(branch, **self.request)

    @abstractmethod
    async def _apply(self, branch: "Branch", /, **kw):
        """override in subclass"""

    def __hash__(self):
        return
