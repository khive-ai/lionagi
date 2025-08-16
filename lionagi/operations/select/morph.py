from dataclasses import dataclass, field
from enum import Enum
from typing import Any, ClassVar

from lionagi.ln import Params
from lionagi.operations.ReAct.morph import OperateFields
from lionagi.utils import DataClass

from ..morph import Morphism, MorphMeta

__all__ = (
    "SelectFields",
    "SelectParams",
    "SelectContext",
    "SelectMorphism",
)


class SelectFields(OperateFields):
    choices: list[str] | type[Enum] | dict[str, Any]
    max_num_selections: int = 1
    branch_kwargs: dict[str, Any] | None = None
    return_branch: bool = False
    verbose: bool = False


@dataclass(slots=True, frozen=True, init=False)
class SelectParams(Params, SelectFields):
    pass


@dataclass(slots=True)
class SelectContext(DataClass, SelectFields):
    pass


_DEFAULT_SELECT_PARAMS = SelectParams()


@dataclass(slots=True, frozen=True)
class SelectMorphism(Morphism):
    ctx_cls: ClassVar[type[DataClass]] = SelectContext

    meta: MorphMeta = field(
        default_factory=lambda: MorphMeta(
            name="select",
            description="A morphism for handling selection operations.",
            version="1.0.0",
        )
    )
    params: SelectParams = _DEFAULT_SELECT_PARAMS
    ctx: SelectContext | None = None

    async def _apply(self, **kw):
        from .select import select

        return await select(self.branch, **kw)
