from dataclasses import dataclass, field
from typing import ClassVar

from lionagi.ln import Params
from lionagi.service import iModel
from lionagi.utils import DataClass

from ..morph import Morphism, MorphMeta

__all__ = (
    "IntepretFields",
    "InterpretParams",
    "IntepretContext",
    "InterpretMorphism",
)


class IntepretFields:
    interpret: bool = False
    """If True, the prompt will be re-interpreted by interpret model before sending to analysis model"""

    interpret_domain: str | None = None
    """Business domain of interest"""

    interpret_style: str | None = None
    """describes the style of the interpretation"""

    interpret_sample: str | None = None
    """Writing sample"""

    interpret_model: iModel | None = None
    """Model used for interpretation."""

    interpret_kwargs: dict | None = None
    """additional kwargs for interpret model"""


@dataclass(slots=True, frozen=True, init=False)
class InterpretParams(Params, IntepretFields):
    pass


@dataclass(slots=True)
class IntepretContext(DataClass, IntepretFields):
    pass


_DEFAULT_INTEPRET_PARAMS = InterpretParams()


@dataclass(slots=True, frozen=True)
class IntepretMorphism(Morphism):
    ctx_cls: ClassVar[type[DataClass]] = IntepretContext

    meta: MorphMeta = field(
        default_factory=lambda: MorphMeta(
            name="select",
            description="A morphism for handling interpretation operations.",
            version="1.0.0",
        )
    )
    params: InterpretParams = _DEFAULT_INTEPRET_PARAMS
    ctx: IntepretContext | None = None

    async def _apply(self, **kw):
        from .interpret import interpret

        return await interpret(self.branch, **kw)
