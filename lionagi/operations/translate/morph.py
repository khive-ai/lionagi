from dataclasses import dataclass, field
from typing import ClassVar, Literal

from lionagi.ln import Params
from lionagi.service.imodel import iModel
from lionagi.utils import DataClass

from ..morph import Morphism, MorphMeta


class TranslateFields:
    text: str
    technique: Literal["SynthLang"] = "SynthLang"
    technique_kwargs: dict = None
    compress: bool = False
    chat_model: iModel = None
    compress_model: iModel = None
    compression_ratio: float = 0.2
    compress_kwargs = None
    verbose: bool = True
    new_branch: bool = True


@dataclass(slots=True, frozen=True, init=False)
class TranslateParams(Params, TranslateFields):
    pass


@dataclass(slots=True)
class TranslateContext(DataClass, TranslateFields):
    pass


_DEFAULT_TRANSLATE_PARAMS = TranslateParams()


@dataclass(slots=True, frozen=True)
class TranslateMorphism(Morphism):
    ctx_cls: ClassVar[type[DataClass]] = TranslateContext

    meta: MorphMeta = field(
        default_factory=lambda: MorphMeta(
            name="translate",
            description="A morphism for handling translation operations.",
            version="1.0.0",
        )
    )
    params: TranslateParams = _DEFAULT_TRANSLATE_PARAMS
    ctx: TranslateContext | None = None

    async def _apply(self, **kw):
        from .translate import translate

        return await translate(self.branch, **kw)
