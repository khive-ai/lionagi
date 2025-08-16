from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal

from pydantic import BaseModel

from lionagi.ln import Params
from lionagi.utils import DataClass

from ..morph import Morphism, MorphMeta

__all__ = (
    "ParseFields",
    "ParseParams",
    "ParseContext",
    "ParseMorphism",
)


class ParseFields:
    text: str
    """The text to be parsed."""

    handle_validation: Literal["raise", "return_value", "return_none"] = (
        "return_value"
    )

    max_retries: int = 3
    """Maximum number of retries for parsing."""

    response_format: type[BaseModel] = None
    """The format of the response expected from the parse operation."""

    similarity_algo = "jaro_winkler"
    similarity_threshold: float = 0.85
    fuzzy_match: bool = True
    handle_unmatched: Literal["ignore", "raise", "remove", "fill", "force"] = (
        "force"
    )
    fill_value: Any = None
    fill_mapping: dict[str, Any] | None = None
    strict: bool = False
    suppress_conversion_errors: bool = False


@dataclass(slots=True, frozen=True, init=False)
class ParseParams(Params, ParseFields):
    pass


@dataclass(slots=True)
class ParseContext(DataClass, ParseFields):
    pass


_DEFAULT_PARSE_PARAMS = ParseParams()


@dataclass(slots=True, frozen=True)
class ParseMorphism(Morphism):
    ctx_cls: ClassVar[type[DataClass]] = ParseContext

    meta: MorphMeta = field(
        default_factory=lambda: MorphMeta(
            name="parse",
            description="A morphism for handling structured text parsing operations.",
            version="1.0.0",
        )
    )
    params: ParseParams = _DEFAULT_PARSE_PARAMS
    ctx: ParseContext | None = None

    async def _apply(self, branch, **kw):
        from .parse import parse

        return await parse(branch, **kw)
