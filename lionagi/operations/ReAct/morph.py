from dataclasses import dataclass, field
from typing import ClassVar, Literal

from pydantic import BaseModel

from lionagi.ln import Params
from lionagi.service.imodel import iModel
from lionagi.utils import DataClass

from ..interpret.morph import IntepretFields
from ..morph import Morphism, MorphMeta
from ..operate.morph import OperateFields

__all__ = (
    "ReActFields",
    "ReActParams",
    "ReActContext",
    "ReActMorphism",
)


class AnalysisFields:
    max_extensions: int | None = 5
    """Number of times to extend the analysis."""

    extension_allowed: bool = True
    """If True, the analysis can be auto extended determined by analysis model."""

    analysis_model: iModel | None = None
    """Model used for analysis."""

    reasoning_effort: Literal["low", "medium", "high"] = None
    """The effort level of the reasoning."""

    return_analysis: bool = False
    """If True, the analysis will be returned in addition to the response."""

    continue_after_failed_response: bool = False
    """If True, the process will continue even if the response generation fails."""

    verbose_analysis: bool = False
    intermediate_response_options: list[BaseModel] | BaseModel = None
    """If provided, the analysis will provide these deliverables as intermediate responses."""
    intermediate_listable: bool = False
    """If True, the intermediate responses will be listable."""


class ReActFields(OperateFields, IntepretFields, AnalysisFields):
    display_as: Literal["json", "yaml"] = "yaml"
    verbose_length: int = None

    response_model: iModel | None = None
    """Model used for the final response generation."""
    response_kwargs: dict | None = None
    """kw used in final response generation"""


@dataclass(slots=True, frozen=True, init=False)
class ReActParams(Params, OperateFields):
    pass


@dataclass(slots=True)
class ReActContext(DataClass, OperateFields):
    pass


_DEFAULT_REACT_PARAMS = ReActParams()


@dataclass(slots=True, frozen=True)
class ReActMorphism(Morphism):
    ctx_cls: ClassVar[type[DataClass]] = ReActContext
    meta: MorphMeta = field(
        default_factory=lambda: MorphMeta(
            name="ReAct",
            description="A morphism for handling self expansive reason action agentic loops.",
            version="1.0.0",
        )
    )
    params: ReActParams = _DEFAULT_REACT_PARAMS
    ctx: ReActContext | None = None

    async def _apply(self, branch, **kw):
        from .ReAct import ReAct

        return await ReAct(branch, **kw)
