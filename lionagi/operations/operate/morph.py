from dataclasses import dataclass, field
from typing import ClassVar, Literal

from lionagi.fields import Instruct
from lionagi.ln import Params
from lionagi.models import FieldModel, ModelParams
from lionagi.protocols.types import Operative, ToolRef
from lionagi.utils import DataClass

from ..communicate.morph import CommunicateFields
from ..morph import Morphism, MorphMeta

__all__ = (
    "OperateFields",
    "OperateParams",
    "OperateContext",
    "OperateMorphism",
)


class OperateFields(CommunicateFields):
    instruct: Instruct | dict = None
    invoke_actions: bool = True
    tools: ToolRef = None

    operative: Operative = None
    """deprecated"""

    return_operative: bool = False
    """deprecated. If True, the operative will be returned instead of the response."""

    actions: bool = False
    """will be deprecated, and replaced by `capabilities` in Operative."""

    reason: bool = False
    """will be deprecated, and replaced by `capabilities` in Operative."""

    action_kwargs: dict = None
    action_strategy: Literal["sequential", "concurrent"] = "concurrent"
    verbose_action: bool = False
    field_models: list[FieldModel] = None
    exclude_fields: list | dict | None = None
    request_params: ModelParams = None
    request_param_kwargs: dict = None
    response_params: ModelParams = None
    response_param_kwargs: dict = None
    handle_validation: Literal["raise", "return_value", "return_none"] = (
        "return_value"
    )


@dataclass(slots=True, frozen=True, init=False)
class OperateParams(Params, OperateFields):
    pass


@dataclass(slots=True)
class OperateContext(DataClass, OperateFields):
    pass


_DEFAULT_OPERATE_PARAMS = OperateParams()


@dataclass(slots=True, frozen=True)
class OperateMorphism(Morphism):
    ctx_cls: ClassVar[type[DataClass]] = OperateContext

    meta: MorphMeta = field(
        default_factory=lambda: MorphMeta(
            name="operate",
            description="A morphism for handling common composable operations.",
            version="1.0.0",
        )
    )
    params: OperateParams = _DEFAULT_OPERATE_PARAMS
    ctx: OperateContext | None = None

    async def _apply(self, branch, **kw):
        from .operate import operate

        return await operate(branch, **kw)
