from dataclasses import dataclass, field
from typing import ClassVar

from pydantic import BaseModel

from lionagi.ln import Params
from lionagi.service import iModel
from lionagi.utils import DataClass

from ..chat.morph import BaseChatFields
from ..morph import Morphism, MorphMeta

__all__ = (
    "CommunicateFields",
    "CommunicateParams",
    "CommunicateContext",
    "CommunicateMorphism",
)


class CommunicateFields(BaseChatFields):
    response_format: type[BaseModel]
    """The format of the response expected from the chat operation."""

    request_model: type[BaseModel]
    """deprecated: Use response_format instead."""
    operative_model: type[BaseModel]
    """deprecated: Use response_format instead."""

    chat_model: iModel
    parse_model: iModel
    skip_validation: bool = False
    num_parse_retries: int = 3
    clear_messages: bool = False


@dataclass(slots=True, frozen=True, init=False)
class CommunicateParams(Params, BaseChatFields):
    pass


@dataclass(slots=True)
class CommunicateContext(DataClass, BaseChatFields):
    pass


_DEFAULT_COMMUNICATE_PARAMS = CommunicateParams()


@dataclass(slots=True, frozen=True)
class CommunicateMorphism(Morphism):
    ctx_cls: ClassVar[type[DataClass]] = CommunicateContext

    meta: MorphMeta = field(
        default_factory=lambda: MorphMeta(
            name="communicate",
            description="A morphism for handling communication operations.",
            version="1.0.0",
        )
    )
    params: CommunicateParams = _DEFAULT_COMMUNICATE_PARAMS
    ctx: CommunicateContext | None = None

    async def _apply(self, branch, **kw):
        from .communicate import communicate

        return await communicate(branch, **kw)
