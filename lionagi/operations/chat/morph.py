from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, JsonValue

from lionagi.ln import Params
from lionagi.protocols.types import Progression, SenderRecipient
from lionagi.service import iModel
from lionagi.utils import DataClass

from ..morph import Morphism, MorphMeta

__all__ = (
    "BaseChatFields",
    "ChatParams",
    "ChatContext",
    "ChatMorphism",
)


class BaseChatFields:
    instruction: JsonValue
    guidance: JsonValue
    context: JsonValue
    sender: SenderRecipient | None = None
    recipient: SenderRecipient | None = None
    request_fields: dict | list
    response_format: type[BaseModel]
    progression: list | Progression
    imodel: iModel
    tool_schemas: list[dict]
    images: list
    image_detail: Literal["low", "high", "auto"] | None = None
    plain_content: str | None = None
    return_ins_res_message: bool = False
    include_token_usage_to_model: bool = False


@dataclass(slots=True, frozen=True, init=False)
class ChatParams(Params, BaseChatFields):
    pass


@dataclass(slots=True)
class ChatContext(DataClass, BaseChatFields):
    pass


_DEFAULT_CHAT_PARAMS = ChatParams()


@dataclass(slots=True, frozen=True)
class ChatMorphism(Morphism):

    meta: MorphMeta = field(
        default_factory=lambda: MorphMeta(
            name="chat",
            description="A morphism for handling chat operations.",
            version="1.0.0",
        )
    )
    params: ChatParams = _DEFAULT_CHAT_PARAMS
    ctx: ChatContext | None = None

    async def _apply(self, **kw):
        from .chat import chat

        return await chat(self.branch, **kw)
