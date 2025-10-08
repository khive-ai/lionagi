from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, ClassVar, TypeAlias

from pydantic import field_serializer, field_validator

from lionagi.ln.types import DataClass
from lionagi.models import HashableModel
from lionagi.protocols.types import IDType, Node


class MessageRole(str, Enum):
    SYSTEM = auto()
    USER = auto()
    ASSISTANT = auto()
    EVENT = auto()
    UNSET = auto()

SenderRecipient: TypeAlias = IDType | MessageRole | str
"""
A union type indicating that a sender or recipient could be:
- A lionagi IDType,
- A string-based role or ID,
- A specific enum role from `MessageRole`.
"""

@dataclass(slots=True)
class BaseMessageContent(DataClass):
    role: MessageRole | None = None
    sender: SenderRecipient | None = None
    recipient: SenderRecipient | None = None
    content: dict | None = None
    properties: dict | None = None





class Message(Node):

    content: BaseMessageContent | None = None
    _content_cls: ClassVar[type[BaseMessageContent]] = BaseMessageContent


    @field_serializer("content")
    def _serialize_content(self, value: Any) -> Any:







    @field_validator("content", mode="before")
    def _validate_content(cls, v):
        if v is None:
            return None
        if isinstance(v, dict):
            return cls._content_cls(**v)
        if isinstance(v, BaseMessageContent):
            return v
        raise TypeError(
            f"Invalid type for content: {type(v)}. "
            "Expected dict or BaseMessageContent instance."
        )
