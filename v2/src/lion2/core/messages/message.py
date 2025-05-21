# Copyright (c) 2023 - 2025, HaiyangLi <quantocean.li at gmail dot com>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any
from uuid import UUID

from pydantic import field_serializer, field_validator
from pydapter.protocols import Embedable, Temporal

from ..types import MessageContent, MessageRole
from .utils import MessageContentType, create_message_content


class Message(Temporal, Embedable):

    role: MessageRole
    "The role of the message sender (e.g., 'user', 'assistant', 'system', 'action')."

    content: MessageContentType
    "Standardized content according to message role."

    sender: str | None = None
    "Identifier for the sender of the message. typically UUID or a name"

    recipient: str | None = None
    "Identifier for the recipient of the message, if any."

    @property
    def rendered(self) -> Any:
        return self.content.rendered

    @property
    def chat_msg(self) -> dict[str, Any]:
        return {
            "role": self.role.value,
            "content": self.content.rendered,
        }

    @field_validator("content", mode="before")
    def _validate_message_content(cls, data: dict | MessageContent):
        return create_message_content(data)

    @field_serializer("content")
    def _serialize_message_content(self, v: MessageContent) -> dict[str, Any]:
        return v.model_dump_json()

    @field_validator("sender", "recipient", mode="before")
    def _validate_sender_recipient(cls, v):
        if v is None:
            return None
        if isinstance(v, str):
            return v
        elif isinstance(v, UUID):
            return str(v)
        raise TypeError(
            f"Sender and recipient must be a string or UUID, got {type(v)}"
        )
