# Copyright (c) 2023 - 2025, HaiyangLi <quantocean.li at gmail dot com>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any
from uuid import UUID

from pydantic import (
    field_serializer,
    field_validator,
    model_validator,
)
from pydapter.core import Adaptable
from pydapter.protocols import Embedable, Temporal
from pydapter.protocols.utils import validate_uuid
from typing_extensions import Self

from .action_request_response import (
    ActionRequestContent,
    ActionResponseContent,
)
from .assistant_response import AssistantResponseContent
from .instruction import InstructionContent
from .system import SystemMessageContent

MessageContentType = (
    SystemMessageContent
    | InstructionContent
    | ActionRequestContent
    | ActionResponseContent
    | AssistantResponseContent
)


class Message(Temporal, Embedable, Adaptable):
    message_content: MessageContentType
    sender: UUID | None = None
    recipient: UUID | None = None

    @property
    def role(self) -> str:
        """
        Returns the role of the message content.
        """
        return self.message_content.role.value

    @property
    def is_action(self) -> bool:
        """
        Returns True if the message content is an action request or response.
        """
        return isinstance(
            self.message_content,
            (ActionRequestContent, ActionResponseContent),
        )

    @model_validator(mode="after")
    def _validate_content(self) -> Self:
        self.content = self.content or self.message_content.rendered
        if self.content is None:
            raise ValueError("Message content cannot be None.")
        return self

    @field_serializer("sender", "recipient")
    def _serialize_sender_recipient(cls, v: UUID) -> str:
        return str(v) if v else None

    @field_validator("sender", "recipient", mode="before")
    def _validate_sender_recipient(cls, v: str) -> UUID:
        return validate_uuid(v)

    @property
    def image_content(self) -> list[dict[str, Any]] | None:
        """
        Extract structured image data from the message content if it is
        represented as a chat message array.
        """
        msg_ = self.message_content.chat_msg
        if isinstance(msg_, dict) and isinstance(msg_["content"], list):
            return [i for i in msg_["content"] if i["type"] == "image_url"]
        return None
