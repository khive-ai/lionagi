# Copyright (c) 2023-2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Any, ClassVar

from pydantic import field_serializer, field_validator

from lionagi.ln.types import DataClass

from .._concepts import Sendable
from ..graph.node import Node
from .base import (
    MessageRole,
    SenderRecipient,
    serialize_sender_recipient,
    validate_sender_recipient,
)


@dataclass(slots=True)
class MessageContent(DataClass):
    """A base class for message content structures."""

    _none_as_sentinel: ClassVar[bool] = True

    @property
    def rendered(self) -> str:
        """Render the content as a string."""
        raise NotImplementedError(
            "Subclasses must implement rendered property."
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MessageContent":
        """Create an instance from a dictionary."""
        raise NotImplementedError(
            "Subclasses must implement from_dict method."
        )


class RoledMessage(Node, Sendable):
    """Base class for all messages with a role and structured content.

    Subclasses must provide a concrete MessageContent type.
    """

    role: MessageRole = MessageRole.UNSET
    content: MessageContent
    sender: SenderRecipient | None = MessageRole.UNSET
    recipient: SenderRecipient | None = MessageRole.UNSET

    @field_serializer("sender", "recipient")
    def _serialize_sender_recipient(self, value: SenderRecipient) -> str:
        return serialize_sender_recipient(value)

    @field_validator("sender", "recipient")
    def _validate_sender_recipient(cls, v):
        if v is None:
            return None
        return validate_sender_recipient(v)

    @property
    def chat_msg(self) -> dict[str, Any] | None:
        """A dictionary representation typically used in chat-based contexts."""
        try:
            role_str = (
                self.role.value
                if isinstance(self.role, MessageRole)
                else str(self.role)
            )
            return {"role": role_str, "content": self.rendered}
        except Exception:
            return None

    @property
    def rendered(self) -> str:
        """Render the message content as a string.

        Delegates to the content's rendered property.
        """
        return self.content.rendered

    @field_validator("role", mode="before")
    def _validate_role(cls, v):
        if isinstance(v, str):
            return MessageRole(v)
        if isinstance(v, MessageRole):
            return v
        return MessageRole.UNSET

    def update(self, sender=None, recipient=None, **kw):
        """Update message fields.

        Args:
            sender: New sender role or ID.
            recipient: New recipient role or ID.
            **kw: Content updates to apply via from_dict() reconstruction.
        """
        if sender:
            self.sender = validate_sender_recipient(sender)
        if recipient:
            self.recipient = validate_sender_recipient(recipient)
        if kw:
            _dict = self.content.to_dict()
            _dict.update(kw)
            self.content = type(self.content).from_dict(_dict)

    def clone(self) -> "RoledMessage":
        """Create a clone with a new ID but reference to original.

        Returns:
            A new message instance with a new ID and deep-copied content,
            with a reference to the original message in metadata.
        """
        # Create a new instance from dict, excluding frozen fields (id, created_at)
        # This allows new id and created_at to be generated
        data = self.to_dict()
        original_id = data.pop("id")
        data.pop("created_at")  # Let new created_at be generated

        # Create new instance
        cloned = type(self).from_dict(data)

        # Store reference to original in metadata
        cloned.metadata["clone_from"] = str(original_id)

        return cloned

    @property
    def image_content(self) -> list[dict[str, Any]] | None:
        """
        Extract structured image data from the message content if it is
        represented as a chat message array.
        """
        msg_ = self.chat_msg
        if isinstance(msg_, dict) and isinstance(msg_["content"], list):
            return [i for i in msg_["content"] if i["type"] == "image_url"]
        return None


# File: lionagi/protocols/messages/message.py
