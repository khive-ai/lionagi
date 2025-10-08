# Copyright (c) 2023-2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from enum import Enum
from typing import Any, TypeAlias

from ..generic.element import ID, Element, IDError, IDType, Observable

__all__ = (
    "MessageRole",
    "MessageField",
    "MESSAGE_FIELDS",
    "validate_sender_recipient",
    "serialize_sender_recipient",
)


class MessageRole(str, Enum):
    """
    Predefined roles for conversation participants or message semantics.
    """

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    UNSET = "unset"
    ACTION = "action"


SenderRecipient: TypeAlias = IDType | MessageRole | str
"""
A union type indicating that a sender or recipient could be:
- A lionagi IDType,
- A string-based role or ID,
- A specific enum role from `MessageRole`.
"""


class MessageField(str, Enum):
    """
    Common field names used in message objects.
    """

    CREATED_AT = "created_at"
    ROLE = "role"
    CONTENT = "content"
    ID = "id"
    SENDER = "sender"
    RECIPIENT = "recipient"
    METADATA = "metadata"


MESSAGE_FIELDS = [i.value for i in MessageField.__members__.values()]


def validate_sender_recipient(value: Any, /) -> SenderRecipient:
    """
    Normalize a sender/recipient value into a recognized type.

    Args:
        value (Any): Input to interpret as a role or ID.

    Returns:
        SenderRecipient: A validated and normalized entity.

    Raises:
        ValueError: If the input cannot be recognized as a role or ID.
    """
    if isinstance(value, MessageRole):
        return value

    if isinstance(value, IDType):
        return value

    if isinstance(value, Observable):
        return value.id

    if value is None:
        return MessageRole.UNSET

    if value in ["system", "user", "unset", "assistant", "action"]:
        return MessageRole(value)

    # Accept plain strings (user names, identifiers, etc)
    if isinstance(value, str):
        # Try to parse as ID first, but allow plain strings as fallback
        try:
            return ID.get_id(value)
        except IDError:
            return value

    raise ValueError("Invalid sender or recipient")


def serialize_sender_recipient(value: Any) -> str | None:
    if not value:
        return None
    # Check instance types first before enum membership
    if isinstance(value, Element):
        return str(value.id)
    if isinstance(value, IDType):
        return str(value)
    if isinstance(value, MessageRole):
        return value.value
    if isinstance(value, str):
        return value
    return str(value)


# File: lionagi/protocols/messages/base.py
