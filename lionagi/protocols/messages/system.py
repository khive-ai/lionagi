# Copyright (c) 2023-2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, ClassVar

from pydantic import Field, field_validator

from .base import SenderRecipient
from .message import Message, MessageContent, MessageRole


@dataclass(slots=True)
class SystemContent(MessageContent):
    """Content for system messages.

    Fields:
        system_message: Main system instruction text
        system_datetime: Optional datetime string
    """

    system_message: str = "You are a helpful AI assistant. Let's think step by step."
    system_datetime: str | None = None
    datetime_factory: Callable[[], str] | None = field(default=None, repr=False)

    @property
    def rendered(self) -> str:
        """Render system message with optional datetime."""
        parts = []
        dt = self.system_datetime
        if dt is None and self.datetime_factory is not None:
            dt = self.datetime_factory()
        if dt:
            parts.append(f"System Time: {dt}")
        parts.append(self.system_message)
        return "\n\n".join(parts)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SystemContent":
        """Construct SystemContent from dictionary."""
        system_message = data.get(
            "system_message",
            cls.__dataclass_fields__["system_message"].default,
        )
        system_datetime = data.get("system_datetime")

        # Handle datetime generation
        if system_datetime is True:
            system_datetime = datetime.now().isoformat(timespec="minutes")
        elif system_datetime is False or system_datetime is None:
            system_datetime = None

        datetime_factory = data.get("datetime_factory")

        return cls(
            system_message=system_message,
            system_datetime=system_datetime,
            datetime_factory=datetime_factory if callable(datetime_factory) else None,
        )


class System(Message):
    """System-level message setting context or policy for the conversation."""

    _role: ClassVar[MessageRole] = MessageRole.SYSTEM
    content: SystemContent = Field(default_factory=SystemContent)
    sender: SenderRecipient | None = MessageRole.SYSTEM
    recipient: SenderRecipient | None = MessageRole.ASSISTANT

    @field_validator("content", mode="before")
    def _validate_content(cls, v):
        if v is None:
            return SystemContent()
        if isinstance(v, dict):
            return SystemContent.from_dict(v)
        if isinstance(v, SystemContent):
            return v
        raise TypeError("content must be dict or SystemContent instance")
