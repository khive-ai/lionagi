# Copyright (c) 2023 - 2025, HaiyangLi <quantocean.li at gmail dot com>
#
# SPDX-License-Identifier: Apache-2.0

from abc import abstractmethod
from enum import Enum
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, Template
from pydantic import BaseModel, ConfigDict, Field

template_path = Path(__file__).parent / "templates"
jinja_env = Environment(loader=FileSystemLoader(template_path))

__all__ = (
    "MessageContent",
    "MessageRole",
    "Message",
    "template_path",
    "jinja_env",
    "Template",
)


class MessageRole(str, Enum):
    """Defines the possible roles a message can have."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    ACTION = "action"


class MessageContent(BaseModel):
    model_config = ConfigDict(
        extra="allow",
        arbitrary_types_allowed=True,
        use_enum_values=True,
    )

    role: MessageRole

    @property
    def rendered(self) -> str:
        return NotImplemented

    def update(self, **kwargs) -> None:
        return NotImplemented

    @property
    def chat_msg(self) -> dict:
        """Returns the message content as a dictionary."""
        return {
            "role": self.role.value,
            "content": self.rendered,
        }
