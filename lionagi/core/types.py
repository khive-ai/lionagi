# Copyright (c) 2023 - 2025, HaiyangLi <quantocean.li at gmail dot com>
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from enum import Enum
from typing import Any

from lionfuncs.parsers import fuzzy_parse_json
from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator

from lionagi.core.core_utils import copy

__all__ = (
    "MessageContent",
    "MessageRole",
    "Message",
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


class ToolRequest(BaseModel):
    function: str
    arguments: dict[str, Any] = Field(default_factory=dict)

    @field_validator("arguments", mode="before")
    def _validate_arguments(cls, v: dict | str | BaseModel) -> dict:
        if isinstance(v, dict):
            return copy(v)
        if isinstance(v, BaseModel):
            return v.model_dump()
        if isinstance(v, str):
            try:
                return fuzzy_parse_json(v.strip(), strict=True)
            except Exception as e:
                raise ValueError("Arguments must be a dictionary.") from e

    @field_validator("function", mode="before")
    def _validate_function(cls, v: Any) -> str:
        if isinstance(v, Callable):
            v = v.__name__
        if hasattr(v, "function"):
            v = v.function
        if not isinstance(v, str):
            raise ValueError("Function must be a string or callable.")
        return v


class ToolResponse(BaseModel):
    function: str
    arguments: dict[str, Any]
    output: BaseModel | JsonValue | None
