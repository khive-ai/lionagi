import json
from enum import Enum
from typing import Any, Literal
from uuid import UUID

from lionfuncs.utils import to_dict
from pydantic import (
    BaseModel,
    JsonValue,
    field_serializer,
    field_validator,
    model_validator,
)
from pydapter.core import Adaptable
from pydapter.protocols.embedable import Embedable
from pydapter.protocols.utils import validate_uuid


class MessageRole(str, Enum):
    """Defines the possible roles a message can have."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL_REQUEST = "tool_request"
    TOOL_RESPONSE = "tool_response"


class ToolRequest(BaseModel):
    """Represents a request from the LLM to call a tool."""

    type: Literal["tool_request"] = "tool_request"
    function: str
    arguments: dict[str, Any]

    @field_validator("arguments")
    def _validate_arguments(cls, v):
        if isinstance(v, dict):
            return v
        else:
            return to_dict(v)


class ToolResponse(BaseModel):
    """Represents a response from a tool."""

    type: Literal["tool_response"] = "tool_response"
    request_id: UUID
    function: str
    result: JsonValue | None = None

    @field_serializer("request_id")
    def _serialize_request_id(cls, v: UUID) -> str:
        return str(v)

    @field_validator("request_id")
    def _validate_request_id(cls, v: str) -> UUID:
        return validate_uuid(v)


MessageContent = ToolRequest | ToolResponse | JsonValue | BaseModel


class Message(Embedable, Adaptable):
    role: MessageRole
    sender: UUID | None = None
    recipient: UUID | None = None
    content: MessageContent | None = None

    @property
    def text_content(self) -> str:
        """Returns the text content of the message."""
        if isinstance(self.content, BaseModel):
            return self.content.model_dump_json()
        if isinstance(self.content, dict):
            return json.dumps(self.content)
        if isinstance(self.content, str):
            return self.content
        return ""

    @field_serializer("sender", "recipient")
    def _serialize_sender_recipient(cls, v: UUID) -> str:
        if not v:
            return None
        return str(v)

    @field_validator("sender", "recipient", mode="before")
    def _validate_sender_recipient(cls, v: str) -> UUID:
        if not v:
            return None
        return validate_uuid(v)

    @model_validator(mode="before")
    def _validate_data(cls, data: dict[str, Any]) -> dict[str, Any]:
        role = data.pop("role", None)
        role = MessageRole(role) if isinstance(role, str) else role

        content = to_dict(data.pop("content", None))

        if role == MessageRole.TOOL_REQUEST:
            try:
                content = ToolRequest(**content)
            except Exception as e:
                raise ValueError(
                    f"Invalid content for TOOL_REQUEST: {e}"
                ) from e
        elif role == MessageRole.TOOL_RESPONSE:
            try:
                content = ToolResponse(**content)
            except Exception as e:
                raise ValueError(
                    f"Invalid content for TOOL_RESPONSE: {e}"
                ) from e
        elif role not in [
            MessageRole.SYSTEM,
            MessageRole.USER,
            MessageRole.ASSISTANT,
        ]:
            raise ValueError(
                f"Invalid role: {role}. Must be one of {list(MessageRole)}."
            )

        return {
            "role": role,
            "content": content,
            **data,
        }
