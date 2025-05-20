"""Message models for different types of messages.

This module contains models for different types of messages used in
conversations with LLMs, including user messages, assistant messages,
tool requests, and tool responses.
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Union

import lionfuncs.utils as utils
from pydantic import Field

from lionagi.models.base import BaseModel


class RoleEnum(str, Enum):
    """Enumeration of message roles."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL_REQUEST = "tool_request"
    TOOL_RESPONSE = "tool_response"


class Message(BaseModel):
    """Base message model for all types of messages."""

    id: str = Field(default_factory=lambda: utils.generate_uuid())
    role: RoleEnum
    sender: Optional[str] = None
    content: Any
    timestamp: str = Field(default_factory=lambda: utils.get_timestamp())
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ToolCallRequest(BaseModel):
    """Model for a tool call request from an LLM."""

    id: str = Field(default_factory=lambda: utils.generate_uuid())
    function_name: str
    arguments: Dict[str, Any]


class ToolCallResponse(BaseModel):
    """Model for a tool call response to an LLM."""

    request_id: str
    result: Any
    is_error: bool = False
    error_message: Optional[str] = None


# Register adapters for the models
Message.register_adapters()
ToolCallRequest.register_adapters()
ToolCallResponse.register_adapters()