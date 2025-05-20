# lionagi.models.message - Message related Pydantic models
# Copyright (c) 2023-present, HaiyangLi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from enum import Enum
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, field_validator # Added field_validator

# Assuming Temporal provides id, created_at, updated_at
# from pydapter.protocols import Temporal # Actual import
# from pydapter.core import Adaptable # Actual import

# For now, let's define placeholders if pydapter is not available for linting
try:
    from pydapter.protocols import Temporal
except ImportError:
    class Temporal(BaseModel): # type: ignore
        id: str = Field(default_factory=lambda: "temp_id")
        created_at: Any = Field(default_factory=lambda: "timestamp")
        updated_at: Any = Field(default_factory=lambda: "timestamp")
        class Config:
            arbitrary_types_allowed = True

try:
    from pydapter.core import Adaptable
except ImportError:
    class Adaptable: # type: ignore
        pass


class MessageRole(str, Enum):
    """Defines the possible roles a message can have."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL_REQUEST = "tool_request"
    TOOL_RESPONSE = "tool_response"

class ToolCallRequest(Temporal):
    """Represents a request from the LLM to call a tool."""
    function_name: str
    arguments: Dict[str, Any]

class ToolCallResponse(Temporal):
    """Represents the result of a tool call."""
    tool_call_id: str
    function_name: str
    result: Any
    is_error: bool = False
    error_message: Optional[str] = None

class Message(Temporal, Adaptable):
    """Represents a message in a conversation."""
    role: MessageRole
    sender: Optional[str] = None
    recipient: Optional[str] = None
    content: Any  # str, List[ToolCallRequest], or ToolCallResponse
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator('content')
    def validate_content_type(cls, v: Any, values: Any) -> Any: # Changed to Pydantic v2 syntax for validator
        # In Pydantic v2, `values` is a `ValidationInfo` object, access data via `values.data`
        role = values.data.get('role')
        if role == MessageRole.TOOL_REQUEST:
            if not isinstance(v, list) or not all(isinstance(tc, ToolCallRequest) for tc in v):
                raise ValueError("Content for TOOL_REQUEST must be a list of ToolCallRequest objects.")
        elif role == MessageRole.TOOL_RESPONSE:
            if not isinstance(v, ToolCallResponse):
                raise ValueError("Content for TOOL_RESPONSE must be a ToolCallResponse object.")
        elif role in [MessageRole.SYSTEM, MessageRole.USER, MessageRole.ASSISTANT]:
            # As per TDS, allowing str or dict for assistant (non-tool) responses.
            # This might need further refinement based on LLM response handling.
            if not isinstance(v, (str, dict)): # Allow dict for assistant
                 raise ValueError(f"Content for {role.value} must be a string or a dict.")
        return v
    
    class Config:
        arbitrary_types_allowed = True


__all__ = [
    "MessageRole",
    "ToolCallRequest",
    "ToolCallResponse",
    "Message"
]