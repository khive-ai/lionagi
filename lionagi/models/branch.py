# lionagi.models.branch - Branch Pydantic model (data part)
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

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
from pydapter.core import Adaptable
from pydapter.protocols import Temporal

# from .message import Message # Forward reference or direct import later
# from .tool import Tool # Forward reference or direct import later


# To avoid circular imports with Message and Tool, define placeholders for now
# These would typically be: from .message import Message; from .tool import Tool
class Message(BaseModel):  # Placeholder
    id: str
    # ... other fields


class Tool(BaseModel):  # Placeholder
    id: str
    name: str
    # ... other fields


class BranchModel(Temporal, Adaptable):
    """
    Pydantic model representing the data state of a Branch.
    The behavioral aspects will be in lionagi.core.branch.Branch.
    """

    name: str | None = None
    user: str | None = None  # User/owner of the branch
    system_message: Message | None = (
        None  # Stores the initial system message object
    )

    # messages: List[Message] = Field(default_factory=list) # This will be managed by a Pile-like structure
    # Instead of storing messages directly, we might store IDs or have MessageManager handle persistence
    # For now, aligning with TDS, which lists `messages: List[Message]`.
    # This implies the MessageManager (Pile-like) would serialize its content here.
    # Or, this list is what the MessageManager populates from its internal store upon BranchModel creation.
    # Let's assume this list represents the current snapshot of messages for serialization.
    message_ids: list[str] = Field(
        default_factory=list
    )  # IDs of messages in order, actual Message objects managed by MessageManager

    tool_configs: list[Tool] = Field(
        default_factory=list
    )  # Registered tool configurations (serializable part of Tool)

    # LLM service configuration for this branch.
    # This could be a direct dict or a model like ServiceInterface if that's made adaptable.
    # TDS Sec 4 Branch model shows: service_config: Optional[Dict[str, Any]] = None
    service_config: dict[str, Any] | None = None

    mailbox_id: str | None = None  # ID of its associated mailbox
    metadata: dict[str, Any] = Field(default_factory=dict)

    # Progression/order of messages will be handled by the MessageManager (Pile-like)
    # and reflected in message_ids for serialization.

    class Config:
        arbitrary_types_allowed = True


__all__ = ["BranchModel"]
