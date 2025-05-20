"""Orchestrator for managing conversation flow and tool calls.

The Orchestrator is responsible for managing the flow of a conversation,
including processing messages, handling tool calls, and updating the
conversation state.
"""

from typing import Any, Dict, List, Optional, Union

import lionfuncs.async_utils as async_utils
from lionfuncs.errors import LionError

from lionagi.core.service_interface import ServiceInterface
from lionagi.core.tool_manager import ToolManager
from lionagi.models.message import Message, ToolCallRequest, ToolCallResponse
from lionagi.models.session import Branch, Session


class Orchestrator:
    """Orchestrator for managing conversation flow and tool calls."""

    def __init__(
        self,
        service_interface: ServiceInterface,
        tool_manager: Optional[ToolManager] = None,
    ):
        """Initialize the Orchestrator.

        Args:
            service_interface: Interface for communicating with LLM services.
            tool_manager: Manager for tool registration and execution.
        """
        self.service_interface = service_interface
        self.tool_manager = tool_manager or ToolManager()

    async def process_message(
        self,
        session: Session,
        message_content: str,
        message_role: str = "user",
        tools: Optional[List[Any]] = None,
    ) -> Message:
        """Process a new message in the conversation.

        Args:
            session: The session containing the conversation.
            message_content: The content of the message.
            message_role: The role of the message (default: "user").
            tools: Optional list of tools to make available for this message.

        Returns:
            The assistant's response message.
        """
        # TODO: Implement message processing logic
        # 1. Create a new message and add it to the current branch
        # 2. Prepare the request for the LLM service
        # 3. Send the request to the LLM service
        # 4. Process the response, handling any tool calls
        # 5. Return the final assistant message
        raise NotImplementedError("This method is not yet implemented.")

    async def _handle_tool_calls(
        self,
        session: Session,
        tool_calls: List[Dict[str, Any]],
    ) -> List[ToolCallResponse]:
        """Handle tool calls from the LLM.

        Args:
            session: The session containing the conversation.
            tool_calls: List of tool calls from the LLM.

        Returns:
            List of tool call responses.
        """
        # TODO: Implement tool call handling logic
        # 1. Convert tool calls to ToolCallRequest objects
        # 2. Execute each tool call using the tool manager
        # 3. Add tool requests and responses to the conversation
        # 4. Return the tool call responses
        raise NotImplementedError("This method is not yet implemented.")

    async def _update_conversation_state(
        self,
        session: Session,
        message: Message,
    ) -> None:
        """Update the conversation state with a new message.

        Args:
            session: The session containing the conversation.
            message: The message to add to the conversation.
        """
        # TODO: Implement conversation state update logic
        # 1. Get the current branch
        # 2. Add the message to the branch
        # 3. Update the branch in the session
        raise NotImplementedError("This method is not yet implemented.")