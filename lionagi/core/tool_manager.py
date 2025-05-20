"""Tool manager for registering and executing tools.

The ToolManager is responsible for registering tools, generating schemas for
them, and executing tool calls.
"""

import inspect
from typing import Any, Callable, Dict, List, Optional, Union

import lionfuncs.async_utils as async_utils
import lionfuncs.schema_utils as schema_utils
from lionfuncs.errors import LionError

from lionagi.models.message import ToolCallRequest, ToolCallResponse


class ToolManager:
    """Manager for tool registration and execution."""

    def __init__(self):
        """Initialize the ToolManager."""
        self._tools: Dict[str, Callable] = {}
        self._schemas: Dict[str, Dict[str, Any]] = {}

    def register_tool(
        self,
        name: str,
        func: Callable,
        schema: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Register a tool with the ToolManager.

        Args:
            name: The name of the tool.
            func: The function to call when the tool is executed.
            schema: Optional schema for the tool. If None, it will be
                generated from the function signature.

        Raises:
            LionError: If a tool with the same name is already registered.
        """
        # TODO: Implement tool registration logic
        # 1. Check if a tool with the same name is already registered
        # 2. Generate a schema if one is not provided
        # 3. Register the tool and its schema
        raise NotImplementedError("This method is not yet implemented.")

    def unregister_tool(self, name: str) -> None:
        """Unregister a tool from the ToolManager.

        Args:
            name: The name of the tool to unregister.

        Raises:
            LionError: If the tool is not registered.
        """
        # TODO: Implement tool unregistration logic
        # 1. Check if the tool is registered
        # 2. Remove the tool and its schema
        raise NotImplementedError("This method is not yet implemented.")

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """Get the schemas for all registered tools.

        Returns:
            A list of tool schemas in the format expected by LLM services.
        """
        # TODO: Implement schema retrieval logic
        # 1. Convert the stored schemas to the format expected by LLM services
        # 2. Return the list of schemas
        raise NotImplementedError("This method is not yet implemented.")

    def get_tool(self, name: str) -> Callable:
        """Get a tool by name.

        Args:
            name: The name of the tool to retrieve.

        Returns:
            The tool function.

        Raises:
            LionError: If the tool is not registered.
        """
        # TODO: Implement tool retrieval logic
        # 1. Check if the tool is registered
        # 2. Return the tool function
        raise NotImplementedError("This method is not yet implemented.")

    async def execute_tool_call(
        self, tool_call: ToolCallRequest
    ) -> ToolCallResponse:
        """Execute a tool call.

        Args:
            tool_call: The tool call to execute.

        Returns:
            The response from the tool.

        Raises:
            LionError: If the tool is not registered or if there is an error
                executing the tool.
        """
        # TODO: Implement tool execution logic
        # 1. Get the tool function
        # 2. Extract the arguments from the tool call
        # 3. Execute the tool function with the arguments
        # 4. Handle any errors
        # 5. Return a ToolCallResponse with the result
        raise NotImplementedError("This method is not yet implemented.")

    def _generate_schema(self, func: Callable) -> Dict[str, Any]:
        """Generate a schema for a function.

        Args:
            func: The function to generate a schema for.

        Returns:
            A schema for the function in the format expected by LLM services.
        """
        # TODO: Implement schema generation logic
        # 1. Use lionfuncs.schema_utils.function_to_openai_schema to generate a schema
        # 2. Return the schema
        raise NotImplementedError("This method is not yet implemented.")