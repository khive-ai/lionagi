from typing import Callable

from lionfuncs.utils import to_list
from pydapter.protocols import Event

from ..types import ToolRequest
from .tool import Tool


class ActionManager:

    def __init__(self, *args, **kwargs):
        self.registry: dict[str, Tool] = {}
        tools = []
        if args:
            tools.extend(to_list(args, dropna=True, flatten=True))
        if kwargs:
            tools.extend(
                to_list(kwargs, dropna=True, flatten=True, use_values=True)
            )

        self.register_tools(tools, update=True)

    def _register_tool(
        self,
        tool: Tool | Callable,
        update: bool = False,
    ):
        if not isinstance(tool, Tool) and callable(tool):
            tool = Tool(func_callable=tool)
        if not isinstance(tool, Tool):
            raise TypeError(
                "Must provide a `Tool` object or a callable function."
            )
        if tool.function in self.registry and not update:
            raise ValueError(f"Tool {tool.function} is already registered.")
        self.registry[tool.function] = tool

    def register_tools(
        self, tools: Tool | Callable | list, update: bool = False
    ):
        tools = [tools] if not isinstance(tools, list) else tools
        for t in tools:
            self._register_tool(t, update=update)

    def __contains__(self, tool: Callable | Tool | str) -> bool:
        if isinstance(tool, Tool):
            return tool.function in self.registry
        elif isinstance(tool, str):
            return tool in self.registry
        elif callable(tool):
            return tool.__name__ in self.registry
        return False

    def get_tool_schemas(
        self,
        tools: bool | str | Tool | list,
    ):
        if tools is True:
            tools = list(self.registry.keys())
        tools = [tools] if not isinstance(tools, list) else tools

        schemas = {}

        for i in tools:
            if isinstance(i, Tool):
                schemas[i.function] = i.tool_schema
            elif isinstance(i, str):
                if i in self.registry:
                    schemas[i] = self.registry[i].tool_schema
                else:
                    raise ValueError(f"Tool {i} is not registered.")
            else:
                raise TypeError(f"Unsupported type {type(i)}")
        return schemas

    def create_tool_calls(
        self, tool_requests: ToolRequest | list[ToolRequest]
    ) -> list[Event]:
        if not isinstance(tool_requests, list):
            tool_requests = [tool_requests]
        return [self._create_tool_call(tool_req) for tool_req in tool_requests]

    def _create_tool_call(self, tool_req: ToolRequest) -> Event:
        if tool_req.function not in self.registry:
            raise ValueError(f"Tool {tool_req.function} is not registered.")

        tool = self.registry[tool_req.function]
        return Event(
            event_invoke_function=tool.invoke,
            event_invoke_kwargs=tool_req.arguments,
            event_type="tool_call",
        )
