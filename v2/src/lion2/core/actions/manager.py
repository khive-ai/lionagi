from typing import Any, Callable

from lionfuncs.utils import to_list
from pydapter.protocols import Event

from ..types import ToolRequest, GenericParams
from .tool import Tool, ToolParameterValidationError, FuncTool


class ActionManager:
    def __init__(self, *args: list[FuncTool], **kwargs: dict[str, FuncTool]) -> None:
        self.registry: dict[str, Tool] = {}
        tools_to_register = []
        if args:
            tools_to_register.extend(to_list(args, dropna=True, flatten=True))
        if kwargs:
            tools_to_register.extend(
                to_list(
                    kwargs,
                    dropna=True,
                    flatten=True,
                    flatten_tuple_set=True,
                    use_values=True,
                )
            )

        if tools_to_register:
            self.register_tools(tools_to_register, update=True)

    def _register_tool(
        self,
        tool_input: FuncTool,
        update: bool = False,
    ) -> None:
        tool: Tool
        if isinstance(tool_input, Tool):
            tool = tool_input
        elif callable(tool_input):
            try:
                tool_name = tool_input.__name__
            except AttributeError:
                tool_name = "unnamed_callable_tool"

            tool = Tool(
                name=tool_name,
                description=f"Tool for function {tool_name}",
                parameters_schema=GenericParams,
                func_callable=tool_input,
            )
        else:
            raise TypeError("Must provide a `Tool` object or a callable function.")

        if tool.name in self.registry and not update:
            raise ValueError(f"Tool {tool.name} is already registered.")
        self.registry[tool.name] = tool

    def register_tools(
        self,
        tools: Tool | Callable[..., Any] | list[Tool | Callable[..., Any]],
        update: bool = False,
    ) -> None:
        processed_tools = to_list(tools, dropna=True)
        for t in processed_tools:
            self._register_tool(t, update=update)

    def __contains__(self, tool_identifier: Callable[..., Any] | Tool | str) -> bool:
        key_to_check: str | None = None
        if isinstance(tool_identifier, Tool):
            key_to_check = tool_identifier.name
        elif isinstance(tool_identifier, str):
            key_to_check = tool_identifier
        elif callable(tool_identifier):
            try:
                key_to_check = tool_identifier.__name__
            except AttributeError:  # Some callables might not have __name__
                return False

        return key_to_check is not None and key_to_check in self.registry

    def get_tool_schemas(
        self,
        tools_filter: bool | str | Tool | list[str | Tool],
    ) -> dict[str, Any]:  # Returns dict of tool_name to its OpenAI schema
        target_tool_names: list[str] = []
        if tools_filter is True:
            target_tool_names = list(self.registry.keys())
        else:
            processed_filter = to_list(tools_filter, dropna=True)
            for item in processed_filter:
                if isinstance(item, Tool):
                    if item.name in self.registry:
                        target_tool_names.append(item.name)
                    else:
                        raise ValueError(
                            f"Tool instance '{item.name}' is not registered."
                        )

                elif isinstance(item, str):
                    if item in self.registry:
                        target_tool_names.append(item)
                    else:
                        raise ValueError(f"Tool name '{item}' is not registered.")
                else:
                    raise TypeError(
                        f"Unsupported type '{type(item)}' in tools_filter. Expected str or Tool."
                    )

        unique_target_tool_names = sorted(list(set(target_tool_names)))

        schemas: dict[str, Any] = {}
        for name in unique_target_tool_names:
            schemas[name] = self.registry[name].to_openai_tool_schema()
        return schemas

    def create_tool_calls(
        self, tool_requests: ToolRequest | list[ToolRequest]
    ) -> list[Event]:
        processed_requests = to_list(tool_requests, dropna=True)
        if not all(isinstance(tr, ToolRequest) for tr in processed_requests):
            raise TypeError("All items in tool_requests must be ToolRequest objects.")

        return [self._create_tool_call(tool_req) for tool_req in processed_requests]

    def _create_tool_call(self, tool_req: ToolRequest) -> Event:
        if tool_req.function not in self.registry:
            raise ValueError(f"Tool '{tool_req.function}' is not registered.")

        tool = self.registry[tool_req.function]

        try:
            arguments_dict = tool_req.arguments or {}
            validated_params = tool.validate_parameters(arguments_dict)
        except ToolParameterValidationError as e:
            raise ValueError(
                f"Parameter validation failed for tool '{tool.name}': {e}"
            ) from e
        except Exception as e:
            raise ValueError(
                f"Unexpected error during parameter validation for tool '{tool.name}': {e}"
            ) from e

        return Event(
            event_invoke_function=tool.invoke,
            event_invoke_args=[validated_params],
            event_invoke_kwargs={},
            event_type="tool_call",
        )
