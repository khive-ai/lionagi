# Tools and Functions

Tools give branches the ability to call your Python functions during LLM interactions. LionAGI automatically generates OpenAI-compatible function schemas from your code, handles both sync and async functions, and supports MCP server integration.

## Registering Tools

The simplest way to add tools is to pass functions directly when creating a Branch:

```python
from lionagi import Branch

def calculate_sum(a: float, b: float) -> float:
    """Add two numbers together."""
    return a + b

def search_records(query: str) -> list[dict]:
    """Search the database for matching records."""
    return [{"id": 1, "name": "example"}]

branch = Branch(tools=[calculate_sum, search_records])
```

You can also register tools after creation:

```python
branch.register_tools([calculate_sum, search_records])

# Or one at a time
branch.register_tools(calculate_sum)
```

To update an existing tool registration, pass `update=True`:

```python
branch.register_tools(calculate_sum, update=True)
```

## How Schema Generation Works

When you register a function, LionAGI calls `function_to_schema()` to auto-generate an OpenAI-compatible tool schema from the function's signature and docstring.

```python
from lionagi.libs.schema.function_to_schema import function_to_schema

def get_weather(city: str, units: str = "celsius") -> dict:
    """Get current weather for a city.

    Args:
        city: The city name to look up.
        units: Temperature units (celsius or fahrenheit).
    """
    return {"city": city, "temperature": 22, "units": units}

schema = function_to_schema(get_weather)
```

The generated schema follows the OpenAI function calling format:

```json
{
  "type": "function",
  "function": {
    "name": "get_weather",
    "description": "Get current weather for a city.",
    "parameters": {
      "type": "object",
      "properties": {
        "city": {"type": "string", "description": "The city name to look up."},
        "units": {"type": "string", "description": "Temperature units (celsius or fahrenheit)."}
      },
      "required": ["city", "units"]
    }
  }
}
```

Python type hints map to JSON Schema types:

| Python | JSON Schema |
|--------|-------------|
| `str` | `string` |
| `int` | `number` |
| `float` | `number` |
| `bool` | `boolean` |
| `list` / `tuple` | `array` |
| `dict` | `object` |

Google-style and reST-style docstrings are both supported for extracting parameter descriptions.

## The Tool Class

For more control, create `Tool` objects directly:

```python
from lionagi.protocols.action.tool import Tool

tool = Tool(func_callable=get_weather)

# Access the auto-generated schema
print(tool.function)          # "get_weather"
print(tool.tool_schema)       # Full OpenAI-format schema
print(tool.required_fields)   # {"city", "units"}
```

### Input Validation with request_options

You can attach a Pydantic model for strict input validation:

```python
from pydantic import BaseModel

class WeatherRequest(BaseModel):
    city: str
    units: str = "celsius"

tool = Tool(
    func_callable=get_weather,
    request_options=WeatherRequest,
)
branch = Branch(tools=[tool])
```

When the LLM calls this tool, arguments are validated against `WeatherRequest` before the function executes.

### Pre/Post Processing

Tools support optional preprocessing of arguments and postprocessing of results:

```python
def clean_args(args: dict) -> dict:
    args["city"] = args["city"].strip().title()
    return args

def format_output(result: dict) -> str:
    return f"{result['city']}: {result['temperature']}deg {result['units']}"

tool = Tool(
    func_callable=get_weather,
    preprocessor=clean_args,
    postprocessor=format_output,
)
```

Both sync and async preprocessors/postprocessors are supported.

### Strict Mode

Enable `strict_func_call=True` to require that the LLM-provided arguments exactly match the function schema's required fields (no extra, no missing):

```python
tool = Tool(func_callable=get_weather, strict_func_call=True)
```

## Async Tool Support

Async functions work identically to sync functions. LionAGI detects whether a function is async and handles it automatically:

```python
import httpx

async def fetch_page(url: str) -> str:
    """Fetch the content of a web page."""
    async with httpx.AsyncClient() as client:
        resp = await client.get(url)
        return resp.text

branch = Branch(tools=[fetch_page])
```

## Using Tools in Operations

### operate() with actions=True

The `operate()` method supports tool calling when you set `actions=True`:

```python
from pydantic import BaseModel

class ResearchResult(BaseModel):
    findings: list[str]
    sources: list[str]

result = await branch.operate(
    instruction="Research renewable energy trends",
    response_format=ResearchResult,
    actions=True,
    action_strategy="concurrent",  # or "sequential"
)
```

### ReAct() for Multi-Step Tool Use

`ReAct()` is designed for iterative tool use with reasoning:

```python
result = await branch.ReAct(
    instruct={"instruction": "Find and analyze the latest sales data"},
    max_extensions=3,
    verbose=True,
)
```

The model reasons about what tool to call, observes the result, and decides whether to call more tools or produce a final answer.

### act() for Direct Execution

`act()` executes tool calls directly without involving the LLM:

```python
responses = await branch.act(
    action_request=[
        {"function": "calculate_sum", "arguments": {"a": 10, "b": 20}},
        {"function": "search_records", "arguments": {"query": "revenue"}},
    ],
    strategy="concurrent",
)
```

## The ActionManager

The `ActionManager` (accessible via `branch.acts`) maintains the tool registry and handles invocation:

```python
# Check registered tools
print(branch.tools)                  # dict[str, Tool]
print("calculate_sum" in branch.acts)  # True

# Get schemas for all registered tools
schemas = branch.acts.schema_list    # list[dict]
```

### FunctionCalling

When a tool is invoked, the ActionManager creates a `FunctionCalling` event that tracks execution status, duration, and results:

```python
from lionagi.protocols.action.function_calling import FunctionCalling

# Internally, ActionManager does:
func_call = branch.acts.match_tool(action_request)  # Creates FunctionCalling
await func_call.invoke()                              # Executes the function
print(func_call.execution.status)                     # COMPLETED or FAILED
print(func_call.execution.response)                   # Function return value
```

## MCP Tool Integration

LionAGI supports [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) for integrating external tool servers.

### Registering MCP Tools

Register tools from a single MCP server:

```python
tools = await branch.acts.register_mcp_server(
    server_config={"server": "search"},
    tool_names=["web_search", "news_search"],
)
```

### Auto-Discovery

Omit `tool_names` to auto-discover all tools from a server:

```python
tools = await branch.acts.register_mcp_server(
    server_config={"server": "search"},
)
```

### Loading from Config File

Load multiple MCP servers from a `.mcp.json` configuration file:

```python
all_tools = await branch.acts.load_mcp_config(
    config_path="/path/to/.mcp.json",
    server_names=["search", "memory"],  # or None for all servers
)
```

### MCP Config Dict

You can also pass an MCP config dict directly when creating a Tool:

```python
tool = Tool(mcp_config={"web_search": {"server": "search"}})
branch.register_tools(tool)
```

!!! warning "MCP Exclusivity"
    A Tool must have either `func_callable` or `mcp_config`, not both. Providing both raises a `ValueError`.

## LionTool Base Class

For complex tools with setup/teardown logic, extend the `LionTool` base class:

```python
from lionagi.tools.base import LionTool

class ReaderTool(LionTool):
    """Tool for reading files."""
    # Implements to_tool() which returns Tool instances
    pass

branch = Branch(tools=[ReaderTool])
```

Built-in LionTool implementations (like `ReaderTool`) are automatically converted to `Tool` instances during registration.

## Next Steps

- [Operations](operations.md) -- using tools with `operate()`, `ReAct()`, and `act()`
- [Messages and Memory](messages-and-memory.md) -- how tool calls appear in conversation history
