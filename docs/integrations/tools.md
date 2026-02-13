# Tool Integration

lionagi converts Python functions into LLM-callable tools automatically. The `Tool` class wraps any callable with schema generation, preprocessing, postprocessing, and argument validation.

## How It Works

When you pass a function to `Branch(tools=[...])`, lionagi:

1. Inspects the function signature and docstring
2. Generates an OpenAI-compatible JSON schema via `function_to_schema()`
3. Wraps it in a `Tool` object registered with the `ActionManager`
4. Sends the schema to the LLM during `operate()` or `ReAct()` calls
5. Automatically invokes the function when the LLM requests it

## Basic Usage

```python
from lionagi import Branch

def calculate_sum(a: float, b: float) -> float:
    """Add two numbers together.

    Args:
        a: First number.
        b: Second number.
    """
    return a + b

branch = Branch(
    tools=[calculate_sum],
    system="You have access to a calculator tool."
)

# operate() with actions=True enables tool invocation
result = await branch.operate(
    instruction="What is 15 + 27?",
    actions=True,
)
```

## The Tool Class

`Tool` extends `Element` and wraps a callable with metadata:

```python
from lionagi.protocols.action.tool import Tool

tool = Tool(
    func_callable=calculate_sum,
    # Optional: pre/post processing
    preprocessor=lambda args: {k: float(v) for k, v in args.items()},
    postprocessor=lambda result: f"The answer is {result}",
    # Optional: strict argument validation
    strict_func_call=False,
)

# Auto-generated schema
print(tool.tool_schema)
# {'type': 'function', 'function': {'name': 'calculate_sum', ...}}

print(tool.function)       # 'calculate_sum'
print(tool.required_fields) # {'a', 'b'}
```

### Key Tool Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `func_callable` | `Callable` | The function to wrap (required) |
| `tool_schema` | `dict` | Custom schema (auto-generated if omitted) |
| `request_options` | `type[BaseModel]` | Pydantic model for input validation |
| `preprocessor` | `Callable` | Transform arguments before execution |
| `postprocessor` | `Callable` | Transform results after execution |
| `strict_func_call` | `bool` | Enforce exact parameter matching |
| `mcp_config` | `dict` | MCP server tool config (see [MCP Servers](mcp-servers.md)) |

## Async Tools

Both sync and async functions are supported. Async functions are awaited automatically:

```python
import httpx

async def fetch_url(url: str) -> str:
    """Fetch content from a URL.

    Args:
        url: The URL to fetch.
    """
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        return response.text[:500]

branch = Branch(tools=[fetch_url])
```

## Registering Tools

Tools can be registered at construction or later:

```python
# At construction
branch = Branch(tools=[func_a, func_b])

# After construction
branch.register_tools([func_c, func_d])

# With update=True to replace existing tools
branch.register_tools([func_a_v2], update=True)

# Check registered tools
print(branch.tools)  # dict of {name: Tool}
```

You can pass raw functions, `Tool` objects, `LionTool` subclasses, or MCP config dicts.

## Pydantic Request Validation

Use `request_options` to validate tool arguments with a Pydantic model:

```python
from pydantic import BaseModel, Field

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    max_results: int = Field(default=5, ge=1, le=20)

def search(query: str, max_results: int = 5) -> str:
    """Search for information."""
    return f"Results for '{query}' (max {max_results})"

tool = Tool(
    func_callable=search,
    request_options=SearchRequest,
)
```

## function_to_schema()

The schema generator extracts function metadata from type hints and docstrings:

```python
from lionagi.libs.schema.function_to_schema import function_to_schema

def example(name: str, count: int) -> bool:
    """Do something with name and count.

    Args:
        name: The name to process.
        count: How many times to process.
    """
    return True

schema = function_to_schema(example)
# {
#   'type': 'function',
#   'function': {
#     'name': 'example',
#     'description': 'Do something with name and count.',
#     'parameters': {
#       'type': 'object',
#       'properties': {
#         'name': {'type': 'string', 'description': 'The name to process.'},
#         'count': {'type': 'number', 'description': 'How many times to process.'}
#       },
#       'required': ['name', 'count']
#     }
#   }
# }
```

Supported type mappings: `str` -> `string`, `int`/`float` -> `number`, `list`/`tuple` -> `array`, `bool` -> `boolean`, `dict` -> `object`.

## Using Tools with operate() and ReAct()

### operate()

```python
result = await branch.operate(
    instruction="Search for Python tutorials",
    actions=True,          # Enable tool calling
    invoke_actions=True,   # Auto-invoke tools (default)
    action_strategy="concurrent",  # or "sequential"
)
```

### ReAct()

ReAct runs a think-act-observe loop, automatically using registered tools:

```python
result = await branch.ReAct(
    instruct={
        "instruction": "Research Python best practices and summarize findings",
        "context": {"focus": "async programming"},
    },
    tools=True,            # Use all registered tools (default)
    max_extensions=3,      # Max reasoning iterations
    verbose=True,
)
```

## Built-in Tools

### ReaderTool

Reads files and URLs using the `docling` document converter. Requires `pip install lionagi[reader]`.

```python
from lionagi.tools import ReaderTool

branch = Branch(
    tools=[ReaderTool],
    system="You can read documents and URLs."
)

result = await branch.ReAct(
    instruct={"instruction": "Read and summarize this document: report.pdf"},
    max_extensions=4,
)
```

ReaderTool supports three actions:

- **open**: Convert a file or URL to text, returns a `doc_id` and length
- **read**: Read a slice of a previously opened document by `doc_id` and offset
- **list_dir**: List files in a directory

## Custom LionTool

For reusable tools with internal state, subclass `LionTool`:

```python
from lionagi.tools.base import LionTool
from lionagi.protocols.action.tool import Tool

class MyStatefulTool(LionTool):
    is_lion_system_tool = True
    system_tool_name = "my_tool"

    def __init__(self):
        super().__init__()
        self.state = {}

    def to_tool(self) -> Tool:
        def my_tool(action: str, key: str, value: str = None) -> str:
            """Stateful key-value store."""
            if action == "set":
                self.state[key] = value
                return f"Set {key}"
            return self.state.get(key, "Not found")

        return Tool(func_callable=my_tool)

# Register like any other tool
branch = Branch(tools=[MyStatefulTool])
```

## Best Practices

- **Type hints**: Always add type hints -- they drive schema generation
- **Docstrings**: Use Google-style docstrings with `Args:` sections for parameter descriptions
- **Return strings**: LLMs work best when tools return string results
- **Error handling**: Wrap tool logic in try/except and return error messages as strings
- **Async preferred**: Use async functions for I/O-bound tools to avoid blocking
