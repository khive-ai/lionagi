# MCP Server Integration

lionagi supports the [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) for connecting to external tool servers. MCP tools are registered and invoked the same way as native tools -- the LLM sees them as regular function calls.

## How It Works

The `Tool` class accepts an `mcp_config` parameter instead of `func_callable`. When the LLM calls the tool, lionagi connects to the MCP server via `fastmcp` and executes the tool remotely. Connections are pooled by `MCPConnectionPool` for reuse.

## Configuration

### .mcp.json File

Define MCP servers in a `.mcp.json` file:

```json
{
  "mcpServers": {
    "search": {
      "command": "python",
      "args": ["-m", "my_search_server"],
      "env": {
        "API_KEY": "your-key"
      }
    },
    "memory": {
      "url": "http://localhost:8000/mcp"
    }
  }
}
```

Servers can be configured with either a `command` (stdio transport) or a `url` (HTTP transport).

### Auto-Discovery

Register all tools from an MCP server automatically:

```python
from lionagi import Branch
from lionagi.protocols.action.manager import load_mcp_tools

# Load tools from .mcp.json
tools = await load_mcp_tools(
    config_path=".mcp.json",
    server_names=["search"],
)

branch = Branch(tools=tools)
```

### Manual Registration via ActionManager

For more control, use the `ActionManager` directly:

```python
branch = Branch()

# Auto-discover and register all tools from a server
registered = await branch.acts.register_mcp_server(
    server_config={"server": "search"},
)
print(registered)  # ['exa_search', 'web_fetch', ...]

# Or register specific tools only
registered = await branch.acts.register_mcp_server(
    server_config={"command": "python", "args": ["-m", "my_server"]},
    tool_names=["search", "fetch"],
)
```

### Inline MCP Config

Register a single MCP tool inline via dict config:

```python
branch = Branch(
    tools=[
        {"my_tool": {"command": "python", "args": ["-m", "server"]}}
    ]
)
```

The dict must have exactly one key (the tool name) mapping to the server config.

## Pydantic Validation for MCP Tools

Add request validation to MCP tools with `request_options`:

```python
from pydantic import BaseModel, Field

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    num_results: int = Field(default=5, ge=1, le=20)

tools = await load_mcp_tools(
    config_path=".mcp.json",
    server_names=["search"],
    request_options_map={
        "search": {"exa_search": SearchRequest}
    },
)
```

## Connection Pool

`MCPConnectionPool` manages MCP client connections:

- Connections are cached by server config and reused across calls
- Stale connections are automatically detected and replaced
- Use `await MCPConnectionPool.cleanup()` to close all connections
- The pool supports `async with` context management

## Requirements

MCP support requires the `fastmcp` package:

```bash
pip install fastmcp
```
