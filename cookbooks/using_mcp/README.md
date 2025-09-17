# Using MCP (Model Context Protocol) with LionAGI

This cookbook demonstrates how to integrate MCP servers with LionAGI for dynamic
tool discovery and execution.

## What is MCP?

The Model Context Protocol (MCP) is an open standard developed by Anthropic that
enables seamless integration between AI models and external tools/data sources.
It provides a unified protocol for:

- Dynamic tool discovery
- Standardized tool execution
- Connection pooling for efficient resource management

## Installation

```bash
pip install "lionagi[mcp]" khivemcp
```

## Quick Start

### 1. Configure Your MCP Server

Create a `.mcp.json` file to define your MCP servers:

```json
{
  "mcpServers": {
    "search": {
      "command": "python",
      "args": ["-m", "khivemcp.cli", "search_group_config.json"],
      "timeout": 300
    }
  }
}
```

### 2. Define Request Wrappers

MCP tools expect requests wrapped in a specific format. Create Pydantic models
to handle this:

```python
from pydantic import BaseModel
from lionagi.service.third_party.exa_models import ExaSearchRequest
from lionagi.service.third_party.pplx_models import PerplexityChatRequest

class ExaRequest(BaseModel):
    request: ExaSearchRequest

class PerplexityRequest(BaseModel):
    request: PerplexityChatRequest
```

### 3. Load and Use MCP Tools

```python
from lionagi import Branch, iModel
from lionagi.protocols.action.manager import load_mcp_tools

# Load tools with Pydantic validation
tools = await load_mcp_tools(
    ".mcp.json",
    server_names=["search"],
    request_options_map={
        "search": {
            "exa_search": ExaRequest,
            "perplexity_search": PerplexityRequest,
        }
    }
)

# Use with Branch for ReAct reasoning
branch = Branch(
    chat_model=iModel(provider="openai", model="gpt-4.1-mini"),
    tools=tools
)

result = await branch.ReAct(
    instruct={
        "instruction": "Research recent AI developments",
        "context": {}
    },
    tools=["search_exa_search"],
    max_extensions=3
)
```

## Files in This Cookbook

- **`.mcp.json`** - MCP server configuration
- **`search_group_config.json`** - Configuration for the search service group
- **`search_group.py`** - MCP server implementation with Exa and Perplexity
  search
- **`react_mcp_with_schema.py`** - Complete example using MCP tools with ReAct
  reasoning

## Key Features

### Dynamic Tool Discovery

MCP servers can be queried for available tools at runtime:

```python
from lionagi.protocols.action.manager import ActionManager

manager = ActionManager()
tools = await manager.register_mcp_server(
    {"server": "search"},
    update=True
)
```

### Connection Pooling

Efficient connection reuse across multiple tool invocations:

```python
from lionagi.service.connections.mcp.wrapper import MCPConnectionPool

# Connections are automatically pooled and reused
MCPConnectionPool.load_config(".mcp.json")
client = await MCPConnectionPool.get_client({"server": "search"})
```

### Type Safety with Pydantic

Full request/response validation using Pydantic models ensures type safety
throughout the pipeline.

## Advanced Configuration

### Environment Variables

Control MCP server logging and debugging:

```bash
export MCP_DEBUG=true           # Enable debug mode
export LOG_LEVEL=ERROR          # Set logging level
export FASTMCP_QUIET=true       # Suppress FastMCP output
```

### Custom MCP Servers

Create your own MCP server by extending `ServiceGroup`:

```python
from khivemcp import ServiceGroup, operation

class MyServiceGroup(ServiceGroup):
    @operation(name="my_tool", schema=MyRequestModel)
    async def my_tool(self, request: MyRequestModel):
        # Tool implementation
        return {"result": "success"}
```

## Troubleshooting

### Common Issues

1. **Schema Mismatch**: Ensure your wrapper models correctly wrap the request
   with a `request` field
2. **Server Not Found**: Check that your `.mcp.json` paths are correct
3. **Verbose Logging**: Set environment variables to suppress server output

### Debug Mode

Enable debug logging to troubleshoot issues:

```python
tools = await load_mcp_tools(
    ".mcp.json",
    server_names=["search"],
    debug=True  # Enable debug mode
)
```

## Resources

- [MCP Specification](https://modelcontextprotocol.io)
- [FastMCP Documentation](https://gofastmcp.com)
- [KhiveMCP](https://github.com/khive-ai/khivemcp)
- [LionAGI Documentation](https://github.com/lion-agi/lionagi)

## License

This cookbook is part of the LionAGI project and follows the same Apache 2.0
license.
