# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict

try:
    from fastmcp import Client as FastMCPClient

    FASTMCP_AVAILABLE = True
except ImportError:
    FASTMCP_AVAILABLE = False
    FastMCPClient = None

# Suppress MCP server logging by default
logging.getLogger("mcp").setLevel(logging.WARNING)
logging.getLogger("fastmcp").setLevel(logging.WARNING)


class MCPConnectionPool:
    """Simple connection pool for MCP clients."""

    _clients: Dict[str, Any] = {}
    _configs: Dict[str, Dict] = {}
    _lock = asyncio.Lock()

    @classmethod
    def load_config(cls, path: str = ".mcp.json") -> None:
        """Load MCP server configurations from file."""
        config_path = Path(path)
        if config_path.exists():
            with open(config_path) as f:
                data = json.load(f)
                cls._configs.update(data.get("mcpServers", {}))

    @classmethod
    async def get_client(cls, server_config: Dict[str, Any]) -> Any:
        """Get or create a pooled MCP client."""
        if not FASTMCP_AVAILABLE:
            raise ImportError(
                "FastMCP not installed. Run: pip install fastmcp"
            )

        # Generate unique key for this config
        if "server" in server_config:
            # Server reference from .mcp.json
            server_name = server_config["server"]
            if server_name not in cls._configs:
                # Try loading config
                cls.load_config()
                if server_name not in cls._configs:
                    raise ValueError(f"Unknown MCP server: {server_name}")

            config = cls._configs[server_name]
            cache_key = f"server:{server_name}"
        else:
            # Inline config - use command as key
            config = server_config
            cache_key = f"inline:{config.get('command')}:{id(config)}"

        # Check if client exists and is connected
        async with cls._lock:
            if cache_key in cls._clients:
                client = cls._clients[cache_key]
                # Simple connectivity check
                if hasattr(client, "is_connected") and client.is_connected():
                    return client
                else:
                    # Remove stale client
                    del cls._clients[cache_key]

            # Create new client
            client = await cls._create_client(config)
            cls._clients[cache_key] = client
            return client

    @classmethod
    async def _create_client(cls, config: Dict[str, Any]) -> Any:
        """Create a new MCP client from config."""
        import subprocess

        # Handle different config formats
        if "url" in config:
            # Direct URL connection
            client = FastMCPClient(config["url"])
        elif "command" in config:
            # Command-based connection
            env = os.environ.copy()
            env.update(config.get("env", {}))

            # Create client with command
            from fastmcp.client.transports import StdioTransport

            # Suppress stderr by default unless debug mode is explicitly set
            stderr_mode = subprocess.DEVNULL
            if (
                config.get("debug", False)
                or os.environ.get("MCP_DEBUG", "").lower() == "true"
            ):
                stderr_mode = None  # Use default stderr

            transport = StdioTransport(
                command=config["command"],
                args=config.get("args", []),
                env=env,
                stderr=stderr_mode,
            )
            client = FastMCPClient(transport)
        else:
            raise ValueError("Config must have 'url' or 'command'")

        # Initialize connection
        await client.__aenter__()
        return client

    @classmethod
    async def cleanup(cls):
        """Clean up all pooled connections."""
        async with cls._lock:
            for client in cls._clients.values():
                try:
                    await client.__aexit__(None, None, None)
                except:
                    pass  # Ignore cleanup errors
            cls._clients.clear()


def create_mcp_tool(mcp_config: Dict[str, Any], tool_name: str) -> Any:
    """Create a callable that wraps MCP tool execution.

    Args:
        mcp_config: MCP server configuration (server reference or inline)
        tool_name: Name of the tool (can be qualified like "server_toolname")

    Returns:
        Async callable that executes the MCP tool
    """

    async def mcp_callable(**kwargs):
        """Execute MCP tool with connection pooling."""
        # Extract the original tool name if it was stored in metadata
        actual_tool_name = mcp_config.get("_original_tool_name", tool_name)

        # Remove metadata before getting client
        config_for_client = {
            k: v for k, v in mcp_config.items() if not k.startswith("_")
        }

        client = await MCPConnectionPool.get_client(config_for_client)

        # MCP tools expect arguments wrapped in "request"
        # If we have Pydantic validation, the kwargs are already validated
        # but need to be wrapped for the MCP call
        if not "request" in kwargs:
            kwargs = {"request": kwargs}

        # Call the tool with the original name
        result = await client.call_tool(actual_tool_name, kwargs)

        # Handle different result types
        if hasattr(result, "content"):
            # CallToolResult object - extract content
            content = result.content
            if isinstance(content, list) and len(content) == 1:
                item = content[0]
                if hasattr(item, "text"):
                    return item.text
                elif isinstance(item, dict) and item.get("type") == "text":
                    return item.get("text", "")
            return content
        elif isinstance(result, list) and len(result) == 1:
            item = result[0]
            if isinstance(item, dict) and item.get("type") == "text":
                return item.get("text", "")

        return result

    # Set function metadata for Tool introspection
    mcp_callable.__name__ = tool_name
    mcp_callable.__doc__ = f"MCP tool: {tool_name}"

    return mcp_callable
