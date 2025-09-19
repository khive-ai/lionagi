# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any

# Suppress MCP server logging by default
logging.getLogger("mcp").setLevel(logging.WARNING)
logging.getLogger("fastmcp").setLevel(logging.WARNING)
logging.getLogger("mcp.server").setLevel(logging.WARNING)
logging.getLogger("mcp.server.lowlevel").setLevel(logging.WARNING)
logging.getLogger("mcp.server.lowlevel.server").setLevel(logging.WARNING)


class MCPConnectionPool:
    """Simple connection pool for MCP clients.

    Security Model:
    This class trusts user-provided MCP server configurations, similar to how
    development tools trust configured language servers or extensions. Users are
    responsible for vetting the MCP servers they choose to run.

    For enhanced security in production:
    - Run MCP servers in sandboxed environments (containers, VMs)
    - Use process isolation and resource limits
    - Monitor server behavior and resource usage
    - Validate server outputs before use
    """

    _clients: dict[str, Any] = {}
    _configs: dict[str, dict] = {}
    _lock = asyncio.Lock()

    async def __aenter__(self):
        """Context manager entry."""
        return self

    async def __aexit__(self, *_):
        """Context manager exit - cleanup connections."""
        await self.cleanup()

    @classmethod
    def load_config(cls, path: str = ".mcp.json") -> None:
        """Load MCP server configurations from file.

        Args:
            path: Path to .mcp.json configuration file

        Raises:
            FileNotFoundError: If config file doesn't exist
            json.JSONDecodeError: If config file has invalid JSON
            ValueError: If config structure is invalid
        """
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"MCP config file not found: {path}")

        try:
            with open(config_path) as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(
                f"Invalid JSON in MCP config file: {e.msg}", e.doc, e.pos
            )

        if not isinstance(data, dict):
            raise ValueError("MCP config must be a JSON object")

        servers = data.get("mcpServers", {})
        if not isinstance(servers, dict):
            raise ValueError("mcpServers must be a dictionary")

        cls._configs.update(servers)

    @classmethod
    async def get_client(cls, server_config: dict[str, Any]) -> Any:
        """Get or create a pooled MCP client."""
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
    async def _create_client(cls, config: dict[str, Any]) -> Any:
        """Create a new MCP client from config.

        Security Note:
        MCP servers are explicitly configured by users via .mcp.json files or API calls.
        The security model trusts user-provided configurations, similar to how IDEs trust
        configured language servers. For additional security, run MCP servers in sandboxed
        environments (containers, VMs) rather than restricting commands at the library level.

        Args:
            config: Server configuration with 'url' or 'command' + optional 'args' and 'env'

        Raises:
            ValueError: If config format is invalid
        """
        # Validate config structure
        if not isinstance(config, dict):
            raise ValueError("Config must be a dictionary")

        if not any(k in config for k in ["url", "command"]):
            raise ValueError("Config must have either 'url' or 'command' key")

        try:
            from fastmcp import Client as FastMCPClient
        except ImportError:
            raise ImportError(
                "FastMCP not installed. Run: pip install fastmcp"
            )

        # Handle different config formats
        if "url" in config:
            # Direct URL connection
            client = FastMCPClient(config["url"])
        elif "command" in config:
            # Command-based connection
            # Validate args if provided
            args = config.get("args", [])
            if not isinstance(args, list):
                raise ValueError("Config 'args' must be a list")

            # Merge environment variables - user config takes precedence
            env = os.environ.copy()
            env.update(config.get("env", {}))

            # Suppress server logging unless debug mode is enabled
            if not (
                config.get("debug", False)
                or os.environ.get("MCP_DEBUG", "").lower() == "true"
            ):
                # Common environment variables to suppress logging
                env.setdefault("LOG_LEVEL", "ERROR")
                env.setdefault("PYTHONWARNINGS", "ignore")
                # Suppress FastMCP server logs
                env.setdefault("FASTMCP_QUIET", "true")
                env.setdefault("MCP_QUIET", "true")

            # Create client with command
            from fastmcp.client.transports import StdioTransport

            transport = StdioTransport(
                command=config["command"],
                args=args,
                env=env,
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
            for cache_key, client in cls._clients.items():
                try:
                    await client.__aexit__(None, None, None)
                except Exception as e:
                    # Log cleanup errors for debugging while continuing cleanup
                    logging.debug(
                        f"Error cleaning up MCP client {cache_key}: {e}"
                    )
            cls._clients.clear()


def create_mcp_tool(mcp_config: dict[str, Any], tool_name: str) -> Any:
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
