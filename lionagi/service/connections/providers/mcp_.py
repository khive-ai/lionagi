# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

from lionagi.service.connections.endpoint import Endpoint
from lionagi.service.connections.endpoint_config import EndpointConfig

# FastMCP imports - conditional import for graceful degradation
try:
    from fastmcp.client import Client as FastMCPClient
    from fastmcp.client.transports import (
        StdioTransport,
        SSETransport,
        StreamableHttpTransport,
    )
    FASTMCP_AVAILABLE = True
except ImportError:
    FASTMCP_AVAILABLE = False
    FastMCPClient = None
    StdioTransport = None
    SSETransport = None
    StreamableHttpTransport = None

logger = logging.getLogger(__name__)

# Default timeouts based on khive_mcp.py patterns
DEFAULT_TIMEOUT = 30.0
INIT_TIMEOUT = 5.0
CLEANUP_TIMEOUT = 2.0


@dataclass
class MCPServerConfig:
    """Configuration for an MCP server."""
    name: str
    command: str
    args: List[str] = None
    env: Dict[str, str] = None
    transport: str = "stdio"
    url: Optional[str] = None
    timeout: float = DEFAULT_TIMEOUT
    disabled: bool = False
    buffer_size: int = 65536

    def __post_init__(self):
        if self.args is None:
            self.args = []
        if self.env is None:
            self.env = {}


class MCPToolDescriptor(BaseModel):
    """Describes an MCP tool for lionagi integration."""
    name: str
    description: str
    server_name: str
    input_schema: Optional[Dict[str, Any]] = None
    parameters: List[str] = Field(default_factory=list)


class MCPToolRequest(BaseModel):
    """Request model for MCP tool execution."""
    server_name: str
    tool_name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)
    timeout: Optional[float] = None


class MCPTransportFactory:
    """Factory for creating MCP transport instances."""
    
    @staticmethod
    def create_transport(config: MCPServerConfig):
        """Create appropriate transport based on configuration."""
        if not FASTMCP_AVAILABLE:
            raise ImportError("FastMCP is not installed. Install with: pip install fastmcp")
        
        transport_type = config.transport.lower()
        
        if transport_type in ["http", "https", "sse"]:
            if not config.url:
                raise ValueError(f"URL required for {transport_type} transport")
            if transport_type in ["http", "https"]:
                return StreamableHttpTransport(config.url)
            else:
                return SSETransport(config.url)
        
        elif transport_type in ["stdio", "pipe"]:
            # Prepare environment
            env = os.environ.copy()
            env.update(config.env)
            env["PYTHONUNBUFFERED"] = "1"  # Force unbuffered output
            
            return StdioTransport(
                command=config.command,
                args=config.args,
                env=env,
            )
        
        else:
            raise ValueError(f"Unsupported transport type: {transport_type}")


class FastMCPClientManager:
    """Manages FastMCP client lifecycle within lionagi patterns."""
    
    def __init__(self, server_config: MCPServerConfig):
        self.config = server_config
        self._transport = None
        self._client = None
        
    @asynccontextmanager
    async def get_client(self):
        """Get a FastMCP client with proper lifecycle management."""
        if not FASTMCP_AVAILABLE:
            raise ImportError("FastMCP is not installed. Install with: pip install fastmcp")
        
        transport = MCPTransportFactory.create_transport(self.config)
        client = FastMCPClient(transport)
        
        connected = False
        try:
            # Initialize with timeout
            await asyncio.wait_for(client.__aenter__(), timeout=INIT_TIMEOUT)
            connected = True
            yield client
            
        except asyncio.TimeoutError:
            raise asyncio.TimeoutError(
                f"Failed to connect to {self.config.name} within {INIT_TIMEOUT}s"
            )
        except Exception as e:
            if connected:
                try:
                    await client.__aexit__(type(e), e, e.__traceback__)
                except Exception:
                    pass  # Ignore cleanup errors
            raise
        else:
            if connected:
                try:
                    await client.__aexit__(None, None, None)
                except Exception:
                    pass  # Ignore cleanup errors


class MCPToolDiscovery:
    """Discovers and converts MCP tools for lionagi integration."""
    
    @staticmethod
    async def discover_tools(
        client_manager: FastMCPClientManager
    ) -> List[MCPToolDescriptor]:
        """Discover tools from MCP server and convert to descriptors."""
        tools = []
        
        try:
            async with client_manager.get_client() as client:
                mcp_tools = await client.list_tools()
                
                for tool in mcp_tools:
                    # Extract parameters from schema if available
                    parameters = []
                    input_schema = None
                    
                    if hasattr(tool, 'inputSchema') and tool.inputSchema:
                        input_schema = tool.inputSchema
                        if isinstance(input_schema, dict) and "properties" in input_schema:
                            parameters = list(input_schema["properties"].keys())
                        elif hasattr(input_schema, "properties"):
                            parameters = list(input_schema.properties.keys())
                    
                    descriptor = MCPToolDescriptor(
                        name=f"{client_manager.config.name}::{tool.name}",
                        description=tool.description or f"MCP tool: {tool.name}",
                        server_name=client_manager.config.name,
                        input_schema=input_schema,
                        parameters=parameters
                    )
                    tools.append(descriptor)
                    
        except Exception as e:
            logger.error(f"Failed to discover tools from {client_manager.config.name}: {e}")
            raise
        
        return tools


# MCP Endpoint Configuration
MCP_ENDPOINT_CONFIG = EndpointConfig(
    name="mcp",
    provider="mcp",
    base_url="mcp://localhost",  # Virtual URL for MCP operations
    endpoint="tool",
    method="POST",
    openai_compatible=False,
    auth_type="none",
    api_key="mcp-internal",
    timeout=DEFAULT_TIMEOUT,
    request_options=MCPToolRequest,
)


class MCPEndpoint(Endpoint):
    """MCP endpoint for lionagi integration."""
    
    def __init__(
        self,
        config: EndpointConfig = MCP_ENDPOINT_CONFIG,
        servers: Optional[List[MCPServerConfig]] = None,
        **kwargs,
    ):
        """Initialize MCP endpoint.
        
        Args:
            config: Endpoint configuration
            servers: List of MCP server configurations
            **kwargs: Additional configuration
        """
        super().__init__(config, **kwargs)
        
        # Initialize server configurations
        self.servers = {}
        self.client_managers = {}
        self.discovered_tools = {}
        
        if servers:
            for server_config in servers:
                self.register_server(server_config)
    
    def register_server(self, server_config: MCPServerConfig):
        """Register an MCP server configuration."""
        if server_config.disabled:
            logger.info(f"Skipping disabled MCP server: {server_config.name}")
            return
        
        self.servers[server_config.name] = server_config
        self.client_managers[server_config.name] = FastMCPClientManager(server_config)
        logger.info(f"Registered MCP server: {server_config.name}")
    
    async def discover_tools(self, server_name: Optional[str] = None) -> List[MCPToolDescriptor]:
        """Discover tools from MCP servers.
        
        Args:
            server_name: Specific server to discover from, or None for all servers
            
        Returns:
            List of discovered tool descriptors
        """
        if not FASTMCP_AVAILABLE:
            raise ImportError("FastMCP is not installed. Install with: pip install fastmcp")
        
        all_tools = []
        servers_to_check = [server_name] if server_name else list(self.servers.keys())
        
        for name in servers_to_check:
            if name not in self.client_managers:
                logger.warning(f"Unknown MCP server: {name}")
                continue
            
            try:
                client_manager = self.client_managers[name]
                tools = await MCPToolDiscovery.discover_tools(client_manager)
                self.discovered_tools[name] = tools
                all_tools.extend(tools)
                logger.info(f"Discovered {len(tools)} tools from {name}")
                
            except Exception as e:
                logger.error(f"Failed to discover tools from {name}: {e}")
                continue
        
        return all_tools
    
    def create_payload(
        self,
        request: Union[dict, BaseModel],
        extra_headers: Optional[dict] = None,
        **kwargs,
    ):
        """Create payload for MCP tool execution."""
        
        # Convert request to dict if BaseModel
        if isinstance(request, BaseModel):
            request_dict = request.model_dump(exclude_none=True)
        else:
            request_dict = request.copy()
        
        # Ensure required fields are present
        if "server_name" not in request_dict:
            raise ValueError("server_name is required for MCP tool execution")
        
        if "tool_name" not in request_dict:
            raise ValueError("tool_name is required for MCP tool execution")
        
        # Validate server exists
        server_name = request_dict["server_name"]
        if server_name not in self.servers:
            raise ValueError(f"Unknown MCP server: {server_name}")
        
        # Set default timeout if not provided
        if "timeout" not in request_dict:
            request_dict["timeout"] = self.servers[server_name].timeout
        
        # Create headers (minimal for MCP)
        headers = extra_headers or {}
        headers.update({
            "Content-Type": "application/json",
            "MCP-Server": server_name,
        })
        
        return (request_dict, headers)
    
    async def _call_aiohttp(self, payload: dict, headers: dict, **kwargs):
        """Execute MCP tool call (overrides base implementation)."""
        if not FASTMCP_AVAILABLE:
            raise ImportError("FastMCP is not installed. Install with: pip install fastmcp")
        
        server_name = payload["server_name"]
        tool_name = payload["tool_name"]
        arguments = payload.get("arguments", {})
        timeout = payload.get("timeout", DEFAULT_TIMEOUT)
        
        if server_name not in self.client_managers:
            raise ValueError(f"Unknown MCP server: {server_name}")
        
        client_manager = self.client_managers[server_name]
        
        try:
            async with asyncio.timeout(timeout):
                async with client_manager.get_client() as client:
                    # Execute the tool
                    result = await client.call_tool(tool_name, arguments)
                    
                    # Format result for lionagi patterns
                    if isinstance(result, list):
                        # Handle MCP content format
                        formatted = []
                        for item in result:
                            if isinstance(item, dict) and item.get("type") == "text":
                                formatted.append(item.get("text", ""))
                            else:
                                formatted.append(item)
                        return {
                            "success": True,
                            "result": formatted,
                            "server": server_name,
                            "tool": tool_name,
                        }
                    else:
                        return {
                            "success": True,
                            "result": result,
                            "server": server_name,
                            "tool": tool_name,
                        }
        
        except asyncio.TimeoutError:
            raise asyncio.TimeoutError(
                f"MCP tool '{tool_name}' on server '{server_name}' timed out after {timeout}s"
            )
        except Exception as e:
            logger.error(f"MCP tool execution failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__,
                "server": server_name,
                "tool": tool_name,
            }
    
    async def list_server_tools(self, server_name: str) -> List[MCPToolDescriptor]:
        """List tools available on a specific server."""
        if server_name not in self.discovered_tools:
            # Discover tools if not cached
            await self.discover_tools(server_name)
        
        return self.discovered_tools.get(server_name, [])
    
    def get_server_status(self, server_name: str) -> dict:
        """Get status information for a server."""
        if server_name not in self.servers:
            return {"status": "unknown", "error": "Server not registered"}
        
        config = self.servers[server_name]
        return {
            "status": "disabled" if config.disabled else "registered",
            "name": server_name,
            "command": config.command,
            "transport": config.transport,
            "timeout": config.timeout,
            "tools_discovered": len(self.discovered_tools.get(server_name, [])),
        }
    
    @classmethod
    def from_config_file(cls, config_path: Union[str, Path], **kwargs) -> "MCPEndpoint":
        """Create MCPEndpoint from khive-style configuration file.
        
        Args:
            config_path: Path to MCP configuration file
            **kwargs: Additional endpoint configuration
            
        Returns:
            Configured MCPEndpoint instance
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"MCP config file not found: {config_path}")
        
        try:
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            
            servers = []
            mcp_servers = config_data.get("mcpServers", {})
            
            for server_name, server_config in mcp_servers.items():
                # Detect transport type
                transport = "stdio"
                url = None
                
                if "url" in server_config:
                    url = server_config["url"]
                    transport = "sse" if url.startswith(("http://", "https://")) else "stdio"
                elif "transport" in server_config:
                    transport = server_config["transport"]
                
                mcp_config = MCPServerConfig(
                    name=server_name,
                    command=server_config.get("command", ""),
                    args=server_config.get("args", []),
                    env=server_config.get("env", {}),
                    transport=transport,
                    url=url,
                    timeout=server_config.get("timeout", DEFAULT_TIMEOUT),
                    disabled=server_config.get("disabled", False),
                )
                servers.append(mcp_config)
            
            return cls(servers=servers, **kwargs)
            
        except (json.JSONDecodeError, KeyError) as e:
            raise ValueError(f"Invalid MCP configuration file: {e}")


__all__ = [
    "MCPEndpoint",
    "MCPServerConfig", 
    "MCPToolDescriptor",
    "MCPToolRequest",
    "MCP_ENDPOINT_CONFIG",
]
