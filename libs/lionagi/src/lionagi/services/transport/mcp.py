# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""MCP transport for Model Context Protocol communication using FastMCP."""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from typing import Any, Protocol
from urllib.parse import urlparse

from lionagi import _err, ln

HAS_FASTMCP = ln.is_import_installed("fastmcp")


class MCPTransport(Protocol):
    """IO boundary for MCP server communication.
    
    Handles Model Context Protocol server connections, tool invocations,
    and resource access through a standardized interface.
    """
    
    async def call_method(
        self,
        server_uri: str,
        method: str,
        params: dict[str, Any],
        *,
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        """Call MCP server method and return result.
        
        Args:
            server_uri: MCP server URI (e.g., "stdio://server-name")
            method: Method/tool name to invoke
            params: Parameters to pass to method
            timeout_s: Optional timeout in seconds
            
        Returns:
            dict: Method result
            
        Raises:
            TransportError: If method call fails or server not found
            TimeoutError: If call exceeds timeout
        """
        ...
    
    async def stream_method(
        self,
        server_uri: str,
        method: str,
        params: dict[str, Any],
        *,
        timeout_s: float | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Call MCP server method and stream results.
        
        Args:
            server_uri: MCP server URI (e.g., "stdio://server-name")
            method: Method/tool name to invoke
            params: Parameters to pass to method
            timeout_s: Optional timeout in seconds
            
        Yields:
            dict: Streamed results from method
            
        Raises:
            TransportError: If method call fails or server not found
            TimeoutError: If call exceeds timeout
        """
        ...


class FastMCPTransport:
    """FastMCP-based transport implementation.
    
    Features:
    - Client connection pooling and reuse
    - Concurrent execution limits via semaphore
    - Proper error mapping to lionagi exceptions
    - Timeout enforcement
    - Resource cleanup on close
    """
    
    def __init__(self, *, max_concurrent: int = 5):
        """Initialize FastMCP transport.
        
        Args:
            max_concurrent: Maximum number of concurrent MCP operations
            
        Raises:
            TransportError: If FastMCP is not installed
        """
        if not HAS_FASTMCP:
            raise _err.TransportError(
                "FastMCP not installed. Install with: pip install fastmcp"
            )
        
        self.max_concurrent = max_concurrent
        self._semaphore = ln.Semaphore(max_concurrent)
        self._clients: dict = {}
    
    async def _get_client(self, server_uri: str):
        """Get or create MCP client for server URI.
        
        Args:
            server_uri: MCP server URI
            
        Returns:
            fastmcp.Client: Client instance
            
        Raises:
            TransportError: If server URI is invalid or server not found
        """
        import fastmcp

        if server_uri in self._clients:
            return self._clients[server_uri]
        
        # Parse server URI to determine transport type
        parsed = urlparse(server_uri)
        
        if parsed.scheme == "stdio":
            # stdio://server-name format
            server_name = parsed.netloc or parsed.path.lstrip('/')
            if not server_name:
                raise _err.TransportError(
                    f"Invalid stdio URI: {server_uri}",
                    context={"server_uri": server_uri}
                )
            
            try:
                # Create stdio client - FastMCP will look for server in MCP config
                client = fastmcp.Client(transport=f"stdio://{server_name}")
                self._clients[server_uri] = client
                return client
                
            except Exception as e:
                raise _err.TransportError(
                    f"Server not found: {server_name}",
                    context={"server_uri": server_uri, "server_name": server_name},
                    cause=e
                )
                
        elif parsed.scheme in ("http", "https"):
            # HTTP transport
            try:
                client = fastmcp.Client(transport=server_uri)
                self._clients[server_uri] = client
                return client
                
            except Exception as e:
                raise _err.TransportError(
                    f"Failed to connect to HTTP MCP server: {server_uri}",
                    context={"server_uri": server_uri},
                    cause=e
                )
        else:
            raise _err.TransportError(
                f"Unsupported MCP transport scheme: {parsed.scheme}",
                context={"server_uri": server_uri, "scheme": parsed.scheme}
            )
    
    async def call_method(
        self,
        server_uri: str,
        method: str,
        params: dict[str, Any],
        *,
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        """Call MCP server method and return result."""
        async with self._semaphore:
            try:
                client = await self._get_client(server_uri)
                
                # Use client context manager for proper connection handling
                async with client:
                    if timeout_s:
                        # Apply timeout using lionagi's fail_after
                        with ln.fail_after(timeout_s):
                            result = await client.call_tool(name=method, arguments=params)
                    else:
                        result = await client.call_tool(name=method, arguments=params)
                    
                    return result
                    
            except _err.LionError:
                # Re-raise lionagi errors as-is
                raise
            except ln.get_cancelled_exc_class() as e:
                # Handle lionagi cancellation (timeout) exceptions
                raise _err.TimeoutError(
                    f"MCP call timed out",
                    context={
                        "server_uri": server_uri,
                        "method": method,
                        "timeout": timeout_s
                    },
                    cause=e
                )
            except Exception as e:
                raise _err.TransportError(
                    f"MCP call failed: {e}",
                    context={
                        "server_uri": server_uri,
                        "method": method,
                        "params": params
                    },
                    cause=e
                )
    
    async def stream_method(
        self,
        server_uri: str,
        method: str,
        params: dict[str, Any],
        *,
        timeout_s: float | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream MCP server method results."""
        async with self._semaphore:
            try:
                client = await self._get_client(server_uri)
                
                start_time = time.time()
                
                async with client:
                    # Check if client supports streaming
                    if not hasattr(client, 'stream_tool'):
                        # Fallback to single call for non-streaming tools
                        result = await self.call_method(
                            server_uri, method, params, timeout_s=timeout_s
                        )
                        yield result
                        return
                    
                    # Stream results
                    async for chunk in client.stream_tool(name=method, arguments=params):
                        # Check timeout on each chunk
                        if timeout_s and (time.time() - start_time) > timeout_s:
                            raise _err.TimeoutError(
                                f"MCP stream timed out after {timeout_s}s",
                                context={
                                    "server_uri": server_uri,
                                    "method": method,
                                    "timeout": timeout_s
                                }
                            )
                        
                        yield chunk
                        
            except _err.LionError:
                # Re-raise lionagi errors as-is
                raise
            except Exception as e:
                raise _err.TransportError(
                    f"MCP stream failed: {e}",
                    context={
                        "server_uri": server_uri,
                        "method": method,
                        "params": params
                    },
                    cause=e
                )
    
    async def close(self) -> None:
        """Close all MCP client connections and cleanup."""
        for client in list(self._clients.values()):
            try:
                if hasattr(client, 'close'):
                    await client.close()
            except Exception:
                # Ignore cleanup errors
                pass
        
        self._clients.clear()
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, _exc_type, _exc_val, _exc_tb):
        """Async context manager exit with cleanup."""
        await self.close()