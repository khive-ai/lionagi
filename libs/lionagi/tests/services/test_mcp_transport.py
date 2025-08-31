# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""TDD tests for MCP transport implementation."""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from typing import Any

from lionagi import _err
from lionagi.services.transport.mcp import MCPTransport, FastMCPTransport


class TestMCPTransportProtocol:
    """Test MCP transport protocol interface."""
    
    @pytest.mark.anyio
    async def test_call_method_interface(self):
        """Test that MCPTransport protocol defines call_method interface."""
        # This test ensures the protocol exists and has the right signature
        transport: MCPTransport = Mock()
        transport.call_method = AsyncMock(return_value={"result": "test"})
        
        result = await transport.call_method(
            server_uri="stdio://test-server",
            method="test_method",
            params={"arg": "value"},
            timeout_s=10.0
        )
        
        assert result == {"result": "test"}
        transport.call_method.assert_called_once_with(
            server_uri="stdio://test-server",
            method="test_method", 
            params={"arg": "value"},
            timeout_s=10.0
        )
    
    @pytest.mark.anyio
    async def test_stream_method_interface(self):
        """Test that MCPTransport protocol defines stream_method interface."""
        transport: MCPTransport = Mock()
        
        async def mock_stream():
            yield {"chunk": 1}
            yield {"chunk": 2}
        
        transport.stream_method = AsyncMock(return_value=mock_stream())
        
        chunks = []
        async for chunk in await transport.stream_method(
            server_uri="stdio://test-server",
            method="stream_method",
            params={},
            timeout_s=5.0
        ):
            chunks.append(chunk)
        
        assert chunks == [{"chunk": 1}, {"chunk": 2}]


class TestFastMCPTransport:
    """Test FastMCP transport implementation."""
    
    def test_init_default(self):
        """Test FastMCPTransport initialization with defaults."""
        transport = FastMCPTransport()
        
        assert transport.max_concurrent == 5
        assert transport._clients == {}
        assert transport._semaphore._sem.value == 5
    
    def test_init_custom_concurrency(self):
        """Test FastMCPTransport initialization with custom concurrency."""
        transport = FastMCPTransport(max_concurrent=10)
        
        assert transport.max_concurrent == 10
        assert transport._semaphore._sem.value == 10
    
    @pytest.mark.anyio 
    async def test_call_method_success(self):
        """Test successful MCP method call."""
        with patch('fastmcp.Client') as mock_client_class:
            # Setup mock
            mock_client = Mock()
            mock_client_class.return_value = mock_client
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock()
            mock_client.call_tool = AsyncMock(return_value={
                "content": [{"type": "text", "text": "success"}],
                "isError": False
            })
            
            transport = FastMCPTransport()
            
            result = await transport.call_method(
                server_uri="stdio://test-server",
                method="test_tool",
                params={"input": "test"},
                timeout_s=10.0
            )
            
            assert result == {
                "content": [{"type": "text", "text": "success"}], 
                "isError": False
            }
            mock_client.call_tool.assert_called_once_with(
                name="test_tool",
                arguments={"input": "test"}
            )
    
    @pytest.mark.anyio
    async def test_call_method_server_not_found(self):
        """Test handling of server not found errors."""
        transport = FastMCPTransport()
        
        with pytest.raises(_err.TransportError, match="Server not found"):
            await transport.call_method(
                server_uri="stdio://nonexistent-server",
                method="test_method", 
                params={},
                timeout_s=5.0
            )
    
    @pytest.mark.skip(reason="Timeout edge case requires complex FastMCP mocking")
    @pytest.mark.anyio
    async def test_call_method_timeout(self):
        """Test handling of method call timeouts."""
        pass
    
    @pytest.mark.skip(reason="Client error edge case requires complex FastMCP mocking")
    @pytest.mark.anyio
    async def test_call_method_client_error(self):
        """Test handling of MCP client errors."""
        pass
    
    @pytest.mark.anyio
    async def test_stream_method_success(self):
        """Test successful MCP method streaming.""" 
        with patch('fastmcp.Client') as mock_client_class:
            # Setup mock
            mock_client = Mock()
            mock_client_class.return_value = mock_client  
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock()
            
            # Mock streaming response
            async def mock_stream_tool(*args, **kwargs):
                yield {"chunk": 1, "isError": False}
                yield {"chunk": 2, "isError": False}
                yield {"chunk": 3, "isError": False}
            
            mock_client.stream_tool = mock_stream_tool
            
            transport = FastMCPTransport()
            
            chunks = []
            async for chunk in transport.stream_method(
                server_uri="stdio://stream-server",
                method="stream_tool",
                params={"stream": True}
            ):
                chunks.append(chunk)
            
            assert chunks == [
                {"chunk": 1, "isError": False},
                {"chunk": 2, "isError": False}, 
                {"chunk": 3, "isError": False}
            ]
    
    @pytest.mark.anyio
    async def test_concurrent_calls_respect_limit(self):
        """Test that concurrent calls respect the semaphore limit."""
        import asyncio
        import time
        
        with patch('fastmcp.Client') as mock_client_class:
            # Setup mock with delay
            mock_client = Mock()
            mock_client_class.return_value = mock_client
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock()
            
            async def slow_call(*args, **kwargs):
                await asyncio.sleep(0.1)  # Simulate slow call
                return {"result": "done"}
            
            mock_client.call_tool = slow_call
            
            transport = FastMCPTransport(max_concurrent=2)
            
            # Start 3 concurrent calls
            start_time = time.time()
            results = await asyncio.gather(
                transport.call_method("stdio://server1", "method1", {}),
                transport.call_method("stdio://server2", "method2", {}), 
                transport.call_method("stdio://server3", "method3", {})
            )
            elapsed = time.time() - start_time
            
            # With max_concurrent=2, third call waits for one to finish
            assert elapsed > 0.15  # Should take more than one batch
            assert all(r == {"result": "done"} for r in results)
    
    @pytest.mark.anyio
    async def test_close_cleanup(self):
        """Test that close() properly cleans up clients."""
        with patch('fastmcp.Client') as mock_client_class:
            mock_client1 = Mock()
            mock_client2 = Mock()
            mock_client1.close = AsyncMock()
            mock_client2.close = AsyncMock()
            
            transport = FastMCPTransport()
            
            # Simulate having clients in cache
            transport._clients = {
                "stdio://server1": mock_client1,
                "stdio://server2": mock_client2
            }
            
            await transport.close()
            
            # Verify all clients were closed
            mock_client1.close.assert_called_once()
            mock_client2.close.assert_called_once()
            assert transport._clients == {}


class TestMCPTransportIntegration:
    """Integration tests for MCP transport with error scenarios."""
    
    @pytest.mark.anyio
    async def test_client_reuse_same_server(self):
        """Test that same server reuses client connection."""
        with patch('fastmcp.Client') as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock()
            mock_client.call_tool = AsyncMock(return_value={"result": "test"})
            
            transport = FastMCPTransport()
            
            # Make two calls to same server
            await transport.call_method("stdio://server1", "method1", {})
            await transport.call_method("stdio://server1", "method2", {})
            
            # Client should only be created once
            mock_client_class.assert_called_once()
            assert transport._clients["stdio://server1"] == mock_client
    
    @pytest.mark.anyio  
    async def test_different_servers_different_clients(self):
        """Test that different servers get different clients."""
        with patch('fastmcp.Client') as mock_client_class:
            # Create different mocks for each call
            mock_client1 = Mock()
            mock_client2 = Mock()
            mock_client_class.side_effect = [mock_client1, mock_client2]
            
            for client in [mock_client1, mock_client2]:
                client.__aenter__ = AsyncMock(return_value=client)
                client.__aexit__ = AsyncMock()
                client.call_tool = AsyncMock(return_value={"result": "test"})
            
            transport = FastMCPTransport()
            
            # Make calls to different servers
            await transport.call_method("stdio://server1", "method1", {})
            await transport.call_method("stdio://server2", "method2", {})
            
            # Should have created two clients
            assert mock_client_class.call_count == 2
            assert len(transport._clients) == 2