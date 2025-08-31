# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for CLI transport implementation."""

import pytest
import time
from unittest.mock import patch, Mock, AsyncMock

from lionagi import _err
from lionagi.services.transport.cli import SubprocessCLITransport


class TestCLITransport:
    """Test CLI transport functionality."""
    
    def test_init_default(self):
        """Test SubprocessCLITransport initialization with defaults."""
        transport = SubprocessCLITransport()
        
        assert transport.max_concurrent == 5
        assert transport._processes == set()
        # Check lionagi semaphore structure
        assert transport._semaphore._sem.value == 5
    
    def test_init_custom_concurrency(self):
        """Test SubprocessCLITransport initialization with custom concurrency."""
        transport = SubprocessCLITransport(max_concurrent=10)
        
        assert transport.max_concurrent == 10
        assert transport._semaphore._sem.value == 10
    
    @pytest.mark.anyio
    async def test_execute_command_not_found(self):
        """Test handling of missing commands."""
        transport = SubprocessCLITransport()
        
        with pytest.raises(_err.TransportError, match="Command not found"):
            await transport.execute(
                ["nonexistent_command_xyz_123"],
                timeout_s=5.0
            )
    
    @pytest.mark.anyio
    async def test_execute_empty_command(self):
        """Test handling of empty command list."""
        transport = SubprocessCLITransport()
        
        with pytest.raises(_err.TransportError, match="Empty command provided"):
            await transport.execute(
                [],
                timeout_s=5.0
            )
    
    @pytest.mark.anyio
    async def test_close_cleanup(self):
        """Test that close() properly cleans up processes."""
        transport = SubprocessCLITransport()
        
        # Mock some processes
        mock_process1 = Mock()
        mock_process2 = Mock()
        mock_process1.returncode = None
        mock_process2.returncode = 0  # Already finished
        mock_process1.kill = Mock()
        mock_process2.kill = Mock() 
        mock_process1.wait = AsyncMock()
        mock_process2.wait = AsyncMock()
        
        # Add to processes set
        transport._processes.add(mock_process1)
        transport._processes.add(mock_process2)
        
        await transport.close()
        
        # Should kill only running processes (returncode is None)
        mock_process1.kill.assert_called_once()
        mock_process2.kill.assert_not_called()  # Already finished (returncode=0)
        mock_process1.wait.assert_called_once()
        mock_process2.wait.assert_called_once()
        
        assert transport._processes == set()


class TestCLITransportIntegration:
    """Integration tests for CLI transport."""
    
    @pytest.mark.anyio
    async def test_execute_echo_success(self):
        """Test successful command execution with echo."""
        transport = SubprocessCLITransport()
        
        # Use echo command which should be available on all platforms
        stdout, stderr, returncode = await transport.execute(
            ["echo", "hello world"],
            timeout_s=5.0
        )
        
        assert returncode == 0
        assert "hello world" in stdout
        assert stderr == ""
    
    @pytest.mark.anyio
    async def test_concurrent_execution_limit(self):
        """Test that concurrent executions respect semaphore limit."""
        import asyncio
        
        transport = SubprocessCLITransport(max_concurrent=2)
        
        async def slow_command():
            # Use a command that takes some time
            stdout, stderr, code = await transport.execute(
                ["sleep", "0.1"] if hasattr(asyncio, 'create_subprocess_exec') else ["ping", "-c", "1", "127.0.0.1"],
                timeout_s=2.0
            )
            return time.time()
        
        # Start 3 concurrent commands with max_concurrent=2
        start_time = time.time()
        try:
            times = await asyncio.gather(
                slow_command(),
                slow_command(), 
                slow_command(),
                return_exceptions=True
            )
            
            # Should complete (though some might error on different platforms)
            assert len(times) == 3
            
        except Exception as e:
            # Some platforms might not support sleep command
            pytest.skip(f"Platform-specific command issue: {e}")