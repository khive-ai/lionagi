# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""CLI transport for subprocess-based communication."""

from __future__ import annotations

import asyncio
import shutil
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Protocol

from lionagi import _err, ln


class CLITransport(Protocol):
    """IO boundary for CLI commands.
    
    Handles subprocess execution, streaming, and error mapping.
    Provides secure subprocess management with timeout enforcement.
    """
    
    async def execute(
        self,
        command: list[str],
        *,
        stdin: str | bytes | None = None,
        env: dict[str, str] | None = None,
        cwd: str | Path | None = None,
        timeout_s: float | None = None,
    ) -> tuple[str, str, int]:
        """Execute CLI command and return output.
        
        Args:
            command: Command and arguments as list (safe from injection)
            stdin: Input data to send to command
            env: Environment variables for command
            cwd: Working directory for command execution
            timeout_s: Maximum execution time in seconds
        
        Returns:
            tuple: (stdout, stderr, returncode)
            
        Raises:
            TransportError: If command not found or execution fails
            TimeoutError: If command exceeds timeout
        """
        ...
    
    async def stream(
        self,
        command: list[str],
        *,
        stdin: str | bytes | None = None,
        env: dict[str, str] | None = None,
        cwd: str | Path | None = None,
        timeout_s: float | None = None,
    ) -> AsyncIterator[str]:
        """Execute CLI command and stream output lines.
        
        Args:
            command: Command and arguments as list (safe from injection)
            stdin: Input data to send to command  
            env: Environment variables for command
            cwd: Working directory for command execution
            timeout_s: Maximum execution time in seconds
            
        Yields:
            str: Output lines from command
            
        Raises:
            TransportError: If command not found or execution fails
            TimeoutError: If command exceeds timeout
        """
        ...


class SubprocessCLITransport:
    """Concrete CLI transport using asyncio subprocess.
    
    Features:
    - Concurrent execution limits via semaphore
    - Process cleanup on timeout/error
    - Secure command validation
    - UTF-8 handling with error recovery
    """
    
    def __init__(self, *, max_concurrent: int = 5):
        """Initialize CLI transport.
        
        Args:
            max_concurrent: Maximum number of concurrent subprocesses
        """
        self.max_concurrent = max_concurrent
        self._semaphore = ln.Semaphore(max_concurrent)
        self._processes: set[asyncio.subprocess.Process] = set()
    
    async def execute(
        self,
        command: list[str],
        *,
        stdin: str | bytes | None = None,
        env: dict[str, str] | None = None,
        cwd: str | Path | None = None,
        timeout_s: float | None = None,
    ) -> tuple[str, str, int]:
        """Execute CLI command with timeout and error handling."""
        
        # Validate command exists
        if not command:
            raise _err.TransportError("Empty command provided")
            
        if not shutil.which(command[0]):
            raise _err.TransportError(
                f"Command not found: {command[0]}",
                context={"command": command[0], "PATH": shutil.which(command[0])}
            )
        
        async with self._semaphore:
            start_time = time.perf_counter()
            
            # Prepare stdin
            stdin_data = None
            if stdin:
                stdin_data = stdin.encode('utf-8') if isinstance(stdin, str) else stdin
            
            # Create subprocess with secure settings
            try:
                process = await asyncio.create_subprocess_exec(
                    *command,
                    stdin=asyncio.subprocess.PIPE if stdin_data else None,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env=env,
                    cwd=str(cwd) if cwd else None,
                    # Security: Never use shell=True
                )
            except (OSError, ValueError) as e:
                raise _err.TransportError(
                    f"Failed to start command: {e}",
                    context={"command": command[0], "cwd": str(cwd) if cwd else None},
                    cause=e,
                )
            
            self._processes.add(process)
            
            try:
                # Execute with timeout
                if timeout_s:
                    with ln.fail_after(timeout_s):
                        stdout, stderr = await process.communicate(input=stdin_data)
                else:
                    stdout, stderr = await process.communicate(input=stdin_data)
                    
            except ln.get_cancelled_exc_class():
                # Handle timeout/cancellation - kill process and re-raise as TimeoutError
                if process.returncode is None:
                    process.kill()
                    await process.wait()
                elapsed = time.perf_counter() - start_time
                raise _err.TimeoutError(
                    f"CLI command timed out after {elapsed:.2f}s",
                    context={
                        "command": command[0], 
                        "timeout": timeout_s,
                        "elapsed": elapsed
                    }
                )
            except Exception:
                # Cleanup process on any other error
                if process.returncode is None:
                    process.kill()
                    await process.wait()
                raise
            finally:
                self._processes.discard(process)
            
            # Decode output with error handling
            try:
                stdout_str = stdout.decode('utf-8') if stdout else ""
            except UnicodeDecodeError:
                stdout_str = stdout.decode('utf-8', errors='replace') if stdout else ""
                
            try:
                stderr_str = stderr.decode('utf-8') if stderr else ""  
            except UnicodeDecodeError:
                stderr_str = stderr.decode('utf-8', errors='replace') if stderr else ""
            
            return stdout_str, stderr_str, process.returncode
    
    async def stream(
        self,
        command: list[str],
        *,
        stdin: str | bytes | None = None,
        env: dict[str, str] | None = None,
        cwd: str | Path | None = None,
        timeout_s: float | None = None,
    ) -> AsyncIterator[str]:
        """Stream CLI output line by line."""
        
        # Validate command
        if not command:
            raise _err.TransportError("Empty command provided")
            
        if not shutil.which(command[0]):
            raise _err.TransportError(
                f"Command not found: {command[0]}",
                context={"command": command[0]}
            )
        
        async with self._semaphore:
            try:
                process = await asyncio.create_subprocess_exec(
                    *command,
                    stdin=asyncio.subprocess.PIPE if stdin else None,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,  # Merge streams for simplicity
                    env=env,
                    cwd=str(cwd) if cwd else None,
                )
            except (OSError, ValueError) as e:
                raise _err.TransportError(
                    f"Failed to start streaming command: {e}",
                    context={"command": command[0], "cwd": str(cwd) if cwd else None},
                    cause=e,
                )
            
            self._processes.add(process)
            
            try:
                # Send stdin if provided
                if stdin:
                    stdin_data = stdin.encode('utf-8') if isinstance(stdin, str) else stdin
                    if process.stdin:
                        process.stdin.write(stdin_data)
                        await process.stdin.drain()
                        process.stdin.close()
                
                # Stream output with optional timeout
                start_time = time.time()
                
                while True:
                    # Check timeout
                    if timeout_s and (time.time() - start_time) > timeout_s:
                        process.kill()
                        raise _err.TimeoutError(
                            f"Stream timed out after {timeout_s}s",
                            context={"command": command[0], "timeout": timeout_s}
                        )
                    
                    # Read line with timeout using lionagi's fail_after
                    try:
                        with ln.fail_after(1.0):  # Short timeout to check main timeout
                            line = await process.stdout.readline()
                    except ln.get_cancelled_exc_class():
                        # Continue to check main timeout and process status
                        if process.returncode is not None:
                            break
                        continue
                    
                    if not line:
                        break
                        
                    # Decode line with error handling
                    try:
                        line_str = line.decode('utf-8').rstrip()
                    except UnicodeDecodeError:
                        line_str = line.decode('utf-8', errors='replace').rstrip()
                    
                    if line_str:  # Skip empty lines
                        yield line_str
                
                # Wait for process completion
                await process.wait()
                
                # Check final exit code
                if process.returncode != 0:
                    raise _err.ServiceError(
                        f"Command failed with exit code {process.returncode}",
                        context={
                            "command": command[0],
                            "exit_code": process.returncode
                        }
                    )
                
            except Exception:
                # Cleanup process on any error
                if process.returncode is None:
                    process.kill()
                    await process.wait()
                raise
            finally:
                self._processes.discard(process)
    
    async def close(self) -> None:
        """Kill all running processes and cleanup."""
        # Kill all processes
        for process in list(self._processes):
            if process.returncode is None:
                process.kill()
        
        # Wait for all to finish
        for process in list(self._processes):
            await process.wait()
        
        self._processes.clear()