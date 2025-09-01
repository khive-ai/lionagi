# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Multi-transport support for lionagi services.

Provides HTTP, CLI, and SDK transport implementations with unified interfaces.
"""

# CLI Transport
from .cli import CLITransport, SubprocessCLITransport

# HTTP Transport
from .http import HTTPTransport, HTTPXTransport

# MCP Transport
from .mcp import FastMCPTransport, MCPTransport

# Backward compatibility: re-export as original Transport
Transport = HTTPTransport  # Legacy alias

__all__ = [
    # HTTP transports
    "HTTPTransport",
    "HTTPXTransport",
    # CLI transports
    "CLITransport",
    "SubprocessCLITransport",
    # MCP transports
    "MCPTransport",
    "FastMCPTransport",
    # Legacy compatibility
    "Transport",  # Alias for HTTPTransport
]
