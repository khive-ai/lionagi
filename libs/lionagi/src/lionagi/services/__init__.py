# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Lionagi services module exports."""

from .core import CallContext, Service
from .endpoint import ChatRequestModel, RequestModel
from .imodel import iModel

# Transport exports
from .transport import Transport  # Legacy alias for HTTPTransport
from .transport import (
    CLITransport,
    FastMCPTransport,
    HTTPTransport,
    HTTPXTransport,
    MCPTransport,
    SubprocessCLITransport,
)

__all__ = [
    # Core service types
    "CallContext",
    "Service",
    "RequestModel",
    "ChatRequestModel",
    "iModel",
    # Transport types
    "Transport",  # Legacy alias
    "HTTPTransport",
    "HTTPXTransport",
    "CLITransport",
    "SubprocessCLITransport",
    "MCPTransport",
    "FastMCPTransport",
]
