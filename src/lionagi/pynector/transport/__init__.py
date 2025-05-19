"""
Transport Abstraction Layer for pynector.

This package provides a flexible, maintainable, and type-safe interface for
network communication that follows the sans-I/O pattern.

The Transport Abstraction Layer includes implementations for:
- HTTP Transport: For HTTP communication using httpx
- SDK Transport: For interacting with AI model provider SDKs (OpenAI and Anthropic)
"""

from pynector.transport.errors import (
    ConnectionError,
    ConnectionRefusedError,
    ConnectionTimeoutError,
    DeserializationError,
    MessageError,
    SerializationError,
    TransportError,
    TransportSpecificError,
)
from pynector.transport.factory import TransportFactory
from pynector.transport.protocol import Message, Transport
from pynector.transport.registry import TransportFactoryRegistry

# Import SDK Transport components
from pynector.transport.sdk import SdkTransport, SdkTransportError, SdkTransportFactory

__all__ = [
    "Transport",
    "Message",
    "TransportFactory",
    "TransportFactoryRegistry",
    "TransportError",
    "ConnectionError",
    "ConnectionTimeoutError",
    "ConnectionRefusedError",
    "MessageError",
    "SerializationError",
    "DeserializationError",
    "TransportSpecificError",
    # SDK Transport
    "SdkTransport",
    "SdkTransportFactory",
    "SdkTransportError",
]
