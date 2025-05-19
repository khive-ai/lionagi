"""
SDK Transport Layer for interacting with AI model provider SDKs.

This module provides a unified interface for interacting with AI model provider SDKs
(such as OpenAI and Anthropic) while conforming to the Transport Protocol defined in
the Transport Abstraction Layer.

The SDK Transport Layer uses the adapter pattern to separate the transport interface
from SDK-specific implementations, making it easy to add support for new SDKs in the
future. It also provides comprehensive error translation, mapping SDK-specific errors
to the Transport error hierarchy.

Components:
    - SdkTransport: The main class implementing the Transport Protocol
    - SDKAdapter: Abstract base class for SDK-specific adapters
    - OpenAIAdapter: Adapter for the OpenAI SDK
    - AnthropicAdapter: Adapter for the Anthropic SDK
    - SdkTransportFactory: Factory for creating SdkTransport instances
    - Error classes: Comprehensive error hierarchy for SDK-specific errors

Example:
    ```python
    from pynector.transport import TransportFactoryRegistry
    from pynector.transport.sdk.factory import SdkTransportFactory

    # Set up registry
    registry = TransportFactoryRegistry()
    registry.register("openai", SdkTransportFactory(sdk_type="openai"))

    # Create a transport
    transport = registry.create_transport("openai")

    # Use the transport with async context manager
    async with transport as t:
        # Send a prompt
        await t.send(b"Tell me a joke about programming")

        # Receive the response
        async for chunk in t.receive():
            print(chunk.decode("utf-8"), end="")
    ```

For more detailed documentation, see the SDK Transport Documentation.
"""

from pynector.transport.sdk.adapter import AnthropicAdapter, OpenAIAdapter, SDKAdapter
from pynector.transport.sdk.errors import (
    AuthenticationError,
    InvalidRequestError,
    PermissionError,
    RateLimitError,
    RequestTooLargeError,
    ResourceNotFoundError,
    SdkTransportError,
)
from pynector.transport.sdk.factory import SdkTransportFactory
from pynector.transport.sdk.transport import SdkTransport

__all__ = [
    "SdkTransport",
    "SDKAdapter",
    "OpenAIAdapter",
    "AnthropicAdapter",
    "SdkTransportFactory",
    "SdkTransportError",
    "AuthenticationError",
    "RateLimitError",
    "InvalidRequestError",
    "ResourceNotFoundError",
    "PermissionError",
    "RequestTooLargeError",
]
