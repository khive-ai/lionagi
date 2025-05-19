# SDK Transport Implementation

The SDK Transport Implementation is a concrete implementation of the Transport
Abstraction Layer for interacting with AI model provider SDKs, such as OpenAI
and Anthropic. It provides a unified interface for making requests to these
services while conforming to the Transport Protocol defined in the Transport
Abstraction Layer.

## Table of Contents

- [Overview](#overview)
- [Components](#components)
  - [SdkTransport](#sdktransport)
  - [SDK Adapters](#sdk-adapters)
  - [SdkTransportFactory](#sdktransportfactory)
  - [SDK Error Hierarchy](#sdk-error-hierarchy)
- [Features](#features)
  - [Adapter Pattern](#adapter-pattern)
  - [Error Translation](#error-translation)
  - [Authentication Management](#authentication-management)
  - [Streaming Support](#streaming-support)
- [Usage Examples](#usage-examples)
  - [Basic Usage](#basic-usage)
  - [OpenAI Example](#openai-example)
  - [Anthropic Example](#anthropic-example)
  - [Streaming Example](#streaming-example)
  - [Error Handling](#error-handling)
  - [Custom Configuration](#custom-configuration)

## Overview

The SDK Transport Implementation provides a complete solution for interacting
with AI model provider SDKs within the Pynector framework. It follows the
Transport Protocol defined in the Transport Abstraction Layer, ensuring
compatibility with the rest of the framework while providing SDK-specific
functionality.

Key benefits of the SDK Transport Implementation include:

- **Unified interface:** Consistent interface for different AI model provider
  SDKs
- **Adapter pattern:** Separation of transport logic from SDK-specific details
- **Error translation:** Mapping of SDK-specific errors to the Transport error
  hierarchy
- **Authentication management:** Secure handling of API keys
- **Streaming support:** Unified streaming interface across different SDKs
- **Configurability:** Support for SDK-specific configuration options

## Components

### SdkTransport

The `SdkTransport` class is the core component of the SDK Transport
Implementation. It implements the Transport Protocol and provides methods for
connecting, disconnecting, sending, and receiving data from AI model provider
SDKs.

```python
class SdkTransport:
    """SDK transport implementation using OpenAI and Anthropic SDKs."""

    def __init__(
        self,
        sdk_type: str = "openai",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 60.0,
        **kwargs: Any
    ):
        """Initialize the transport with configuration options."""
        ...
```

The `SdkTransport` class provides the following methods:

- `connect()`: Establishes the connection to the SDK
- `disconnect()`: Closes the connection to the SDK
- `send(data)`: Sends data to the SDK
- `receive()`: Receives data from the SDK as an async iterator

### SDK Adapters

The SDK adapters provide a consistent interface for interacting with different
AI model provider SDKs. Each SDK has a corresponding adapter class that
translates between the Transport Protocol and the SDK-specific API.

#### SDKAdapter

The `SDKAdapter` is an abstract base class that defines the interface for all
SDK-specific adapters:

```python
class SDKAdapter(abc.ABC):
    """Base adapter class for SDK-specific implementations."""

    @abc.abstractmethod
    async def complete(self, prompt: str, model: Optional[str] = None, **kwargs: Any) -> str:
        """Generate a completion for the given prompt."""
        pass

    @abc.abstractmethod
    async def stream(self, prompt: str, model: Optional[str] = None, **kwargs: Any) -> AsyncIterator[bytes]:
        """Stream a completion for the given prompt."""
        pass
```

#### OpenAIAdapter

The `OpenAIAdapter` implements the `SDKAdapter` interface for the OpenAI SDK:

```python
class OpenAIAdapter(SDKAdapter):
    """Adapter for the OpenAI SDK."""

    def __init__(self, client: openai.AsyncOpenAI):
        """Initialize the adapter with an OpenAI client."""
        self.client = client

    async def complete(self, prompt: str, model: Optional[str] = None, **kwargs: Any) -> str:
        """Generate a completion using the OpenAI API."""
        ...

    async def stream(self, prompt: str, model: Optional[str] = None, **kwargs: Any) -> AsyncIterator[bytes]:
        """Stream a completion using the OpenAI API."""
        ...
```

#### AnthropicAdapter

The `AnthropicAdapter` implements the `SDKAdapter` interface for the Anthropic
SDK:

```python
class AnthropicAdapter(SDKAdapter):
    """Adapter for the Anthropic SDK."""

    def __init__(self, client: anthropic.AsyncAnthropic):
        """Initialize the adapter with an Anthropic client."""
        self.client = client

    async def complete(self, prompt: str, model: Optional[str] = None, **kwargs: Any) -> str:
        """Generate a completion using the Anthropic API."""
        ...

    async def stream(self, prompt: str, model: Optional[str] = None, **kwargs: Any) -> AsyncIterator[bytes]:
        """Stream a completion using the Anthropic API."""
        ...
```

### SdkTransportFactory

The `SdkTransportFactory` class implements the TransportFactory Protocol for
creating SDK transport instances. It follows the Factory Method pattern,
providing a way to create SDK transport instances with default configuration.

```python
class SdkTransportFactory:
    """Factory for creating SDK transport instances."""

    def __init__(
        self,
        sdk_type: str = "openai",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 60.0,
        **kwargs: Any
    ):
        """Initialize the factory with default configuration options."""
        ...

    def create_transport(self, **kwargs: Any) -> SdkTransport:
        """Create a new SDK transport instance."""
        ...
```

### SDK Error Hierarchy

The SDK Transport Implementation defines a comprehensive error hierarchy for
SDK-specific errors. This makes it easier to handle specific error conditions.

```
TransportError
└── TransportSpecificError
    └── SdkTransportError
        ├── AuthenticationError
        ├── RateLimitError
        ├── InvalidRequestError
        ├── ResourceNotFoundError
        ├── PermissionError
        └── RequestTooLargeError
```

- **SdkTransportError**: Base class for all SDK transport errors
- **AuthenticationError**: Error indicating an authentication failure
- **RateLimitError**: Error indicating a rate limit was exceeded
- **InvalidRequestError**: Error indicating an invalid request
- **ResourceNotFoundError**: Error indicating a resource was not found
- **PermissionError**: Error indicating a permission issue
- **RequestTooLargeError**: Error indicating a request was too large

## Features

### Adapter Pattern

The SDK Transport Implementation uses the adapter pattern to provide a
consistent interface to different SDKs. This pattern allows the transport to
work with different SDKs without changing its interface, making it easy to add
support for new SDKs in the future.

The adapter pattern is implemented through the `SDKAdapter` abstract base class
and its concrete implementations (`OpenAIAdapter` and `AnthropicAdapter`). Each
adapter translates between the Transport Protocol and the SDK-specific API.

### Error Translation

The SDK Transport Implementation includes comprehensive error translation,
mapping SDK-specific errors to the Transport error hierarchy. This ensures a
consistent error handling experience regardless of the underlying SDK.

Error translation is implemented through the `_translate_error` method in the
`SdkTransport` class. This method examines the error type and maps it to the
appropriate Transport error class.

For example, OpenAI's `AuthenticationError` is mapped to the Transport's
`AuthenticationError`, and Anthropic's API status code 429 is mapped to the
Transport's `RateLimitError`.

### Authentication Management

Authentication in the SDK Transport Layer is handled through API keys. These can
be provided directly or sourced from environment variables.

For OpenAI, the priority for authentication is:

1. API key provided to the constructor
2. `OPENAI_API_KEY` environment variable

For Anthropic, the priority for authentication is:

1. API key provided to the constructor
2. `ANTHROPIC_API_KEY` environment variable

This approach provides flexibility while maintaining security, as API keys are
never logged or exposed in error messages.

### Streaming Support

The SDK Transport Implementation provides a unified streaming interface across
different SDKs. This allows clients to consume streaming responses consistently,
regardless of the underlying SDK.

Streaming is implemented through the `stream` method in the SDK adapters and
exposed through the `receive` method in the `SdkTransport` class. This method
returns an async iterator that yields chunks of the response as they are
received.

## Usage Examples

### Basic Usage

Here's a basic example of how to use the SDK Transport Implementation:

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

### OpenAI Example

Here's an example of how to use the SDK Transport Implementation with OpenAI:

```python
from pynector.transport import TransportFactoryRegistry
from pynector.transport.sdk.factory import SdkTransportFactory

# Set up registry
registry = TransportFactoryRegistry()
registry.register(
    "openai",
    SdkTransportFactory(
        sdk_type="openai",
        model="gpt-4o",
        temperature=0.7,
        max_tokens=1000
    )
)

# Create a transport
transport = registry.create_transport("openai")

# Use the transport with async context manager
async with transport as t:
    # Send a prompt
    await t.send(b"Explain quantum computing in simple terms")

    # Receive the response
    async for chunk in t.receive():
        print(chunk.decode("utf-8"), end="")
```

### Anthropic Example

Here's an example of how to use the SDK Transport Implementation with Anthropic:

```python
from pynector.transport import TransportFactoryRegistry
from pynector.transport.sdk.factory import SdkTransportFactory

# Set up registry
registry = TransportFactoryRegistry()
registry.register(
    "anthropic",
    SdkTransportFactory(
        sdk_type="anthropic",
        model="claude-3-opus-20240229",
        temperature=0.5,
        max_tokens=2000
    )
)

# Create a transport
transport = registry.create_transport("anthropic")

# Use the transport with async context manager
async with transport as t:
    # Send a prompt
    await t.send(b"Write a short story about a robot that learns to love")

    # Receive the response
    async for chunk in t.receive():
        print(chunk.decode("utf-8"), end="")
```

### Streaming Example

Here's an example of how to use the streaming interface:

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
    await t.send(b"Generate a long response about the history of artificial intelligence")

    # Stream the response
    async for chunk in t.receive():
        # Process each chunk as it arrives
        print(chunk.decode("utf-8"), end="", flush=True)
```

### Error Handling

Here's an example of how to handle errors:

```python
from pynector.transport import TransportFactoryRegistry
from pynector.transport.sdk.factory import SdkTransportFactory
from pynector.transport.sdk.errors import (
    SdkTransportError,
    AuthenticationError,
    RateLimitError,
    InvalidRequestError,
    ResourceNotFoundError,
    PermissionError,
    RequestTooLargeError
)
from pynector.transport.errors import ConnectionError, ConnectionTimeoutError

# Set up registry
registry = TransportFactoryRegistry()
registry.register("openai", SdkTransportFactory(sdk_type="openai"))

# Create a transport
transport = registry.create_transport("openai")

try:
    async with transport as t:
        try:
            # Send a prompt
            await t.send(b"Tell me a joke about programming")

            # Receive the response
            async for chunk in t.receive():
                print(chunk.decode("utf-8"), end="")
        except AuthenticationError:
            print("Authentication failed. Check your API key.")
        except RateLimitError:
            print("Rate limit exceeded. Try again later.")
        except InvalidRequestError as e:
            print(f"Invalid request: {e}")
        except ResourceNotFoundError:
            print("Resource not found. Check your model name.")
        except SdkTransportError as e:
            print(f"SDK error: {e}")
except ConnectionError as e:
    print(f"Connection error: {e}")
except ConnectionTimeoutError as e:
    print(f"Connection timeout: {e}")
```

### Custom Configuration

Here's an example of how to use custom configuration options:

```python
from pynector.transport import TransportFactoryRegistry
from pynector.transport.sdk.factory import SdkTransportFactory

# Set up registry with custom configuration
registry = TransportFactoryRegistry()
registry.register(
    "openai",
    SdkTransportFactory(
        sdk_type="openai",
        api_key="your-api-key",  # Directly provide API key
        base_url="https://custom-openai-endpoint.com/v1",  # Custom endpoint
        timeout=30.0,  # Custom timeout
        model="gpt-4o",  # Default model
        temperature=0.7,  # Model-specific parameter
        max_tokens=1000,  # Model-specific parameter
        organization="your-organization-id"  # OpenAI-specific parameter
    )
)

# Create a transport with additional configuration
transport = registry.create_transport(
    "openai",
    model="gpt-3.5-turbo",  # Override the default model
    temperature=0.9  # Override the default temperature
)

# Use the transport with async context manager
async with transport as t:
    # Send a prompt
    await t.send(b"Tell me a joke about programming")

    # Receive the response
    async for chunk in t.receive():
        print(chunk.decode("utf-8"), end="")
```
