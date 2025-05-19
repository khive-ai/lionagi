# Transport Abstraction Layer

The Transport Abstraction Layer is a core component of Pynector that provides a
flexible and maintainable interface for network communication. It follows the
sans-I/O pattern, which separates I/O concerns from protocol logic, making it
easier to test, maintain, and extend.

## Table of Contents

- [Design Philosophy](#design-philosophy)
- [Components](#components)
  - [Transport Protocol](#transport-protocol)
  - [Message Protocol](#message-protocol)
  - [Error Hierarchy](#error-hierarchy)
  - [Message Implementations](#message-implementations)
  - [Transport Factory](#transport-factory)
  - [Transport Factory Registry](#transport-factory-registry)
- [Transport Implementations](#transport-implementations)
  - [HTTP Transport](#http-transport)
- [Usage Examples](#usage-examples)
  - [Basic Usage](#basic-usage)
  - [Error Handling](#error-handling)
  - [Using Multiple Transports](#using-multiple-transports)
  - [Implementing Custom Transports](#implementing-custom-transports)
  - [Implementing Custom Message Formats](#implementing-custom-message-formats)

## Design Philosophy

The Transport Abstraction Layer is designed with the following principles in
mind:

### Sans-I/O Pattern

The sans-I/O pattern separates I/O concerns from protocol logic. This means that
the protocol implementation doesn't directly perform I/O operations, but instead
defines how to interpret and generate data. This separation has several
benefits:

- **Testability**: Protocol logic can be tested without actual I/O, making tests
  faster and more reliable.
- **Flexibility**: The same protocol implementation can be used with different
  I/O mechanisms (synchronous, asynchronous, etc.).
- **Maintainability**: Changes to I/O mechanisms don't affect protocol logic,
  and vice versa.

### Protocol-Based Design

The Transport Abstraction Layer uses Python's Protocol classes (from the
`typing` module) to define interfaces. This enables static type checking and
makes it clear what methods a class must implement to satisfy the interface.

### Async Context Management

The Transport Abstraction Layer uses async context managers for resource
handling. This ensures that resources are properly acquired and released, even
in the presence of exceptions.

## Components

### Transport Protocol

The Transport Protocol defines the interface for all transport implementations.
It includes methods for connecting, disconnecting, sending, and receiving
messages.

```python
from collections.abc import AsyncIterator
from typing import Protocol, TypeVar

T = TypeVar("T")

class Transport(Protocol, Generic[T]):
    """Protocol defining the interface for transport implementations."""

    async def connect(self) -> None:
        """Establish the connection to the remote endpoint."""
        ...

    async def disconnect(self) -> None:
        """Close the connection to the remote endpoint."""
        ...

    async def send(self, message: T) -> None:
        """Send a message over the transport."""
        ...

    async def receive(self) -> AsyncIterator[T]:
        """Receive messages from the transport."""
        ...

    async def __aenter__(self) -> "Transport[T]":
        """Enter the async context, establishing the connection."""
        ...

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit the async context, closing the connection."""
        ...
```

### Message Protocol

The Message Protocol defines the interface for message serialization and
deserialization. It includes methods for converting messages to and from bytes,
as well as accessing message headers and payload.

```python
from typing import Any, Protocol, TypeVar

M = TypeVar("M", bound="Message")

class Message(Protocol):
    """Protocol defining the interface for message serialization/deserialization."""

    def serialize(self) -> bytes:
        """Convert the message to bytes for transmission."""
        ...

    @classmethod
    def deserialize(cls: type[M], data: bytes) -> M:
        """Create a message from received bytes."""
        ...

    def get_headers(self) -> dict[str, Any]:
        """Get the message headers."""
        ...

    def get_payload(self) -> Any:
        """Get the message payload."""
        ...
```

### Error Hierarchy

The Transport Abstraction Layer defines a comprehensive error hierarchy for
transport-related errors. This makes it easier to handle specific error
conditions.

```
TransportError
├── ConnectionError
│   ├── ConnectionTimeoutError
│   └── ConnectionRefusedError
├── MessageError
│   ├── SerializationError
│   └── DeserializationError
└── TransportSpecificError
```

- **TransportError**: Base class for all transport-related errors.
- **ConnectionError**: Error indicating a connection problem.
  - **ConnectionTimeoutError**: Error indicating a connection timeout.
  - **ConnectionRefusedError**: Error indicating a connection was refused.
- **MessageError**: Error related to message handling.
  - **SerializationError**: Error during message serialization.
  - **DeserializationError**: Error during message deserialization.
- **TransportSpecificError**: Base class for transport-specific errors.

### Message Implementations

The Transport Abstraction Layer includes two message implementations:

#### JsonMessage

The `JsonMessage` class implements the Message protocol with JSON serialization.
It's suitable for text-based protocols and human-readable messages.

```python
from typing import Any

from pynector.transport.errors import DeserializationError, SerializationError

class JsonMessage:
    """JSON-serialized message implementation."""

    content_type: ClassVar[str] = "application/json"

    def __init__(self, headers: dict[str, Any], payload: Any):
        """Initialize a new JSON message."""
        self.headers = headers
        self.payload = payload

    def serialize(self) -> bytes:
        """Convert the message to bytes for transmission."""
        data = {"headers": self.headers, "payload": self.payload}
        try:
            return json.dumps(data).encode("utf-8")
        except (TypeError, ValueError) as e:
            raise SerializationError(f"Failed to serialize JSON message: {e}")

    @classmethod
    def deserialize(cls, data: bytes) -> "JsonMessage":
        """Create a message from received bytes."""
        try:
            parsed = json.loads(data.decode("utf-8"))
            return cls(
                headers=parsed.get("headers", {}),
                payload=parsed.get("payload", None)
            )
        except json.JSONDecodeError as e:
            raise DeserializationError(f"Invalid JSON data: {e}")
        except UnicodeDecodeError as e:
            raise DeserializationError(f"Invalid UTF-8 encoding: {e}")

    def get_headers(self) -> dict[str, Any]:
        """Get the message headers."""
        return self.headers

    def get_payload(self) -> Any:
        """Get the message payload."""
        return self.payload
```

#### BinaryMessage

The `BinaryMessage` class implements the Message protocol with binary
serialization. It's suitable for binary protocols and efficient transmission.

```python
from typing import Any

from pynector.transport.errors import DeserializationError, SerializationError

class BinaryMessage:
    """Binary message implementation."""

    content_type: ClassVar[str] = "application/octet-stream"

    def __init__(self, headers: dict[str, Any], payload: bytes):
        """Initialize a new binary message."""
        self.headers = headers
        self.payload = payload

    def serialize(self) -> bytes:
        """Convert the message to bytes for transmission."""
        # Simple format: 4-byte header length + header JSON + payload
        try:
            header_json = json.dumps(self.headers).encode("utf-8")
            header_len = len(header_json)
            return header_len.to_bytes(4, byteorder="big") + header_json + self.payload
        except (TypeError, ValueError) as e:
            raise SerializationError(f"Failed to serialize binary message: {e}")

    @classmethod
    def deserialize(cls, data: bytes) -> "BinaryMessage":
        """Create a message from received bytes."""
        try:
            if len(data) < 4:
                raise DeserializationError("Message too short")

            header_len = int.from_bytes(data[:4], byteorder="big")
            if len(data) < 4 + header_len:
                raise DeserializationError("Message truncated")

            header_json = data[4 : 4 + header_len]
            headers = json.loads(header_json.decode("utf-8"))
            payload = data[4 + header_len :]

            return cls(headers=headers, payload=payload)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise DeserializationError(f"Invalid binary message format: {e}")
        except (ValueError, OverflowError) as e:
            raise DeserializationError(f"Invalid binary message structure: {e}")

    def get_headers(self) -> dict[str, Any]:
        """Get the message headers."""
        return self.headers

    def get_payload(self) -> bytes:
        """Get the message payload."""
        return self.payload
```

### Transport Factory

The Transport Factory defines the interface for creating transport instances. It
follows the Factory Method pattern, which provides a way to create objects
without specifying the exact class of object that will be created.

```python
from typing import Any, Protocol, TypeVar

T = TypeVar("T")

class TransportFactory(Protocol, Generic[T]):
    """Protocol defining the interface for transport factories."""

    def create_transport(self, **kwargs: Any) -> T:
        """Create a new transport instance."""
        ...
```

### Transport Factory Registry

The Transport Factory Registry provides a registry for transport factories. It
allows for dynamic registration and lookup of transport factories.

```python
from typing import Any

from pynector.transport.factory import TransportFactory
from pynector.transport.protocol import Transport

class TransportFactoryRegistry:
    """Registry for transport factories."""

    def __init__(self):
        """Initialize a new transport factory registry."""
        self._factories = {}

    def register(self, name: str, factory: TransportFactory) -> None:
        """Register a transport factory."""
        self._factories[name] = factory

    def get(self, name: str) -> TransportFactory:
        """Get a transport factory by name."""
        return self._factories[name]

    def create_transport(self, name: str, **kwargs: Any) -> Transport:
        """Create a transport using a registered factory."""
        factory = self.get(name)
        return factory.create_transport(**kwargs)
```

## Transport Implementations

The Transport Abstraction Layer includes implementations for common transport
protocols.

### HTTP Transport

The HTTP Transport Implementation provides a complete solution for HTTP
communication within the Pynector framework. It is built on the `httpx` library
and provides a robust, feature-rich HTTP client with support for modern HTTP
features.

Key features of the HTTP Transport Implementation include:

- **Async-first design**: Built on `httpx.AsyncClient` for efficient
  asynchronous HTTP communication
- **Connection pooling**: Reuses connections for improved performance
- **Comprehensive error handling**: Maps HTTP errors to the Transport error
  hierarchy
- **Retry mechanism**: Automatically retries failed requests with exponential
  backoff
- **Support for modern HTTP features**: Includes query parameters, headers, form
  data, JSON, and file uploads
- **Streaming support**: Efficiently handles large responses with streaming

For detailed documentation on the HTTP Transport Implementation, see
[HTTP Transport Documentation](http_transport.md).

## Usage Examples

### Basic Usage

Here's a basic example of how to use the Transport Abstraction Layer:

```python
from pynector.transport import TransportFactoryRegistry
from pynector.transport.message import JsonMessage

# Set up registry
registry = TransportFactoryRegistry()
registry.register("my_transport", MyTransportFactory())

# Create a transport
transport = registry.create_transport("my_transport", host="example.com", port=8080)

# Use the transport with async context manager
async with transport as t:
    # Send a message
    await t.send(JsonMessage({"content-type": "application/json"}, {"data": "Hello, World!"}))

    # Receive messages
    async for message in t.receive():
        print(f"Received: {message.get_payload()}")
```

### Error Handling

Here's an example of how to handle errors:

```python
from pynector.transport import (
    TransportFactoryRegistry,
    ConnectionError,
    ConnectionTimeoutError,
    ConnectionRefusedError,
    MessageError,
    SerializationError,
    DeserializationError,
)
from pynector.transport.message import JsonMessage

# Set up registry
registry = TransportFactoryRegistry()
registry.register("my_transport", MyTransportFactory())

# Create a transport
transport = registry.create_transport("my_transport", host="example.com", port=8080)

try:
    async with transport as t:
        try:
            # Send a message
            await t.send(JsonMessage({"content-type": "application/json"}, {"data": "Hello, World!"}))
        except ConnectionError as e:
            print(f"Connection error while sending: {e}")
        except SerializationError as e:
            print(f"Serialization error: {e}")

        try:
            # Receive messages
            async for message in t.receive():
                print(f"Received: {message.get_payload()}")
        except ConnectionError as e:
            print(f"Connection error while receiving: {e}")
        except DeserializationError as e:
            print(f"Deserialization error: {e}")
except ConnectionTimeoutError as e:
    print(f"Connection timeout: {e}")
except ConnectionRefusedError as e:
    print(f"Connection refused: {e}")
except ConnectionError as e:
    print(f"Other connection error: {e}")
```

### Using Multiple Transports

Here's an example of how to use multiple transports:

```python
from pynector.transport import TransportFactoryRegistry
from pynector.transport.message import JsonMessage, BinaryMessage

# Set up registry
registry = TransportFactoryRegistry()
registry.register("json_transport", JsonTransportFactory())
registry.register("binary_transport", BinaryTransportFactory())

# Create transports
json_transport = registry.create_transport("json_transport", host="example.com", port=8080)
binary_transport = registry.create_transport("binary_transport", host="example.org", port=9090)

# Use both transports
async with json_transport as jt, binary_transport as bt:
    # Send messages
    await jt.send(JsonMessage({"content-type": "application/json"}, {"data": "Hello, JSON!"}))
    await bt.send(BinaryMessage({"content-type": "application/octet-stream"}, b"Hello, Binary!"))

    # Receive messages from both
    json_messages = [msg async for msg in jt.receive()]
    binary_messages = [msg async for msg in bt.receive()]

    # Process messages
    for msg in json_messages:
        print(f"JSON message: {msg.get_payload()}")

    for msg in binary_messages:
        print(f"Binary message: {msg.get_payload()}")
```

### Implementing Custom Transports

To implement a custom transport, you need to create a class that satisfies the
Transport protocol:

```python
from collections.abc import AsyncIterator
from typing import Any

from pynector.transport import ConnectionError, TransportError
from pynector.transport.message import JsonMessage

class MyJsonTransport:
    """Custom JSON transport implementation."""

    def __init__(self, host: str, port: int):
        """Initialize a new transport."""
        self.host = host
        self.port = port
        self.connected = False
        self.connection = None

    async def connect(self) -> None:
        """Establish the connection to the remote endpoint."""
        try:
            # Implement connection logic here
            self.connection = await some_library.connect(self.host, self.port)
            self.connected = True
        except some_library.ConnectionError as e:
            raise ConnectionError(f"Failed to connect to {self.host}:{self.port}: {e}")
        except some_library.TimeoutError as e:
            raise ConnectionTimeoutError(f"Connection to {self.host}:{self.port} timed out: {e}")

    async def disconnect(self) -> None:
        """Close the connection to the remote endpoint."""
        if self.connected and self.connection:
            try:
                await self.connection.close()
            finally:
                self.connected = False
                self.connection = None

    async def send(self, message: JsonMessage) -> None:
        """Send a message over the transport."""
        if not self.connected or not self.connection:
            raise ConnectionError("Not connected")

        try:
            data = message.serialize()
            await self.connection.send(data)
        except some_library.ConnectionError as e:
            self.connected = False
            self.connection = None
            raise ConnectionError(f"Connection lost while sending: {e}")
        except Exception as e:
            raise TransportError(f"Error sending message: {e}")

    async def receive(self) -> AsyncIterator[JsonMessage]:
        """Receive messages from the transport."""
        if not self.connected or not self.connection:
            raise ConnectionError("Not connected")

        try:
            while self.connected:
                data = await self.connection.receive()
                if not data:  # Connection closed
                    self.connected = False
                    self.connection = None
                    break

                yield JsonMessage.deserialize(data)
        except some_library.ConnectionError as e:
            self.connected = False
            self.connection = None
            raise ConnectionError(f"Connection lost while receiving: {e}")
        except Exception as e:
            raise TransportError(f"Error receiving message: {e}")

    async def __aenter__(self) -> "MyJsonTransport":
        """Enter the async context, establishing the connection."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit the async context, closing the connection."""
        await self.disconnect()
```

Then, create a factory for your transport:

```python
from typing import Any

from pynector.transport.factory import TransportFactory

class MyJsonTransportFactory:
    """Factory for creating MyJsonTransport instances."""

    def create_transport(self, **kwargs: Any) -> MyJsonTransport:
        """Create a new transport instance."""
        host = kwargs.get("host")
        port = kwargs.get("port")

        if not host:
            raise ValueError("Host is required")
        if not port:
            raise ValueError("Port is required")

        return MyJsonTransport(host=host, port=port)
```

Finally, register your factory with the registry:

```python
from pynector.transport import TransportFactoryRegistry

# Set up registry
registry = TransportFactoryRegistry()
registry.register("my_json", MyJsonTransportFactory())

# Create a transport
transport = registry.create_transport("my_json", host="example.com", port=8080)
```

### Implementing Custom Message Formats

To implement a custom message format, you need to create a class that satisfies
the Message protocol:

```python
from typing import Any, ClassVar

from pynector.transport.errors import DeserializationError, SerializationError

class MyCustomMessage:
    """Custom message implementation."""

    content_type: ClassVar[str] = "application/x-custom"

    def __init__(self, headers: dict[str, Any], payload: Any):
        """Initialize a new custom message."""
        self.headers = headers
        self.payload = payload

    def serialize(self) -> bytes:
        """Convert the message to bytes for transmission."""
        # Implement your custom serialization logic here
        try:
            # Example: Simple format with header length + header + payload
            header_bytes = some_library.serialize(self.headers)
            payload_bytes = some_library.serialize(self.payload)
            header_len = len(header_bytes)
            return header_len.to_bytes(4, byteorder="big") + header_bytes + payload_bytes
        except Exception as e:
            raise SerializationError(f"Failed to serialize custom message: {e}")

    @classmethod
    def deserialize(cls, data: bytes) -> "MyCustomMessage":
        """Create a message from received bytes."""
        try:
            # Implement your custom deserialization logic here
            if len(data) < 4:
                raise DeserializationError("Message too short")

            header_len = int.from_bytes(data[:4], byteorder="big")
            if len(data) < 4 + header_len:
                raise DeserializationError("Message truncated")

            header_bytes = data[4 : 4 + header_len]
            payload_bytes = data[4 + header_len :]

            headers = some_library.deserialize(header_bytes)
            payload = some_library.deserialize(payload_bytes)

            return cls(headers=headers, payload=payload)
        except Exception as e:
            raise DeserializationError(f"Failed to deserialize custom message: {e}")

    def get_headers(self) -> dict[str, Any]:
        """Get the message headers."""
        return self.headers

    def get_payload(self) -> Any:
        """Get the message payload."""
        return self.payload
```

Then, you can use your custom message format with any transport that supports
it:

```python
from pynector.transport import TransportFactoryRegistry

# Set up registry
registry = TransportFactoryRegistry()
registry.register("my_transport", MyTransportFactory())

# Create a transport
transport = registry.create_transport("my_transport", host="example.com", port=8080)

# Use the transport with your custom message format
async with transport as t:
    # Send a message
    await t.send(MyCustomMessage({"content-type": "application/x-custom"}, {"data": "Hello, Custom!"}))

    # Receive messages
    async for message in t.receive():
        print(f"Received: {message.get_payload()}")
```
