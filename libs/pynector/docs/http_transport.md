# HTTP Transport Implementation

The HTTP Transport Implementation is a concrete implementation of the Transport
Abstraction Layer for HTTP communication. It provides a robust, feature-rich
HTTP client based on the `httpx` library, enabling efficient and reliable HTTP
communication with support for modern HTTP features.

## Table of Contents

- [Overview](#overview)
- [Components](#components)
  - [HTTPTransport](#httptransport)
  - [HttpMessage](#httpmessage)
  - [HTTPTransportFactory](#httptransportfactory)
  - [HTTP Error Hierarchy](#http-error-hierarchy)
- [Features](#features)
  - [Connection Pooling](#connection-pooling)
  - [Retry Mechanism](#retry-mechanism)
  - [Error Handling](#error-handling)
  - [HTTP Features Support](#http-features-support)
  - [Streaming Support](#streaming-support)
- [Usage Examples](#usage-examples)
  - [Basic HTTP GET Request](#basic-http-get-request)
  - [HTTP POST with JSON Data](#http-post-with-json-data)
  - [File Upload](#file-upload)
  - [Custom Headers and Authentication](#custom-headers-and-authentication)
  - [Handling Errors](#handling-errors)
  - [Streaming Responses](#streaming-responses)
  - [Configuring Retries](#configuring-retries)

## Overview

The HTTP Transport Implementation provides a complete solution for HTTP
communication within the Pynector framework. It follows the Transport Protocol
defined in the Transport Abstraction Layer, ensuring compatibility with the rest
of the framework while providing HTTP-specific functionality.

Key benefits of the HTTP Transport Implementation include:

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

## Components

### HTTPTransport

The `HTTPTransport` class is the core component of the HTTP Transport
Implementation. It implements the Transport Protocol and provides methods for
connecting, disconnecting, sending, and receiving HTTP messages.

```python
class HTTPTransport(Transport[T], Generic[T]):
    """HTTP transport implementation using httpx.AsyncClient."""

    def __init__(
        self,
        base_url: str = "",
        headers: Optional[dict[str, str]] = None,
        timeout: Union[float, httpx.Timeout] = 10.0,
        max_retries: int = 3,
        retry_backoff_factor: float = 0.5,
        retry_status_codes: Optional[set[int]] = None,
        follow_redirects: bool = True,
        verify_ssl: bool = True,
        http2: bool = False,
    ):
        """Initialize the transport with configuration options."""
        ...
```

The `HTTPTransport` class provides the following methods:

- `connect()`: Establishes the connection by initializing the AsyncClient
- `disconnect()`: Closes the connection by closing the AsyncClient
- `send(message)`: Sends an HTTP message
- `receive()`: Receives HTTP responses
- `stream_response(message)`: Streams a response from the HTTP transport

### HttpMessage

The `HttpMessage` class implements the Message Protocol for HTTP communication.
It handles serialization and deserialization of HTTP messages, including
headers, payload, and binary content.

```python
class HttpMessage:
    """HTTP message implementation."""

    content_type: ClassVar[str] = "application/json"

    def __init__(
        self,
        method: str = "GET",
        url: str = "",
        headers: Optional[dict[str, str]] = None,
        params: Optional[dict[str, Any]] = None,
        json_data: Optional[Any] = None,
        form_data: Optional[dict[str, Any]] = None,
        files: Optional[dict[str, Any]] = None,
        content: Optional[Union[str, bytes]] = None,
    ):
        """Initialize an HTTP message."""
        ...
```

The `HttpMessage` class provides the following methods:

- `serialize()`: Converts the message to bytes for transmission
- `deserialize(data)`: Creates a message from received bytes
- `get_headers()`: Gets the message headers
- `get_payload()`: Gets the message payload

### HTTPTransportFactory

The `HTTPTransportFactory` class implements the TransportFactory Protocol for
creating HTTP transport instances. It follows the Factory Method pattern,
providing a way to create HTTP transport instances with default configuration.

```python
class HTTPTransportFactory:
    """Factory for creating HTTP transport instances."""

    def __init__(
        self,
        base_url: str,
        message_type: type[T],
        default_headers: Optional[dict[str, str]] = None,
        default_timeout: float = 30.0,
        default_max_retries: int = 3,
        default_retry_backoff_factor: float = 0.5,
        default_retry_status_codes: Optional[set[int]] = None,
        default_follow_redirects: bool = True,
        default_verify_ssl: bool = True,
        default_http2: bool = False,
    ):
        """Initialize the factory with default configuration."""
        ...
```

The `HTTPTransportFactory` class provides the following method:

- `create_transport(**kwargs)`: Creates a new HTTP transport instance with the
  specified configuration

### HTTP Error Hierarchy

The HTTP Transport Implementation defines a comprehensive error hierarchy for
HTTP-specific errors. This makes it easier to handle specific error conditions.

```
TransportError
└── TransportSpecificError
    └── HTTPTransportError
        ├── HTTPStatusError
        │   ├── HTTPClientError
        │   │   ├── HTTPUnauthorizedError (401)
        │   │   ├── HTTPForbiddenError (403)
        │   │   ├── HTTPNotFoundError (404)
        │   │   ├── HTTPTimeoutError (408)
        │   │   └── HTTPTooManyRequestsError (429)
        │   └── HTTPServerError (5xx)
        └── CircuitOpenError
```

- **HTTPTransportError**: Base class for HTTP transport-specific errors
- **HTTPStatusError**: Error representing an HTTP response status error
  - **HTTPClientError**: HTTP client error (4xx)
    - **HTTPUnauthorizedError**: HTTP unauthorized error (401)
    - **HTTPForbiddenError**: HTTP forbidden error (403)
    - **HTTPNotFoundError**: HTTP not found error (404)
    - **HTTPTimeoutError**: HTTP timeout error (408)
    - **HTTPTooManyRequestsError**: HTTP too many requests error (429)
  - **HTTPServerError**: HTTP server error (5xx)
- **CircuitOpenError**: Error indicating that the circuit breaker is open

## Features

### Connection Pooling

The HTTP Transport Implementation uses `httpx.AsyncClient` for connection
pooling. This means that connections are reused for multiple requests to the
same host, improving performance by reducing the overhead of establishing new
connections.

Connection pooling is handled automatically by the `HTTPTransport` class, which
maintains a single `AsyncClient` instance for the lifetime of the transport.

### Retry Mechanism

The HTTP Transport Implementation includes a configurable retry mechanism with
exponential backoff. This means that failed requests are automatically retried
with increasing delays between attempts, improving reliability in the face of
transient errors.

The retry mechanism is configured with the following parameters:

- `max_retries`: Maximum number of retry attempts (default: 3)
- `retry_backoff_factor`: Factor for exponential backoff (default: 0.5)
- `retry_status_codes`: HTTP status codes that should trigger a retry (default:
  429, 500, 502, 503, 504)

The retry mechanism also handles network-related errors, such as connection
errors and timeouts.

### Error Handling

The HTTP Transport Implementation includes comprehensive error handling, mapping
HTTP errors to the Transport error hierarchy. This makes it easier to handle
specific error conditions.

HTTP status codes are mapped to specific error classes:

- 401: `HTTPUnauthorizedError`
- 403: `HTTPForbiddenError`
- 404: `HTTPNotFoundError`
- 408: `HTTPTimeoutError`
- 429: `HTTPTooManyRequestsError`
- 4xx: `HTTPClientError`
- 5xx: `HTTPServerError`

Network-related errors are mapped to the Transport error hierarchy:

- `httpx.ConnectError`: `ConnectionError`
- `httpx.ConnectTimeout`: `ConnectionTimeoutError`
- `httpx.ReadTimeout`, `httpx.WriteTimeout`: `ConnectionTimeoutError`

### HTTP Features Support

The HTTP Transport Implementation supports a wide range of HTTP features:

- **Query parameters**: URL query parameters for GET requests
- **Headers**: Custom HTTP headers for all requests
- **Form data**: Form data for POST requests
- **JSON**: JSON data for request bodies
- **Files**: File uploads for multipart/form-data requests
- **Raw content**: Raw content for request bodies
- **Streaming**: Streaming responses for large payloads

### Streaming Support

The HTTP Transport Implementation includes support for streaming responses,
which is useful for handling large payloads efficiently. Streaming is
implemented using the `stream_response` method, which returns an async iterator
yielding chunks of the response as they are received.

## Usage Examples

### Basic HTTP GET Request

```python
from pynector.transport import TransportFactoryRegistry
from pynector.transport.http import HttpMessage, HTTPTransportFactory

# Set up registry
registry = TransportFactoryRegistry()
registry.register(
    "http",
    HTTPTransportFactory(
        base_url="https://api.example.com",
        message_type=HttpMessage,
    ),
)

# Create a transport
transport = registry.create_transport("http")

# Use the transport with async context manager
async with transport as t:
    # Create a GET request message
    message = HttpMessage(
        method="GET",
        url="/users",
        params={"limit": 10},
    )

    # Send the message
    await t.send(message)

    # Receive the response
    async for response in t.receive():
        data = response.get_payload()["data"]
        print(f"Received {len(data)} users")
```

### HTTP POST with JSON Data

```python
from pynector.transport import TransportFactoryRegistry
from pynector.transport.http import HttpMessage, HTTPTransportFactory

# Set up registry
registry = TransportFactoryRegistry()
registry.register(
    "http",
    HTTPTransportFactory(
        base_url="https://api.example.com",
        message_type=HttpMessage,
    ),
)

# Create a transport
transport = registry.create_transport("http")

# Use the transport with async context manager
async with transport as t:
    # Create a POST request message with JSON data
    message = HttpMessage(
        method="POST",
        url="/users",
        headers={"Content-Type": "application/json"},
        json_data={
            "name": "John Doe",
            "email": "john.doe@example.com",
        },
    )

    # Send the message
    await t.send(message)

    # Receive the response
    async for response in t.receive():
        data = response.get_payload()["data"]
        print(f"Created user with ID: {data['id']}")
```

### File Upload

```python
from pynector.transport import TransportFactoryRegistry
from pynector.transport.http import HttpMessage, HTTPTransportFactory

# Set up registry
registry = TransportFactoryRegistry()
registry.register(
    "http",
    HTTPTransportFactory(
        base_url="https://api.example.com",
        message_type=HttpMessage,
    ),
)

# Create a transport
transport = registry.create_transport("http")

# Use the transport with async context manager
async with transport as t:
    # Create a POST request message with file upload
    with open("avatar.png", "rb") as f:
        file_content = f.read()

    message = HttpMessage(
        method="POST",
        url="/users/1/avatar",
        files={"avatar": ("avatar.png", file_content, "image/png")},
    )

    # Send the message
    await t.send(message)

    # Receive the response
    async for response in t.receive():
        data = response.get_payload()["data"]
        print(f"Uploaded avatar: {data['avatar_url']}")
```

### Custom Headers and Authentication

```python
from pynector.transport import TransportFactoryRegistry
from pynector.transport.http import HttpMessage, HTTPTransportFactory

# Set up registry with default headers for authentication
registry = TransportFactoryRegistry()
registry.register(
    "http",
    HTTPTransportFactory(
        base_url="https://api.example.com",
        message_type=HttpMessage,
        default_headers={
            "Authorization": "Bearer YOUR_API_KEY",
            "User-Agent": "Pynector/1.0",
        },
    ),
)

# Create a transport
transport = registry.create_transport("http")

# Use the transport with async context manager
async with transport as t:
    # Create a GET request message
    message = HttpMessage(
        method="GET",
        url="/protected-resource",
    )

    # Send the message
    await t.send(message)

    # Receive the response
    async for response in t.receive():
        data = response.get_payload()["data"]
        print(f"Received protected data: {data}")
```

### Handling Errors

```python
from pynector.transport import TransportFactoryRegistry
from pynector.transport.http import (
    HttpMessage,
    HTTPTransportFactory,
    HTTPNotFoundError,
    HTTPUnauthorizedError,
    HTTPServerError,
)

# Set up registry
registry = TransportFactoryRegistry()
registry.register(
    "http",
    HTTPTransportFactory(
        base_url="https://api.example.com",
        message_type=HttpMessage,
    ),
)

# Create a transport
transport = registry.create_transport("http")

try:
    async with transport as t:
        # Create a GET request message
        message = HttpMessage(
            method="GET",
            url="/users/999",  # Non-existent user
        )

        try:
            # Send the message
            await t.send(message)

            # Receive the response
            async for response in t.receive():
                data = response.get_payload()["data"]
                print(f"Received user: {data}")
        except HTTPNotFoundError:
            print("User not found")
        except HTTPUnauthorizedError:
            print("Authentication required")
        except HTTPServerError:
            print("Server error occurred")
except Exception as e:
    print(f"Transport error: {e}")
```

### Streaming Responses

```python
from pynector.transport import TransportFactoryRegistry
from pynector.transport.http import HttpMessage, HTTPTransportFactory

# Set up registry
registry = TransportFactoryRegistry()
registry.register(
    "http",
    HTTPTransportFactory(
        base_url="https://api.example.com",
        message_type=HttpMessage,
    ),
)

# Create a transport
transport = registry.create_transport("http")

# Use the transport with async context manager
async with transport as t:
    # Create a GET request message for a large file
    message = HttpMessage(
        method="GET",
        url="/large-file",
    )

    # Stream the response
    with open("large-file.dat", "wb") as f:
        async for chunk in t.stream_response(message):
            f.write(chunk)
            print(f"Received {len(chunk)} bytes")
```

### Configuring Retries

```python
from pynector.transport import TransportFactoryRegistry
from pynector.transport.http import HttpMessage, HTTPTransportFactory

# Set up registry with custom retry configuration
registry = TransportFactoryRegistry()
registry.register(
    "http",
    HTTPTransportFactory(
        base_url="https://api.example.com",
        message_type=HttpMessage,
        default_max_retries=5,
        default_retry_backoff_factor=1.0,
        default_retry_status_codes={429, 500, 502, 503, 504, 408},
    ),
)

# Create a transport
transport = registry.create_transport("http")

# Use the transport with async context manager
async with transport as t:
    # Create a GET request message
    message = HttpMessage(
        method="GET",
        url="/flaky-endpoint",  # Endpoint that might fail transiently
    )

    # Send the message (will retry up to 5 times with exponential backoff)
    await t.send(message)

    # Receive the response
    async for response in t.receive():
        data = response.get_payload()["data"]
        print(f"Received data after retries: {data}")
```
