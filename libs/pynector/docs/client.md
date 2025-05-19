# Core Pynector Client

The `Pynector` class is the main entry point for using the Pynector library. It
provides a high-level interface that integrates the Transport Abstraction Layer,
Structured Concurrency, and Optional Observability components into a cohesive,
user-friendly API.

## Key Features

- **Flexible Transport Integration**: Works with both built-in and custom
  transports
- **Efficient Batch Processing**: Parallel request processing with concurrency
  limits
- **Optional Observability**: Integrated tracing and logging with no-op
  fallbacks
- **Resource Safety**: Proper async resource management with context managers
- **Robust Error Handling**: Specific exception types and retry mechanisms
- **Configuration Hierarchy**: Supports instance config, environment variables,
  and defaults

## Installation

```bash
# Basic installation
pip install pynector

# With observability features
pip install pynector[observability]
```

## Basic Usage

### Creating a Client

```python
from pynector import Pynector

# Create a client with default HTTP transport
client = Pynector(
    transport_type="http",
    base_url="https://api.example.com",
    headers={"Content-Type": "application/json"}
)

# Using as an async context manager (recommended)
async with Pynector(transport_type="http", base_url="https://api.example.com") as client:
    # Client is automatically connected and will be properly closed when exiting the context
    response = await client.request({"path": "/users", "method": "GET"})
```

### Making Requests

```python
# Simple request
response = await client.request(
    {"path": "/users", "method": "GET", "params": {"limit": 10}}
)

# Request with timeout
try:
    response = await client.request(
        {"path": "/users", "method": "GET"},
        timeout=5.0  # 5 second timeout
    )
except TimeoutError:
    print("Request timed out")

# Request with retry for transient errors
response = await client.request_with_retry(
    {"path": "/users", "method": "GET"},
    max_retries=3,
    retry_delay=1.0  # Initial delay, will increase exponentially
)
```

### Batch Requests

```python
# Create a batch of requests
requests = [
    ({"path": "/users/1", "method": "GET"}, {}),
    ({"path": "/users/2", "method": "GET"}, {}),
    ({"path": "/users/3", "method": "GET"}, {})
]

# Process requests in parallel with concurrency limit
responses = await client.batch_request(
    requests,
    max_concurrency=2,  # Process at most 2 requests at a time
    timeout=10.0,       # 10 second timeout for the entire batch
    raise_on_error=False  # Return exceptions instead of raising them
)

# Check results
for i, response in enumerate(responses):
    if isinstance(response, Exception):
        print(f"Request {i} failed: {response}")
    else:
        print(f"Request {i} succeeded: {response}")
```

### Resource Management

```python
# Manual resource management
client = Pynector(transport_type="http", base_url="https://api.example.com")
try:
    response = await client.request({"path": "/users", "method": "GET"})
finally:
    await client.aclose()  # Ensure resources are properly released

# Using async context manager (recommended)
async with Pynector(transport_type="http", base_url="https://api.example.com") as client:
    response = await client.request({"path": "/users", "method": "GET"})
    # Resources automatically released when exiting the context
```

## Advanced Usage

### Custom Configuration

```python
# Instance configuration
client = Pynector(
    transport_type="http",
    config={
        "timeout": 30.0,
        "retry_count": 3,
        "max_connections": 10
    }
)

# Environment variables (set before creating client)
# PYNECTOR_TIMEOUT=30.0
# PYNECTOR_RETRY_COUNT=3
client = Pynector(transport_type="http")
# Will use environment variables for configuration
```

### Using a Custom Transport

```python
from pynector.transport.protocol import Transport

# Create a custom transport implementation
class MyCustomTransport(Transport):
    async def connect(self) -> None:
        # Implementation
        pass

    async def disconnect(self) -> None:
        # Implementation
        pass

    async def send(self, data, **options) -> None:
        # Implementation
        pass

    async def receive(self):
        # Implementation
        yield b"response data"

# Use the custom transport
custom_transport = MyCustomTransport()
client = Pynector(transport=custom_transport)
```

### Integrating with Telemetry

```python
# Configure telemetry (optional)
from pynector.telemetry import configure_telemetry

configure_telemetry(
    service_name="my-service",
    log_level="INFO"
)

# Create client with telemetry enabled
client = Pynector(
    transport_type="http",
    base_url="https://api.example.com",
    enable_telemetry=True  # Default is True
)

# Telemetry will automatically capture spans and logs for requests
```

### HTTP Transport Specific Options

```python
# Create client with HTTP transport specific options
client = Pynector(
    transport_type="http",
    base_url="https://api.example.com",
    headers={"Authorization": "Bearer token"},
    timeout=30.0,
    follow_redirects=True,
    verify=True,  # Verify SSL certificates
    proxies={"http": "http://proxy.example.com:8080"}
)
```

### SDK Transport Specific Options

```python
# Create client with SDK transport for OpenAI
client = Pynector(
    transport_type="sdk",
    provider="openai",
    api_key="your-api-key",
    organization="your-org-id"  # Optional
)

# Create client with SDK transport for Anthropic
client = Pynector(
    transport_type="sdk",
    provider="anthropic",
    api_key="your-api-key"
)
```

## Error Handling

The Pynector client provides a comprehensive error hierarchy for handling
different types of errors:

```python
from pynector.errors import (
    PynectorError,       # Base exception for all Pynector errors
    ConfigurationError,  # Error in client or transport configuration
    TransportError,      # Error in transport layer (e.g., connection error)
    TimeoutError,        # Request timed out
)

try:
    response = await client.request({"path": "/users", "method": "GET"})
except TimeoutError:
    print("Request timed out")
except TransportError as e:
    print(f"Transport error: {e}")
except ConfigurationError as e:
    print(f"Configuration error: {e}")
except PynectorError as e:
    print(f"Other Pynector error: {e}")
```

## API Reference

### Pynector Class

```python
class Pynector:
    def __init__(
        self,
        transport: Optional[Transport] = None,
        transport_type: str = "http",
        enable_telemetry: bool = True,
        config: Optional[dict[str, Any]] = None,
        **transport_options,
    ):
        """Initialize the Pynector instance.

        Args:
            transport: Optional pre-configured transport instance to use.
            transport_type: Type of transport to create if transport is not provided.
            enable_telemetry: Whether to enable telemetry features.
            config: Configuration options for the client.
            **transport_options: Additional options passed to the transport factory.
        """
        # ...

    async def request(
        self, data: Any, timeout: Optional[float] = None, **options
    ) -> Any:
        """Send a single request and return the response.

        Args:
            data: The data to send.
            timeout: Optional timeout in seconds for this specific request.
            **options: Additional options for the request.

        Returns:
            The response data.

        Raises:
            TransportError: If there is an error with the transport.
            TimeoutError: If the request times out.
            PynectorError: For other errors.
        """
        # ...

    async def batch_request(
        self,
        requests: list[tuple[Any, dict]],
        max_concurrency: Optional[int] = None,
        timeout: Optional[float] = None,
        raise_on_error: bool = False,
        **options,
    ) -> list[Any]:
        """Send multiple requests in parallel and return the responses.

        Args:
            requests: List of (data, options) tuples.
            max_concurrency: Maximum number of concurrent requests.
            timeout: Optional timeout in seconds for the entire batch.
            raise_on_error: Whether to raise on the first error.
            **options: Additional options for all requests.

        Returns:
            List of responses or exceptions.

        Raises:
            TimeoutError: If the batch times out and raise_on_error is True.
            PynectorError: For other errors if raise_on_error is True.
        """
        # ...

    async def request_with_retry(
        self, data: Any, max_retries: int = 3, retry_delay: float = 1.0, **options
    ) -> Any:
        """Send a request with retry for transient errors.

        Args:
            data: The data to send
            max_retries: The maximum number of retry attempts
            retry_delay: The initial delay between retries (will be exponentially increased)
            **options: Additional options for the request

        Returns:
            The response data

        Raises:
            TransportError: If all retry attempts fail
            TimeoutError: If the request times out after all retry attempts
            PynectorError: For other errors
        """
        # ...

    async def aclose(self) -> None:
        """Close the Pynector instance and release resources."""
        # ...

    async def __aenter__(self) -> "Pynector":
        """Enter the async context."""
        # ...

    async def __aexit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        """Exit the async context."""
        # ...
```

## Performance Considerations

### Concurrency Limits

When using `batch_request`, the `max_concurrency` parameter controls how many
requests can be processed concurrently. This is important for:

1. **Resource Management**: Prevents overwhelming the system with too many
   concurrent connections
2. **Rate Limiting**: Helps stay within API rate limits
3. **Performance Optimization**: Finding the optimal concurrency level for your
   specific workload

```python
# Example: Finding optimal concurrency
import time
import asyncio
from pynector import Pynector

async def benchmark_concurrency():
    client = Pynector(transport_type="http", base_url="https://api.example.com")

    # Create a batch of 100 requests
    requests = [({"path": "/test", "method": "GET"}, {}) for _ in range(100)]

    # Test different concurrency limits
    concurrency_limits = [1, 5, 10, 20, 50, 100]
    results = []

    for limit in concurrency_limits:
        start_time = time.time()
        await client.batch_request(requests, max_concurrency=limit)
        duration = time.time() - start_time
        results.append((limit, duration))
        print(f"Concurrency {limit}: {duration:.2f} seconds")

    await client.aclose()
    return results

asyncio.run(benchmark_concurrency())
```

### Timeout Handling

Proper timeout handling is crucial for preventing resource leaks and ensuring
responsiveness:

```python
# Global timeout in configuration
client = Pynector(
    transport_type="http",
    base_url="https://api.example.com",
    config={"timeout": 10.0}  # Global default timeout
)

# Per-request timeout (overrides global)
response = await client.request(
    {"path": "/users", "method": "GET"},
    timeout=5.0  # This request has a 5-second timeout
)

# Batch request timeout (applies to the entire batch)
responses = await client.batch_request(
    requests,
    timeout=15.0  # The entire batch must complete within 15 seconds
)
```

## Integration with Other Components

The Pynector client integrates seamlessly with other components of the library:

### Transport Abstraction Layer

```python
# HTTP Transport
client = Pynector(
    transport_type="http",
    base_url="https://api.example.com",
    headers={"Content-Type": "application/json"}
)

# SDK Transport
client = Pynector(
    transport_type="sdk",
    provider="openai",
    api_key="your-api-key"
)
```

### Structured Concurrency

The client uses AnyIO's structured concurrency primitives internally for:

- Task groups in batch requests
- Timeout handling with cancellation scopes
- Concurrency limiting with capacity limiters

### Optional Observability

```python
# Configure telemetry
from pynector.telemetry import configure_telemetry

configure_telemetry(
    service_name="my-service",
    log_level="INFO"
)

# Create client with telemetry
client = Pynector(
    transport_type="http",
    base_url="https://api.example.com",
    enable_telemetry=True
)

# Each request will automatically:
# - Create a span with request/response attributes
# - Log request start/completion/errors
# - Propagate trace context
```
