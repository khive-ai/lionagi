"""
Core Pynector client implementation.

This module provides the main entry point for using the Pynector library. The `Pynector` class
integrates the Transport Abstraction Layer, Structured Concurrency, and Optional Observability
components into a cohesive, user-friendly API.

The client supports:
- Multiple transport types (HTTP, SDK, custom)
- Batch request processing with concurrency control
- Timeout handling and retry mechanisms
- Optional telemetry with tracing and logging
- Proper resource management with async context managers

Basic usage:
```python
from pynector import Pynector

async with Pynector(
    transport_type="http",
    base_url="https://api.example.com",
    headers={"Content-Type": "application/json"}
) as client:
    response = await client.request({"path": "/users", "method": "GET"})
```

For more detailed documentation, see the client documentation in the docs/ directory.
"""

from types import TracebackType
from typing import Any, Optional, TypeVar

import anyio

from pynector.config import get_env_config
from pynector.errors import (
    ConfigurationError,
    PynectorError,
    TimeoutError,
    TransportError,
)
from pynector.telemetry import get_telemetry
from pynector.transport.protocol import Transport
from pynector.transport.registry import get_transport_factory_registry

T = TypeVar("T")


class Pynector:
    """
    The core client class for making requests through various transports with support for
    batch processing, timeouts, and optional observability.

    Key features:
    - Flexible transport integration (HTTP, SDK, custom)
    - Efficient batch processing with concurrency limits
    - Timeout handling and retry mechanisms
    - Optional telemetry with tracing and logging
    - Proper resource management with async context managers
    - Configuration hierarchy (instance, environment, defaults)

    The client can be used as an async context manager (recommended) or with explicit
    resource management using the aclose() method.
    """

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
        # Store configuration
        self._config = config or {}
        self._transport_type = transport_type
        self._transport_options = transport_options

        # Set up transport
        self._transport = transport
        self._owns_transport = transport is None
        self._transport_initialized = False

        # Set up telemetry
        self._tracer, self._logger = (
            get_telemetry("pynector.core") if enable_telemetry else (None, None)
        )

        # Validate configuration
        self._validate_config()

    def _validate_config(self) -> None:
        """Validate the configuration."""
        # Validate transport type if we need to create a transport
        if self._owns_transport:
            factory_registry = get_transport_factory_registry()
            if self._transport_type not in factory_registry.get_registered_names():
                raise ConfigurationError(
                    f"Invalid transport type: {self._transport_type}. "
                    f"Available types: {', '.join(factory_registry.get_registered_names())}"
                )

    def _get_config(self, key: str, default: Any = None) -> Any:
        """Get a configuration value from the hierarchy.

        Args:
            key: The configuration key
            default: The default value if not found

        Returns:
            The configuration value
        """
        # 1. Check instance configuration
        if key in self._config:
            return self._config[key]

        # 2. Check environment variables
        env_value = get_env_config(key)
        if env_value is not None:
            return env_value

        # 3. Return default value
        return default

    async def _get_transport(self) -> Transport:
        """Get or create a transport instance.

        Returns:
            The transport instance

        Raises:
            ConfigurationError: If the transport cannot be created
        """
        if self._transport is None or not self._transport_initialized:
            try:
                # Get transport factory
                factory_registry = get_transport_factory_registry()
                factory = factory_registry.get(self._transport_type)

                # Create transport if needed
                if self._transport is None:
                    self._transport = factory.create_transport(
                        **self._transport_options
                    )

                # Connect the transport
                await self._transport.connect()
                self._transport_initialized = True

                if self._logger:
                    self._logger.info(
                        "transport.connected",
                        transport_type=self._transport_type,
                        owns_transport=self._owns_transport,
                    )

            except Exception as e:
                if self._logger:
                    self._logger.error(
                        "transport.connection_failed",
                        transport_type=self._transport_type,
                        error=str(e),
                        error_type=type(e).__name__,
                    )
                if isinstance(e, PynectorError):
                    raise
                elif isinstance(e, ConnectionError):
                    raise TransportError(f"Connection error: {e}") from e
                else:
                    raise ConfigurationError(
                        f"Failed to initialize transport: {e}"
                    ) from e

        return self._transport

    async def request(
        self, data: Any, timeout: Optional[float] = None, **options
    ) -> Any:
        """Send a single request and return the response.

        Args:
            data: The data to send. The format depends on the transport type.
                For HTTP transport, typically a dict with 'path', 'method', and optional
                'params', 'headers', 'json', etc.
            timeout: Optional timeout in seconds for this specific request.
                Overrides the global timeout from configuration.
            **options: Additional options for the request, passed to the transport.
                For HTTP transport, can include 'headers', 'params', etc.

        Returns:
            The response data. The format depends on the transport type.
            For HTTP transport, typically a dict or bytes.

        Raises:
            TransportError: If there is an error with the transport.
            TimeoutError: If the request times out.
            PynectorError: For other errors.

        Example:
            ```python
            # Basic request
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
            ```
        """
        # Start span if tracing is enabled
        if self._tracer:
            with self._tracer.start_as_current_span("pynector.request") as span:
                span.set_attribute("request.size", len(str(data)))
                if options:
                    span.set_attribute("request.options", str(options))

                try:
                    result = await self._perform_request_with_timeout(
                        data, timeout, **options
                    )
                    span.set_attribute("response.size", len(str(result)))
                    return result
                except Exception as e:
                    span.record_exception(e)
                    raise
        else:
            return await self._perform_request_with_timeout(data, timeout, **options)

    async def _perform_request_with_timeout(
        self, data: Any, timeout: Optional[float] = None, **options
    ) -> Any:
        """Perform a request with timeout handling.

        Args:
            data: The data to send
            timeout: Optional timeout in seconds
            **options: Additional options for the request

        Returns:
            The response data

        Raises:
            TimeoutError: If the request times out
            TransportError: If there is an error with the transport
            PynectorError: For other errors
        """
        # Log request if logging is enabled
        if self._logger:
            self._logger.info(
                "request.start",
                data_size=len(str(data)),
                timeout=timeout,
                options=str(options),
            )

        # Get timeout from options, instance config, or default
        if timeout is None:
            timeout = self._get_config("timeout")

        try:
            if timeout:
                # Use move_on_after instead of fail_after
                with anyio.move_on_after(float(timeout)) as scope:
                    result = await self._perform_request(data, **options)

                # If scope.cancel_called is True, a timeout occurred
                if scope.cancel_called:
                    if self._logger:
                        self._logger.error(
                            "request.timeout", timeout=timeout, data_size=len(str(data))
                        )
                    raise TimeoutError(f"Request timed out after {timeout} seconds")
            else:
                result = await self._perform_request(data, **options)

            # Log success if logging is enabled
            if self._logger:
                self._logger.info(
                    "request.complete",
                    data_size=len(str(data)),
                    result_size=len(str(result)),
                )

            return result

        except Exception as e:
            # Log error if logging is enabled
            if self._logger:
                self._logger.error(
                    "request.error", error=str(e), error_type=type(e).__name__
                )

            # Re-raise the exception
            raise

    async def _perform_request(self, data: Any, **options) -> Any:
        """Perform the actual request.

        Args:
            data: The data to send
            **options: Additional options for the request

        Returns:
            The response data

        Raises:
            TransportError: If there is an error with the transport
            PynectorError: For other errors
        """
        transport = await self._get_transport()

        try:
            await transport.send(data, **options)
            result = b""
            async for chunk in transport.receive():
                result += chunk
            return result
        except ConnectionError as e:
            raise TransportError(f"Connection error: {e}") from e
        except Exception as e:
            if isinstance(e, PynectorError):
                raise
            raise PynectorError(f"Unexpected error: {e}") from e

    async def batch_request(
        self,
        requests: list[tuple[Any, dict]],
        max_concurrency: Optional[int] = None,
        timeout: Optional[float] = None,
        raise_on_error: bool = False,
        **options,
    ) -> list[Any]:
        """Send multiple requests in parallel and return the responses.

        This method uses structured concurrency to process multiple requests in parallel,
        with optional concurrency limits and timeout handling.

        Args:
            requests: List of (data, options) tuples. Each tuple contains the data to send
                and a dict of options specific to that request.
            max_concurrency: Maximum number of concurrent requests. If None, all requests
                are processed concurrently. Use this to limit resource usage or respect
                rate limits.
            timeout: Optional timeout in seconds for the entire batch. If the batch doesn't
                complete within this time, remaining requests are cancelled.
            raise_on_error: Whether to raise on the first error. If False (default), errors
                are returned as exceptions in the result list. If True, the first error
                encountered will be raised immediately.
            **options: Additional options for all requests. These are merged with the
                request-specific options, with request-specific options taking precedence.

        Returns:
            List of responses or exceptions. The list has the same length as the input
            requests list, with responses in the same order. If a request fails and
            raise_on_error is False, the corresponding item will be an exception.

        Raises:
            TimeoutError: If the batch times out and raise_on_error is True.
            PynectorError: For other errors if raise_on_error is True.

        Example:
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
        """
        # Start span if tracing is enabled
        if self._tracer:
            with self._tracer.start_as_current_span("pynector.batch_request") as span:
                span.set_attribute("request.count", len(requests))
                if max_concurrency:
                    span.set_attribute("max_concurrency", max_concurrency)
                if timeout:
                    span.set_attribute("timeout", timeout)

                try:
                    results = await self._perform_batch_request(
                        requests, max_concurrency, timeout, raise_on_error, **options
                    )
                    span.set_attribute(
                        "success_count",
                        sum(1 for r in results if not isinstance(r, Exception)),
                    )
                    span.set_attribute(
                        "error_count",
                        sum(1 for r in results if isinstance(r, Exception)),
                    )
                    return results
                except Exception as e:
                    span.record_exception(e)
                    raise
        else:
            return await self._perform_batch_request(
                requests, max_concurrency, timeout, raise_on_error, **options
            )

    async def _perform_batch_request(
        self,
        requests: list[tuple[Any, dict]],
        max_concurrency: Optional[int] = None,
        timeout: Optional[float] = None,
        raise_on_error: bool = False,
        **options,
    ) -> list[Any]:
        """Perform multiple requests in parallel.

        Args:
            requests: List of (data, options) tuples
            max_concurrency: Maximum number of concurrent requests
            timeout: Optional timeout in seconds for the entire batch
            raise_on_error: Whether to raise on the first error
            **options: Additional options for all requests

        Returns:
            List of responses or exceptions

        Raises:
            TimeoutError: If the batch times out and raise_on_error is True
            PynectorError: For other errors if raise_on_error is True
        """
        results = [None] * len(requests)

        # Log batch request if logging is enabled
        if self._logger:
            self._logger.info(
                "batch_request.start",
                request_count=len(requests),
                max_concurrency=max_concurrency,
                timeout=timeout,
            )

        # Create a capacity limiter if max_concurrency is specified
        limiter = anyio.CapacityLimiter(max_concurrency) if max_concurrency else None

        async def process_request(index, data, request_options):
            try:
                # Merge options
                merged_options = options.copy()
                merged_options.update(request_options)

                # Process with limiter if specified
                if limiter:
                    async with limiter:
                        result = await self.request(data, **merged_options)
                else:
                    result = await self.request(data, **merged_options)

                results[index] = result
            except Exception as e:
                results[index] = e
                if raise_on_error:
                    raise

        try:
            # Apply timeout if specified
            if timeout:
                # Use move_on_after instead of fail_after for timeout handling
                with anyio.move_on_after(timeout) as scope:
                    async with anyio.create_task_group() as tg:
                        for i, (data, request_options) in enumerate(requests):
                            # Call start_soon without awaiting it
                            tg.start_soon(process_request, i, data, request_options)

                # If the scope was cancelled, it means we had a timeout
                if scope.cancel_called:
                    if self._logger:
                        self._logger.error(
                            "batch_request.timeout",
                            timeout=timeout,
                            request_count=len(requests),
                        )

                    if raise_on_error:
                        raise TimeoutError(
                            f"Batch request timed out after {timeout} seconds"
                        )

                    # Fill remaining results with timeout errors
                    for i, result in enumerate(results):
                        if result is None:
                            results[i] = TimeoutError(
                                f"Request timed out after {timeout} seconds"
                            )
            else:
                async with anyio.create_task_group() as tg:
                    for i, (data, request_options) in enumerate(requests):
                        # Call start_soon without awaiting it
                        tg.start_soon(process_request, i, data, request_options)

            # Log completion if logging is enabled
            if self._logger:
                success_count = sum(1 for r in results if not isinstance(r, Exception))
                error_count = sum(1 for r in results if isinstance(r, Exception))
                self._logger.info(
                    "batch_request.complete",
                    request_count=len(requests),
                    success_count=success_count,
                    error_count=error_count,
                )

            return results

        except Exception as e:
            # Handle exceptions other than timeouts
            if self._logger:
                self._logger.error(
                    "batch_request.error", error=str(e), error_type=type(e).__name__
                )

            # Re-raise if we're supposed to raise on error
            if raise_on_error:
                raise

            # Otherwise fill in any remaining results with the error
            for i, result in enumerate(results):
                if result is None:
                    results[i] = e

            return results

    async def request_with_retry(
        self, data: Any, max_retries: int = 3, retry_delay: float = 1.0, **options
    ) -> Any:
        """Send a request with retry for transient errors.

        This method automatically retries requests that fail with TransportError,
        using exponential backoff between attempts.

        Args:
            data: The data to send. The format depends on the transport type.
            max_retries: The maximum number of retry attempts. The total number of
                attempts will be max_retries + 1 (the initial attempt plus retries).
            retry_delay: The initial delay between retries in seconds. This delay
                is exponentially increased for subsequent retries (1x, 2x, 4x, 8x, etc.).
            **options: Additional options for the request, passed to the transport.

        Returns:
            The response data. The format depends on the transport type.

        Raises:
            TransportError: If all retry attempts fail
            TimeoutError: If the request times out after all retry attempts
            PynectorError: For other errors

        Example:
            ```python
            # Request with retry for transient errors
            try:
                response = await client.request_with_retry(
                    {"path": "/users", "method": "GET"},
                    max_retries=3,
                    retry_delay=1.0  # Initial delay, will increase exponentially
                )
            except TransportError as e:
                print(f"All retry attempts failed: {e}")
            ```
        """
        last_error = None

        # Start span if tracing is enabled
        if self._tracer:
            with self._tracer.start_as_current_span(
                "pynector.request_with_retry"
            ) as span:
                span.set_attribute("max_retries", max_retries)
                span.set_attribute("retry_delay", retry_delay)

                for attempt in range(max_retries):
                    span.set_attribute("attempt", attempt + 1)

                    try:
                        result = await self.request(data, **options)
                        span.set_attribute("successful_attempt", attempt + 1)
                        return result
                    except TransportError as e:
                        last_error = e
                        span.record_exception(e)

                        if attempt < max_retries - 1:
                            # Calculate backoff delay
                            delay = retry_delay * (2**attempt)
                            span.set_attribute(f"retry_delay_{attempt}", delay)

                            # Wait before retrying
                            await anyio.sleep(delay)
                        else:
                            break

                # If we get here, all retries failed
                raise last_error
        else:
            # Non-traced version
            for attempt in range(max_retries):
                try:
                    return await self.request(data, **options)
                except TransportError as e:
                    last_error = e

                    if attempt < max_retries - 1:
                        # Wait before retrying (with exponential backoff)
                        await anyio.sleep(retry_delay * (2**attempt))
                    else:
                        break

            # If we get here, all retries failed
            raise last_error

    async def aclose(self) -> None:
        """Close the Pynector instance and release resources.

        This method disconnects the transport if it is owned by this instance
        (i.e., if it was created by this instance rather than passed in).
        It should be called when the client is no longer needed to ensure
        proper resource cleanup.

        Example:
            ```python
            client = Pynector(transport_type="http", base_url="https://api.example.com")
            try:
                response = await client.request({"path": "/users", "method": "GET"})
            finally:
                await client.aclose()  # Ensure resources are properly released
            ```

        Note:
            When using the client as an async context manager, this method is
            called automatically when exiting the context.
        """
        if self._owns_transport and self._transport is not None:
            if self._logger:
                self._logger.info("client.closing")

            try:
                await self._transport.disconnect()
                if self._logger:
                    self._logger.info("client.closed")
            except Exception as e:
                if self._logger:
                    self._logger.error(
                        "client.close_error", error=str(e), error_type=type(e).__name__
                    )
                raise
            finally:
                self._transport = None
                self._transport_initialized = False

    async def __aenter__(self) -> "Pynector":
        """Enter the async context.

        Initializes and connects the transport if needed. This method is called
        automatically when using the client as an async context manager.

        Example:
            ```python
            async with Pynector(transport_type="http", base_url="https://api.example.com") as client:
                # Transport is automatically connected
                response = await client.request({"path": "/users", "method": "GET"})
                # Transport is automatically disconnected when exiting the context
            ```

        Returns:
            The Pynector instance.
        """
        # Only get the transport, don't call __aenter__ on it again
        # since it will be connected in _get_transport
        await self._get_transport()
        return self

    async def __aexit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        """Exit the async context.

        Disconnects the transport if it is owned by this instance and releases
        resources. This method is called automatically when exiting the async
        context manager.

        This ensures proper resource cleanup even if an exception occurs within
        the context.
        """
        if self._owns_transport and self._transport is not None:
            await self._transport.disconnect()
            self._transport = None
            self._transport_initialized = False
