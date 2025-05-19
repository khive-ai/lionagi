"""
HTTP transport implementation for the Transport Abstraction Layer.

This module provides an implementation of the Transport Protocol using the
httpx library, enabling efficient and reliable HTTP communication.
"""

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any, Generic, Optional, TypeVar, Union

import httpx

from pynector.transport.errors import (
    ConnectionError,
    ConnectionTimeoutError,
    MessageError,
    TransportError,
)
from pynector.transport.http.errors import (
    HTTPClientError,
    HTTPForbiddenError,
    HTTPNotFoundError,
    HTTPServerError,
    HTTPStatusError,
    HTTPTimeoutError,
    HTTPTooManyRequestsError,
    HTTPTransportError,
    HTTPUnauthorizedError,
)
from pynector.transport.protocol import Message, Transport

T = TypeVar("T", bound=Message)


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
        """Initialize the transport with configuration options.

        Args:
            base_url: The base URL for all requests
            headers: Default headers to include in all requests
            timeout: Request timeout in seconds or httpx.Timeout instance
            max_retries: Maximum number of retry attempts for transient errors
            retry_backoff_factor: Factor for exponential backoff between retries
            retry_status_codes: HTTP status codes that should trigger a retry
                (default: 429, 500, 502, 503, 504)
            follow_redirects: Whether to automatically follow redirects
            verify_ssl: Whether to verify SSL certificates
            http2: Whether to enable HTTP/2 support
        """
        self.base_url = base_url
        self.headers = headers or {}
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_backoff_factor = retry_backoff_factor
        self.retry_status_codes = retry_status_codes or {429, 500, 502, 503, 504}
        self.follow_redirects = follow_redirects
        self.verify_ssl = verify_ssl
        self.http2 = http2

        self._client: Optional[httpx.AsyncClient] = None
        self._message_type: Optional[type[T]] = None
        self._last_response: Optional[httpx.Response] = None

    async def connect(self) -> None:
        """Establish the connection by initializing the AsyncClient.

        Raises:
            ConnectionError: If the connection cannot be established
            ConnectionTimeoutError: If the connection attempt times out
        """
        if self._client is None:
            try:
                self._client = httpx.AsyncClient(
                    base_url=self.base_url,
                    headers=self.headers,
                    timeout=self.timeout,
                    follow_redirects=self.follow_redirects,
                    verify=self.verify_ssl,
                    http2=self.http2,
                )
            except httpx.ConnectError as e:
                raise ConnectionError(f"Failed to establish connection: {e}")
            except httpx.ConnectTimeout as e:
                raise ConnectionTimeoutError(f"Connection attempt timed out: {e}")
            except Exception as e:
                raise TransportError(f"Unexpected error during connection: {e}")

    async def disconnect(self) -> None:
        """Close the connection by closing the AsyncClient."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def send(self, message: T) -> None:
        """Send a message over the HTTP transport.

        Args:
            message: The message to send

        Raises:
            ConnectionError: If the connection is closed or broken
            HTTPTransportError: For HTTP-specific errors
            MessageError: For message serialization errors
        """
        if self._client is None:
            raise ConnectionError("Transport not connected")

        # Store message type for deserialization
        if self._message_type is None:
            self._message_type = type(message)

        # Extract request parameters from message
        headers = self._extract_headers(message)
        method, url, request_kwargs = self._prepare_request(message)

        # Implement retry logic with exponential backoff
        retry_count = 0
        while True:
            try:
                response = await self._client.request(
                    method=method, url=url, headers=headers, **request_kwargs
                )

                # Store the response for later use in receive()
                self._last_response = response

                # Raise for status but handle retryable errors separately
                if response.status_code >= 400:
                    if (
                        response.status_code in self.retry_status_codes
                        and retry_count < self.max_retries
                    ):
                        retry_count += 1
                        backoff_time = self.retry_backoff_factor * (
                            2 ** (retry_count - 1)
                        )
                        await asyncio.sleep(backoff_time)
                        continue

                    # Map HTTP status codes to appropriate exceptions
                    self._handle_error_response(response)

                # Success
                break

            except (httpx.ConnectError, httpx.ReadTimeout, httpx.WriteTimeout) as e:
                # Network-related errors (potentially transient)
                retry_count += 1
                if retry_count > self.max_retries:
                    if isinstance(e, httpx.ConnectError):
                        raise ConnectionError(f"Connection failed: {e}")
                    elif isinstance(e, httpx.TimeoutException):
                        raise ConnectionTimeoutError(f"Request timed out: {e}")
                    else:
                        raise HTTPTransportError(f"HTTP request failed: {e}")

                backoff_time = self.retry_backoff_factor * (2 ** (retry_count - 1))
                await asyncio.sleep(backoff_time)

    async def receive(self) -> AsyncIterator[T]:
        """Receive messages from the HTTP transport.

        Returns:
            An async iterator yielding messages as they are received

        Raises:
            ConnectionError: If the connection is closed or broken
            HTTPTransportError: For HTTP-specific errors
            MessageError: For message deserialization errors
        """
        if self._client is None:
            raise ConnectionError("Transport not connected")

        if self._message_type is None:
            raise HTTPTransportError("No message type has been set")

        # For HTTP, receive is typically called after a send and yields a single response
        response_data = await self._get_next_response()
        if response_data:
            try:
                message = self._message_type.deserialize(response_data)
                yield message
            except Exception as e:
                raise MessageError(f"Failed to deserialize message: {e}")

    async def _get_next_response(self) -> Optional[bytes]:
        """Get the next response from the HTTP transport.

        Returns:
            The response data as bytes, or None if no response is available

        Raises:
            HTTPTransportError: If there is an error getting the response
        """
        if self._last_response is None:
            return None

        try:
            # Create a message that matches the HttpMessage format
            # Extract JSON data if possible
            try:
                data = self._last_response.json()
            except Exception:
                data = self._last_response.text

            response_data = {
                "headers": dict(self._last_response.headers),
                "payload": {
                    "method": "GET",  # Default method for responses
                    "url": str(self._last_response.url),
                    "data": data,
                },
            }
            return json.dumps(response_data).encode("utf-8")
        except Exception as e:
            raise HTTPTransportError(f"Failed to process response: {e}")

    def _extract_headers(self, message: T) -> dict[str, str]:
        """Extract headers from the message.

        Args:
            message: The message to extract headers from (HttpMessage or dict)

        Returns:
            A dictionary of header name to header value
        """
        # Merge default headers with message headers
        # Convert any non-string values to strings
        headers = {**self.headers}

        # Handle different message types (HttpMessage or dict)
        if hasattr(message, "get_headers") and callable(
            getattr(message, "get_headers")
        ):
            # HttpMessage object
            message_headers = message.get_headers()
        elif isinstance(message, dict) and "headers" in message:
            # Dictionary with headers field
            message_headers = message["headers"]
        else:
            # No headers or dictionary without headers field
            message_headers = {}

        for name, value in message_headers.items():
            if isinstance(value, str):
                headers[name] = value
            else:
                headers[name] = str(value)

        # Set content-type if not already set
        if hasattr(message, "content_type") and "content-type" not in {
            k.lower() for k in headers
        }:
            headers["Content-Type"] = getattr(message, "content_type")

        return headers

    def _prepare_request(self, message: T) -> tuple[str, str, dict[str, Any]]:
        """Prepare request parameters from the message.

        Args:
            message: The message to prepare request parameters from (HttpMessage or dict)

        Returns:
            A tuple of (method, url, request_kwargs)
        """
        # Default values
        method = "POST"
        url = ""
        request_kwargs = {}

        # Handle different message types (HttpMessage object or dict)
        if hasattr(message, "get_payload") and callable(
            getattr(message, "get_payload")
        ):
            # Extract data from HttpMessage object
            payload = message.get_payload()
            if isinstance(payload, dict):
                method = payload.get("method", method).upper()
                url = payload.get("url", payload.get("path", url))
                # Extract additional kwargs for the request
                for key, value in payload.items():
                    if key not in ("method", "url", "path"):
                        request_kwargs[key] = value
        elif isinstance(message, dict):
            # Handle dict message directly
            method = message.get("method", method).upper()
            url = message.get("url", message.get("path", url))

            # Extract common HTTP parameters that might be in message
            if "params" in message:
                request_kwargs["params"] = message["params"]

            if "json" in message:
                request_kwargs["json"] = message["json"]

            if "data" in message:
                request_kwargs["data"] = message["data"]

            if "files" in message:
                request_kwargs["files"] = message["files"]

            # Extract additional kwargs for the request
            for key, value in message.items():
                if key not in (
                    "method",
                    "url",
                    "path",
                    "headers",
                    "params",
                    "json",
                    "data",
                    "files",
                ):
                    request_kwargs[key] = value

        return method, url, request_kwargs

    def _handle_error_response(self, response: httpx.Response) -> None:
        """Handle error responses by raising appropriate exceptions.

        Args:
            response: The error response

        Raises:
            HTTPTransportError: With appropriate status code and message
        """
        http_error = HTTPStatusError(
            response=response,
            message=f"HTTP error {response.status_code}: {response.reason_phrase}",
        )

        # Map HTTP status codes to specific error types
        if response.status_code == 401:
            raise HTTPUnauthorizedError(
                response, f"Unauthorized: {response.reason_phrase}"
            )
        elif response.status_code == 403:
            raise HTTPForbiddenError(response, f"Forbidden: {response.reason_phrase}")
        elif response.status_code == 404:
            raise HTTPNotFoundError(response, f"Not found: {response.reason_phrase}")
        elif response.status_code == 408:
            raise HTTPTimeoutError(
                response, f"Request timeout: {response.reason_phrase}"
            )
        elif response.status_code == 429:
            raise HTTPTooManyRequestsError(
                response, f"Too many requests: {response.reason_phrase}"
            )
        elif 400 <= response.status_code < 500:
            raise HTTPClientError(response, f"Client error: {response.reason_phrase}")
        elif 500 <= response.status_code < 600:
            raise HTTPServerError(response, f"Server error: {response.reason_phrase}")
        else:
            raise http_error

    async def stream_response(self, message: T) -> AsyncIterator[bytes]:
        """Stream a response from the HTTP transport.

        Args:
            message: The message to send

        Returns:
            An async iterator yielding chunks of the response as they are received

        Raises:
            ConnectionError: If the connection is closed or broken
            HTTPTransportError: For HTTP-specific errors
        """
        if self._client is None:
            raise ConnectionError("Transport not connected")

        # Extract request parameters from message
        headers = self._extract_headers(message)
        method, url, request_kwargs = self._prepare_request(message)

        # Don't set streaming mode as a parameter - it's implied by using the stream() method

        try:
            async with self._client.stream(
                method=method, url=url, headers=headers, **request_kwargs
            ) as response:
                if response.status_code >= 400:
                    self._handle_error_response(response)

                # In httpx, aiter_bytes() returns an async iterator
                async for chunk in response.aiter_bytes():
                    yield chunk
        except httpx.HTTPStatusError as e:
            self._handle_error_response(e.response)
        except httpx.ConnectError as e:
            raise ConnectionError(f"Connection failed: {e}")
        except httpx.TimeoutException as e:
            raise ConnectionTimeoutError(f"Request timed out: {e}")
        except Exception as e:
            raise HTTPTransportError(f"HTTP request failed: {e}")

    async def __aenter__(self) -> "HTTPTransport[T]":
        """Enter the async context, establishing the connection.

        Returns:
            The transport instance

        Raises:
            ConnectionError: If the connection cannot be established
            ConnectionTimeoutError: If the connection attempt times out
        """
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit the async context, closing the connection."""
        await self.disconnect()
