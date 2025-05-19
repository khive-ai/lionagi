"""
Factory for creating HTTP transport instances.

This module provides a factory for creating and configuring HTTP transport
instances according to the TransportFactory protocol.
"""

from typing import Any, Optional, TypeVar

from pynector.transport.http.transport import HTTPTransport
from pynector.transport.protocol import Message

T = TypeVar("T", bound=Message)


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
        """Initialize the factory with default configuration.

        Args:
            base_url: The base URL for all requests
            message_type: The message type to use for deserialization
            default_headers: Default headers to include in all requests
            default_timeout: Default request timeout in seconds
            default_max_retries: Default maximum number of retry attempts
            default_retry_backoff_factor: Default factor for exponential backoff
            default_retry_status_codes: Default HTTP status codes that should trigger a retry
            default_follow_redirects: Default whether to automatically follow redirects
            default_verify_ssl: Default whether to verify SSL certificates
            default_http2: Default whether to enable HTTP/2 support
        """
        self.base_url = base_url
        self.message_type = message_type
        self.default_headers = default_headers or {}
        self.default_timeout = default_timeout
        self.default_max_retries = default_max_retries
        self.default_retry_backoff_factor = default_retry_backoff_factor
        self.default_retry_status_codes = default_retry_status_codes or {
            429,
            500,
            502,
            503,
            504,
        }
        self.default_follow_redirects = default_follow_redirects
        self.default_verify_ssl = default_verify_ssl
        self.default_http2 = default_http2

    def create_transport(self, **kwargs: Any) -> HTTPTransport[T]:
        """Create a new HTTP transport instance.

        Args:
            **kwargs: HTTP transport configuration options.
                - headers: Optional[Dict[str, str]] - Additional headers to include
                - timeout: Optional[Union[float, httpx.Timeout]] - Request timeout
                - max_retries: Optional[int] - Maximum number of retry attempts
                - retry_backoff_factor: Optional[float] - Factor for exponential backoff
                - retry_status_codes: Optional[set[int]] - Status codes that trigger a retry
                - follow_redirects: Optional[bool] - Whether to follow redirects
                - verify_ssl: Optional[bool] - Whether to verify SSL certificates
                - http2: Optional[bool] - Whether to enable HTTP/2 support

        Returns:
            A new HTTP transport instance.
        """
        # Merge defaults with provided options
        headers = {**self.default_headers}
        if "headers" in kwargs and kwargs["headers"]:
            headers.update(kwargs["headers"])

        timeout = kwargs.get("timeout", self.default_timeout)
        max_retries = kwargs.get("max_retries", self.default_max_retries)
        retry_backoff_factor = kwargs.get(
            "retry_backoff_factor", self.default_retry_backoff_factor
        )
        retry_status_codes = kwargs.get(
            "retry_status_codes", self.default_retry_status_codes
        )
        follow_redirects = kwargs.get("follow_redirects", self.default_follow_redirects)
        verify_ssl = kwargs.get("verify_ssl", self.default_verify_ssl)
        http2 = kwargs.get("http2", self.default_http2)

        transport = HTTPTransport[T](
            base_url=self.base_url,
            headers=headers,
            timeout=timeout,
            max_retries=max_retries,
            retry_backoff_factor=retry_backoff_factor,
            retry_status_codes=retry_status_codes,
            follow_redirects=follow_redirects,
            verify_ssl=verify_ssl,
            http2=http2,
        )

        # Set message type for deserialization
        transport._message_type = self.message_type

        return transport
