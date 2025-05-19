"""
HTTP-specific error classes for the Transport Abstraction Layer.

This module defines a hierarchy of HTTP-specific error classes that extend
the base Transport error hierarchy, providing detailed error information
for HTTP communication.
"""

from typing import Any

from pynector.transport.errors import TransportSpecificError


class HTTPTransportError(TransportSpecificError):
    """Base class for HTTP transport-specific errors."""

    pass


class HTTPStatusError(HTTPTransportError):
    """Error representing an HTTP response status error."""

    def __init__(self, response: Any, message: str):
        """Initialize with response and message.

        Args:
            response: The HTTP response object
            message: The error message
        """
        self.response = response
        self.status_code = response.status_code
        super().__init__(message)


class HTTPClientError(HTTPStatusError):
    """HTTP client error (4xx)."""

    pass


class HTTPServerError(HTTPStatusError):
    """HTTP server error (5xx)."""

    pass


class HTTPUnauthorizedError(HTTPClientError):
    """HTTP unauthorized error (401)."""

    pass


class HTTPForbiddenError(HTTPClientError):
    """HTTP forbidden error (403)."""

    pass


class HTTPNotFoundError(HTTPClientError):
    """HTTP not found error (404)."""

    pass


class HTTPTimeoutError(HTTPClientError):
    """HTTP timeout error (408)."""

    pass


class HTTPTooManyRequestsError(HTTPClientError):
    """HTTP too many requests error (429)."""

    pass


class CircuitOpenError(HTTPTransportError):
    """Error indicating that the circuit breaker is open."""

    pass
