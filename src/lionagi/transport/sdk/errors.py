"""
Error classes for the SDK Transport Layer.

This module defines a hierarchy of error classes for the SDK Transport Layer,
providing a consistent approach to SDK-specific errors.
"""

from pynector.transport.errors import TransportSpecificError


class SdkTransportError(TransportSpecificError):
    """Base class for all SDK transport errors."""

    pass


class AuthenticationError(SdkTransportError):
    """Error indicating an authentication failure."""

    pass


class RateLimitError(SdkTransportError):
    """Error indicating a rate limit was exceeded."""

    pass


class InvalidRequestError(SdkTransportError):
    """Error indicating an invalid request."""

    pass


class ResourceNotFoundError(SdkTransportError):
    """Error indicating a resource was not found."""

    pass


class PermissionError(SdkTransportError):
    """Error indicating a permission issue."""

    pass


class RequestTooLargeError(SdkTransportError):
    """Error indicating a request was too large."""

    pass
