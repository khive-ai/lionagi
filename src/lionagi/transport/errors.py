"""
Error classes for the Transport Abstraction Layer.

This module defines a hierarchy of error classes for the Transport Abstraction Layer,
providing a consistent approach to transport-specific errors.
"""


class TransportError(Exception):
    """Base class for all transport-related errors."""

    pass


class ConnectionError(TransportError):
    """Error indicating a connection problem."""

    pass


class ConnectionTimeoutError(ConnectionError):
    """Error indicating a connection timeout."""

    pass


class ConnectionRefusedError(ConnectionError):
    """Error indicating a connection was refused."""

    pass


class MessageError(TransportError):
    """Error related to message handling."""

    pass


class SerializationError(MessageError):
    """Error during message serialization."""

    pass


class DeserializationError(MessageError):
    """Error during message deserialization."""

    pass


class TransportSpecificError(TransportError):
    """Base class for transport-specific errors."""

    pass
