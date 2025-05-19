"""
Protocol definitions for the Transport Abstraction Layer.

This module defines the Protocol classes that form the core interfaces
of the Transport Abstraction Layer, following the sans-I/O pattern.
"""

from collections.abc import AsyncIterator
from typing import Any, Generic, Protocol, TypeVar

T = TypeVar("T")
M = TypeVar("M", bound="Message")


class Transport(Protocol, Generic[T]):
    """Protocol defining the interface for transport implementations."""

    async def connect(self) -> None:
        """Establish the connection to the remote endpoint.

        Raises:
            ConnectionError: If the connection cannot be established.
            TimeoutError: If the connection attempt times out.
        """
        ...

    async def disconnect(self) -> None:
        """Close the connection to the remote endpoint.

        This method should be idempotent and safe to call multiple times.
        """
        ...

    async def send(self, message: T) -> None:
        """Send a message over the transport.

        Args:
            message: The message to send.

        Raises:
            ConnectionError: If the connection is closed or broken.
            TransportError: For other transport-specific errors.
        """
        ...

    async def receive(self) -> AsyncIterator[T]:
        """Receive messages from the transport.

        Returns:
            An async iterator yielding messages as they are received.

        Raises:
            ConnectionError: If the connection is closed or broken.
            TransportError: For other transport-specific errors.
        """
        ...

    async def __aenter__(self) -> "Transport[T]":
        """Enter the async context, establishing the connection.

        Returns:
            The transport instance.

        Raises:
            ConnectionError: If the connection cannot be established.
            TimeoutError: If the connection attempt times out.
        """
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit the async context, closing the connection."""
        await self.disconnect()


class Message(Protocol):
    """Protocol defining the interface for message serialization/deserialization."""

    def serialize(self) -> bytes:
        """Convert the message to bytes for transmission.

        Returns:
            The serialized message as bytes.

        Raises:
            SerializationError: If the message cannot be serialized.
        """
        ...

    @classmethod
    def deserialize(cls: type[M], data: bytes) -> M:
        """Create a message from received bytes.

        Args:
            data: The serialized message as bytes.

        Returns:
            The deserialized message.

        Raises:
            DeserializationError: If the data cannot be deserialized.
        """
        ...

    def get_headers(self) -> dict[str, Any]:
        """Get the message headers.

        Returns:
            A dictionary of header name to header value.
        """
        ...

    def get_payload(self) -> Any:
        """Get the message payload.

        Returns:
            The message payload.
        """
        ...
