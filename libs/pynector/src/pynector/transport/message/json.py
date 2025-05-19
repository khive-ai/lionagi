"""
JSON message implementation for the Transport Abstraction Layer.

This module provides a JSON-serialized message implementation that conforms
to the Message protocol.
"""

import json
from typing import Any, ClassVar

from pynector.transport.errors import DeserializationError, SerializationError


class JsonMessage:
    """JSON-serialized message implementation."""

    content_type: ClassVar[str] = "application/json"

    def __init__(self, headers: dict[str, Any], payload: Any):
        """Initialize a new JSON message.

        Args:
            headers: The message headers.
            payload: The message payload.
        """
        self.headers = headers
        self.payload = payload

    def serialize(self) -> bytes:
        """Convert the message to bytes for transmission.

        Returns:
            The serialized message as bytes.

        Raises:
            SerializationError: If the message cannot be serialized.
        """
        data = {"headers": self.headers, "payload": self.payload}
        try:
            return json.dumps(data).encode("utf-8")
        except (TypeError, ValueError) as e:
            raise SerializationError(f"Failed to serialize JSON message: {e}")

    @classmethod
    def deserialize(cls, data: bytes) -> "JsonMessage":
        """Create a message from received bytes.

        Args:
            data: The serialized message as bytes.

        Returns:
            The deserialized message.

        Raises:
            DeserializationError: If the data cannot be deserialized.
        """
        try:
            parsed = json.loads(data.decode("utf-8"))
            return cls(
                headers=parsed.get("headers", {}), payload=parsed.get("payload", None)
            )
        except json.JSONDecodeError as e:
            raise DeserializationError(f"Invalid JSON data: {e}")
        except UnicodeDecodeError as e:
            raise DeserializationError(f"Invalid UTF-8 encoding: {e}")

    def get_headers(self) -> dict[str, Any]:
        """Get the message headers.

        Returns:
            A dictionary of header name to header value.
        """
        return self.headers

    def get_payload(self) -> Any:
        """Get the message payload.

        Returns:
            The message payload.
        """
        return self.payload
