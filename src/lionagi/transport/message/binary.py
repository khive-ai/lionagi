"""
Binary message implementation for the Transport Abstraction Layer.

This module provides a binary message implementation that conforms
to the Message protocol.
"""

import json
from typing import Any, ClassVar

from pynector.transport.errors import DeserializationError, SerializationError


class BinaryMessage:
    """Binary message implementation."""

    content_type: ClassVar[str] = "application/octet-stream"

    def __init__(self, headers: dict[str, Any], payload: bytes):
        """Initialize a new binary message.

        Args:
            headers: The message headers.
            payload: The message payload as bytes.
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
        # Simple format: 4-byte header length + header JSON + payload
        try:
            header_json = json.dumps(self.headers).encode("utf-8")
            header_len = len(header_json)
            return header_len.to_bytes(4, byteorder="big") + header_json + self.payload
        except (TypeError, ValueError) as e:
            raise SerializationError(f"Failed to serialize binary message: {e}")

    @classmethod
    def deserialize(cls, data: bytes) -> "BinaryMessage":
        """Create a message from received bytes.

        Args:
            data: The serialized message as bytes.

        Returns:
            The deserialized message.

        Raises:
            DeserializationError: If the data cannot be deserialized.
        """
        try:
            if len(data) < 4:
                raise DeserializationError("Message too short")

            header_len = int.from_bytes(data[:4], byteorder="big")
            if len(data) < 4 + header_len:
                raise DeserializationError("Message truncated")

            header_json = data[4 : 4 + header_len]
            headers = json.loads(header_json.decode("utf-8"))
            payload = data[4 + header_len :]

            return cls(headers=headers, payload=payload)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise DeserializationError(f"Invalid binary message format: {e}")
        except (ValueError, OverflowError) as e:
            raise DeserializationError(f"Invalid binary message structure: {e}")

    def get_headers(self) -> dict[str, Any]:
        """Get the message headers.

        Returns:
            A dictionary of header name to header value.
        """
        return self.headers

    def get_payload(self) -> bytes:
        """Get the message payload.

        Returns:
            The message payload as bytes.
        """
        return self.payload
