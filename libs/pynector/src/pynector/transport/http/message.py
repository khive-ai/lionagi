"""
HTTP message implementation for the Transport Abstraction Layer.

This module provides an implementation of the Message Protocol for HTTP
communication, handling serialization and deserialization of HTTP messages.
"""

import json
from typing import Any, ClassVar, Optional, Union

from pynector.transport.errors import DeserializationError, SerializationError


class HttpMessage:
    """HTTP message implementation."""

    content_type: ClassVar[str] = "application/json"

    def __init__(
        self,
        method: str = "GET",
        url: str = "",
        headers: Optional[dict[str, str]] = None,
        params: Optional[dict[str, Any]] = None,
        json_data: Optional[Any] = None,
        form_data: Optional[dict[str, Any]] = None,
        files: Optional[dict[str, Any]] = None,
        content: Optional[Union[str, bytes]] = None,
    ):
        """Initialize an HTTP message.

        Args:
            method: The HTTP method (GET, POST, etc.)
            url: The URL path or full URL
            headers: HTTP headers
            params: Query parameters
            json_data: JSON data for the request body
            form_data: Form data for the request body
            files: Files to upload
            content: Raw content for the request body
        """
        self.headers = headers or {}
        self.payload = {"method": method, "url": url}

        if params:
            self.payload["params"] = params
        if json_data is not None:
            self.payload["json"] = json_data
        if form_data:
            self.payload["data"] = form_data
        if files:
            self.payload["files"] = files
        if content:
            self.payload["content"] = (
                content if isinstance(content, bytes) else content.encode("utf-8")
            )

    def serialize(self) -> bytes:
        """Serialize the message to bytes.

        Returns:
            The serialized message as bytes

        Raises:
            SerializationError: If the message cannot be serialized
        """
        try:
            data = {
                "headers": self.headers,
                "payload": {k: v for k, v in self.payload.items() if k != "content"},
            }

            # Handle binary content separately
            if "content" in self.payload:
                if isinstance(self.payload["content"], bytes):
                    # For now, we'll just encode binary content as base64 in JSON
                    # In a real implementation, you might want to use a more efficient
                    # binary serialization format like msgpack or protobuf
                    import base64

                    data["payload"]["content"] = base64.b64encode(
                        self.payload["content"]
                    ).decode("utf-8")
                    data["payload"]["content_encoding"] = "base64"

            return json.dumps(data).encode("utf-8")
        except Exception as e:
            raise SerializationError(f"Failed to serialize HTTP message: {e}")

    @classmethod
    def deserialize(cls, data: bytes) -> "HttpMessage":
        """Deserialize bytes to a message.

        Args:
            data: The serialized message as bytes

        Returns:
            The deserialized message

        Raises:
            DeserializationError: If the data cannot be deserialized
        """
        try:
            parsed = json.loads(data.decode("utf-8"))
            headers = parsed.get("headers", {})
            payload = parsed.get("payload", {})

            # Handle base64-encoded content
            content = None
            if "content" in payload and payload.get("content_encoding") == "base64":
                import base64

                content = base64.b64decode(payload["content"])
                del payload["content"]
                del payload["content_encoding"]

            method = payload.get("method", "GET")
            url = payload.get("url", "")
            params = payload.get("params")
            json_data = payload.get("json")
            form_data = payload.get("data")
            files = payload.get("files")

            return cls(
                method=method,
                url=url,
                headers=headers,
                params=params,
                json_data=json_data,
                form_data=form_data,
                files=files,
                content=content,
            )
        except json.JSONDecodeError as e:
            raise DeserializationError(f"Invalid JSON data: {e}")
        except Exception as e:
            raise DeserializationError(f"Failed to deserialize HTTP message: {e}")

    def get_headers(self) -> dict[str, Any]:
        """Get the message headers.

        Returns:
            A dictionary of header name to header value
        """
        return self.headers

    def get_payload(self) -> Any:
        """Get the message payload.

        Returns:
            The message payload
        """
        return self.payload
