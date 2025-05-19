"""
Tests for the HTTP message implementation.

This module contains tests for the HttpMessage class, which handles
serialization and deserialization of HTTP messages.
"""

import json

import pytest

from pynector.transport.errors import DeserializationError
from pynector.transport.http.message import HttpMessage


class TestHttpMessage:
    """Tests for the HttpMessage class."""

    def test_init_default_values(self):
        """Test HttpMessage initialization with default values."""
        message = HttpMessage()
        assert message.headers == {}
        assert message.payload["method"] == "GET"
        assert message.payload["url"] == ""
        assert "params" not in message.payload
        assert "json" not in message.payload
        assert "data" not in message.payload
        assert "files" not in message.payload
        assert "content" not in message.payload

    def test_init_custom_values(self):
        """Test HttpMessage initialization with custom values."""
        message = HttpMessage(
            method="POST",
            url="/api/data",
            headers={"X-Test": "test"},
            params={"q": "test"},
            json_data={"data": "test"},
            form_data={"field": "value"},
            files={"file": "file_content"},
            content=b"raw content",
        )
        assert message.headers == {"X-Test": "test"}
        assert message.payload["method"] == "POST"
        assert message.payload["url"] == "/api/data"
        assert message.payload["params"] == {"q": "test"}
        assert message.payload["json"] == {"data": "test"}
        assert message.payload["data"] == {"field": "value"}
        assert message.payload["files"] == {"file": "file_content"}
        assert message.payload["content"] == b"raw content"

    def test_serialize(self):
        """Test HttpMessage serialization."""
        message = HttpMessage(
            method="GET",
            url="/api/data",
            headers={"X-Test": "test"},
            params={"q": "test"},
        )
        data = message.serialize()
        assert isinstance(data, bytes)
        parsed = json.loads(data.decode("utf-8"))
        assert parsed["headers"] == {"X-Test": "test"}
        assert parsed["payload"]["method"] == "GET"
        assert parsed["payload"]["url"] == "/api/data"
        assert parsed["payload"]["params"] == {"q": "test"}

    def test_serialize_with_binary_content(self):
        """Test HttpMessage serialization with binary content."""
        binary_content = b"\x00\x01\x02\x03"
        message = HttpMessage(method="POST", url="/api/data", content=binary_content)
        data = message.serialize()
        parsed = json.loads(data.decode("utf-8"))
        # Binary content should be base64 encoded
        assert "content_encoding" in parsed["payload"]
        assert parsed["payload"]["content_encoding"] == "base64"

    def test_deserialize(self):
        """Test HttpMessage deserialization."""
        data = json.dumps(
            {
                "headers": {"X-Test": "test"},
                "payload": {
                    "method": "GET",
                    "url": "/api/data",
                    "params": {"q": "test"},
                },
            }
        ).encode("utf-8")

        message = HttpMessage.deserialize(data)
        assert message.headers == {"X-Test": "test"}
        assert message.payload["method"] == "GET"
        assert message.payload["url"] == "/api/data"
        assert message.payload["params"] == {"q": "test"}

    def test_deserialize_with_binary_content(self):
        """Test HttpMessage deserialization with binary content."""
        import base64

        binary_content = b"\x00\x01\x02\x03"
        encoded_content = base64.b64encode(binary_content).decode("utf-8")

        data = json.dumps(
            {
                "headers": {},
                "payload": {
                    "method": "POST",
                    "url": "/api/data",
                    "content": encoded_content,
                    "content_encoding": "base64",
                },
            }
        ).encode("utf-8")

        message = HttpMessage.deserialize(data)
        assert message.payload["method"] == "POST"
        assert message.payload["url"] == "/api/data"
        assert message.payload["content"] == binary_content

    def test_deserialize_invalid_json(self):
        """Test HttpMessage deserialization with invalid JSON."""
        with pytest.raises(DeserializationError):
            HttpMessage.deserialize(b"invalid json")

    def test_get_headers(self):
        """Test HttpMessage.get_headers method."""
        message = HttpMessage(headers={"X-Test": "test"})
        assert message.get_headers() == {"X-Test": "test"}

    def test_get_payload(self):
        """Test HttpMessage.get_payload method."""
        message = HttpMessage(method="GET", url="/api/data")
        payload = message.get_payload()
        assert payload["method"] == "GET"
        assert payload["url"] == "/api/data"
