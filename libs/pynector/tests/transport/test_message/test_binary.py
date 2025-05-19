"""
Tests for the BinaryMessage implementation.

This module contains tests for the BinaryMessage class.
"""

import json

import pytest

from pynector.transport.errors import DeserializationError, SerializationError
from pynector.transport.message.binary import BinaryMessage


def test_binary_message_init():
    """Test BinaryMessage initialization."""
    headers = {"content-type": "application/octet-stream"}
    payload = b"binary data"

    message = BinaryMessage(headers, payload)

    assert message.headers == headers
    assert message.payload == payload
    assert message.content_type == "application/octet-stream"


def test_binary_message_serialize():
    """Test BinaryMessage serialization."""
    headers = {"content-type": "application/octet-stream"}
    payload = b"binary data"

    message = BinaryMessage(headers, payload)
    serialized = message.serialize()

    # Verify it's bytes
    assert isinstance(serialized, bytes)

    # Verify format: 4-byte header length + header JSON + payload
    header_len = int.from_bytes(serialized[:4], byteorder="big")
    header_json = serialized[4 : 4 + header_len]
    message_payload = serialized[4 + header_len :]

    deserialized_headers = json.loads(header_json.decode("utf-8"))
    assert deserialized_headers == headers
    assert message_payload == payload


def test_binary_message_deserialize():
    """Test BinaryMessage deserialization."""
    headers = {"content-type": "application/octet-stream"}
    payload = b"binary data"

    # Create serialized data
    header_json = json.dumps(headers).encode("utf-8")
    header_len = len(header_json)
    data = header_len.to_bytes(4, byteorder="big") + header_json + payload

    message = BinaryMessage.deserialize(data)

    assert message.headers == headers
    assert message.payload == payload


def test_binary_message_serialize_error():
    """Test BinaryMessage serialization error."""
    # Create a message with non-serializable headers
    headers = {"circular": None}
    headers["circular"] = headers  # Create circular reference
    payload = b"binary data"

    message = BinaryMessage(headers, payload)

    with pytest.raises(SerializationError):
        message.serialize()


def test_binary_message_deserialize_error():
    """Test BinaryMessage deserialization error."""
    # Test cases for deserialization errors

    # 1. Message too short
    with pytest.raises(DeserializationError, match="too short"):
        BinaryMessage.deserialize(b"123")

    # 2. Message truncated
    header_len = (1000).to_bytes(4, byteorder="big")
    with pytest.raises(DeserializationError, match="truncated"):
        BinaryMessage.deserialize(header_len + b"short")

    # 3. Invalid header JSON
    header_len = (5).to_bytes(4, byteorder="big")
    with pytest.raises(DeserializationError, match="Invalid"):
        BinaryMessage.deserialize(header_len + b"not{json" + b"payload")


def test_binary_message_get_methods():
    """Test BinaryMessage get_headers and get_payload methods."""
    headers = {"content-type": "application/octet-stream"}
    payload = b"binary data"

    message = BinaryMessage(headers, payload)

    assert message.get_headers() == headers
    assert message.get_payload() == payload


def test_binary_message_roundtrip():
    """Test that BinaryMessage serialization/deserialization roundtrip works."""
    headers = {"content-type": "application/octet-stream", "id": "123"}
    payload = b"binary data with special chars: \x00\x01\x02\xff"

    message = BinaryMessage(headers, payload)
    serialized = message.serialize()
    deserialized = BinaryMessage.deserialize(serialized)

    assert deserialized.get_headers() == headers
    assert deserialized.get_payload() == payload


def test_binary_message_empty_payload():
    """Test BinaryMessage with empty payload."""
    headers = {"content-type": "application/octet-stream"}
    payload = b""

    message = BinaryMessage(headers, payload)
    serialized = message.serialize()
    deserialized = BinaryMessage.deserialize(serialized)

    assert deserialized.get_headers() == headers
    assert deserialized.get_payload() == payload


def test_binary_message_large_payload():
    """Test BinaryMessage with large payload."""
    headers = {"content-type": "application/octet-stream"}
    payload = b"x" * 10000  # 10KB payload

    message = BinaryMessage(headers, payload)
    serialized = message.serialize()
    deserialized = BinaryMessage.deserialize(serialized)

    assert deserialized.get_headers() == headers
    assert deserialized.get_payload() == payload
    assert len(deserialized.get_payload()) == 10000


def test_binary_message_complex_headers():
    """Test BinaryMessage with complex headers."""
    headers = {
        "content-type": "application/octet-stream",
        "metadata": {
            "id": 123,
            "timestamp": "2025-05-05T12:00:00Z",
            "tags": ["binary", "test", "data"],
        },
    }
    payload = b"binary data"

    message = BinaryMessage(headers, payload)
    serialized = message.serialize()
    deserialized = BinaryMessage.deserialize(serialized)

    assert deserialized.get_headers() == headers
    assert deserialized.get_payload() == payload
