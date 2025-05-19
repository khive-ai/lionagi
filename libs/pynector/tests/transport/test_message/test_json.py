"""
Tests for the JsonMessage implementation.

This module contains tests for the JsonMessage class.
"""

import json

import pytest

from pynector.transport.errors import DeserializationError, SerializationError
from pynector.transport.message.json import JsonMessage


def test_json_message_init():
    """Test JsonMessage initialization."""
    headers = {"content-type": "application/json"}
    payload = {"name": "test", "value": 123}

    message = JsonMessage(headers, payload)

    assert message.headers == headers
    assert message.payload == payload
    assert message.content_type == "application/json"


def test_json_message_serialize():
    """Test JsonMessage serialization."""
    headers = {"content-type": "application/json"}
    payload = {"name": "test", "value": 123}

    message = JsonMessage(headers, payload)
    serialized = message.serialize()

    # Verify it's bytes
    assert isinstance(serialized, bytes)

    # Verify content
    deserialized = json.loads(serialized.decode("utf-8"))
    assert deserialized["headers"] == headers
    assert deserialized["payload"] == payload


def test_json_message_deserialize():
    """Test JsonMessage deserialization."""
    headers = {"content-type": "application/json"}
    payload = {"name": "test", "value": 123}
    data = json.dumps({"headers": headers, "payload": payload}).encode("utf-8")

    message = JsonMessage.deserialize(data)

    assert message.headers == headers
    assert message.payload == payload


def test_json_message_serialize_error():
    """Test JsonMessage serialization error."""
    # Create a message with non-serializable content
    headers = {"content-type": "application/json"}
    payload = {"circular": None}
    payload["circular"] = payload  # Create circular reference

    message = JsonMessage(headers, payload)

    with pytest.raises(SerializationError):
        message.serialize()


def test_json_message_deserialize_error():
    """Test JsonMessage deserialization error."""
    # Invalid JSON data
    data = b"not valid json"

    with pytest.raises(DeserializationError):
        JsonMessage.deserialize(data)

    # Invalid UTF-8 data
    data = b"\xff\xff\xff"

    with pytest.raises(DeserializationError):
        JsonMessage.deserialize(data)


def test_json_message_get_methods():
    """Test JsonMessage get_headers and get_payload methods."""
    headers = {"content-type": "application/json"}
    payload = {"name": "test", "value": 123}

    message = JsonMessage(headers, payload)

    assert message.get_headers() == headers
    assert message.get_payload() == payload


def test_json_message_roundtrip():
    """Test that JsonMessage serialization/deserialization roundtrip works."""
    headers = {"content-type": "application/json", "id": "123"}
    payload = {"name": "test", "value": 123, "nested": {"key": "value"}}

    message = JsonMessage(headers, payload)
    serialized = message.serialize()
    deserialized = JsonMessage.deserialize(serialized)

    assert deserialized.get_headers() == headers
    assert deserialized.get_payload() == payload


def test_json_message_empty_payload():
    """Test JsonMessage with empty payload."""
    headers = {"content-type": "application/json"}
    payload = None

    message = JsonMessage(headers, payload)
    serialized = message.serialize()
    deserialized = JsonMessage.deserialize(serialized)

    assert deserialized.get_headers() == headers
    assert deserialized.get_payload() == payload


def test_json_message_complex_types():
    """Test JsonMessage with complex types."""
    headers = {"content-type": "application/json"}
    payload = {
        "string": "test",
        "integer": 123,
        "float": 123.45,
        "boolean": True,
        "null": None,
        "array": [1, 2, 3],
        "object": {"key": "value"},
    }

    message = JsonMessage(headers, payload)
    serialized = message.serialize()
    deserialized = JsonMessage.deserialize(serialized)

    assert deserialized.get_headers() == headers
    assert deserialized.get_payload() == payload
