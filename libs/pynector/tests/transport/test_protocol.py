"""
Tests for the Transport and Message protocols.

This module contains tests for the Transport and Message protocol interfaces.
"""

import inspect

import pytest

from pynector.transport.protocol import Message, Transport


def test_transport_protocol_methods():
    """Test that Transport protocol has all required methods."""
    assert hasattr(Transport, "connect")
    assert hasattr(Transport, "disconnect")
    assert hasattr(Transport, "send")
    assert hasattr(Transport, "receive")
    assert hasattr(Transport, "__aenter__")
    assert hasattr(Transport, "__aexit__")

    # Check method signatures
    assert inspect.iscoroutinefunction(Transport.connect)
    assert inspect.iscoroutinefunction(Transport.disconnect)
    assert inspect.iscoroutinefunction(Transport.send)
    assert inspect.iscoroutinefunction(Transport.__aenter__)
    assert inspect.iscoroutinefunction(Transport.__aexit__)

    # Check return type annotations
    assert Transport.connect.__annotations__.get("return") is None
    assert Transport.disconnect.__annotations__.get("return") is None
    assert Transport.send.__annotations__.get("return") is None
    # receive should return an AsyncIterator
    assert "AsyncIterator" in str(Transport.receive.__annotations__.get("return"))


# Create a mock implementation to test runtime compatibility
class MockTransport:
    async def connect(self) -> None:
        pass

    async def disconnect(self) -> None:
        pass

    async def send(self, message) -> None:
        pass

    async def receive(self):
        yield b"test"

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.disconnect()


@pytest.mark.asyncio
async def test_transport_implementation():
    """Test that a concrete implementation works with the protocol."""
    transport = MockTransport()

    # Test async context manager
    async with transport as t:
        await t.send("test")
        messages = [msg async for msg in t.receive()]
        assert messages == [b"test"]


def test_message_protocol_methods():
    """Test that Message protocol has all required methods."""
    assert hasattr(Message, "serialize")
    assert hasattr(Message, "deserialize")
    assert hasattr(Message, "get_headers")
    assert hasattr(Message, "get_payload")

    # Check method signatures
    assert not inspect.iscoroutinefunction(Message.serialize)
    assert not inspect.iscoroutinefunction(Message.deserialize)
    assert not inspect.iscoroutinefunction(Message.get_headers)
    assert not inspect.iscoroutinefunction(Message.get_payload)

    # Check return type annotations
    assert Message.serialize.__annotations__.get("return") is bytes
    assert "dict" in str(Message.get_headers.__annotations__.get("return"))


# Create a mock implementation to test compatibility
class MockMessage:
    def __init__(self, headers, payload):
        self.headers = headers
        self.payload = payload

    def serialize(self) -> bytes:
        return b"test"

    @classmethod
    def deserialize(cls, data: bytes) -> "MockMessage":
        return cls({}, "test")

    def get_headers(self):
        return self.headers

    def get_payload(self):
        return self.payload


def test_message_implementation():
    """Test that a concrete implementation works with the protocol."""
    message = MockMessage({"content-type": "text/plain"}, "Hello, world!")

    assert message.serialize() == b"test"
    assert message.get_headers() == {"content-type": "text/plain"}
    assert message.get_payload() == "Hello, world!"

    deserialized = MockMessage.deserialize(b"test")
    assert isinstance(deserialized, MockMessage)
