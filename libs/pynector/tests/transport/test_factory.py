"""
Tests for the TransportFactory protocol.

This module contains tests for the TransportFactory protocol.
"""

from typing import Any

import pytest

from pynector.transport.factory import TransportFactory


# Create a mock transport for testing
class MockTransport:
    def __init__(self, host: str, port: int, **kwargs):
        self.host = host
        self.port = port
        self.options = kwargs

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


# Create a mock factory for testing
class MockTransportFactory:
    def __init__(self, default_host: str = "localhost"):
        self.default_host = default_host

    def create_transport(self, **kwargs: Any) -> MockTransport:
        host = kwargs.get("host", self.default_host)
        port = kwargs.get("port")

        if port is None:
            raise ValueError("Port is required")

        # Extract the parameters we need and pass the rest as options
        kwargs_copy = kwargs.copy()
        if "port" in kwargs_copy:
            del kwargs_copy["port"]
        if "host" in kwargs_copy:
            del kwargs_copy["host"]
        return MockTransport(host, port, **kwargs_copy)


def test_factory_protocol():
    """Test that the TransportFactory protocol is correctly defined."""
    assert hasattr(TransportFactory, "create_transport")


def test_mock_factory_implementation():
    """Test that a concrete factory implementation works."""
    factory = MockTransportFactory()

    # Test with valid arguments
    transport = factory.create_transport(port=8080)
    assert isinstance(transport, MockTransport)
    assert transport.host == "localhost"
    assert transport.port == 8080

    # Test with custom host
    transport = factory.create_transport(host="example.com", port=8080)
    assert transport.host == "example.com"

    # Test with missing required argument
    with pytest.raises(ValueError):
        factory.create_transport()

    # Test with additional arguments
    transport = factory.create_transport(port=8080, timeout=30)
    assert transport.options.get("timeout") == 30


def test_factory_with_default_values():
    """Test factory with default values."""
    factory = MockTransportFactory(default_host="api.example.com")

    transport = factory.create_transport(port=8080)
    assert transport.host == "api.example.com"
    assert transport.port == 8080


def test_factory_error_handling():
    """Test factory error handling."""
    factory = MockTransportFactory()

    # Test with missing required argument
    with pytest.raises(ValueError, match="Port is required"):
        factory.create_transport(host="example.com")

    # Test with missing required argument
    with pytest.raises(ValueError, match="Port is required"):
        factory.create_transport(host="example.com")


class CustomMockTransportFactory:
    """A custom factory implementation for testing."""

    def __init__(self, transport_class):
        self.transport_class = transport_class

    def create_transport(self, **kwargs: Any) -> Any:
        return self.transport_class(**kwargs)


def test_custom_factory():
    """Test a custom factory implementation."""

    class CustomTransport:
        def __init__(self, **kwargs):
            self.options = kwargs

    factory = CustomMockTransportFactory(CustomTransport)
    transport = factory.create_transport(option1="value1", option2="value2")

    assert isinstance(transport, CustomTransport)
    assert transport.options["option1"] == "value1"
    assert transport.options["option2"] == "value2"
