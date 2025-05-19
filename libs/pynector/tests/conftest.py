"""
Pytest configuration for pynector tests.

This module contains fixtures and configuration for pytest.
"""

from collections.abc import AsyncGenerator
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from pynector.transport.factory import TransportFactory
from pynector.transport.http.factory import HTTPTransportFactory
from pynector.transport.http.message import HttpMessage
from pynector.transport.protocol import Transport
from pynector.transport.registry import get_transport_factory_registry

# We don't need to define our own event_loop fixture as pytest-asyncio provides one
# This was causing a deprecation warning


# Mock Transport Protocol implementation
class MockTransport(Transport):
    """Mock transport for testing."""

    def __init__(self, responses=None, raise_on_connect=None, raise_on_send=None):
        self.responses = responses or [b"mock response"]
        self.raise_on_connect = raise_on_connect
        self.raise_on_send = raise_on_send
        self.connected = False
        self.sent_data = []
        self.connect_count = 0
        self.disconnect_count = 0
        self.send_count = 0

    async def connect(self) -> None:
        self.connect_count += 1
        if self.raise_on_connect:
            raise self.raise_on_connect
        self.connected = True

    async def disconnect(self) -> None:
        self.disconnect_count += 1
        self.connected = False

    async def send(self, message: Any, **options) -> None:
        self.send_count += 1
        if self.raise_on_send:
            raise self.raise_on_send
        self.sent_data.append((message, options))

    async def receive(self) -> AsyncGenerator[bytes, None]:
        for response in self.responses:
            yield response

    async def __aenter__(self) -> "MockTransport":
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.disconnect()


# Mock Transport Factory
class MockTransportFactory(TransportFactory):
    """Mock transport factory for testing."""

    def __init__(self, transport=None):
        self.transport = transport or MockTransport()
        self.create_count = 0

    def create_transport(self, **kwargs) -> Transport:
        self.create_count += 1
        self.last_kwargs = kwargs
        return self.transport


@pytest.fixture
def mock_transport():
    """Fixture providing a mock transport."""
    return MockTransport()


@pytest.fixture
def mock_transport_factory(mock_transport):
    """Fixture providing a mock transport factory."""
    return MockTransportFactory(mock_transport)


@pytest.fixture
def register_mock_transport_factory(mock_transport_factory):
    """Fixture that registers a mock transport factory."""
    registry = get_transport_factory_registry()
    registry.register("mock", mock_transport_factory)
    yield
    # Clean up
    if "mock" in registry._factories:
        del registry._factories["mock"]


# Mock Telemetry
@pytest.fixture
def mock_telemetry(monkeypatch):
    """Fixture providing mock telemetry components."""
    mock_tracer = MagicMock()
    mock_span = AsyncMock()
    mock_span.__enter__.return_value = mock_span
    mock_span.__aenter__.return_value = mock_span
    mock_tracer.start_as_current_span.return_value = mock_span
    mock_tracer.start_span.return_value = mock_span

    mock_logger = MagicMock()

    mock_get_telemetry = MagicMock(return_value=(mock_tracer, mock_logger))
    monkeypatch.setattr("pynector.client.get_telemetry", mock_get_telemetry)

    return mock_tracer, mock_logger


# Anyio Backend Selection - only use asyncio
@pytest.fixture(params=["asyncio"])
def anyio_backend(request):
    """Fixture to run tests with different anyio backends."""
    return request.param


@pytest.fixture(autouse=True)
def register_http_transport_factory():
    """Fixture that automatically registers the HTTP transport factory."""
    registry = get_transport_factory_registry()
    # Only register if not already registered
    if "http" not in registry.get_registered_names():
        http_factory = HTTPTransportFactory(
            base_url="http://localhost", message_type=HttpMessage
        )
        registry.register("http", http_factory)
    yield
    # Clean up is not necessary since each test gets a clean registry


def pytest_configure(config):
    """Register custom marks with pytest."""
    config.addinivalue_line("markers", "performance: mark test as a performance test")
    # Register the performance mark in pytest
    import pytest

    pytest.mark.performance = pytest.mark.performance
