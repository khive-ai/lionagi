"""
Tests for the TransportFactoryRegistry.

This module contains tests for the TransportFactoryRegistry class.
"""

import pytest

from pynector.transport.registry import TransportFactoryRegistry


# Create mock factories for testing
class MockHttpTransportFactory:
    def create_transport(self, **kwargs):
        return "http_transport"


class MockWebSocketTransportFactory:
    def create_transport(self, **kwargs):
        return "websocket_transport"


def test_registry_init():
    """Test TransportFactoryRegistry initialization."""
    registry = TransportFactoryRegistry()
    assert registry._factories == {}


def test_registry_register():
    """Test registering factories."""
    registry = TransportFactoryRegistry()

    http_factory = MockHttpTransportFactory()
    ws_factory = MockWebSocketTransportFactory()

    registry.register("http", http_factory)
    registry.register("websocket", ws_factory)

    assert "http" in registry._factories
    assert "websocket" in registry._factories
    assert registry._factories["http"] is http_factory
    assert registry._factories["websocket"] is ws_factory


def test_registry_get():
    """Test getting factories by name."""
    registry = TransportFactoryRegistry()

    http_factory = MockHttpTransportFactory()
    registry.register("http", http_factory)

    retrieved = registry.get("http")
    assert retrieved is http_factory

    # Test getting non-existent factory
    with pytest.raises(KeyError):
        registry.get("nonexistent")


def test_registry_create_transport():
    """Test creating transports through the registry."""
    registry = TransportFactoryRegistry()

    http_factory = MockHttpTransportFactory()
    ws_factory = MockWebSocketTransportFactory()

    registry.register("http", http_factory)
    registry.register("websocket", ws_factory)

    http_transport = registry.create_transport("http", host="example.com")
    assert http_transport == "http_transport"

    ws_transport = registry.create_transport("websocket", url="ws://example.com")
    assert ws_transport == "websocket_transport"

    # Test creating with non-existent factory
    with pytest.raises(KeyError):
        registry.create_transport("nonexistent")


def test_registry_overwrite():
    """Test overwriting a registered factory."""
    registry = TransportFactoryRegistry()

    factory1 = MockHttpTransportFactory()
    factory2 = MockHttpTransportFactory()

    registry.register("http", factory1)
    assert registry.get("http") is factory1

    # Overwrite with a new factory
    registry.register("http", factory2)
    assert registry.get("http") is factory2


def test_registry_multiple_factories():
    """Test registering and using multiple factories."""
    registry = TransportFactoryRegistry()

    # Register multiple factories
    factories = {
        "http": MockHttpTransportFactory(),
        "websocket": MockWebSocketTransportFactory(),
        "custom": MockHttpTransportFactory(),
    }

    for name, factory in factories.items():
        registry.register(name, factory)

    # Verify all factories are registered
    for name, factory in factories.items():
        assert registry.get(name) is factory

    # Create transports from all factories
    for name in factories:
        transport = registry.create_transport(name)
        if name in ["http", "custom"]:
            assert transport == "http_transport"
        else:
            assert transport == "websocket_transport"
