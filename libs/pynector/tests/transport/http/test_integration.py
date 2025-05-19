"""
Integration tests for the HTTP transport with the Transport Abstraction Layer.

This module contains tests that verify the HTTP transport correctly integrates
with the Transport Abstraction Layer.
"""

import pytest

from pynector.transport.http.factory import HTTPTransportFactory
from pynector.transport.http.message import HttpMessage
from pynector.transport.http.transport import HTTPTransport
from pynector.transport.registry import TransportFactoryRegistry
from tests.transport.http.mock_server import MockHTTPServer


@pytest.mark.asyncio
class TestHTTPTransportIntegration:
    """Integration tests for the HTTP transport with the Transport Abstraction Layer."""

    async def test_http_transport_with_registry(self):
        """Test HTTPTransport with TransportFactoryRegistry."""
        # Create registry and factory
        registry = TransportFactoryRegistry()
        factory = HTTPTransportFactory(
            base_url="https://example.com", message_type=HttpMessage
        )

        # Register factory
        registry.register("http", factory)

        # Create transport using registry
        transport = registry.create_transport("http")

        # Verify transport properties
        assert isinstance(transport, HTTPTransport)
        assert transport.base_url == "https://example.com"
        assert transport._message_type == HttpMessage

    async def test_http_transport_with_registry_custom_options(self):
        """Test HTTPTransport with TransportFactoryRegistry and custom options."""
        # Create registry and factory
        registry = TransportFactoryRegistry()
        factory = HTTPTransportFactory(
            base_url="https://example.com",
            message_type=HttpMessage,
            default_headers={"User-Agent": "pynector/1.0"},
        )

        # Register factory
        registry.register("http", factory)

        # Create transport using registry with custom options
        transport = registry.create_transport(
            "http", headers={"X-Test": "test"}, timeout=5.0
        )

        # Verify transport properties
        assert isinstance(transport, HTTPTransport)
        assert transport.base_url == "https://example.com"
        assert transport._message_type == HttpMessage
        assert transport.headers == {"User-Agent": "pynector/1.0", "X-Test": "test"}
        assert transport.timeout == 5.0

    @pytest.mark.skip(reason="Mock server issues")
    async def test_http_transport_registry_end_to_end(self):
        """Test HTTPTransport with TransportFactoryRegistry end-to-end."""
        # Set up mock server
        async with MockHTTPServer() as server:
            server.add_route("/test", {"data": "test"}, status_code=200)

            # Create registry and factory
            registry = TransportFactoryRegistry()
            factory = HTTPTransportFactory(
                base_url=server.url, message_type=HttpMessage
            )

            # Register factory
            registry.register("http", factory)

            # Create transport using registry
            transport = registry.create_transport("http")

            # Create message
            message = HttpMessage(method="GET", url="/test")

            # Send and receive
            async with transport:
                await transport.send(message)

                # Verify response
                received = False
                async for response in transport.receive():
                    received = True
                    payload = response.get_payload()
                    assert payload["status_code"] == 200
                    assert "data" in payload["data"]

                assert received, "No response was received"

    async def test_multiple_transports_from_registry(self):
        """Test creating multiple transports from the same factory in registry."""
        # Create registry and factory
        registry = TransportFactoryRegistry()
        factory = HTTPTransportFactory(
            base_url="https://example.com", message_type=HttpMessage
        )

        # Register factory
        registry.register("http", factory)

        # Create multiple transports
        transport1 = registry.create_transport("http")
        transport2 = registry.create_transport("http", timeout=5.0)

        # Verify they are different instances with correct properties
        assert transport1 is not transport2
        assert transport1.timeout == 30.0  # Default
        assert transport2.timeout == 5.0  # Custom
