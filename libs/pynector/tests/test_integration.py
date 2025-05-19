"""
Integration tests for the Pynector client.

These tests verify the integration between the Pynector client and other components
such as transports and telemetry.
"""

import pytest

from pynector import Pynector


@pytest.mark.anyio
@pytest.mark.skip("Skip due to connection issues in CI")
async def test_integration_with_http_transport():
    """Test integration with HTTP transport."""
    # Create a mock HTTP server
    from tests.transport.http.mock_server import MockHTTPServer

    async with MockHTTPServer() as server:
        # Create client with HTTP transport
        client = Pynector(
            transport_type="http",
            base_url=server.base_url,
            headers={"Content-Type": "application/json"},
        )

        # Configure server to return a specific response
        server.add_route("/test", {"result": "success"})

        # Make a request
        response = await client.request({"path": "/test", "method": "GET"})

        # Verify response
        assert isinstance(response, dict)
        assert response["result"] == "success"

        # Verify server received the request
        assert len(server.requests) == 1
        assert server.requests[0]["path"] == "/test"
        assert server.requests[0]["method"] == "GET"


@pytest.mark.anyio
@pytest.mark.skip("Skip due to connection issues in CI")
async def test_integration_batch_request():
    """Test batch request with real concurrency."""
    # Create a mock HTTP server
    from tests.transport.http.mock_server import MockHTTPServer

    async with MockHTTPServer() as server:
        # Create client with HTTP transport
        client = Pynector(
            transport_type="http",
            base_url=server.base_url,
            headers={"Content-Type": "application/json"},
        )

        # Configure server to return different responses
        server.add_route("/1", {"id": 1})
        server.add_route("/2", {"id": 2})
        server.add_route("/3", {"id": 3})

        # Create batch request
        requests = [
            ({"path": "/1", "method": "GET"}, {}),
            ({"path": "/2", "method": "GET"}, {}),
            ({"path": "/3", "method": "GET"}, {}),
        ]

        # Make batch request
        responses = await client.batch_request(requests, max_concurrency=2)

        # Verify responses
        assert len(responses) == 3
        assert all(not isinstance(r, Exception) for r in responses)
        assert responses[0]["id"] == 1
        assert responses[1]["id"] == 2
        assert responses[2]["id"] == 3

        # Verify server received all requests
        assert len(server.requests) == 3
        assert {req["path"] for req in server.requests} == {"/1", "/2", "/3"}


@pytest.mark.anyio
@pytest.mark.skip("Skip due to connection issues in CI")
async def test_integration_with_telemetry():
    """Test integration with telemetry."""
    # Skip if OpenTelemetry is not available
    pytest.importorskip("opentelemetry")

    # Configure telemetry
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

    from pynector.telemetry import configure_telemetry

    # Set up a tracer provider with a simple processor
    tracer_provider = TracerProvider()
    span_processor = SimpleSpanProcessor(ConsoleSpanExporter())
    tracer_provider.add_span_processor(span_processor)
    trace.set_tracer_provider(tracer_provider)

    # Configure telemetry
    configure_telemetry(service_name="test-service")

    # Create a mock HTTP server
    from tests.transport.http.mock_server import MockHTTPServer

    async with MockHTTPServer() as server:
        # Create client with HTTP transport and telemetry
        client = Pynector(
            transport_type="http", base_url=server.base_url, enable_telemetry=True
        )

        # Configure server to return a response
        server.add_route("/test", {"result": "success"})

        # Make a request
        response = await client.request({"path": "/test", "method": "GET"})

        # Verify response
        assert response["result"] == "success"


@pytest.mark.anyio
@pytest.mark.performance
@pytest.mark.skip("Skip due to connection issues in CI")
async def test_performance_batch_request():
    """Test performance of batch request with different concurrency limits."""
    # Create a mock HTTP server
    import time

    from tests.transport.http.mock_server import MockHTTPServer

    async with MockHTTPServer() as server:
        # Create client with HTTP transport
        client = Pynector(
            transport_type="http",
            base_url=server.base_url,
            headers={"Content-Type": "application/json"},
        )

        # Configure server to return a response with delay
        async def delayed_handler():
            import asyncio

            await asyncio.sleep(0.01)
            return {"result": "success"}

        server.add_route("/test", delayed_handler)

        # Create a large batch of requests
        requests = [({"path": "/test", "method": "GET"}, {}) for _ in range(50)]

        # Test with different concurrency limits
        concurrency_limits = [1, 5, 10, 20, 50]  # No limit = 50
        results = []

        for limit in concurrency_limits:
            start_time = time.time()
            await client.batch_request(requests, max_concurrency=limit)
            duration = time.time() - start_time
            results.append((limit, duration))

        # Verify that higher concurrency leads to faster execution
        # This is a simple check - in real tests we'd want more sophisticated analysis
        durations = [duration for _, duration in results]
        assert durations[0] > durations[-1]  # Sequential should be slower than parallel
