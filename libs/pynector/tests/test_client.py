"""
Unit tests for the Pynector client class.
"""

from unittest.mock import AsyncMock

import anyio
import pytest

from pynector import Pynector
from pynector.errors import ConfigurationError, TimeoutError, TransportError
from tests.conftest import MockTransport


# Initialization Tests
@pytest.mark.anyio
async def test_init_with_defaults():
    """Test initialization with default parameters."""
    client = Pynector(transport_type="sdk")  # Use sdk since http is not available
    assert client._transport is None
    assert client._transport_type == "sdk"
    assert client._owns_transport is True
    assert client._config == {}


@pytest.mark.anyio
async def test_init_with_custom_config():
    """Test initialization with custom configuration."""
    config = {"timeout": 30.0, "retry_count": 3}
    client = Pynector(config=config, transport_type="sdk")  # Use sdk instead of http
    assert client._config == config
    assert client._get_config("timeout") == 30.0


@pytest.mark.anyio
async def test_init_with_transport():
    """Test initialization with a pre-configured transport."""
    transport = AsyncMock()
    client = Pynector(transport=transport)
    assert client._transport is transport
    assert client._owns_transport is False


@pytest.mark.anyio
async def test_init_with_invalid_transport_type(register_mock_transport_factory):
    """Test initialization with an invalid transport type."""
    with pytest.raises(ConfigurationError):
        Pynector(transport_type="invalid")


# Configuration Tests
@pytest.mark.anyio
async def test_get_config_from_instance():
    """Test getting configuration from instance config."""
    client = Pynector(
        config={"timeout": 30.0}, transport_type="sdk"
    )  # Use sdk instead of http
    assert client._get_config("timeout") == 30.0


@pytest.mark.anyio
async def test_get_config_from_env(monkeypatch):
    """Test getting configuration from environment variables."""
    monkeypatch.setenv("PYNECTOR_TIMEOUT", "45.0")
    client = Pynector(transport_type="sdk")  # Use sdk instead of http
    assert client._get_config("timeout") == "45.0"

    # Instance config should override environment
    client = Pynector(
        config={"timeout": 30.0}, transport_type="sdk"
    )  # Use sdk instead of http
    assert client._get_config("timeout") == 30.0


# Transport Management Tests
@pytest.mark.anyio
async def test_get_transport_creates_transport(register_mock_transport_factory):
    """Test that _get_transport creates a transport when needed."""
    client = Pynector(transport_type="mock")
    transport = await client._get_transport()
    assert transport is not None
    assert client._transport_initialized is True
    assert transport.connect_count == 1


@pytest.mark.anyio
async def test_get_transport_reuses_transport(register_mock_transport_factory):
    """Test that _get_transport reuses an existing transport."""
    client = Pynector(transport_type="mock")
    transport1 = await client._get_transport()
    transport2 = await client._get_transport()
    assert transport1 is transport2
    assert transport1.connect_count == 1  # Should only connect once


@pytest.mark.anyio
async def test_get_transport_with_connection_error(
    register_mock_transport_factory, mock_transport_factory
):
    """Test _get_transport with a connection error."""
    mock_transport_factory.transport.raise_on_connect = ConnectionError(
        "Connection failed"
    )
    client = Pynector(transport_type="mock")
    with pytest.raises(TransportError):
        await client._get_transport()


# Request Method Tests
@pytest.mark.anyio
async def test_request_success(register_mock_transport_factory, mock_transport):
    """Test successful request."""
    mock_transport.responses = [b"test response"]
    client = Pynector(transport_type="mock")
    response = await client.request({"test": "data"})
    assert response == b"test response"
    assert mock_transport.send_count == 1
    assert mock_transport.sent_data[0][0] == {"test": "data"}


@pytest.mark.anyio
async def test_request_with_options(register_mock_transport_factory, mock_transport):
    """Test request with additional options."""
    client = Pynector(transport_type="mock")
    await client.request({"test": "data"}, headers={"X-Test": "value"})
    assert mock_transport.sent_data[0][1]["headers"] == {"X-Test": "value"}


@pytest.mark.anyio
async def test_request_with_timeout(register_mock_transport_factory, mock_transport):
    """Test request with timeout."""
    client = Pynector(transport_type="mock")

    # Mock a slow response that will trigger timeout
    async def slow_receive():
        await anyio.sleep(0.5)
        yield b"too late"

    mock_transport.receive = slow_receive

    with pytest.raises(TimeoutError):
        await client.request({"test": "data"}, timeout=0.1)


@pytest.mark.anyio
async def test_request_with_transport_error(
    register_mock_transport_factory, mock_transport
):
    """Test request with transport error."""
    mock_transport.raise_on_send = ConnectionError("Send failed")
    client = Pynector(transport_type="mock")
    with pytest.raises(TransportError):
        await client.request({"test": "data"})


@pytest.mark.anyio
async def test_request_with_telemetry(register_mock_transport_factory, mock_telemetry):
    """Test request with telemetry."""
    mock_tracer, mock_logger = mock_telemetry
    client = Pynector(transport_type="mock")
    await client.request({"test": "data"})

    # Verify span was created
    assert mock_tracer.start_as_current_span.called
    assert mock_tracer.start_as_current_span.call_args[0][0] == "pynector.request"

    # Verify logging
    assert mock_logger.info.called


# Batch Request Tests
@pytest.mark.anyio
async def test_batch_request_success(register_mock_transport_factory, mock_transport):
    """Test successful batch request."""
    mock_transport.responses = [b"response 1", b"response 2", b"response 3"]
    client = Pynector(transport_type="mock")

    requests = [({"id": 1}, {}), ({"id": 2}, {}), ({"id": 3}, {})]

    responses = await client.batch_request(requests)
    assert len(responses) == 3
    assert all(not isinstance(r, Exception) for r in responses)
    assert mock_transport.send_count == 3


@pytest.mark.anyio
async def test_batch_request_with_max_concurrency(
    register_mock_transport_factory, mock_transport
):
    """Test batch request with max concurrency."""
    client = Pynector(transport_type="mock")

    requests = [({"id": i}, {}) for i in range(10)]

    # With max_concurrency=2, requests should be processed in batches
    await client.batch_request(requests, max_concurrency=2)
    assert mock_transport.send_count == 10


@pytest.mark.anyio
async def test_batch_request_with_timeout(
    register_mock_transport_factory, mock_transport
):
    """Test batch request with timeout."""
    client = Pynector(transport_type="mock")

    # Mock a slow response that will trigger timeout
    async def slow_receive():
        await anyio.sleep(0.5)
        yield b"too late"

    mock_transport.receive = slow_receive

    requests = [({"id": i}, {}) for i in range(5)]

    # Without raise_on_error, should return TimeoutError objects
    responses = await client.batch_request(requests, timeout=0.1)
    assert len(responses) == 5
    assert all(isinstance(r, TimeoutError) for r in responses)

    # With raise_on_error, should raise TimeoutError
    with pytest.raises(TimeoutError):
        await client.batch_request(requests, timeout=0.1, raise_on_error=True)


@pytest.mark.anyio
async def test_batch_request_with_partial_errors(
    register_mock_transport_factory, mock_transport_factory
):
    """Test batch request with some requests failing."""

    # Create a transport that fails on specific requests
    class PartialFailTransport(MockTransport):
        async def send(self, message, **options):
            self.send_count += 1
            self.sent_data.append((message, options))
            if isinstance(message, dict) and message.get("id") == 2:
                raise ConnectionError("Failed for id 2")

    mock_transport_factory.transport = PartialFailTransport()
    client = Pynector(transport_type="mock")

    requests = [({"id": 1}, {}), ({"id": 2}, {}), ({"id": 3}, {})]

    # Without raise_on_error, should return mix of responses and exceptions
    responses = await client.batch_request(requests)
    assert len(responses) == 3
    assert not isinstance(responses[0], Exception)
    assert isinstance(responses[1], TransportError)
    assert not isinstance(responses[2], Exception)


# Resource Management Tests
@pytest.mark.anyio
async def test_aclose(register_mock_transport_factory, mock_transport):
    """Test aclose method."""
    client = Pynector(transport_type="mock")
    await client._get_transport()  # Initialize transport
    await client.aclose()
    assert mock_transport.disconnect_count == 1
    assert client._transport is None
    assert client._transport_initialized is False


@pytest.mark.anyio
async def test_aclose_with_external_transport(mock_transport):
    """Test aclose with external transport."""
    client = Pynector(transport=mock_transport)
    await client.aclose()
    # Should not disconnect external transport
    assert mock_transport.disconnect_count == 0


@pytest.mark.anyio
async def test_async_context_manager(register_mock_transport_factory, mock_transport):
    """Test async context manager protocol."""
    # In the current implementation, connect is called when:
    # 1. _get_transport is called in __aenter__ which initializes the transport
    async with Pynector(transport_type="mock") as client:
        assert (
            mock_transport.connect_count == 1
        )  # Connect is called once during __aenter__
        await client.request({"test": "data"})

    # Should disconnect on exit
    assert mock_transport.disconnect_count == 1


# Retry Utility Tests
@pytest.mark.anyio
async def test_request_with_retry_success(
    register_mock_transport_factory, mock_transport
):
    """Test request_with_retry with immediate success."""
    client = Pynector(transport_type="mock")
    response = await client.request_with_retry({"test": "data"})
    assert response == b"mock response"
    assert mock_transport.send_count == 1


@pytest.mark.anyio
async def test_request_with_retry_after_failure(
    register_mock_transport_factory, mock_transport_factory
):
    """Test request_with_retry with success after failure."""

    # Create a transport that fails on first attempt
    class RetryTransport(MockTransport):
        def __init__(self):
            super().__init__()
            self.attempts = 0

        async def send(self, message, **options):
            self.send_count += 1
            self.sent_data.append((message, options))
            self.attempts += 1
            if self.attempts == 1:
                raise ConnectionError("First attempt fails")

    mock_transport_factory.transport = RetryTransport()
    client = Pynector(transport_type="mock")

    # Should succeed on second attempt
    response = await client.request_with_retry({"test": "data"}, retry_delay=0.01)
    assert response == b"mock response"
    assert mock_transport_factory.transport.send_count == 2


@pytest.mark.anyio
async def test_request_with_retry_max_retries(
    register_mock_transport_factory, mock_transport
):
    """Test request_with_retry with max retries exceeded."""
    mock_transport.raise_on_send = ConnectionError("Always fails")
    client = Pynector(transport_type="mock")

    with pytest.raises(TransportError):
        await client.request_with_retry(
            {"test": "data"}, max_retries=3, retry_delay=0.01
        )

    # Should have attempted 3 times
    assert mock_transport.send_count == 3
