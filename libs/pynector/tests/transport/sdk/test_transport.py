"""
Tests for the SdkTransport class.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from pynector.transport.errors import (
    ConnectionError,
    ConnectionRefusedError,
    ConnectionTimeoutError,
)
from pynector.transport.sdk.adapter import AnthropicAdapter, OpenAIAdapter
from pynector.transport.sdk.errors import (
    AuthenticationError,
    InvalidRequestError,
    RateLimitError,
    ResourceNotFoundError,
    SdkTransportError,
)
from pynector.transport.sdk.transport import SdkTransport


# Custom error classes for testing
class MockOpenAIAuthError(Exception):
    """Mock OpenAI authentication error."""

    pass


class MockOpenAIRateLimitError(Exception):
    """Mock OpenAI rate limit error."""

    pass


class MockOpenAITimeoutError(Exception):
    """Mock OpenAI timeout error."""

    pass


class MockOpenAIConnectionError(Exception):
    """Mock OpenAI connection error."""

    pass


class MockOpenAIBadRequestError(Exception):
    """Mock OpenAI bad request error."""

    pass


class MockOpenAINotFoundError(Exception):
    """Mock OpenAI not found error."""

    pass


# Set module and name attributes
MockOpenAIAuthError.__module__ = "openai"
setattr(MockOpenAIAuthError, "__name__", "AuthenticationError")
MockOpenAIRateLimitError.__module__ = "openai"
setattr(MockOpenAIRateLimitError, "__name__", "RateLimitError")
MockOpenAITimeoutError.__module__ = "openai"
setattr(MockOpenAITimeoutError, "__name__", "APITimeoutError")
MockOpenAIConnectionError.__module__ = "openai"
setattr(MockOpenAIConnectionError, "__name__", "APIConnectionError")
MockOpenAIBadRequestError.__module__ = "openai"
setattr(MockOpenAIBadRequestError, "__name__", "BadRequestError")
MockOpenAINotFoundError.__module__ = "openai"
setattr(MockOpenAINotFoundError, "__name__", "NotFoundError")


# Test initialization
def test_sdk_transport_init():
    """Test SdkTransport initialization."""
    # Default initialization
    transport = SdkTransport()
    assert transport.sdk_type == "openai"
    assert transport.api_key is None
    assert transport.base_url is None
    assert transport.timeout == 60.0
    assert transport.config == {}
    assert transport._client is None
    assert transport._adapter is None

    # Custom initialization
    transport = SdkTransport(
        sdk_type="anthropic",
        api_key="test-key",
        base_url="https://example.com",
        timeout=30.0,
        model="claude-3-opus-20240229",
    )
    assert transport.sdk_type == "anthropic"
    assert transport.api_key == "test-key"
    assert transport.base_url == "https://example.com"
    assert transport.timeout == 30.0
    assert transport.config == {"model": "claude-3-opus-20240229"}
    assert transport._client is None
    assert transport._adapter is None


# Test connect method
@pytest.mark.asyncio
async def test_sdk_transport_connect_openai():
    """Test SdkTransport connect method with OpenAI."""
    with patch("openai.AsyncOpenAI") as mock_openai:
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        transport = SdkTransport(sdk_type="openai", api_key="test-key")
        await transport.connect()

        # Verify client was created
        mock_openai.assert_called_once()
        assert transport._client is mock_client
        assert isinstance(transport._adapter, OpenAIAdapter)


@pytest.mark.asyncio
async def test_sdk_transport_connect_anthropic():
    """Test SdkTransport connect method with Anthropic."""
    with patch("anthropic.AsyncAnthropic") as mock_anthropic:
        mock_client = MagicMock()
        mock_anthropic.return_value = mock_client

        transport = SdkTransport(sdk_type="anthropic", api_key="test-key")
        await transport.connect()

        # Verify client was created
        mock_anthropic.assert_called_once()
        assert transport._client is mock_client
        assert isinstance(transport._adapter, AnthropicAdapter)


@pytest.mark.asyncio
async def test_sdk_transport_connect_unsupported():
    """Test SdkTransport connect method with unsupported SDK type."""
    transport = SdkTransport(sdk_type="unsupported")

    with pytest.raises(ConnectionError, match="Unsupported SDK type"):
        await transport.connect()


@pytest.mark.asyncio
async def test_sdk_transport_connect_error():
    """Test SdkTransport connect method with connection error."""
    with patch("openai.AsyncOpenAI") as mock_openai:
        mock_openai.side_effect = httpx.ConnectError("Connection refused")

        transport = SdkTransport(sdk_type="openai")

        with pytest.raises(ConnectionError, match="Connection refused"):
            await transport.connect()


# Test disconnect method
@pytest.mark.asyncio
async def test_sdk_transport_disconnect():
    """Test SdkTransport disconnect method."""
    with patch("openai.AsyncOpenAI") as mock_openai:
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        transport = SdkTransport(sdk_type="openai")
        await transport.connect()

        assert transport._client is not None
        assert transport._adapter is not None

        await transport.disconnect()

        assert transport._client is None
        assert transport._adapter is None


# Test send method
@pytest.mark.asyncio
async def test_sdk_transport_send():
    """Test SdkTransport send method."""
    # Setup mock adapter
    mock_adapter = MagicMock()
    mock_adapter.complete = AsyncMock()

    # Create transport with mock adapter
    transport = SdkTransport(sdk_type="openai", model="gpt-4o")
    transport._adapter = mock_adapter

    # Test send method
    await transport.send(b"Test prompt")

    # Verify adapter was called correctly
    mock_adapter.complete.assert_called_once_with("Test prompt", model="gpt-4o")


@pytest.mark.asyncio
async def test_sdk_transport_send_not_connected():
    """Test SdkTransport send method when not connected."""
    transport = SdkTransport()

    with pytest.raises(ConnectionError, match="not connected"):
        await transport.send(b"Test prompt")


@pytest.mark.asyncio
async def test_sdk_transport_send_error():
    """Test SdkTransport send method with error."""
    # Setup mock adapter
    mock_adapter = MagicMock()
    mock_adapter.complete = AsyncMock(
        side_effect=MockOpenAIAuthError("Invalid API key")
    )

    # Create transport with mock adapter
    transport = SdkTransport(sdk_type="openai")
    transport._adapter = mock_adapter

    # Test send method
    with pytest.raises(AuthenticationError, match="Authentication failed"):
        await transport.send(b"Test prompt")


# Test receive method
@pytest.mark.asyncio
async def test_sdk_transport_receive():
    """Test SdkTransport receive method."""
    # Setup mock adapter
    mock_adapter = MagicMock()

    async def mock_stream(*args, **kwargs):
        yield b"Test "
        yield b"response"

    mock_adapter.stream = mock_stream

    # Create transport with mock adapter
    transport = SdkTransport(sdk_type="openai", prompt="Custom prompt", model="gpt-4o")
    transport._adapter = mock_adapter

    # Test receive method
    chunks = []
    async for chunk in transport.receive():
        chunks.append(chunk)

    # Verify result
    assert chunks == [b"Test ", b"response"]


@pytest.mark.asyncio
async def test_sdk_transport_receive_not_connected():
    """Test SdkTransport receive method when not connected."""
    transport = SdkTransport()

    with pytest.raises(ConnectionError, match="not connected"):
        async for _ in transport.receive():
            pass


@pytest.mark.asyncio
async def test_sdk_transport_receive_error():
    """Test SdkTransport receive method with error."""
    # Setup mock adapter
    mock_adapter = MagicMock()

    # Create a proper async generator that raises an exception
    class MockAsyncIterator:
        def __aiter__(self):
            return self

        async def __anext__(self):
            raise MockOpenAIRateLimitError("Rate limit exceeded")

    mock_adapter.stream = MagicMock(return_value=MockAsyncIterator())

    # Create transport with mock adapter
    transport = SdkTransport(sdk_type="openai")
    transport._adapter = mock_adapter

    # Test receive method
    with pytest.raises(RateLimitError, match="Rate limit exceeded"):
        async for _ in transport.receive():
            pass
            pass


# Test error translation
def test_sdk_transport_translate_error():
    """Test SdkTransport error translation."""
    transport = SdkTransport()

    # Test OpenAI errors using custom exceptions
    assert isinstance(
        transport._translate_error(MockOpenAIAuthError("test")), AuthenticationError
    )
    assert isinstance(
        transport._translate_error(MockOpenAIRateLimitError("test")), RateLimitError
    )
    assert isinstance(
        transport._translate_error(MockOpenAITimeoutError("test")),
        ConnectionTimeoutError,
    )
    assert isinstance(
        transport._translate_error(MockOpenAIConnectionError("test")), ConnectionError
    )
    assert isinstance(
        transport._translate_error(MockOpenAIBadRequestError("test")),
        InvalidRequestError,
    )
    assert isinstance(
        transport._translate_error(MockOpenAINotFoundError("test")),
        ResourceNotFoundError,
    )

    # Test httpx errors
    assert isinstance(
        transport._translate_error(httpx.TimeoutException("test")),
        ConnectionTimeoutError,
    )
    assert isinstance(
        transport._translate_error(httpx.ConnectError("test")), ConnectionRefusedError
    )
    assert isinstance(
        transport._translate_error(httpx.RequestError("test")), ConnectionError
    )

    # Test default case
    assert isinstance(transport._translate_error(Exception("test")), SdkTransportError)


# Test async context manager
@pytest.mark.asyncio
async def test_sdk_transport_context_manager():
    """Test SdkTransport async context manager."""
    with patch.object(SdkTransport, "connect", AsyncMock()) as mock_connect:
        with patch.object(SdkTransport, "disconnect", AsyncMock()) as mock_disconnect:
            transport = SdkTransport()

            async with transport as t:
                assert t is transport
                mock_connect.assert_called_once()
                mock_disconnect.assert_not_called()

            mock_disconnect.assert_called_once()
