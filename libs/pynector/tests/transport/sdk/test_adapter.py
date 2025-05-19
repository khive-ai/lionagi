"""
Tests for SDK adapter implementations.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from pynector.transport.sdk.adapter import AnthropicAdapter, OpenAIAdapter, SDKAdapter


# Test the abstract base class
def test_sdk_adapter_abstract():
    """Test that SDKAdapter is an abstract base class."""
    with pytest.raises(TypeError):
        SDKAdapter()  # Should raise TypeError because it's abstract


# Mock classes for testing
class MockOpenAIClient:
    def __init__(self):
        self.chat = MagicMock()
        self.chat.completions = MagicMock()
        self.chat.completions.create = AsyncMock()
        self.chat.completions.stream = AsyncMock()


class MockAnthropicClient:
    def __init__(self):
        self.messages = MagicMock()
        self.messages.create = AsyncMock()
        self.messages.create.with_streaming_response = AsyncMock()


# OpenAI adapter tests
@pytest.mark.asyncio
async def test_openai_adapter_complete():
    """Test OpenAI adapter complete method."""
    # Setup mock
    client = MockOpenAIClient()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Test response"
    client.chat.completions.create.return_value = mock_response

    # Create adapter
    adapter = OpenAIAdapter(client)

    # Test complete method
    result = await adapter.complete("Test prompt", model="gpt-4o")

    # Verify result
    assert result == "Test response"

    # Verify client was called correctly
    client.chat.completions.create.assert_called_once()
    call_args = client.chat.completions.create.call_args[1]
    assert call_args["messages"] == [{"role": "user", "content": "Test prompt"}]
    assert call_args["model"] == "gpt-4o"


@pytest.mark.asyncio
async def test_openai_adapter_stream():
    """Test OpenAI adapter stream method."""
    # Setup mock
    client = MockOpenAIClient()

    # Create a proper async iterator for testing
    class MockAsyncIterator:
        def __init__(self, items):
            self.items = items

        def __aiter__(self):
            return self

        async def __anext__(self):
            if not self.items:
                raise StopAsyncIteration
            return self.items.pop(0)

    # Create mock items
    mock_items = [
        MagicMock(type="content.delta", delta="Test "),
        MagicMock(type="content.delta", delta="response"),
        MagicMock(type="not.content.delta", delta="ignored"),
    ]

    # Create a mock context manager
    mock_cm = AsyncMock()
    mock_cm.__aenter__.return_value = MockAsyncIterator(mock_items)

    # Make the stream method return our mock context manager
    client.chat.completions.stream = MagicMock(return_value=mock_cm)

    # Create adapter
    adapter = OpenAIAdapter(client)

    # Test stream method
    chunks = []
    async for chunk in adapter.stream("Test prompt", model="gpt-4o"):
        chunks.append(chunk)

    # Verify result
    assert chunks == [b"Test ", b"response"]

    # Verify client was called correctly
    client.chat.completions.stream.assert_called_once()
    call_args = client.chat.completions.stream.call_args[1]
    assert call_args["messages"] == [{"role": "user", "content": "Test prompt"}]
    assert call_args["model"] == "gpt-4o"


# Anthropic adapter tests
@pytest.mark.asyncio
async def test_anthropic_adapter_complete():
    """Test Anthropic adapter complete method."""
    # Setup mock
    client = MockAnthropicClient()
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="Test response")]
    client.messages.create.return_value = mock_response

    # Create adapter
    adapter = AnthropicAdapter(client)

    # Test complete method
    result = await adapter.complete("Test prompt", model="claude-3-opus-20240229")

    # Verify result
    assert result == "Test response"

    # Verify client was called correctly
    client.messages.create.assert_called_once()
    call_args = client.messages.create.call_args[1]
    assert call_args["messages"] == [{"role": "user", "content": "Test prompt"}]
    assert call_args["model"] == "claude-3-opus-20240229"


@pytest.mark.asyncio
async def test_anthropic_adapter_stream():
    """Test Anthropic adapter stream method."""
    # Setup mock
    client = MockAnthropicClient()

    # Create a proper async iterator for testing
    class MockTextIterator:
        def __init__(self, items):
            self.items = items.copy()

        def __aiter__(self):
            return self

        async def __anext__(self):
            if not self.items:
                raise StopAsyncIteration
            return self.items.pop(0)

    # Create a mock response with iter_text method
    mock_response = AsyncMock()
    mock_response.iter_text = MagicMock(
        return_value=MockTextIterator(["Test ", "response"])
    )

    # Create a mock context manager
    mock_cm = AsyncMock()
    mock_cm.__aenter__.return_value = mock_response

    # Make the with_streaming_response method return our mock context manager
    client.messages.create.with_streaming_response = MagicMock(return_value=mock_cm)

    # Create adapter
    adapter = AnthropicAdapter(client)

    # Test stream method
    chunks = []
    async for chunk in adapter.stream("Test prompt", model="claude-3-opus-20240229"):
        chunks.append(chunk)

    # Verify result
    assert chunks == [b"Test ", b"response"]

    # Verify client was called correctly
    client.messages.create.with_streaming_response.assert_called_once()
    call_args = client.messages.create.with_streaming_response.call_args[1]
    assert call_args["messages"] == [{"role": "user", "content": "Test prompt"}]
    assert call_args["model"] == "claude-3-opus-20240229"
