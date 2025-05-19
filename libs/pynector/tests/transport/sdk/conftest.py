"""
Fixtures for SDK transport tests.
"""

import asyncio
from collections.abc import AsyncIterator
from typing import Any, Optional

import pytest

from pynector.transport.sdk.adapter import SDKAdapter


class MockAdapter(SDKAdapter):
    """Mock adapter for testing."""

    def __init__(
        self, responses: dict[str, Any] = None, errors: dict[str, Exception] = None
    ):
        """Initialize the mock adapter with responses and errors.

        Args:
            responses: Mapping of prompts to responses.
            errors: Mapping of prompts to errors.
        """
        self.responses = responses or {}
        self.errors = errors or {}
        self.complete_calls = []
        self.stream_calls = []

    async def complete(
        self, prompt: str, model: Optional[str] = None, **kwargs: Any
    ) -> str:
        """Generate a completion for the given prompt."""
        # Store the call for verification
        self.complete_calls.append((prompt, model, kwargs))

        # Check if this prompt should raise an error
        if prompt in self.errors:
            raise self.errors[prompt]

        # Return the response
        return self.responses.get(prompt, f"Mock response to: {prompt}")

    async def stream(
        self, prompt: str, model: Optional[str] = None, **kwargs: Any
    ) -> AsyncIterator[bytes]:
        """Stream a completion for the given prompt."""
        self.stream_calls.append((prompt, model, kwargs))

        if prompt in self.errors:
            raise self.errors[prompt]

        response = self.responses.get(prompt, f"Mock response to: {prompt}")
        chunks = response.split()

        for chunk in chunks:
            yield chunk.encode("utf-8")
            await asyncio.sleep(0.01)  # Simulate streaming delay


@pytest.fixture
def mock_adapter():
    """Fixture for creating a mock adapter."""
    return MockAdapter(
        responses={"hello": "Hello, world!", "test": "This is a test response."},
        errors={
            "error": ValueError("Test error"),
            "auth_error": Exception("Authentication failed"),
        },
    )


@pytest.fixture
def mock_transport(mock_adapter):
    """Fixture for creating a transport with a mock adapter."""
    from pynector.transport.sdk.transport import SdkTransport

    transport = SdkTransport(sdk_type="mock")
    transport._adapter = mock_adapter

    return transport
