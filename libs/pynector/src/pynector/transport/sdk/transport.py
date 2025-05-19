"""
SDK transport implementation.

This module provides a transport implementation for interacting with AI model provider SDKs,
such as OpenAI and Anthropic, following the Transport Protocol.
"""

import os
from collections.abc import AsyncIterator
from typing import Any, Optional

import anthropic
import httpx
import openai

from pynector.transport.errors import (
    ConnectionError,
    ConnectionRefusedError,
    ConnectionTimeoutError,
)
from pynector.transport.sdk.adapter import AnthropicAdapter, OpenAIAdapter
from pynector.transport.sdk.errors import (
    AuthenticationError,
    InvalidRequestError,
    PermissionError,
    RateLimitError,
    RequestTooLargeError,
    ResourceNotFoundError,
    SdkTransportError,
)


class SdkTransport:
    """SDK transport implementation using OpenAI and Anthropic SDKs."""

    def __init__(
        self,
        sdk_type: str = "openai",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 60.0,
        **kwargs: Any,
    ):
        """Initialize the transport with configuration options.

        Args:
            sdk_type: The SDK type to use. Can be "openai" or "anthropic".
            api_key: The API key to use. If not provided, will use environment variables.
            base_url: The base URL to use. If not provided, will use the default.
            timeout: The timeout in seconds for API calls.
            **kwargs: Additional SDK-specific configuration options.
        """
        self.sdk_type = sdk_type
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self.config = kwargs
        self._client = None
        self._adapter = None

    async def connect(self) -> None:
        """Establish the connection to the SDK.

        Raises:
            ConnectionError: If the connection cannot be established.
            TimeoutError: If the connection attempt times out.
        """
        if self._client is not None:
            return

        try:
            if self.sdk_type == "openai":
                self._client = await self._create_openai_client()
                self._adapter = OpenAIAdapter(self._client)
            elif self.sdk_type == "anthropic":
                self._client = await self._create_anthropic_client()
                self._adapter = AnthropicAdapter(self._client)
            else:
                raise ValueError(f"Unsupported SDK type: {self.sdk_type}")
        except Exception as e:
            raise self._translate_connection_error(e)

    async def disconnect(self) -> None:
        """Close the connection to the SDK."""
        self._client = None
        self._adapter = None

    async def send(self, data: bytes) -> None:
        """Send data over the transport.

        Args:
            data: The data to send.

        Raises:
            ConnectionError: If the connection is closed or broken.
            TransportError: For other transport-specific errors.
        """
        if self._adapter is None:
            raise ConnectionError("Transport not connected")

        try:
            prompt = data.decode("utf-8")
            model = self.config.get("model")
            kwargs = {k: v for k, v in self.config.items() if k != "model"}
            await self._adapter.complete(prompt, model=model, **kwargs)
        except Exception as e:
            raise self._translate_error(e)

    async def receive(self) -> AsyncIterator[bytes]:
        """Receive data from the transport.

        Returns:
            An async iterator yielding data as it is received.

        Raises:
            ConnectionError: If the connection is closed or broken.
            TransportError: For other transport-specific errors.
        """
        if self._adapter is None:
            raise ConnectionError("Transport not connected")

        try:
            prompt = self.config.get("prompt", "Generate a response")
            model = self.config.get("model")
            kwargs = {
                k: v for k, v in self.config.items() if k not in ["prompt", "model"]
            }
            async for chunk in self._adapter.stream(prompt, model=model, **kwargs):
                yield chunk
        except Exception as e:
            raise self._translate_error(e)

    async def _create_openai_client(self) -> openai.AsyncOpenAI:
        """Create an OpenAI client.

        Returns:
            The OpenAI client.

        Raises:
            ConnectionError: If the client cannot be created.
        """
        try:
            return openai.AsyncOpenAI(
                api_key=self.api_key or os.environ.get("OPENAI_API_KEY"),
                base_url=self.base_url,
                timeout=self.timeout,
                **{
                    k: v
                    for k, v in self.config.items()
                    if k in ["organization", "max_retries"]
                },
            )
        except Exception as e:
            raise ConnectionError(f"Failed to create OpenAI client: {str(e)}")

    async def _create_anthropic_client(self) -> anthropic.AsyncAnthropic:
        """Create an Anthropic client.

        Returns:
            The Anthropic client.

        Raises:
            ConnectionError: If the client cannot be created.
        """
        try:
            return anthropic.AsyncAnthropic(
                api_key=self.api_key or os.environ.get("ANTHROPIC_API_KEY"),
                base_url=self.base_url,
                timeout=httpx.Timeout(self.timeout),
                **{k: v for k, v in self.config.items() if k in ["auth_token"]},
            )
        except Exception as e:
            raise ConnectionError(f"Failed to create Anthropic client: {str(e)}")

    def _translate_connection_error(self, error: Exception) -> Exception:
        """Translate SDK connection errors to Transport errors.

        Args:
            error: The SDK error.

        Returns:
            The translated error.
        """
        if isinstance(error, httpx.TimeoutException):
            return ConnectionTimeoutError(f"Connection timeout: {str(error)}")
        elif isinstance(error, httpx.ConnectError):
            return ConnectionRefusedError(f"Connection refused: {str(error)}")
        else:
            return ConnectionError(f"Connection error: {str(error)}")

    def _translate_error(self, error: Exception) -> Exception:
        """Translate SDK errors to Transport errors.

        Args:
            error: The SDK error.

        Returns:
            The translated error.
        """
        error_class_name = error.__class__.__name__
        error_module = getattr(error, "__module__", "")

        # OpenAI errors
        if error_module.startswith("openai"):
            if error_class_name == "AuthenticationError":
                return AuthenticationError(f"Authentication failed: {str(error)}")
            elif error_class_name == "RateLimitError":
                return RateLimitError(f"Rate limit exceeded: {str(error)}")
            elif error_class_name == "APITimeoutError":
                return ConnectionTimeoutError(f"API timeout: {str(error)}")
            elif error_class_name == "APIConnectionError":
                return ConnectionError(f"API connection error: {str(error)}")
            elif error_class_name == "BadRequestError":
                return InvalidRequestError(f"Bad request: {str(error)}")
            elif error_class_name == "NotFoundError":
                return ResourceNotFoundError(f"Resource not found: {str(error)}")

        # Anthropic errors
        elif error_module.startswith("anthropic"):
            if error_class_name == "APIStatusError":
                status_code = getattr(error, "status_code", None)
                if status_code == 401:
                    return AuthenticationError(f"Authentication failed: {str(error)}")
                elif status_code == 403:
                    return PermissionError(f"Permission denied: {str(error)}")
                elif status_code == 404:
                    return ResourceNotFoundError(f"Resource not found: {str(error)}")
                elif status_code == 429:
                    return RateLimitError(f"Rate limit exceeded: {str(error)}")
                elif status_code == 400:
                    return InvalidRequestError(f"Bad request: {str(error)}")
                elif status_code == 413:
                    return RequestTooLargeError(f"Request too large: {str(error)}")

        # httpx errors
        elif isinstance(error, httpx.TimeoutException):
            return ConnectionTimeoutError(f"Connection timeout: {str(error)}")
        elif isinstance(error, httpx.ConnectError):
            return ConnectionRefusedError(f"Connection refused: {str(error)}")
        elif isinstance(error, httpx.RequestError):
            return ConnectionError(f"Request error: {str(error)}")

        # Default case
        return SdkTransportError(f"SDK error: {str(error)}")

    async def __aenter__(self) -> "SdkTransport":
        """Enter the async context, establishing the connection.

        Returns:
            The transport instance.

        Raises:
            ConnectionError: If the connection cannot be established.
            TimeoutError: If the connection attempt times out.
        """
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit the async context, closing the connection."""
        await self.disconnect()
