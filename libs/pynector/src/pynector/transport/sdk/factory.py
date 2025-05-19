"""
Factory for creating SDK transport instances.

This module provides a factory for creating and configuring SDK transport instances,
implementing the TransportFactory protocol.
"""

from typing import Any, Optional

from pynector.transport.sdk.transport import SdkTransport


class SdkTransportFactory:
    """Factory for creating SDK transport instances."""

    def __init__(
        self,
        sdk_type: str = "openai",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 60.0,
        **kwargs: Any,
    ):
        """Initialize the factory with default configuration options.

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
        self.default_config = kwargs

    def create_transport(self, **kwargs: Any) -> SdkTransport:
        """Create a new SDK transport instance.

        Args:
            **kwargs: Additional configuration options that override the defaults.

        Returns:
            A new SDK transport instance.

        Raises:
            ValueError: If the configuration is invalid.
        """
        # Merge kwargs with default_config, with kwargs taking precedence
        config = {**self.default_config, **kwargs}

        # Extract specific parameters
        sdk_type = kwargs.get("sdk_type", self.sdk_type)
        api_key = kwargs.get("api_key", self.api_key)
        base_url = kwargs.get("base_url", self.base_url)
        timeout = kwargs.get("timeout", self.timeout)

        # Remove extracted parameters from config to avoid duplication
        for key in ["sdk_type", "api_key", "base_url", "timeout"]:
            config.pop(key, None)

        return SdkTransport(
            sdk_type=sdk_type,
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            **config,
        )
