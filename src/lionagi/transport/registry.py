"""
Registry for transport factories.

This module provides a registry for transport factories, allowing for
dynamic registration and lookup of transport factories.
"""

import importlib
from typing import Any

from pynector.transport.factory import TransportFactory
from pynector.transport.protocol import Transport


class TransportFactoryRegistry:
    """Registry for transport factories."""

    def __init__(self):
        """Initialize a new transport factory registry."""
        self._factories = {}

    def register(self, name: str, factory: TransportFactory) -> None:
        """Register a transport factory.

        Args:
            name: The name to register the factory under.
            factory: The factory instance.
        """
        self._factories[name] = factory

    def get(self, name: str) -> TransportFactory:
        """Get a transport factory by name.

        Args:
            name: The name of the factory to get.

        Returns:
            The factory instance.

        Raises:
            KeyError: If no factory is registered with the given name.
        """
        return self._factories[name]

    def create_transport(self, name: str, **kwargs: Any) -> Transport:
        """Create a transport using a registered factory.

        Args:
            name: The name of the factory to use.
            **kwargs: Transport-specific configuration options.

        Returns:
            A new transport instance.

        Raises:
            KeyError: If no factory is registered with the given name.
        """
        factory = self.get(name)
        return factory.create_transport(**kwargs)

    def get_registered_names(self) -> list[str]:
        """Get a list of all registered factory names.

        Returns:
            A list of all registered factory names.
        """
        return list(self._factories.keys())


# Global registry instance
_registry = None


def get_transport_factory_registry() -> TransportFactoryRegistry:
    """Get the global transport factory registry.

    Returns:
        The global transport factory registry.
    """
    global _registry
    if _registry is None:
        _registry = TransportFactoryRegistry()

        # Try to register built-in transport factories
        # We use importlib to avoid issues with circular imports
        # or missing optional dependencies
        try:
            http_module = importlib.import_module("pynector.transport.http.factory")
            http_factory = getattr(http_module, "HttpTransportFactory")
            _registry.register("http", http_factory())
        except (ImportError, AttributeError):
            # HTTP transport is optional
            pass

        try:
            sdk_module = importlib.import_module("pynector.transport.sdk.factory")
            sdk_factory = getattr(sdk_module, "SdkTransportFactory")
            _registry.register("sdk", sdk_factory())
        except (ImportError, AttributeError):
            # SDK transport is optional
            pass

    return _registry
