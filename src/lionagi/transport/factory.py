"""
Factory for creating transport instances.

This module defines the TransportFactory protocol, which is responsible for
creating and configuring transport instances.
"""

from typing import Any, Generic, Protocol, TypeVar

T = TypeVar("T")


class TransportFactory(Protocol, Generic[T]):
    """Protocol defining the interface for transport factories."""

    def create_transport(self, **kwargs: Any) -> T:
        """Create a new transport instance.

        Args:
            **kwargs: Transport-specific configuration options.

        Returns:
            A new transport instance.

        Raises:
            ValueError: If the configuration is invalid.
        """
        ...
