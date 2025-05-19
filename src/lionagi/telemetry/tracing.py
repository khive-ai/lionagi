"""
Tracing implementations for the telemetry module.

This module provides tracing implementations, including no-op implementations
for when OpenTelemetry is not available.
"""

from typing import Any, Optional

from pynector.telemetry import HAS_OPENTELEMETRY

# Import at module level for patching in tests
if HAS_OPENTELEMETRY:
    from opentelemetry.context import detach
else:
    # Define dummy function for patching in tests
    def detach(token):
        pass


class NoOpSpan:
    """No-op implementation of a span."""

    def __init__(self, name: str = "", attributes: Optional[dict[str, Any]] = None):
        """
        Initialize a new no-op span.

        Args:
            name: The name of the span
            attributes: Optional attributes to set on the span
        """
        self.name = name
        self.attributes = attributes or {}

    def __enter__(self):
        """Enter the span context."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the span context."""
        pass

    async def __aenter__(self):
        """Enter the async span context."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit the async span context."""
        pass

    def set_attribute(self, key: str, value: Any) -> None:
        """
        Set an attribute on the span.

        Args:
            key: The attribute key
            value: The attribute value
        """
        self.attributes[key] = value

    def add_event(self, name: str, attributes: Optional[dict[str, Any]] = None) -> None:
        """
        Add an event to the span.

        Args:
            name: The event name
            attributes: Optional attributes for the event
        """
        pass

    def record_exception(self, exception: Exception) -> None:
        """
        Record an exception in the span.

        Args:
            exception: The exception to record
        """
        pass

    def set_status(self, status) -> None:
        """
        Set the status of the span.

        Args:
            status: The status to set
        """
        pass


class AsyncSpanWrapper:
    """Wrapper to make a regular span work as an async context manager."""

    def __init__(self, span, token=None):
        """
        Initialize a new async span wrapper.

        Args:
            span: The span to wrap
            token: Optional context token to detach when exiting
        """
        self.span = span
        self.token = token

    async def __aenter__(self):
        """Enter the async span context."""
        self.span.__enter__()
        return self.span

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit the async span context."""
        try:
            self.span.__exit__(exc_type, exc_val, exc_tb)
        finally:
            if self.token is not None:
                # Use the module-level detach function
                detach(self.token)

    # Delegate methods to the wrapped span
    def set_attribute(self, key, value):
        """Set an attribute on the span."""
        return self.span.set_attribute(key, value)

    def add_event(self, name, attributes=None):
        """Add an event to the span."""
        return self.span.add_event(name, attributes)

    def record_exception(self, exception):
        """Record an exception in the span."""
        return self.span.record_exception(exception)

    def set_status(self, status):
        """Set the status of the span."""
        return self.span.set_status(status)
