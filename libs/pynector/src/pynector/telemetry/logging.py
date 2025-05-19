"""
Logging implementations for the telemetry module.

This module provides logging implementations, including no-op implementations
for when structlog is not available.
"""

from typing import Any


class NoOpLogger:
    """No-op implementation of a logger."""

    def __init__(self, name: str = ""):
        """
        Initialize a new no-op logger.

        Args:
            name: The name of the logger
        """
        self.name = name

    def debug(self, event: str, **kwargs: Any) -> None:
        """
        Log a debug message (no-op).

        Args:
            event: The event name
            **kwargs: Additional key-value pairs to include in the log
        """
        pass

    def info(self, event: str, **kwargs: Any) -> None:
        """
        Log an info message (no-op).

        Args:
            event: The event name
            **kwargs: Additional key-value pairs to include in the log
        """
        pass

    def warning(self, event: str, **kwargs: Any) -> None:
        """
        Log a warning message (no-op).

        Args:
            event: The event name
            **kwargs: Additional key-value pairs to include in the log
        """
        pass

    def error(self, event: str, **kwargs: Any) -> None:
        """
        Log an error message (no-op).

        Args:
            event: The event name
            **kwargs: Additional key-value pairs to include in the log
        """
        pass

    def critical(self, event: str, **kwargs: Any) -> None:
        """
        Log a critical message (no-op).

        Args:
            event: The event name
            **kwargs: Additional key-value pairs to include in the log
        """
        pass
