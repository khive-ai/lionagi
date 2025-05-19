"""
Pytest configuration for telemetry tests.

This module provides fixtures and configuration for testing the telemetry module.
"""

from unittest.mock import MagicMock


class AsyncContextManagerMock:
    """
    A helper class for mocking async context managers in tests.

    This class implements the __aenter__ and __aexit__ methods with proper
    __await__ support, which is required for async context managers.
    """

    def __init__(self, return_value=None):
        """
        Initialize a new async context manager mock.

        Args:
            return_value: The value to return from __aenter__
        """
        self.return_value = return_value
        self.enter_called = False
        self.exit_called = False
        self.exit_args = None

    async def __aenter__(self, *args, **kwargs):
        """Enter the async context."""
        self.enter_called = True
        return self.return_value

    async def __aexit__(self, exc_type=None, exc_val=None, exc_tb=None):
        """Exit the async context."""
        self.exit_called = True
        self.exit_args = (exc_type, exc_val, exc_tb)
        return False  # Don't suppress exceptions

    def __await__(self):
        """
        Implement __await__ to make the object awaitable.
        This is required for proper async context manager support.
        """

        async def _await_impl():
            return self

        return _await_impl().__await__()


def create_autospec_mock_for_async_cm(return_value=None):
    """
    Create a MagicMock that can be used as an async context manager.

    Args:
        return_value: The value to return from __aenter__

    Returns:
        A MagicMock with __aenter__ and __aexit__ methods that work in async with
    """
    mock = MagicMock()
    cm = AsyncContextManagerMock(return_value)

    mock.__aenter__ = cm.__aenter__
    mock.__aexit__ = cm.__aexit__
    mock.__await__ = cm.__await__

    return mock
