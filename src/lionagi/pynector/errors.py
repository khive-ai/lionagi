"""
Error class definitions for the Pynector library.
"""


class PynectorError(Exception):
    """Base class for all Pynector errors."""

    pass


class TransportError(PynectorError):
    """Error related to transport operations."""

    pass


class ConfigurationError(PynectorError):
    """Error related to configuration."""

    pass


class TimeoutError(PynectorError):
    """Error related to timeouts."""

    pass
