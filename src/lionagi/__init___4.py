"""
pynector - Python Connector Library

A flexible, maintainable, and type-safe library for network communication.
"""

from pynector.client import Pynector
from pynector.errors import (
    ConfigurationError,
    PynectorError,
    TimeoutError,
    TransportError,
)
from pynector.telemetry import configure_telemetry

__version__ = "0.1.0"

__all__ = [
    "Pynector",
    "PynectorError",
    "TransportError",
    "ConfigurationError",
    "TimeoutError",
    "configure_telemetry",
]
