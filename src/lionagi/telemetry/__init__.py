"""
Telemetry module for pynector.

This module provides optional observability features for pynector, including
tracing and structured logging. It uses OpenTelemetry for tracing and structlog
for logging, but both are optional dependencies.
"""

# Try to import OpenTelemetry
try:
    # Import only what we need
    from opentelemetry.trace.status import Status, StatusCode

    HAS_OPENTELEMETRY = True
except ImportError:
    HAS_OPENTELEMETRY = False

    # Define status codes for no-op implementation
    class StatusCode:
        ERROR = 1
        OK = 0

    class Status:
        def __init__(self, status_code):
            self.status_code = status_code


# Try to import structlog
try:
    # Just check if it's available
    import importlib.util

    HAS_STRUCTLOG = importlib.util.find_spec("structlog") is not None
except ImportError:
    HAS_STRUCTLOG = False

from pynector.telemetry.config import configure_telemetry

# Import after defining HAS_* flags
from pynector.telemetry.facade import LoggingFacade, TracingFacade

# Define what's available for import
__all__ = [
    "TracingFacade",
    "LoggingFacade",
    "configure_telemetry",
    "get_telemetry",
    "HAS_OPENTELEMETRY",
    "HAS_STRUCTLOG",
    "Status",
    "StatusCode",
]


def get_telemetry(name: str) -> tuple:
    """
    Get tracer and logger instances for the given name.

    Args:
        name: The name to use for the tracer and logger

    Returns:
        A tuple containing a tracer and logger
    """
    return TracingFacade(name), LoggingFacade(name)
