"""
Configuration functions for the telemetry module.

This module provides functions for configuring telemetry with sensible defaults,
including functions for reading configuration from environment variables.
"""

import logging
import os
from typing import Any, Optional

from pynector.telemetry import HAS_OPENTELEMETRY, HAS_STRUCTLOG

# Import these at module level for patching in tests
if HAS_OPENTELEMETRY:
    from opentelemetry import trace  # noqa: F401
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    # Import exporters at module level for patching in tests
    try:
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter,
        )
    except ImportError:
        # Define dummy class for patching in tests
        class OTLPSpanExporter:
            def __init__(self, endpoint=None):
                self.endpoint = endpoint

    try:
        from opentelemetry.sdk.trace.export import ConsoleSpanExporter
    except ImportError:
        # Define dummy class for patching in tests
        class ConsoleSpanExporter:
            pass

    try:
        from opentelemetry.exporter.zipkin.json import ZipkinExporter
    except ImportError:
        # Define dummy class for patching in tests
        class ZipkinExporter:
            def __init__(self, endpoint=None):
                self.endpoint = endpoint

else:
    # Define dummy classes for patching in tests
    class Resource:
        @staticmethod
        def create(attributes):
            return None

    class TracerProvider:
        def __init__(self, resource=None):
            self.resource = resource

        def add_span_processor(self, processor):
            pass

    class BatchSpanProcessor:
        def __init__(self, exporter):
            self.exporter = exporter

    class OTLPSpanExporter:
        def __init__(self, endpoint=None):
            self.endpoint = endpoint

    class ConsoleSpanExporter:
        pass

    class ZipkinExporter:
        def __init__(self, endpoint=None):
            self.endpoint = endpoint


# Import structlog at module level for patching in tests
if HAS_STRUCTLOG:
    import logging

    import structlog
else:
    # Define dummy modules for patching in tests
    class DummyStructlog:
        def configure(self, **kwargs):
            pass

        def get_logger(self, name):
            return None

        class stdlib:
            class LoggerFactory:
                pass

            class BoundLogger:
                pass

        class contextvars:
            def merge_contextvars(self, logger, method_name, event_dict):
                return event_dict

        class processors:
            def add_log_level(self, logger, method_name, event_dict):
                return event_dict

            def TimeStamper(self, fmt=None):
                def processor(logger, method_name, event_dict):
                    return event_dict

                return processor

            def JSONRenderer(self):
                def processor(logger, method_name, event_dict):
                    return event_dict

                return processor

    structlog = DummyStructlog()

    class DummyLogging:
        DEBUG = 10
        INFO = 20
        WARNING = 30
        ERROR = 40
        CRITICAL = 50

        def basicConfig(self, **kwargs):
            pass

    logging = DummyLogging()


def get_env_bool(name: str, default: bool = False) -> bool:
    """
    Get a boolean value from an environment variable.

    Args:
        name: The name of the environment variable
        default: The default value if the environment variable is not set

    Returns:
        The boolean value
    """
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in ("true", "1", "yes", "y", "t")


def get_env_dict(name: str, default: Optional[dict[str, str]] = None) -> dict[str, str]:
    """
    Get a dictionary from a comma-separated environment variable.

    Args:
        name: The name of the environment variable
        default: The default value if the environment variable is not set

    Returns:
        The dictionary
    """
    value = os.environ.get(name)
    if not value:
        return default or {}

    result = {}
    for pair in value.split(","):
        if "=" in pair:
            key, val = pair.split("=", 1)
            result[key.strip()] = val.strip()
    return result


def configure_telemetry(
    service_name: Optional[str] = None,
    resource_attributes: Optional[dict[str, str]] = None,
    trace_enabled: Optional[bool] = None,
    log_level: str = "INFO",
    log_processors: Optional[list[Any]] = None,
    trace_exporters: Optional[list[str]] = None,
) -> bool:
    """
    Configure OpenTelemetry and structlog with sensible defaults.

    Args:
        service_name: The name of the service
        resource_attributes: Additional resource attributes
        trace_enabled: Whether tracing is enabled
        log_level: The log level
        log_processors: Additional log processors
        trace_exporters: The trace exporters to use

    Returns:
        True if telemetry is enabled (either tracing or logging), False otherwise
    """
    # Check if dependencies are available
    if not (HAS_OPENTELEMETRY or HAS_STRUCTLOG):
        return False

    # Determine if tracing is enabled
    if trace_enabled is None:
        trace_enabled = not get_env_bool("OTEL_SDK_DISABLED", False)

    # Get service name
    if service_name is None:
        service_name = os.environ.get("OTEL_SERVICE_NAME", "unknown_service")

    # Get resource attributes
    if resource_attributes is None:
        resource_attributes = {}

    env_attrs = get_env_dict("OTEL_RESOURCE_ATTRIBUTES")
    resource_attributes = {**env_attrs, **resource_attributes}

    # Ensure service name is in resource attributes
    resource_attributes["service.name"] = service_name

    # Track if any telemetry was configured
    telemetry_configured = False

    # Configure OpenTelemetry if enabled and available
    if HAS_OPENTELEMETRY:
        if trace_enabled:
            try:
                # Create resource with service info
                resource = Resource.create(resource_attributes)

                # Create and set tracer provider
                tracer_provider = TracerProvider(resource=resource)
                # Use imported trace from the conditional import at the top
                from opentelemetry import trace as otel_trace

                otel_trace.set_tracer_provider(tracer_provider)

                # Configure exporters
                _configure_exporters(tracer_provider, trace_exporters)
                telemetry_configured = True
            except ImportError:
                # If OpenTelemetry SDK is not available, disable tracing
                trace_enabled = False
        else:
            # If trace_enabled is False, return False when OpenTelemetry is the only dependency
            if not HAS_STRUCTLOG:
                return False

    # Configure structlog if available
    if HAS_STRUCTLOG:
        try:
            _configure_structlog(log_level, log_processors)
            telemetry_configured = True
        except ImportError:
            # If structlog is not available, we can't configure it
            pass

    # Return True if any telemetry was configured, but False if trace_enabled is False and OpenTelemetry is the only dependency
    if HAS_OPENTELEMETRY and not HAS_STRUCTLOG and not trace_enabled:
        return False
    return telemetry_configured


def _configure_exporters(tracer_provider, exporters=None):
    """
    Configure trace exporters.

    Args:
        tracer_provider: The tracer provider to configure
        exporters: The exporters to use, or None to use the default
    """
    if not HAS_OPENTELEMETRY:
        return

    try:
        # Determine which exporters to use
        if exporters is None:
            exporter_env = os.environ.get("OTEL_TRACES_EXPORTER", "otlp")
            exporters = [ex.strip() for ex in exporter_env.split(",")]

        # Configure each exporter
        for exporter_name in exporters:
            if exporter_name == "otlp":
                try:
                    # OTLP exporter
                    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                        OTLPSpanExporter,
                    )

                    otlp_endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
                    if otlp_endpoint:
                        exporter = OTLPSpanExporter(endpoint=otlp_endpoint)
                    else:
                        exporter = OTLPSpanExporter()
                    tracer_provider.add_span_processor(BatchSpanProcessor(exporter))
                except ImportError:
                    # OTLP exporter not available
                    pass

            elif exporter_name == "console":
                try:
                    # Console exporter
                    from opentelemetry.sdk.trace.export import ConsoleSpanExporter

                    tracer_provider.add_span_processor(
                        BatchSpanProcessor(ConsoleSpanExporter())
                    )
                except ImportError:
                    # Console exporter not available
                    pass

            elif exporter_name == "zipkin":
                try:
                    # Zipkin exporter
                    from opentelemetry.exporter.zipkin.json import ZipkinExporter

                    zipkin_endpoint = os.environ.get("OTEL_EXPORTER_ZIPKIN_ENDPOINT")
                    if zipkin_endpoint:
                        exporter = ZipkinExporter(endpoint=zipkin_endpoint)
                    else:
                        exporter = ZipkinExporter()
                    tracer_provider.add_span_processor(BatchSpanProcessor(exporter))
                except ImportError:
                    # Zipkin exporter not available
                    pass
    except ImportError:
        # OpenTelemetry SDK not available
        pass


def _configure_structlog(log_level, processors=None):
    """
    Configure structlog.

    Args:
        log_level: The log level
        processors: Additional processors to add
    """
    if not HAS_STRUCTLOG:
        return

    try:
        import structlog

        # Set up logging
        logging.basicConfig(
            format="%(message)s",
            level=getattr(logging, log_level),
        )

        # Define custom processor to add trace context
        def add_trace_context(_, __, event_dict):
            """Add trace context to log entries if available."""
            if HAS_OPENTELEMETRY:
                try:
                    from opentelemetry import trace

                    current_span = trace.get_current_span()
                    if current_span:
                        context = current_span.get_span_context()
                        if hasattr(context, "is_valid") and context.is_valid:
                            event_dict["trace_id"] = format(context.trace_id, "032x")
                            event_dict["span_id"] = format(context.span_id, "016x")
                except ImportError:
                    # OpenTelemetry not available
                    pass
            return event_dict

        # Define processors
        default_processors = [
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            add_trace_context,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ]

        # Add custom processors
        if processors:
            default_processors.extend(processors)

        # Configure structlog
        structlog.configure(
            processors=default_processors,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
    except ImportError:
        # structlog not available
        pass
