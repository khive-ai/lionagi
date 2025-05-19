"""
Telemetry facades for the telemetry module.

This module provides facade classes for tracing and logging operations,
which provide a consistent API regardless of whether the optional dependencies
are available.
"""

from contextlib import asynccontextmanager
from typing import Any, ContextManager, Optional

from pynector.telemetry import HAS_OPENTELEMETRY, HAS_STRUCTLOG, Status, StatusCode
from pynector.telemetry.logging import NoOpLogger
from pynector.telemetry.tracing import AsyncSpanWrapper, NoOpSpan

# Import these at module level for patching in tests
if HAS_OPENTELEMETRY:
    from opentelemetry import trace
    from opentelemetry.context import attach, detach, get_current

# Import structlog at module level for patching in tests
if HAS_STRUCTLOG:
    import structlog
else:
    # Define dummy functions for patching in tests
    def attach(context):
        return None

    def detach(token):
        pass

    def get_current():
        return {}


class TracingFacade:
    """Facade for tracing operations."""

    def __init__(self, name: str):
        """
        Initialize a new tracing facade.

        Args:
            name: The name of the tracer
        """
        self.name = name
        if HAS_OPENTELEMETRY:
            self.tracer = trace.get_tracer(name)
        else:
            self.tracer = None

    def start_span(
        self, name: str, attributes: Optional[dict[str, Any]] = None
    ) -> ContextManager:
        """
        Start a new span.

        Args:
            name: The name of the span
            attributes: Optional attributes to set on the span

        Returns:
            A context manager that will end the span when exited
        """
        if HAS_OPENTELEMETRY and self.tracer:
            return self.tracer.start_span(name, attributes=attributes)
        return NoOpSpan(name, attributes)

    def start_as_current_span(
        self, name: str, attributes: Optional[dict[str, Any]] = None
    ) -> ContextManager:
        """
        Start a new span and set it as the current span.

        Args:
            name: The name of the span
            attributes: Optional attributes to set on the span

        Returns:
            A context manager that will end the span when exited
        """
        if HAS_OPENTELEMETRY and self.tracer:
            return self.tracer.start_as_current_span(name, attributes=attributes)
        return NoOpSpan(name, attributes)

    @asynccontextmanager
    async def start_async_span(
        self, name: str, attributes: Optional[dict[str, Any]] = None
    ):
        """
        Start a new span for async operations.

        Args:
            name: The name of the span
            attributes: Optional attributes to set on the span

        Returns:
            An async context manager that will end the span when exited
        """
        if HAS_OPENTELEMETRY and self.tracer:
            # Use a wrapper to make a regular span work as an async context manager
            span = self.tracer.start_span(name, attributes=attributes)
            wrapper = AsyncSpanWrapper(span)
            try:
                await wrapper.__aenter__()
                yield wrapper
            finally:
                await wrapper.__aexit__(None, None, None)
        else:
            span = NoOpSpan(name, attributes)
            try:
                await span.__aenter__()
                yield span
            finally:
                await span.__aexit__(None, None, None)

    @asynccontextmanager
    async def start_as_current_async_span(
        self, name: str, attributes: Optional[dict[str, Any]] = None
    ):
        """
        Start a new span for async operations and set it as the current span.

        Args:
            name: The name of the span
            attributes: Optional attributes to set on the span

        Returns:
            An async context manager that will end the span when exited
        """
        # Determine which type of span to create before the yield
        use_opentelemetry = False
        token = None
        wrapper = None

        if HAS_OPENTELEMETRY and self.tracer:
            try:
                # Capture current context
                token = attach(get_current())
                # Start a new span as the current span
                span = self.tracer.start_as_current_span(name, attributes=attributes)
                wrapper = AsyncSpanWrapper(span, token)
                use_opentelemetry = True
            except ImportError:
                # Fallback if opentelemetry is not available
                if token is not None:
                    detach(token)
                    token = None

        # Use the appropriate span based on what was set up above
        try:
            if use_opentelemetry and wrapper is not None:
                await wrapper.__aenter__()
                yield wrapper
            else:
                # Use NoOpSpan as fallback
                span = NoOpSpan(name, attributes)
                await span.__aenter__()
                yield span
        finally:
            if use_opentelemetry and wrapper is not None:
                await wrapper.__aexit__(None, None, None)
            elif not use_opentelemetry:
                await span.__aexit__(None, None, None)


class LoggingFacade:
    """Facade for logging operations."""

    def __init__(self, name: str):
        """
        Initialize a new logging facade.

        Args:
            name: The name of the logger
        """
        self.name = name
        if HAS_STRUCTLOG:
            self.logger = structlog.get_logger(name)
        else:
            self.logger = NoOpLogger(name)

    def _add_trace_context(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """
        Add trace context to log entries if available.

        Args:
            kwargs: The keyword arguments to add trace context to

        Returns:
            The updated keyword arguments
        """
        if HAS_OPENTELEMETRY:
            try:
                # Add trace context to logs
                current_span = trace.get_current_span()
                if current_span:
                    context = current_span.get_span_context()
                    if hasattr(context, "is_valid") and context.is_valid:
                        kwargs["trace_id"] = format(context.trace_id, "032x")
                        kwargs["span_id"] = format(context.span_id, "016x")
            except ImportError:
                # OpenTelemetry not available
                pass
        return kwargs

    def debug(self, event: str, **kwargs: Any) -> None:
        """
        Log a debug message.

        Args:
            event: The event name
            **kwargs: Additional key-value pairs to include in the log
        """
        kwargs = self._add_trace_context(kwargs)
        if HAS_STRUCTLOG:
            self.logger.debug(event, **kwargs)

    def info(self, event: str, **kwargs: Any) -> None:
        """
        Log an info message.

        Args:
            event: The event name
            **kwargs: Additional key-value pairs to include in the log
        """
        kwargs = self._add_trace_context(kwargs)
        if HAS_STRUCTLOG:
            self.logger.info(event, **kwargs)

    def warning(self, event: str, **kwargs: Any) -> None:
        """
        Log a warning message.

        Args:
            event: The event name
            **kwargs: Additional key-value pairs to include in the log
        """
        kwargs = self._add_trace_context(kwargs)
        if HAS_STRUCTLOG:
            self.logger.warning(event, **kwargs)

    def error(self, event: str, **kwargs: Any) -> None:
        """
        Log an error message.

        Args:
            event: The event name
            **kwargs: Additional key-value pairs to include in the log
        """
        kwargs = self._add_trace_context(kwargs)
        if HAS_OPENTELEMETRY:
            try:
                # Mark span as error
                current_span = trace.get_current_span()
                if current_span:
                    current_span.set_status(Status(StatusCode.ERROR))
            except ImportError:
                # OpenTelemetry not available
                pass

        if HAS_STRUCTLOG:
            self.logger.error(event, **kwargs)

    def critical(self, event: str, **kwargs: Any) -> None:
        """
        Log a critical message.

        Args:
            event: The event name
            **kwargs: Additional key-value pairs to include in the log
        """
        kwargs = self._add_trace_context(kwargs)
        if HAS_OPENTELEMETRY:
            try:
                # Mark span as error
                current_span = trace.get_current_span()
                if current_span:
                    current_span.set_status(Status(StatusCode.ERROR))
            except ImportError:
                # OpenTelemetry not available
                pass

        if HAS_STRUCTLOG:
            self.logger.critical(event, **kwargs)
