# Copyright (c) 2025 HaiyangLi (Ocean) - All Rights Reserved
# PRIVATE - Lion Ecosystem (Language Interoperable Network)
# Core telemetry facade with vendor-neutral interface for observability

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from time import perf_counter
from typing import Any, ContextManager, Protocol


class Telemetry(Protocol):
    """Vendor-neutral telemetry interface for metrics and tracing.

    This Protocol provides a stable interface that can be implemented
    by OpenTelemetry, Prometheus, or any other observability backend
    without requiring changes to application code.
    """

    def counter(self, name: str, value: float = 1.0, **labels: Any) -> None:
        """Increment a counter metric.

        Args:
            name: Metric name (e.g., 'service.call.success_total')
            value: Amount to increment (default: 1.0)
            **labels: Key-value labels for metric dimensions
        """
        ...

    def histogram(self, name: str, value: float, **labels: Any) -> None:
        """Record a value in a histogram metric.

        Args:
            name: Metric name (e.g., 'service.call.duration_s')
            value: Value to record
            **labels: Key-value labels for metric dimensions
        """
        ...

    def gauge(self, name: str, value: float, **labels: Any) -> None:
        """Set a gauge metric to a specific value.

        Args:
            name: Metric name (e.g., 'executor.queue.size')
            value: Current value to set
            **labels: Key-value labels for metric dimensions
        """
        ...

    @contextmanager
    def span(self, name: str, **attrs: Any) -> ContextManager[None]:
        """Create a distributed tracing span.

        Args:
            name: Span name (e.g., 'service.call', 'transport.http')
            **attrs: Span attributes for context

        Yields:
            Context manager for the span lifecycle
        """
        ...


class _NoopTelemetry:
    """Default no-op telemetry implementation.

    Provides zero-overhead telemetry interface when no backend is configured.
    All operations are no-ops with minimal performance impact.
    """

    def counter(self, name: str, value: float = 1.0, **labels: Any) -> None:
        pass

    def histogram(self, name: str, value: float, **labels: Any) -> None:
        pass

    def gauge(self, name: str, value: float, **labels: Any) -> None:
        pass

    @contextmanager
    def span(self, name: str, **attrs: Any) -> ContextManager[None]:
        yield


# Global telemetry instance - defaults to no-op
_telemetry: Telemetry = _NoopTelemetry()


def set_telemetry(telemetry_impl: Telemetry) -> None:
    """Set the global telemetry implementation.

    This allows swapping in OpenTelemetry, Prometheus, or other backends
    without changing application code.

    Args:
        telemetry_impl: Telemetry implementation to use
    """
    global _telemetry
    _telemetry = telemetry_impl


def get_telemetry() -> Telemetry:
    """Get the current telemetry implementation.

    Returns:
        Current telemetry implementation (defaults to no-op)
    """
    return _telemetry


# Convenience alias for shorter imports
t = get_telemetry


class TelemetryTimer:
    """Context manager for timing operations with automatic histogram recording.

    Example:
        with TelemetryTimer('service.call.duration_s', service='openai'):
            result = await service.call(request)
    """

    def __init__(self, metric_name: str, **labels: Any):
        self.metric_name = metric_name
        self.labels = labels
        self.start_time: float | None = None

    def __enter__(self) -> TelemetryTimer:
        self.start_time = perf_counter()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self.start_time is not None:
            duration = perf_counter() - self.start_time
            get_telemetry().histogram(self.metric_name, duration, **self.labels)


def timed(metric_name: str, **labels: Any) -> TelemetryTimer:
    """Create a timer context manager for measuring operation duration.

    Args:
        metric_name: Histogram metric name for duration recording
        **labels: Labels to attach to the metric

    Returns:
        Context manager that records duration on exit

    Example:
        with timed('service.call.duration_s', service='openai', model='gpt-4'):
            result = await service.call(request)
    """
    return TelemetryTimer(metric_name, **labels)
