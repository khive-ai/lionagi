"""Tests for the dummy classes in the configuration module."""

import os
from unittest.mock import MagicMock, patch


def test_configure_exporters_with_zipkin():
    """Test _configure_exporters function with zipkin exporter."""
    # Import the function directly
    from pynector.telemetry.config import _configure_exporters

    # Mock OpenTelemetry availability
    with patch("pynector.telemetry.config.HAS_OPENTELEMETRY", True):
        # Create mocks for all required components
        mock_tracer_provider = MagicMock()
        mock_batch_processor = MagicMock()
        mock_zipkin_exporter = MagicMock()

        # Create the necessary imports and patch the actual implementation
        with (
            patch(
                "pynector.telemetry.config.BatchSpanProcessor",
                return_value=mock_batch_processor,
            ),
            patch(
                "pynector.telemetry.config.ZipkinExporter",
                return_value=mock_zipkin_exporter,
            ),
            patch.dict(os.environ, {"OTEL_TRACES_EXPORTER": "zipkin"}, clear=True),
        ):
            # Call the function
            _configure_exporters(mock_tracer_provider)

            # Verify that the provider has the span processor added
            assert mock_tracer_provider.add_span_processor.called
            # We can't check the exact object because it's created inside the function


def test_configure_exporters_with_multiple_exporters():
    """Test _configure_exporters function with multiple exporters."""
    # Import the function directly
    from pynector.telemetry.config import _configure_exporters

    # Mock OpenTelemetry availability
    with patch("pynector.telemetry.config.HAS_OPENTELEMETRY", True):
        # Create mocks for all required components
        mock_tracer_provider = MagicMock()
        mock_batch_processor = MagicMock()
        mock_otlp_exporter = MagicMock()
        mock_console_exporter = MagicMock()

        # Create the necessary imports and patch the actual implementation
        with (
            patch(
                "pynector.telemetry.config.BatchSpanProcessor",
                return_value=mock_batch_processor,
            ),
            patch(
                "pynector.telemetry.config.OTLPSpanExporter",
                return_value=mock_otlp_exporter,
            ),
            patch(
                "pynector.telemetry.config.ConsoleSpanExporter",
                return_value=mock_console_exporter,
            ),
            patch.dict(
                os.environ, {"OTEL_TRACES_EXPORTER": "otlp,console"}, clear=True
            ),
        ):
            # Call the function
            _configure_exporters(mock_tracer_provider)

            # Verify that the provider has the span processor added multiple times
            assert mock_tracer_provider.add_span_processor.call_count >= 2


def test_configure_exporters_with_endpoint():
    """Test _configure_exporters function with endpoint configuration."""
    # Import the function directly
    from pynector.telemetry.config import _configure_exporters

    # Mock OpenTelemetry availability
    with patch("pynector.telemetry.config.HAS_OPENTELEMETRY", True):
        # Create mocks for all required components
        mock_tracer_provider = MagicMock()
        mock_batch_processor = MagicMock()
        mock_otlp_exporter = MagicMock()

        # Create the necessary imports and patch the actual implementation
        with (
            patch(
                "pynector.telemetry.config.BatchSpanProcessor",
                return_value=mock_batch_processor,
            ),
            patch(
                "pynector.telemetry.config.OTLPSpanExporter",
                return_value=mock_otlp_exporter,
            ),
            patch.dict(
                os.environ,
                {
                    "OTEL_TRACES_EXPORTER": "otlp",
                    "OTEL_EXPORTER_OTLP_ENDPOINT": "http://localhost:4317",
                },
                clear=True,
            ),
        ):
            # Call the function
            _configure_exporters(mock_tracer_provider)

            # Verify that the provider has the span processor added
            assert mock_tracer_provider.add_span_processor.called


def test_configure_exporters_with_zipkin_endpoint():
    """Test _configure_exporters function with zipkin endpoint configuration."""
    # Import the function directly
    from pynector.telemetry.config import _configure_exporters

    # Mock OpenTelemetry availability
    with patch("pynector.telemetry.config.HAS_OPENTELEMETRY", True):
        # Create mocks for all required components
        mock_tracer_provider = MagicMock()
        mock_batch_processor = MagicMock()
        mock_zipkin_exporter = MagicMock()

        # Create the necessary imports and patch the actual implementation
        with (
            patch(
                "pynector.telemetry.config.BatchSpanProcessor",
                return_value=mock_batch_processor,
            ),
            patch(
                "pynector.telemetry.config.ZipkinExporter",
                return_value=mock_zipkin_exporter,
            ),
            patch.dict(
                os.environ,
                {
                    "OTEL_TRACES_EXPORTER": "zipkin",
                    "OTEL_EXPORTER_ZIPKIN_ENDPOINT": "http://localhost:9411/api/v2/spans",
                },
                clear=True,
            ),
        ):
            # Call the function
            _configure_exporters(mock_tracer_provider)

            # Verify that the provider has the span processor added
            assert mock_tracer_provider.add_span_processor.called
