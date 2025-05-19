"""Additional tests for the configuration module to improve coverage."""

import os
from unittest.mock import MagicMock, patch


def test_configure_exporters_with_otel():
    """Test _configure_exporters function with OTLP exporter."""
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
            patch.dict(os.environ, {"OTEL_TRACES_EXPORTER": "otlp"}, clear=True),
        ):
            # Call the function
            _configure_exporters(mock_tracer_provider)

            # Verify that the provider has the span processor added
            assert mock_tracer_provider.add_span_processor.called
            # We can't check the exact object because it's created inside the function


def test_configure_exporters_with_console():
    """Test _configure_exporters function with console exporter."""
    # Import the function directly
    from pynector.telemetry.config import _configure_exporters

    # Mock OpenTelemetry availability
    with patch("pynector.telemetry.config.HAS_OPENTELEMETRY", True):
        # Create mocks for all required components
        mock_tracer_provider = MagicMock()
        mock_batch_processor = MagicMock()
        mock_console_exporter = MagicMock()

        # Create the necessary imports and patch the actual implementation
        with (
            patch(
                "pynector.telemetry.config.BatchSpanProcessor",
                return_value=mock_batch_processor,
            ),
            patch(
                "pynector.telemetry.config.ConsoleSpanExporter",
                return_value=mock_console_exporter,
            ),
            patch.dict(os.environ, {"OTEL_TRACES_EXPORTER": "console"}, clear=True),
        ):
            # Call the function
            _configure_exporters(mock_tracer_provider)

            # Verify that the provider has the span processor added
            assert mock_tracer_provider.add_span_processor.called
            # We can't check the exact object because it's created inside the function


def test_configure_structlog_detailed():
    """Test _configure_structlog function in detail."""
    from pynector.telemetry.config import _configure_structlog

    # Mock structlog availability
    with patch("pynector.telemetry.config.HAS_STRUCTLOG", True):
        # Mock the required modules
        mock_structlog = MagicMock()
        mock_logging = MagicMock()

        # Mock the specific functions we need
        mock_merge_contextvars = MagicMock()
        mock_add_log_level = MagicMock()
        mock_time_stamper = MagicMock()
        mock_json_renderer = MagicMock()
        mock_logger_factory = MagicMock()
        mock_configure = MagicMock()

        # Create properly typed mock objects for the structlog module
        mock_structlog.contextvars.merge_contextvars = mock_merge_contextvars
        mock_structlog.processors.add_log_level = mock_add_log_level
        mock_structlog.processors.TimeStamper = mock_time_stamper
        mock_structlog.processors.JSONRenderer = mock_json_renderer
        mock_structlog.stdlib.LoggerFactory = mock_logger_factory
        mock_structlog.configure = mock_configure

        # Setup logging module
        mock_logging.INFO = 20
        mock_logging.DEBUG = 10
        mock_logging.WARNING = 30
        mock_logging.ERROR = 40
        mock_logging.CRITICAL = 50
        mock_logging.basicConfig = MagicMock()

        # Mock the modules in sys.modules
        with patch.dict(
            "sys.modules", {"structlog": mock_structlog, "logging": mock_logging}
        ):
            # Patch the modules directly
            with (
                patch("pynector.telemetry.config.structlog", mock_structlog),
                patch("pynector.telemetry.config.logging", mock_logging),
            ):
                # Test with default values
                _configure_structlog("INFO")

                # Verify that the basic logging was configured
                mock_logging.basicConfig.assert_called_once_with(
                    format="%(message)s", level=mock_logging.INFO
                )

                # Verify that structlog was configured
                mock_configure.assert_called_once()

                # Test with custom processors
                mock_configure.reset_mock()
                mock_logging.basicConfig.reset_mock()

                custom_processor = MagicMock()
                _configure_structlog("DEBUG", [custom_processor])

                mock_logging.basicConfig.assert_called_once_with(
                    format="%(message)s", level=mock_logging.DEBUG
                )

                mock_configure.assert_called_once()

                # Test with ImportError during configuration
                mock_configure.reset_mock()
                mock_logging.basicConfig.reset_mock()
                mock_configure.side_effect = ImportError("Test import error")

                # Should handle the exception gracefully
                _configure_structlog("INFO")

                mock_logging.basicConfig.assert_called_once()
                mock_configure.assert_called_once()


def test_dummy_classes_more_coverage():
    """Test more dummy classes to increase coverage."""
    from pynector.telemetry.config import HAS_OPENTELEMETRY, HAS_STRUCTLOG

    if not HAS_OPENTELEMETRY:
        from pynector.telemetry.config import (
            BatchSpanProcessor,
            ConsoleSpanExporter,
            OTLPSpanExporter,
        )

        # Test BatchSpanProcessor with specific options
        span_processor = BatchSpanProcessor(
            "exporter", max_queue_size=100, schedule_delay_millis=5000
        )
        assert span_processor.exporter == "exporter"
        assert span_processor.max_queue_size == 100
        assert span_processor.schedule_delay_millis == 5000

        # Test OTLPSpanExporter with specific options
        otlp_exporter = OTLPSpanExporter(
            endpoint="http://localhost:4317", headers={"key": "value"}, timeout=10
        )
        assert otlp_exporter.endpoint == "http://localhost:4317"
        assert otlp_exporter.headers == {"key": "value"}
        assert otlp_exporter.timeout == 10

        # Test ConsoleSpanExporter with specific options
        console_exporter = ConsoleSpanExporter(service_name="test-service")
        assert console_exporter.service_name == "test-service"

    if not HAS_STRUCTLOG:
        from pynector.telemetry.config import structlog

        # Test bound logger methods
        logger = structlog.get_logger("test_logger")
        logger.debug("test message", key="value")
        logger.info("test message", key="value")
        logger.warning("test message", key="value")
        logger.error("test message", key="value")
        logger.critical("test message", key="value")


def test_get_env_bool_edge_cases():
    """Test get_env_bool with more edge cases."""
    from pynector.telemetry.config import get_env_bool

    # Test with various truthy values
    with patch.dict(os.environ, {"TEST_BOOL": "TRUE"}):  # uppercase
        assert get_env_bool("TEST_BOOL") is True

    with patch.dict(os.environ, {"TEST_BOOL": "True"}):  # title case
        assert get_env_bool("TEST_BOOL") is True

    # Note: The function only checks for "true", "1", "yes", "y", "t" (case-insensitive)
    # So "ON" is not recognized as a truthy value
    with patch.dict(os.environ, {"TEST_BOOL": "yes"}):  # yes
        assert get_env_bool("TEST_BOOL") is True

    # Test with various falsy values
    with patch.dict(os.environ, {"TEST_BOOL": "FALSE"}):  # uppercase
        assert get_env_bool("TEST_BOOL") is False

    with patch.dict(os.environ, {"TEST_BOOL": "False"}):  # title case
        assert get_env_bool("TEST_BOOL") is False

    with patch.dict(os.environ, {"TEST_BOOL": "no"}):  # no
        assert get_env_bool("TEST_BOOL") is False
