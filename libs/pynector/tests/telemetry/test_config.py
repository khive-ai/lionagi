"""Tests for the configuration module."""

import os
from unittest.mock import MagicMock, patch


def test_get_env_bool():
    """Test get_env_bool function."""
    from pynector.telemetry.config import get_env_bool

    # Test with environment variable set to true
    with patch.dict(os.environ, {"TEST_BOOL": "true"}):
        assert get_env_bool("TEST_BOOL") is True

    # Test with environment variable set to false
    with patch.dict(os.environ, {"TEST_BOOL": "false"}):
        assert get_env_bool("TEST_BOOL") is False

    # Test with environment variable set to 1
    with patch.dict(os.environ, {"TEST_BOOL": "1"}):
        assert get_env_bool("TEST_BOOL") is True

    # Test with environment variable set to 0
    with patch.dict(os.environ, {"TEST_BOOL": "0"}):
        assert get_env_bool("TEST_BOOL") is False

    # Test with environment variable not set
    with patch.dict(os.environ, {}, clear=True):
        assert get_env_bool("TEST_BOOL") is False
        assert get_env_bool("TEST_BOOL", True) is True

    # Test with other values
    with patch.dict(os.environ, {"TEST_BOOL": "yes"}):
        assert get_env_bool("TEST_BOOL") is True

    with patch.dict(os.environ, {"TEST_BOOL": "y"}):
        assert get_env_bool("TEST_BOOL") is True

    with patch.dict(os.environ, {"TEST_BOOL": "t"}):
        assert get_env_bool("TEST_BOOL") is True

    with patch.dict(os.environ, {"TEST_BOOL": "other"}):
        assert get_env_bool("TEST_BOOL") is False


def test_get_env_dict():
    """Test get_env_dict function."""
    from pynector.telemetry.config import get_env_dict

    # Test with environment variable set
    with patch.dict(os.environ, {"TEST_DICT": "key1=value1,key2=value2"}):
        result = get_env_dict("TEST_DICT")
        assert result == {"key1": "value1", "key2": "value2"}

    # Test with environment variable not set
    with patch.dict(os.environ, {}, clear=True):
        result = get_env_dict("TEST_DICT")
        assert result == {}

        # Test with default value
        result = get_env_dict("TEST_DICT", {"default_key": "default_value"})
        assert result == {"default_key": "default_value"}

    # Test with invalid format
    with patch.dict(os.environ, {"TEST_DICT": "invalid_format"}):
        result = get_env_dict("TEST_DICT")
        assert result == {}

    # Test with empty string
    with patch.dict(os.environ, {"TEST_DICT": ""}):
        result = get_env_dict("TEST_DICT")
        assert result == {}

    # Test with multiple equals signs
    with patch.dict(os.environ, {"TEST_DICT": "key1=value1=extra,key2=value2"}):
        result = get_env_dict("TEST_DICT")
        assert result == {"key1": "value1=extra", "key2": "value2"}


def test_configure_telemetry_no_dependencies():
    """Test configure_telemetry function with no dependencies."""
    # Mock dependencies availability
    with (
        patch("pynector.telemetry.config.HAS_OPENTELEMETRY", False),
        patch("pynector.telemetry.config.HAS_STRUCTLOG", False),
    ):
        from pynector.telemetry.config import configure_telemetry

        assert configure_telemetry() is False


def test_configure_telemetry_with_opentelemetry():
    """Test configure_telemetry function with OpenTelemetry."""
    # Mock dependencies availability
    with (
        patch("pynector.telemetry.config.HAS_OPENTELEMETRY", True),
        patch("pynector.telemetry.config.HAS_STRUCTLOG", False),
    ):
        # Mock OpenTelemetry modules
        mock_resource = MagicMock()
        mock_resource_create = MagicMock(return_value=mock_resource)
        mock_tracer_provider = MagicMock()
        mock_tracer_provider_class = MagicMock(return_value=mock_tracer_provider)
        mock_set_tracer_provider = MagicMock()
        mock_configure_exporters = MagicMock()

        with (
            patch("pynector.telemetry.config.Resource.create", mock_resource_create),
            patch(
                "pynector.telemetry.config.TracerProvider",
                mock_tracer_provider_class,
            ),
            patch(
                "pynector.telemetry.config.trace.set_tracer_provider",
                mock_set_tracer_provider,
            ),
            patch(
                "pynector.telemetry.config._configure_exporters",
                mock_configure_exporters,
            ),
        ):
            from pynector.telemetry.config import configure_telemetry

            # Test with default values
            with patch.dict(os.environ, {}, clear=True):
                result = configure_telemetry()

                assert result is True
                mock_resource_create.assert_called_once_with(
                    {"service.name": "unknown_service"}
                )
                mock_tracer_provider_class.assert_called_once_with(
                    resource=mock_resource
                )
                mock_set_tracer_provider.assert_called_once_with(mock_tracer_provider)
                mock_configure_exporters.assert_called_once_with(
                    mock_tracer_provider, None
                )

            # Test with custom values
            mock_resource_create.reset_mock()
            mock_tracer_provider_class.reset_mock()
            mock_set_tracer_provider.reset_mock()
            mock_configure_exporters.reset_mock()

            result = configure_telemetry(
                service_name="test-service",
                resource_attributes={"key": "value"},
                trace_enabled=True,
                trace_exporters=["console"],
            )

            assert result is True
            mock_resource_create.assert_called_once_with(
                {"key": "value", "service.name": "test-service"}
            )
            mock_tracer_provider_class.assert_called_once_with(resource=mock_resource)
            mock_set_tracer_provider.assert_called_once_with(mock_tracer_provider)
            mock_configure_exporters.assert_called_once_with(
                mock_tracer_provider, ["console"]
            )

            # Test with trace_enabled=False
            mock_resource_create.reset_mock()
            mock_tracer_provider_class.reset_mock()
            mock_set_tracer_provider.reset_mock()
            mock_configure_exporters.reset_mock()

            result = configure_telemetry(trace_enabled=False)

            # When structlog is not available, result should be False
            assert result is False
            mock_resource_create.assert_not_called()
            mock_tracer_provider_class.assert_not_called()
            mock_set_tracer_provider.assert_not_called()
            mock_configure_exporters.assert_not_called()

            # Test with ImportError during configuration
            mock_resource_create.reset_mock()
            mock_tracer_provider_class.reset_mock()
            mock_set_tracer_provider.reset_mock()
            mock_configure_exporters.reset_mock()

            with patch(
                "pynector.telemetry.config.Resource.create", side_effect=ImportError
            ):
                result = configure_telemetry(trace_enabled=True)

                # When structlog is not available, result should be False
                assert result is False


def test_configure_telemetry_with_structlog():
    """Test configure_telemetry function with structlog."""
    # Mock dependencies availability
    with (
        patch("pynector.telemetry.config.HAS_OPENTELEMETRY", False),
        patch("pynector.telemetry.config.HAS_STRUCTLOG", True),
    ):
        # Mock structlog module
        mock_configure_structlog = MagicMock()

        with patch(
            "pynector.telemetry.config._configure_structlog",
            mock_configure_structlog,
        ):
            from pynector.telemetry.config import configure_telemetry

            # Test with default values
            mock_configure_structlog.reset_mock()
            result = configure_telemetry()

            # When structlog is available, result should be True
            assert result is True
            mock_configure_structlog.assert_called_once_with("INFO", None)

            # Test with custom values
            mock_configure_structlog.reset_mock()

            result = configure_telemetry(
                log_level="DEBUG", log_processors=["custom_processor"]
            )

            # When structlog is available, result should be True
            assert result is True
            mock_configure_structlog.assert_called_once_with(
                "DEBUG", ["custom_processor"]
            )

            # Test with ImportError during configuration
            mock_configure_structlog.reset_mock()
            mock_configure_structlog.side_effect = ImportError

            # Import get_env_bool for the test

            result = configure_telemetry()

            # When structlog configuration fails, result should be False
            assert result is False
            mock_configure_structlog.assert_called_once()


def test_configure_telemetry_with_both():
    """Test configure_telemetry function with both OpenTelemetry and structlog."""
    # Mock dependencies availability
    with (
        patch("pynector.telemetry.config.HAS_OPENTELEMETRY", True),
        patch("pynector.telemetry.config.HAS_STRUCTLOG", True),
    ):
        # Mock OpenTelemetry modules
        mock_resource = MagicMock()
        mock_resource_create = MagicMock(return_value=mock_resource)
        mock_tracer_provider = MagicMock()
        mock_tracer_provider_class = MagicMock(return_value=mock_tracer_provider)
        mock_set_tracer_provider = MagicMock()
        mock_configure_exporters = MagicMock()

        # Mock structlog module
        mock_configure_structlog = MagicMock()

        with (
            patch("pynector.telemetry.config.Resource.create", mock_resource_create),
            patch(
                "pynector.telemetry.config.TracerProvider",
                mock_tracer_provider_class,
            ),
            patch(
                "pynector.telemetry.config.trace.set_tracer_provider",
                mock_set_tracer_provider,
            ),
            patch(
                "pynector.telemetry.config._configure_exporters",
                mock_configure_exporters,
            ),
            patch(
                "pynector.telemetry.config._configure_structlog",
                mock_configure_structlog,
            ),
        ):
            from pynector.telemetry.config import configure_telemetry

            # Test with default values
            with patch.dict(os.environ, {}, clear=True):
                result = configure_telemetry()

                assert result is True
                mock_resource_create.assert_called_once_with(
                    {"service.name": "unknown_service"}
                )
                mock_tracer_provider_class.assert_called_once_with(
                    resource=mock_resource
                )
                mock_set_tracer_provider.assert_called_once_with(mock_tracer_provider)
                mock_configure_exporters.assert_called_once_with(
                    mock_tracer_provider, None
                )
                mock_configure_structlog.assert_called_once_with("INFO", None)

            # Test with trace_enabled=False
            mock_resource_create.reset_mock()
            mock_tracer_provider_class.reset_mock()
            mock_set_tracer_provider.reset_mock()
            mock_configure_exporters.reset_mock()
            mock_configure_structlog.reset_mock()

            result = configure_telemetry(trace_enabled=False)

            # When structlog is available, result should be True even if trace_enabled is False
            assert result is True
            mock_resource_create.assert_not_called()
            mock_tracer_provider_class.assert_not_called()
            mock_set_tracer_provider.assert_not_called()
            mock_configure_exporters.assert_not_called()
            mock_configure_structlog.assert_called_once_with("INFO", None)


def test_configure_exporters_simple():
    """Test _configure_exporters function with simple cases."""
    from pynector.telemetry.config import _configure_exporters

    # Test with OpenTelemetry not available
    with patch("pynector.telemetry.config.HAS_OPENTELEMETRY", False):
        mock_tracer_provider = MagicMock()
        _configure_exporters(mock_tracer_provider)
        mock_tracer_provider.add_span_processor.assert_not_called()


def test_configure_structlog_simple():
    """Test _configure_structlog function with simple cases."""
    from pynector.telemetry.config import _configure_structlog

    # Test with structlog not available
    with patch("pynector.telemetry.config.HAS_STRUCTLOG", False):
        _configure_structlog("INFO")
        # No exception should be raised

    # Test with structlog available but ImportError
    with (
        patch("pynector.telemetry.config.HAS_STRUCTLOG", True),
        patch("pynector.telemetry.config.structlog.configure", side_effect=ImportError),
    ):
        _configure_structlog("INFO")
        # No exception should be raised


def test_dummy_classes():
    """Test the dummy classes defined in the module."""
    from pynector.telemetry.config import HAS_OPENTELEMETRY, HAS_STRUCTLOG

    # Test OpenTelemetry dummy classes
    if not HAS_OPENTELEMETRY:
        from pynector.telemetry.config import (
            BatchSpanProcessor,
            ConsoleSpanExporter,
            OTLPSpanExporter,
            Resource,
            TracerProvider,
            ZipkinExporter,
        )

        # Test Resource.create
        resource = Resource.create({"key": "value"})
        assert resource is None

        # Test TracerProvider
        provider = TracerProvider(resource="test")
        assert provider.resource == "test"

        # Test BatchSpanProcessor
        processor = BatchSpanProcessor("exporter")
        assert processor.exporter == "exporter"

        # Test add_span_processor
        provider.add_span_processor(processor)
        # No exception should be raised

        # Test OTLPSpanExporter
        exporter = OTLPSpanExporter(endpoint="http://localhost:4317")
        assert exporter.endpoint == "http://localhost:4317"

        # Test ConsoleSpanExporter
        _ = ConsoleSpanExporter()
        # No exception should be raised

        # Test ZipkinExporter
        zipkin_exporter = ZipkinExporter(endpoint="http://localhost:9411")
        assert zipkin_exporter.endpoint == "http://localhost:9411"

    # Test structlog dummy classes
    if not HAS_STRUCTLOG:
        from pynector.telemetry.config import structlog

        # Test configure
        structlog.configure(processors=["test"])
        # No exception should be raised

        # Test get_logger
        logger = structlog.get_logger("test")
        assert logger is None

        # Test processors
        event_dict = {"key": "value"}
        result = structlog.contextvars.merge_contextvars(None, None, event_dict)
        assert result == event_dict

        result = structlog.processors.add_log_level(None, None, event_dict)
        assert result == event_dict

        timestamper = structlog.processors.TimeStamper(fmt="iso")
        result = timestamper(None, None, event_dict)
        assert result == event_dict

        renderer = structlog.processors.JSONRenderer()
        result = renderer(None, None, event_dict)
        assert result == event_dict


def test_dummy_logging():
    """Test the dummy logging module."""
    from pynector.telemetry.config import HAS_STRUCTLOG

    if not HAS_STRUCTLOG:
        from pynector.telemetry.config import logging

        # Test logging levels
        assert logging.DEBUG == 10
        assert logging.INFO == 20
        assert logging.WARNING == 30
        assert logging.ERROR == 40
        assert logging.CRITICAL == 50

        # Test basicConfig
        logging.basicConfig(level=logging.INFO, format="%(message)s")
        # No exception should be raised
