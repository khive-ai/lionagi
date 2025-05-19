"""Additional tests for the telemetry facades to improve coverage."""

from unittest.mock import MagicMock, patch

import pytest


@pytest.mark.asyncio
async def test_tracing_facade_start_as_current_async_span_exception():
    """Test TracingFacade.start_as_current_async_span exception handling."""
    # Mock OpenTelemetry availability
    with patch("pynector.telemetry.facade.HAS_OPENTELEMETRY", True):
        # Mock the trace module and context with attach raising an exception
        mock_span = MagicMock()
        mock_tracer = MagicMock()
        mock_tracer.start_as_current_span.return_value = mock_span
        mock_get_tracer = MagicMock(return_value=mock_tracer)
        mock_context = MagicMock()
        mock_attach = MagicMock(side_effect=Exception("Test exception"))
        mock_detach = MagicMock()
        mock_get_current = MagicMock(return_value=mock_context)

        with (
            patch("pynector.telemetry.facade.trace.get_tracer", mock_get_tracer),
            patch("pynector.telemetry.facade.attach", mock_attach),
            patch("pynector.telemetry.facade.detach", mock_detach),
            patch("pynector.telemetry.facade.get_current", mock_get_current),
        ):
            from pynector.telemetry.facade import TracingFacade

            tracer = TracingFacade("test_tracer")

            # Test exception handling
            with pytest.raises(Exception):
                async with tracer.start_as_current_async_span(
                    "test_span", {"key": "value"}
                ) as _:
                    pass  # Should not reach here

            # Verify attach was called but not detach (since attach failed)
            mock_attach.assert_called_once_with(mock_context)
            mock_detach.assert_not_called()


@pytest.mark.asyncio
async def test_tracing_facade_start_as_current_async_span_import_error():
    """Test TracingFacade.start_as_current_async_span with import error."""
    # Mock OpenTelemetry availability but get_current raises ImportError
    with patch("pynector.telemetry.facade.HAS_OPENTELEMETRY", True):
        # Mock the trace module and context with get_current raising ImportError
        mock_span = MagicMock()
        mock_tracer = MagicMock()
        mock_tracer.start_as_current_span.return_value = mock_span
        mock_get_tracer = MagicMock(return_value=mock_tracer)
        mock_attach = MagicMock()
        mock_detach = MagicMock()
        mock_get_current = MagicMock(side_effect=ImportError("Test import error"))

        with (
            patch("pynector.telemetry.facade.trace.get_tracer", mock_get_tracer),
            patch("pynector.telemetry.facade.attach", mock_attach),
            patch("pynector.telemetry.facade.detach", mock_detach),
            patch("pynector.telemetry.facade.get_current", mock_get_current),
        ):
            from pynector.telemetry.facade import TracingFacade
            from pynector.telemetry.tracing import NoOpSpan

            tracer = TracingFacade("test_tracer")

            # Test ImportError fallback
            async with tracer.start_as_current_async_span(
                "test_span", {"key": "value"}
            ) as span:
                assert isinstance(span, NoOpSpan)
                assert span.name == "test_span"
                assert span.attributes == {"key": "value"}

            # Verify get_current was called but not attach or detach
            mock_get_current.assert_called_once()
            mock_attach.assert_not_called()
            mock_detach.assert_not_called()


def test_logging_facade_methods_with_import_error():
    """Test LoggingFacade methods with ImportError when getting trace context."""
    # Mock dependencies availability
    with (
        patch("pynector.telemetry.facade.HAS_STRUCTLOG", True),
        patch("pynector.telemetry.facade.HAS_OPENTELEMETRY", True),
    ):
        # Mock the structlog logger
        mock_logger = MagicMock()
        mock_get_logger = MagicMock(return_value=mock_logger)

        # Mock OpenTelemetry with get_current_span raising ImportError
        mock_get_current_span = MagicMock(side_effect=ImportError("Test import error"))

        with (
            patch("pynector.telemetry.facade.structlog.get_logger", mock_get_logger),
            patch(
                "pynector.telemetry.facade.trace.get_current_span",
                mock_get_current_span,
            ),
        ):
            from pynector.telemetry.facade import LoggingFacade

            logger = LoggingFacade("test_logger")

            # Test debug method
            logger.debug("test_event", key="value")
            mock_logger.debug.assert_called_once_with("test_event", key="value")
            assert mock_get_current_span.call_count >= 1

            # Reset mocks
            mock_logger.debug.reset_mock()
            mock_get_current_span.reset_mock()

            # Test info method
            logger.info("test_event", key="value")
            mock_logger.info.assert_called_once_with("test_event", key="value")
            assert mock_get_current_span.call_count >= 1

            # Reset mocks
            mock_logger.info.reset_mock()
            mock_get_current_span.reset_mock()

            # Test warning method
            logger.warning("test_event", key="value")
            mock_logger.warning.assert_called_once_with("test_event", key="value")
            assert mock_get_current_span.call_count >= 1

            # Reset mocks
            mock_logger.warning.reset_mock()
            mock_get_current_span.reset_mock()

            # Test error method (should call get_current_span twice: once for trace context, once for setting status)
            logger.error("test_event", key="value")
            mock_logger.error.assert_called_once_with("test_event", key="value")
            assert mock_get_current_span.call_count == 2

            # Reset mocks
            mock_logger.error.reset_mock()
            mock_get_current_span.reset_mock()

            # Test critical method (should call get_current_span twice: once for trace context, once for setting status)
            logger.critical("test_event", key="value")
            mock_logger.critical.assert_called_once_with("test_event", key="value")
            assert mock_get_current_span.call_count == 2


def test_logging_facade_error_methods_with_import_error():
    """Test LoggingFacade error and critical methods with ImportError when setting status."""
    # Mock dependencies availability
    with (
        patch("pynector.telemetry.facade.HAS_STRUCTLOG", True),
        patch("pynector.telemetry.facade.HAS_OPENTELEMETRY", True),
    ):
        # Mock the structlog logger
        mock_logger = MagicMock()
        mock_get_logger = MagicMock(return_value=mock_logger)

        # Mock OpenTelemetry with span but set_status raising ImportError
        mock_span = MagicMock()
        mock_context = MagicMock()
        mock_context.is_valid = True
        mock_context.trace_id = 0x1234567890ABCDEF1234567890ABCDEF
        mock_context.span_id = 0x1234567890ABCDEF
        mock_span.get_span_context.return_value = mock_context
        mock_span.set_status = MagicMock(side_effect=ImportError("Test import error"))
        mock_get_current_span = MagicMock(return_value=mock_span)

        with (
            patch("pynector.telemetry.facade.structlog.get_logger", mock_get_logger),
            patch(
                "pynector.telemetry.facade.trace.get_current_span",
                mock_get_current_span,
            ),
            patch(
                "pynector.telemetry.facade.trace.Status",
                MagicMock(side_effect=ImportError("Test import error")),
            ),
        ):
            from pynector.telemetry.facade import LoggingFacade

            logger = LoggingFacade("test_logger")

            # Test error method (should handle ImportError when setting status)
            logger.error("test_event", key="value")
            mock_logger.error.assert_called_once_with(
                "test_event",
                key="value",
                trace_id="1234567890abcdef1234567890abcdef",
                span_id="1234567890abcdef",
            )

            # Reset mocks
            mock_logger.error.reset_mock()

            # Test critical method (should handle ImportError when setting status)
            logger.critical("test_event", key="value")
            mock_logger.critical.assert_called_once_with(
                "test_event",
                key="value",
                trace_id="1234567890abcdef1234567890abcdef",
                span_id="1234567890abcdef",
            )
