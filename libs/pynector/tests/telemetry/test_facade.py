"""Tests for the telemetry facades."""

from unittest.mock import MagicMock, patch

import pytest


@pytest.mark.parametrize("has_opentelemetry", [True, False])
def test_tracing_facade_init(has_opentelemetry):
    """Test TracingFacade initialization."""
    # Mock OpenTelemetry availability
    with patch("pynector.telemetry.facade.HAS_OPENTELEMETRY", has_opentelemetry):
        if has_opentelemetry:
            # Mock the trace module
            mock_tracer = MagicMock()
            mock_get_tracer = MagicMock(return_value=mock_tracer)
            with patch("pynector.telemetry.facade.trace.get_tracer", mock_get_tracer):
                from pynector.telemetry.facade import TracingFacade

                tracer = TracingFacade("test_tracer")

                assert tracer.name == "test_tracer"
                assert tracer.tracer == mock_tracer
                mock_get_tracer.assert_called_once_with("test_tracer")
        else:
            from pynector.telemetry.facade import TracingFacade

            tracer = TracingFacade("test_tracer")

            assert tracer.name == "test_tracer"
            assert tracer.tracer is None


@pytest.mark.parametrize("has_opentelemetry", [True, False])
def test_tracing_facade_start_span(has_opentelemetry):
    """Test TracingFacade.start_span."""
    # Mock OpenTelemetry availability
    with patch("pynector.telemetry.facade.HAS_OPENTELEMETRY", has_opentelemetry):
        if has_opentelemetry:
            # Mock the trace module
            mock_span = MagicMock()
            mock_tracer = MagicMock()
            mock_tracer.start_span.return_value = mock_span
            mock_get_tracer = MagicMock(return_value=mock_tracer)
            with patch("pynector.telemetry.facade.trace.get_tracer", mock_get_tracer):
                from pynector.telemetry.facade import TracingFacade

                tracer = TracingFacade("test_tracer")

                span = tracer.start_span("test_span", {"key": "value"})

                assert span == mock_span
                mock_tracer.start_span.assert_called_once_with(
                    "test_span", attributes={"key": "value"}
                )
        else:
            from pynector.telemetry.facade import TracingFacade
            from pynector.telemetry.tracing import NoOpSpan

            tracer = TracingFacade("test_tracer")

            span = tracer.start_span("test_span", {"key": "value"})

            assert isinstance(span, NoOpSpan)
            assert span.name == "test_span"
            assert span.attributes == {"key": "value"}


@pytest.mark.parametrize("has_opentelemetry", [True, False])
def test_tracing_facade_start_as_current_span(has_opentelemetry):
    """Test TracingFacade.start_as_current_span."""
    # Mock OpenTelemetry availability
    with patch("pynector.telemetry.facade.HAS_OPENTELEMETRY", has_opentelemetry):
        if has_opentelemetry:
            # Mock the trace module
            mock_span = MagicMock()
            mock_tracer = MagicMock()
            mock_tracer.start_as_current_span.return_value = mock_span
            mock_get_tracer = MagicMock(return_value=mock_tracer)
            with patch("pynector.telemetry.facade.trace.get_tracer", mock_get_tracer):
                from pynector.telemetry.facade import TracingFacade

                tracer = TracingFacade("test_tracer")

                span = tracer.start_as_current_span("test_span", {"key": "value"})

                assert span == mock_span
                mock_tracer.start_as_current_span.assert_called_once_with(
                    "test_span", attributes={"key": "value"}
                )
        else:
            from pynector.telemetry.facade import TracingFacade
            from pynector.telemetry.tracing import NoOpSpan

            tracer = TracingFacade("test_tracer")

            span = tracer.start_as_current_span("test_span", {"key": "value"})

            assert isinstance(span, NoOpSpan)
            assert span.name == "test_span"
            assert span.attributes == {"key": "value"}


@pytest.mark.asyncio
@pytest.mark.parametrize("has_opentelemetry", [True, False])
async def test_tracing_facade_start_async_span(has_opentelemetry):
    """Test TracingFacade.start_async_span."""
    # Mock OpenTelemetry availability
    with patch("pynector.telemetry.facade.HAS_OPENTELEMETRY", has_opentelemetry):
        if has_opentelemetry:
            # Mock the trace module
            mock_span = MagicMock()
            mock_tracer = MagicMock()
            mock_tracer.start_span.return_value = mock_span
            mock_get_tracer = MagicMock(return_value=mock_tracer)
            with patch("pynector.telemetry.facade.trace.get_tracer", mock_get_tracer):
                from pynector.telemetry.facade import TracingFacade
                from pynector.telemetry.tracing import AsyncSpanWrapper

                tracer = TracingFacade("test_tracer")

                async with tracer.start_async_span(
                    "test_span", {"key": "value"}
                ) as span_wrapper:
                    assert isinstance(span_wrapper, AsyncSpanWrapper)
                    assert span_wrapper.span == mock_span
                mock_tracer.start_span.assert_called_once_with(
                    "test_span", attributes={"key": "value"}
                )
        else:
            from pynector.telemetry.facade import TracingFacade
            from pynector.telemetry.tracing import NoOpSpan

            tracer = TracingFacade("test_tracer")

            async with tracer.start_async_span("test_span", {"key": "value"}) as span:
                assert isinstance(span, NoOpSpan)
                assert span.name == "test_span"
                assert span.attributes == {"key": "value"}


@pytest.mark.asyncio
@pytest.mark.parametrize("has_opentelemetry", [True, False])
async def test_tracing_facade_start_as_current_async_span(has_opentelemetry):
    """Test TracingFacade.start_as_current_async_span."""
    # Mock OpenTelemetry availability
    with patch("pynector.telemetry.facade.HAS_OPENTELEMETRY", has_opentelemetry):
        if has_opentelemetry:
            # Mock the trace module and context
            mock_span = MagicMock()
            mock_tracer = MagicMock()
            mock_tracer.start_as_current_span.return_value = mock_span
            mock_get_tracer = MagicMock(return_value=mock_tracer)
            mock_token = MagicMock()
            mock_context = MagicMock()
            mock_attach = MagicMock(return_value=mock_token)
            mock_detach = MagicMock()
            mock_get_current = MagicMock(return_value=mock_context)

            with (
                patch("pynector.telemetry.facade.trace.get_tracer", mock_get_tracer),
                patch("pynector.telemetry.facade.attach", mock_attach),
                patch("pynector.telemetry.facade.detach", mock_detach),
                patch("pynector.telemetry.facade.get_current", mock_get_current),
            ):
                from pynector.telemetry.facade import TracingFacade
                from pynector.telemetry.tracing import AsyncSpanWrapper

                tracer = TracingFacade("test_tracer")

                async with tracer.start_as_current_async_span(
                    "test_span", {"key": "value"}
                ) as span_wrapper:
                    assert isinstance(span_wrapper, AsyncSpanWrapper)
                    assert span_wrapper.span == mock_span
                    assert span_wrapper.token == mock_token

                mock_attach.assert_called_once_with(mock_context)
                mock_tracer.start_as_current_span.assert_called_once_with(
                    "test_span", attributes={"key": "value"}
                )
        else:
            from pynector.telemetry.facade import TracingFacade
            from pynector.telemetry.tracing import NoOpSpan

            tracer = TracingFacade("test_tracer")

            async with tracer.start_as_current_async_span(
                "test_span", {"key": "value"}
            ) as span:
                assert isinstance(span, NoOpSpan)
                assert span.name == "test_span"
                assert span.attributes == {"key": "value"}


@pytest.mark.parametrize("has_structlog", [True, False])
def test_logging_facade_init(has_structlog):
    """Test LoggingFacade initialization."""
    # Mock structlog availability
    with patch("pynector.telemetry.facade.HAS_STRUCTLOG", has_structlog):
        if has_structlog:
            # Mock the structlog module
            mock_logger = MagicMock()
            mock_get_logger = MagicMock(return_value=mock_logger)
            with patch(
                "pynector.telemetry.facade.structlog.get_logger", mock_get_logger
            ):
                from pynector.telemetry.facade import LoggingFacade

                logger = LoggingFacade("test_logger")

                assert logger.name == "test_logger"
                assert logger.logger == mock_logger
                mock_get_logger.assert_called_once_with("test_logger")
        else:
            from pynector.telemetry.facade import LoggingFacade
            from pynector.telemetry.logging import NoOpLogger

            logger = LoggingFacade("test_logger")

            assert logger.name == "test_logger"
            assert isinstance(logger.logger, NoOpLogger)


@pytest.mark.parametrize(
    "has_structlog,has_opentelemetry",
    [(True, True), (True, False), (False, True), (False, False)],
)
def test_logging_facade_methods(has_structlog, has_opentelemetry):
    """Test LoggingFacade methods."""
    # Mock dependencies availability
    with (
        patch("pynector.telemetry.facade.HAS_STRUCTLOG", has_structlog),
        patch("pynector.telemetry.facade.HAS_OPENTELEMETRY", has_opentelemetry),
    ):
        if has_structlog:
            # Mock the structlog logger
            mock_logger = MagicMock()
            mock_get_logger = MagicMock(return_value=mock_logger)
            with patch(
                "pynector.telemetry.facade.structlog.get_logger", mock_get_logger
            ):
                if has_opentelemetry:
                    # Mock OpenTelemetry for trace context
                    mock_span = MagicMock()
                    mock_context = MagicMock()
                    mock_context.is_valid = True
                    mock_context.trace_id = 0x1234567890ABCDEF1234567890ABCDEF
                    mock_context.span_id = 0x1234567890ABCDEF
                    mock_span.get_span_context.return_value = mock_context
                    mock_get_current_span = MagicMock(return_value=mock_span)

                    with patch(
                        "pynector.telemetry.facade.trace.get_current_span",
                        mock_get_current_span,
                    ):
                        from pynector.telemetry.facade import LoggingFacade

                        logger = LoggingFacade("test_logger")

                        # Test debug
                        logger.debug("test_event", key="value")
                        mock_logger.debug.assert_called_once_with(
                            "test_event",
                            key="value",
                            trace_id="1234567890abcdef1234567890abcdef",
                            span_id="1234567890abcdef",
                        )

                        # Test info
                        logger.info("test_event", key="value")
                        mock_logger.info.assert_called_once_with(
                            "test_event",
                            key="value",
                            trace_id="1234567890abcdef1234567890abcdef",
                            span_id="1234567890abcdef",
                        )

                        # Test warning
                        logger.warning("test_event", key="value")
                        mock_logger.warning.assert_called_once_with(
                            "test_event",
                            key="value",
                            trace_id="1234567890abcdef1234567890abcdef",
                            span_id="1234567890abcdef",
                        )

                        # Test error
                        mock_span.set_status.reset_mock()
                        logger.error("test_event", key="value")
                        mock_logger.error.assert_called_once_with(
                            "test_event",
                            key="value",
                            trace_id="1234567890abcdef1234567890abcdef",
                            span_id="1234567890abcdef",
                        )
                        mock_span.set_status.assert_called_once()

                        # Test critical
                        mock_span.set_status.reset_mock()
                        logger.critical("test_event", key="value")
                        mock_logger.critical.assert_called_once_with(
                            "test_event",
                            key="value",
                            trace_id="1234567890abcdef1234567890abcdef",
                            span_id="1234567890abcdef",
                        )
                        mock_span.set_status.assert_called_once()
                else:
                    # No OpenTelemetry
                    from pynector.telemetry.facade import LoggingFacade

                    logger = LoggingFacade("test_logger")

                    # Test debug
                    logger.debug("test_event", key="value")
                    mock_logger.debug.assert_called_once_with("test_event", key="value")

                    # Test info
                    logger.info("test_event", key="value")
                    mock_logger.info.assert_called_once_with("test_event", key="value")

                    # Test warning
                    logger.warning("test_event", key="value")
                    mock_logger.warning.assert_called_once_with(
                        "test_event", key="value"
                    )

                    # Test error
                    logger.error("test_event", key="value")
                    mock_logger.error.assert_called_once_with("test_event", key="value")

                    # Test critical
                    logger.critical("test_event", key="value")
                    mock_logger.critical.assert_called_once_with(
                        "test_event", key="value"
                    )
        else:
            # No structlog, all methods should be no-ops
            from pynector.telemetry.facade import LoggingFacade

            logger = LoggingFacade("test_logger")

            # All methods should not raise
            logger.debug("test_event", key="value")
            logger.info("test_event", key="value")
            logger.warning("test_event", key="value")
            logger.error("test_event", key="value")
            logger.critical("test_event", key="value")
