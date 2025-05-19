"""Tests for the logging module."""


def test_noop_logger_init():
    """Test NoOpLogger initialization."""
    from pynector.telemetry.logging import NoOpLogger

    logger = NoOpLogger("test_logger")
    assert logger.name == "test_logger"

    # Test with default value
    logger = NoOpLogger()
    assert logger.name == ""


def test_noop_logger_methods():
    """Test NoOpLogger methods."""
    from pynector.telemetry.logging import NoOpLogger

    logger = NoOpLogger("test_logger")

    # All methods should be no-ops and not raise exceptions
    logger.debug("test_event", key="value")
    logger.info("test_event", key="value")
    logger.warning("test_event", key="value")
    logger.error("test_event", key="value")
    logger.critical("test_event", key="value")
