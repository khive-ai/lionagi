"""
Tests for the error hierarchy.

This module contains tests for the error hierarchy of the Transport Abstraction Layer.
"""

import pytest

from pynector.transport.errors import (
    ConnectionError,
    ConnectionRefusedError,
    ConnectionTimeoutError,
    DeserializationError,
    MessageError,
    SerializationError,
    TransportError,
    TransportSpecificError,
)


def test_error_hierarchy():
    """Test that the error hierarchy is correctly implemented."""
    # Test inheritance
    assert issubclass(ConnectionError, TransportError)
    assert issubclass(ConnectionTimeoutError, ConnectionError)
    assert issubclass(ConnectionRefusedError, ConnectionError)
    assert issubclass(MessageError, TransportError)
    assert issubclass(SerializationError, MessageError)
    assert issubclass(DeserializationError, MessageError)
    assert issubclass(TransportSpecificError, TransportError)

    # Test instantiation
    error = TransportError("Test error")
    assert str(error) == "Test error"

    # Test specific error types
    timeout_error = ConnectionTimeoutError("Connection timed out")
    assert isinstance(timeout_error, ConnectionError)
    assert isinstance(timeout_error, TransportError)
    assert str(timeout_error) == "Connection timed out"


def test_error_handling():
    """Test error handling in a typical use case."""
    try:
        raise ConnectionTimeoutError("Connection timed out after 30s")
    except ConnectionError as e:
        assert "timed out" in str(e)
    except TransportError:
        pytest.fail("ConnectionTimeoutError should be caught by ConnectionError")

    try:
        raise SerializationError("Failed to serialize message")
    except MessageError as e:
        assert "serialize" in str(e)
    except TransportError:
        pytest.fail("SerializationError should be caught by MessageError")


def test_custom_transport_error():
    """Test creating a custom transport-specific error."""

    class CustomTransportError(TransportSpecificError):
        """Custom transport error for testing."""

        pass

    error = CustomTransportError("Custom error")
    assert isinstance(error, TransportSpecificError)
    assert isinstance(error, TransportError)
    assert str(error) == "Custom error"

    # Test catching with parent types
    try:
        raise CustomTransportError("Custom error")
    except TransportError as e:
        assert "Custom error" == str(e)
    else:
        pytest.fail("CustomTransportError should be caught by TransportError")
