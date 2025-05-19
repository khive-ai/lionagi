"""
Tests for SDK-specific error classes.
"""

import pytest

from pynector.transport.errors import TransportSpecificError
from pynector.transport.sdk.errors import (
    AuthenticationError,
    InvalidRequestError,
    PermissionError,
    RateLimitError,
    RequestTooLargeError,
    ResourceNotFoundError,
    SdkTransportError,
)


def test_error_hierarchy():
    """Test that the error hierarchy is correctly implemented."""
    # Test inheritance
    assert issubclass(SdkTransportError, TransportSpecificError)
    assert issubclass(AuthenticationError, SdkTransportError)
    assert issubclass(RateLimitError, SdkTransportError)
    assert issubclass(InvalidRequestError, SdkTransportError)
    assert issubclass(ResourceNotFoundError, SdkTransportError)
    assert issubclass(PermissionError, SdkTransportError)
    assert issubclass(RequestTooLargeError, SdkTransportError)

    # Test instantiation
    error = SdkTransportError("Test error")
    assert str(error) == "Test error"

    # Test specific error types
    auth_error = AuthenticationError("Invalid API key")
    assert isinstance(auth_error, SdkTransportError)
    assert isinstance(auth_error, TransportSpecificError)
    assert str(auth_error) == "Invalid API key"


def test_error_handling():
    """Test error handling in a typical use case."""
    try:
        raise AuthenticationError("Invalid API key")
    except SdkTransportError as e:
        assert "Invalid API key" in str(e)
    except TransportSpecificError:
        pytest.fail("AuthenticationError should be caught by SdkTransportError")

    try:
        raise RateLimitError("Rate limit exceeded")
    except SdkTransportError as e:
        assert "Rate limit" in str(e)
    except TransportSpecificError:
        pytest.fail("RateLimitError should be caught by SdkTransportError")
