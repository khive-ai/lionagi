"""
Tests for the HTTP-specific error classes.

This module contains tests for the HTTP-specific error classes that extend
the base transport error hierarchy.
"""

from unittest.mock import MagicMock

from pynector.transport.errors import TransportSpecificError
from pynector.transport.http.errors import (
    CircuitOpenError,
    HTTPClientError,
    HTTPForbiddenError,
    HTTPNotFoundError,
    HTTPServerError,
    HTTPStatusError,
    HTTPTimeoutError,
    HTTPTooManyRequestsError,
    HTTPTransportError,
    HTTPUnauthorizedError,
)


class TestHTTPErrors:
    """Tests for the HTTP-specific error classes."""

    def test_http_transport_error_inheritance(self):
        """Test that HTTPTransportError inherits from TransportSpecificError."""
        error = HTTPTransportError("Test error")
        assert isinstance(error, TransportSpecificError)
        assert str(error) == "Test error"

    def test_http_status_error(self):
        """Test HTTPStatusError initialization and properties."""
        mock_response = MagicMock()
        mock_response.status_code = 400

        error = HTTPStatusError(mock_response, "HTTP error 400")
        assert isinstance(error, HTTPTransportError)
        assert error.response == mock_response
        assert error.status_code == 400
        assert str(error) == "HTTP error 400"

    def test_http_client_error(self):
        """Test HTTPClientError initialization and inheritance."""
        mock_response = MagicMock()
        mock_response.status_code = 400

        error = HTTPClientError(mock_response, "Client error")
        assert isinstance(error, HTTPStatusError)
        assert error.status_code == 400
        assert str(error) == "Client error"

    def test_http_server_error(self):
        """Test HTTPServerError initialization and inheritance."""
        mock_response = MagicMock()
        mock_response.status_code = 500

        error = HTTPServerError(mock_response, "Server error")
        assert isinstance(error, HTTPStatusError)
        assert error.status_code == 500
        assert str(error) == "Server error"

    def test_specific_http_errors(self):
        """Test specific HTTP error classes."""
        mock_response = MagicMock()

        # Test 401 Unauthorized
        mock_response.status_code = 401
        error = HTTPUnauthorizedError(mock_response, "Unauthorized")
        assert isinstance(error, HTTPClientError)
        assert error.status_code == 401
        assert str(error) == "Unauthorized"

        # Test 403 Forbidden
        mock_response.status_code = 403
        error = HTTPForbiddenError(mock_response, "Forbidden")
        assert isinstance(error, HTTPClientError)
        assert error.status_code == 403
        assert str(error) == "Forbidden"

        # Test 404 Not Found
        mock_response.status_code = 404
        error = HTTPNotFoundError(mock_response, "Not found")
        assert isinstance(error, HTTPClientError)
        assert error.status_code == 404
        assert str(error) == "Not found"

        # Test 408 Timeout
        mock_response.status_code = 408
        error = HTTPTimeoutError(mock_response, "Timeout")
        assert isinstance(error, HTTPClientError)
        assert error.status_code == 408
        assert str(error) == "Timeout"

        # Test 429 Too Many Requests
        mock_response.status_code = 429
        error = HTTPTooManyRequestsError(mock_response, "Too many requests")
        assert isinstance(error, HTTPClientError)
        assert error.status_code == 429
        assert str(error) == "Too many requests"

    def test_circuit_open_error(self):
        """Test CircuitOpenError initialization and inheritance."""
        error = CircuitOpenError("Circuit is open")
        assert isinstance(error, HTTPTransportError)
        assert str(error) == "Circuit is open"
