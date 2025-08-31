# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Comprehensive tests for HTTP transport error mapping functionality.

Tests P0 requirements from TDD specification:
- V1_Transport_HTTPX (Error Mapping): HTTP status code to error type mapping
- Exception boundary validation at transport layer

Ocean's requirement: Write PROPER tests that validate actual behavior, NOT trivial tests.
These tests validate critical error classification that determines retry behavior.
"""

from unittest.mock import AsyncMock

import httpx
import msgspec
import msgspec.json
import pytest

from lionagi import _err

# Error types from _err module
# Original: from lionagi.errors import (
    _err.NonRetryableError,
    RateLimitError,
    _err.RetryableError,
    TransportError,
)
from lionagi.services.transport import HTTPXTransport


class TestV1TransportHTTPXErrorMapping:
    """Test suite: V1_Transport_HTTPX (Error Mapping)

    Validates HTTP status code to service exception mapping at the transport boundary.
    Critical for ensuring proper retry behavior and error classification throughout the service pipeline.
    """

    @pytest.mark.anyio
    async def test_http_status_retryable_errors(self):
        """Test: HTTP status codes that map to _err.RetryableError (CRITICAL)

        Validates that 429 and 5xx status codes map to _err.RetryableError for retry middleware.
        This is critical for proper resilience behavior.
        """
        transport = HTTPXTransport()

        # Define retryable status codes and their expected behavior
        retryable_statuses = [
            (
                429,
                "Rate limited",
            ),  # Rate limit - special case, should be RateLimitError
            (500, "Internal Server Error"),  # Server errors are retryable
            (502, "Bad Gateway"),
            (503, "Service Unavailable"),
            (504, "Gateway Timeout"),
            (520, "Unknown Error"),  # Custom 5xx codes
            (599, "Network Connect Timeout Error"),  # Edge of 5xx range
        ]

        for status_code, reason in retryable_statuses:
            # Create mock response
            mock_response_data = {
                "error": f"Test {status_code} error",
                "code": status_code,
            }
            mock_response_content = msgspec.json.encode(mock_response_data)

            # Mock httpx client to return the status code
            mock_response = httpx.Response(
                status_code=status_code,
                headers={"content-type": "application/json"},
                content=mock_response_content,
                request=httpx.Request("POST", "https://api.test.com/v1/chat"),
            )

            # Special handling for 429 which should be RateLimitError
            if status_code == 429:
                # Mock response with Retry-After header
                mock_response = httpx.Response(
                    status_code=429,
                    headers={"content-type": "application/json", "retry-after": "60"},
                    content=mock_response_content,
                    request=httpx.Request("POST", "https://api.test.com/v1/chat"),
                )

                # Test that 429 maps to RateLimitError specifically
                with pytest.raises(RateLimitError) as exc_info:
                    transport._check_response_status(mock_response)

                error = exc_info.value
                assert isinstance(
                    error, RateLimitError
                ), f"429 should map to RateLimitError, got {type(error)}"
                assert (
                    error.retry_after == 60.0
                ), f"Expected retry_after=60.0, got {error.retry_after}"
                assert error.retryable is True, "RateLimitError should be retryable"

            else:
                # Test that 5xx maps to _err.RetryableError
                with pytest.raises(_err.RetryableError) as exc_info:
                    transport._check_response_status(mock_response)

                error = exc_info.value
                assert isinstance(
                    error, _err.RetryableError
                ), f"{status_code} should map to _err.RetryableError, got {type(error)}"
                assert error.retryable is True, f"{status_code} error should be retryable"
                assert str(status_code) in str(
                    error
                ), f"Error message should contain status code {status_code}"

    @pytest.mark.anyio
    async def test_http_status_non_retryable_errors(self):
        """Test: HTTP status codes that map to _err.NonRetryableError (CRITICAL)

        Validates that 4xx (except 429) status codes map to _err.NonRetryableError.
        Critical for preventing infinite retries on client errors.
        """
        transport = HTTPXTransport()

        # Define non-retryable status codes (4xx except 429)
        non_retryable_statuses = [
            (400, "Bad Request"),
            (401, "Unauthorized"),
            (403, "Forbidden"),
            (404, "Not Found"),
            (405, "Method Not Allowed"),
            (406, "Not Acceptable"),
            (408, "Request Timeout"),  # Client timeout, non-retryable
            (409, "Conflict"),
            (410, "Gone"),
            (413, "Payload Too Large"),
            (414, "URI Too Long"),
            (415, "Unsupported Media Type"),
            (422, "Unprocessable Entity"),
            (423, "Locked"),
            (424, "Failed Dependency"),
            (428, "Precondition Required"),
            # Note: 429 is handled separately as RateLimitError
            (431, "Request Header Fields Too Large"),
            (451, "Unavailable For Legal Reasons"),
            (499, "Client Closed Request"),  # Edge of 4xx range
        ]

        for status_code, reason in non_retryable_statuses:
            # Create mock response with error details
            mock_response_data = {
                "error": {
                    "message": f"Test {status_code} client error",
                    "type": "client_error",
                    "code": status_code,
                }
            }
            mock_response_content = msgspec.json.encode(mock_response_data)

            mock_response = httpx.Response(
                status_code=status_code,
                headers={"content-type": "application/json"},
                content=mock_response_content,
                request=httpx.Request("POST", "https://api.test.com/v1/chat"),
            )

            # Test that 4xx (except 429) maps to _err.NonRetryableError
            with pytest.raises(_err.NonRetryableError) as exc_info:
                transport._check_response_status(mock_response)

            error = exc_info.value
            assert isinstance(
                error, _err.NonRetryableError
            ), f"{status_code} should map to _err.NonRetryableError, got {type(error)}"
            assert error.retryable is False, f"{status_code} error should not be retryable"
            assert str(status_code) in str(
                error
            ), f"Error message should contain status code {status_code}"

            # Verify error context contains useful debugging information
            assert "status_code" in error.context
            assert error.context["status_code"] == status_code
            assert "url" in error.context

    @pytest.mark.anyio
    async def test_httpx_exception_mapping(self):
        """Test: HTTPX exception to service error mapping (CRITICAL)

        Validates that low-level HTTPX exceptions are properly mapped to service errors.
        Critical for proper error handling at the transport boundary.
        """
        transport = HTTPXTransport()

        # Mock different HTTPX exceptions and test their mapping
        test_cases = [
            # TimeoutException -> TransportError
            (
                httpx.TimeoutException("Request timed out"),
                TransportError,
                "Request timed out",
            ),
            # NetworkError -> _err.RetryableError
            (httpx.NetworkError("Connection failed"), _err.RetryableError, "Network error"),
            # ConnectError (subclass of NetworkError) -> _err.RetryableError
            (httpx.ConnectError("Connection refused"), _err.RetryableError, "Network error"),
            # ReadError (subclass of NetworkError) -> _err.RetryableError
            (httpx.ReadError("Connection broken"), _err.RetryableError, "Network error"),
        ]

        for httpx_exception, expected_error_type, expected_message_part in test_cases:
            # Test send_json exception mapping
            transport._client.request = AsyncMock(side_effect=httpx_exception)

            with pytest.raises(expected_error_type) as exc_info:
                await transport.send_json(
                    method="POST",
                    url="https://api.test.com/v1/test",
                    headers={},
                    json={"test": "data"},
                    timeout_s=30.0,
                )

            error = exc_info.value
            assert (
                expected_message_part.lower() in str(error).lower()
            ), f"Expected '{expected_message_part}' in error message: {error}"

            # Validate error context contains request details
            assert "method" in error.context
            assert "url" in error.context
            assert error.context["method"] == "POST"

    @pytest.mark.anyio
    async def test_json_decode_error_mapping(self):
        """Test: JSON decode error handling (CRITICAL)

        Validates proper handling of invalid JSON responses.
        Critical for preventing crashes when APIs return malformed data.
        """
        transport = HTTPXTransport()

        # Mock successful HTTP response but invalid JSON content
        invalid_json_content = b"{ invalid json content }"
        mock_response = httpx.Response(
            status_code=200,
            headers={"content-type": "application/json"},
            content=invalid_json_content,
            request=httpx.Request("POST", "https://api.test.com/v1/test"),
        )

        transport._client.request = AsyncMock(return_value=mock_response)

        # Should raise TransportError for JSON decode failure
        with pytest.raises(TransportError) as exc_info:
            await transport.send_json(
                method="POST",
                url="https://api.test.com/v1/test",
                headers={},
                json={"test": "data"},
                timeout_s=30.0,
            )

        error = exc_info.value
        assert "Invalid JSON response" in str(error)
        assert "content_preview" in error.context
        # Should include preview of invalid content for debugging
        assert "invalid json" in error.context["content_preview"]

    @pytest.mark.anyio
    async def test_successful_response_no_error(self):
        """Test: Successful responses don't raise errors (CRITICAL)

        Validates that successful status codes (2xx) don't trigger error mapping.
        Regression test to ensure error checking doesn't interfere with success cases.
        """
        transport = HTTPXTransport()

        successful_statuses = [200, 201, 202, 204, 206]

        for status_code in successful_statuses:
            # Mock successful response
            response_data = {"result": "success", "status": status_code}
            mock_response = httpx.Response(
                status_code=status_code,
                headers={"content-type": "application/json"},
                content=msgspec.json.encode(response_data),
                request=httpx.Request("POST", "https://api.test.com/v1/test"),
            )

            transport._client.request = AsyncMock(return_value=mock_response)

            # Should not raise any exception
            result = await transport.send_json(
                method="POST",
                url="https://api.test.com/v1/test",
                headers={},
                json={"test": "data"},
                timeout_s=30.0,
            )

            # Should return parsed JSON
            assert result == response_data
            assert result["status"] == status_code

    @pytest.mark.anyio
    async def test_rate_limit_retry_after_parsing(self):
        """Test: Rate limit Retry-After header parsing (CRITICAL)

        Validates proper parsing of Retry-After header for rate limit errors.
        Critical for implementing proper backoff in retry middleware.
        """
        transport = HTTPXTransport()

        # Test different Retry-After header formats
        retry_after_cases = [
            ("60", 60.0),  # String seconds
            ("120", 120.0),  # Different value
            ("3600", 3600.0),  # Hour
        ]

        for retry_after_str, expected_seconds in retry_after_cases:
            mock_response = httpx.Response(
                status_code=429,
                headers={
                    "content-type": "application/json",
                    "retry-after": retry_after_str,
                },
                content=b'{"error": "Rate limited"}',
                request=httpx.Request("POST", "https://api.test.com/v1/test"),
            )

            with pytest.raises(RateLimitError) as exc_info:
                transport._check_response_status(mock_response)

            error = exc_info.value
            assert (
                error.retry_after == expected_seconds
            ), f"Expected retry_after={expected_seconds}, got {error.retry_after}"
            assert "retry_after" in error.context
            assert error.context["retry_after"] == expected_seconds

    @pytest.mark.anyio
    async def test_rate_limit_missing_retry_after(self):
        """Test: Rate limit without Retry-After header uses default.

        Validates fallback behavior when Retry-After header is missing.
        """
        transport = HTTPXTransport()

        mock_response = httpx.Response(
            status_code=429,
            headers={"content-type": "application/json"},  # No Retry-After
            content=b'{"error": "Rate limited"}',
            request=httpx.Request("POST", "https://api.test.com/v1/test"),
        )

        with pytest.raises(RateLimitError) as exc_info:
            transport._check_response_status(mock_response)

        error = exc_info.value
        # Should use default retry_after value (60 seconds from transport.py)
        assert (
            error.retry_after == 60.0
        ), f"Expected default retry_after=60.0, got {error.retry_after}"

    @pytest.mark.anyio
    async def test_streaming_error_mapping(self):
        """Test: Error mapping during streaming responses (CRITICAL)

        Validates that errors during streaming are properly mapped.
        Critical for streaming service implementations.
        """
        transport = HTTPXTransport()

        # Mock streaming with HTTP error status
        mock_response = httpx.Response(
            status_code=500,
            headers={"content-type": "application/json"},
            content=b'{"error": "Server error during stream"}',
            request=httpx.Request("POST", "https://api.test.com/v1/stream"),
        )

        # Create an async context manager for the response
        class MockAsyncContextManager:
            async def __aenter__(self):
                return mock_response

            async def __aexit__(self, exc_type, exc_val, exc_tb):
                return None

        # Mock the stream method to return the context manager directly
        def mock_stream(*args, **kwargs):
            return MockAsyncContextManager()

        transport._client.stream = mock_stream

        # Should raise _err.RetryableError for 5xx during streaming
        with pytest.raises(_err.RetryableError) as exc_info:
            async for chunk in transport.stream_json(
                method="POST",
                url="https://api.test.com/v1/stream",
                headers={},
                json={"test": "data"},
                timeout_s=30.0,
            ):
                pass  # Should not reach this

        error = exc_info.value
        assert "500" in str(error)
        assert "operation" in error.context
        assert error.context["operation"] == "streaming"

    @pytest.mark.anyio
    async def test_error_context_information(self):
        """Test: Error context contains debugging information (CRITICAL)

        Validates that all errors include sufficient context for debugging.
        Critical for production troubleshooting and observability.
        """
        transport = HTTPXTransport()

        # Test with a 403 Forbidden response
        error_response_data = {
            "error": {
                "message": "Insufficient permissions",
                "type": "authentication_error",
                "details": {"required_scope": "admin", "provided_scope": "user"},
            }
        }
        mock_response_content = msgspec.json.encode(error_response_data)

        mock_response = httpx.Response(
            status_code=403,
            headers={
                "content-type": "application/json",
                "x-request-id": "req_123456789",
                "x-ratelimit-remaining": "99",
            },
            content=mock_response_content,
            request=httpx.Request("POST", "https://api.example.com/v1/protected"),
        )

        with pytest.raises(_err.NonRetryableError) as exc_info:
            transport._check_response_status(mock_response)

        error = exc_info.value

        # Validate comprehensive error context
        required_context_fields = ["status_code", "url", "headers", "response_preview"]
        for field in required_context_fields:
            assert field in error.context, f"Error context missing required field: {field}"

        # Validate specific context values
        assert error.context["status_code"] == 403
        assert "api.example.com" in error.context["url"]
        assert "x-request-id" in error.context["headers"]
        assert error.context["headers"]["x-request-id"] == "req_123456789"

        # Response preview should contain error details (truncated for large responses)
        response_preview = error.context["response_preview"]
        assert "Insufficient permissions" in response_preview
        assert len(response_preview) <= 500 + len(
            "... [truncated]"
        ), "Response preview should be truncated"

    @pytest.mark.anyio
    async def test_response_truncation_behavior(self):
        """Test: Large response body truncation in error context.

        Validates that very large error responses are properly truncated.
        Prevents memory issues and log spam from huge error responses.
        """
        transport = HTTPXTransport()

        # Create a large error response (over 500 chars)
        large_error_data = {
            "error": {
                "message": "Very long error message " + "x" * 1000,  # Much longer than 500 chars
                "details": {"field_" + str(i): f"error_{i}" for i in range(100)},
            }
        }
        large_response_content = msgspec.json.encode(large_error_data)

        mock_response = httpx.Response(
            status_code=400,
            headers={"content-type": "application/json"},
            content=large_response_content,
            request=httpx.Request("POST", "https://api.test.com/v1/test"),
        )

        with pytest.raises(_err.NonRetryableError) as exc_info:
            transport._check_response_status(mock_response)

        error = exc_info.value
        response_preview = error.context["response_preview"]

        # Should be truncated at around 500 characters + truncation marker
        assert (
            len(response_preview) <= 520
        ), f"Response preview too long: {len(response_preview)} chars"
        assert response_preview.endswith(
            "... [truncated]"
        ), "Large response should be truncated with marker"

        # Should still contain beginning of error message
        assert "Very long error message" in response_preview
