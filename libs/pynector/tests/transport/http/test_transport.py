"""
Tests for the HTTP transport implementation.

This module contains tests for the HTTPTransport class, which implements
the Transport Protocol for HTTP communication.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from pynector.transport.errors import ConnectionError, ConnectionTimeoutError
from pynector.transport.http.errors import (
    HTTPClientError,
    HTTPServerError,
    HTTPTransportError,
)
from pynector.transport.http.message import HttpMessage
from pynector.transport.http.transport import HTTPTransport
from tests.transport.http.mock_server import MockHTTPServer


class TestHTTPTransport:
    """Tests for the HTTPTransport class."""

    def test_init_default_values(self):
        """Test HTTPTransport initialization with default values."""
        transport = HTTPTransport()
        assert transport.base_url == ""
        assert transport.headers == {}
        assert transport.timeout == 10.0
        assert transport.max_retries == 3
        assert transport.retry_backoff_factor == 0.5
        assert transport.retry_status_codes == {429, 500, 502, 503, 504}
        assert transport.follow_redirects is True
        assert transport.verify_ssl is True
        assert transport.http2 is False
        assert transport._client is None
        assert transport._message_type is None

    def test_init_custom_values(self):
        """Test HTTPTransport initialization with custom values."""
        transport = HTTPTransport(
            base_url="https://example.com",
            headers={"User-Agent": "pynector/1.0"},
            timeout=5.0,
            max_retries=2,
            retry_backoff_factor=1.0,
            retry_status_codes={500, 502},
            follow_redirects=False,
            verify_ssl=False,
            http2=True,
        )
        assert transport.base_url == "https://example.com"
        assert transport.headers == {"User-Agent": "pynector/1.0"}
        assert transport.timeout == 5.0
        assert transport.max_retries == 2
        assert transport.retry_backoff_factor == 1.0
        assert transport.retry_status_codes == {500, 502}
        assert transport.follow_redirects is False
        assert transport.verify_ssl is False
        assert transport.http2 is True

    @pytest.mark.asyncio
    async def test_connect(self):
        """Test connect method creates AsyncClient."""
        transport = HTTPTransport()
        await transport.connect()

        assert transport._client is not None
        assert isinstance(transport._client, httpx.AsyncClient)

        await transport.disconnect()

    @pytest.mark.asyncio
    async def test_disconnect(self):
        """Test disconnect method closes AsyncClient."""
        transport = HTTPTransport()
        await transport.connect()

        client = transport._client
        assert client is not None

        # Patch the aclose method to verify it's called
        with patch.object(client, "aclose", new_callable=AsyncMock) as mock_aclose:
            await transport.disconnect()
            mock_aclose.assert_called_once()

        assert transport._client is None

    @pytest.mark.asyncio
    async def test_connect_error(self):
        """Test connect method handles errors."""
        with patch(
            "httpx.AsyncClient", side_effect=httpx.ConnectError("Connection error")
        ):
            transport = HTTPTransport()
            with pytest.raises(ConnectionError, match="Failed to establish connection"):
                await transport.connect()

    @pytest.mark.asyncio
    async def test_connect_timeout(self):
        """Test connect method handles timeouts."""
        with patch(
            "httpx.AsyncClient", side_effect=httpx.ConnectTimeout("Connection timeout")
        ):
            transport = HTTPTransport()
            with pytest.raises(
                ConnectionTimeoutError, match="Connection attempt timed out"
            ):
                await transport.connect()

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test async context manager protocol."""
        async with HTTPTransport() as transport:
            assert transport._client is not None
            assert isinstance(transport._client, httpx.AsyncClient)

        assert transport._client is None

    @pytest.mark.asyncio
    async def test_send_not_connected(self):
        """Test send when not connected raises ConnectionError."""
        transport = HTTPTransport()
        message = HttpMessage(method="GET", url="/test")

        with pytest.raises(ConnectionError, match="Transport not connected"):
            await transport.send(message)

    @pytest.mark.asyncio
    async def test_send_basic(self):
        """Test basic send functionality."""
        transport = HTTPTransport()
        message = HttpMessage(method="GET", url="/test")

        # Mock the AsyncClient.request method
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200

        with patch.object(
            httpx.AsyncClient, "request", return_value=mock_response
        ) as mock_request:
            await transport.connect()
            await transport.send(message)

            # Verify request was called with correct parameters
            mock_request.assert_called_once()
            # In newer versions of unittest.mock, call_args is a mock._Call object
            # We need to access the args and kwargs differently
            call_args = mock_request.call_args
            assert call_args.kwargs["method"] == "GET"  # method
            assert call_args.kwargs["url"] == "/test"  # url

            await transport.disconnect()

    @pytest.mark.asyncio
    async def test_send_retry_success(self):
        """Test send with retry logic (success after retry)."""
        transport = HTTPTransport(max_retries=2)
        message = HttpMessage(method="GET", url="/test")

        # First request fails with 503, second succeeds with 200
        mock_response_fail = MagicMock(spec=httpx.Response)
        mock_response_fail.status_code = 503
        mock_response_fail.reason_phrase = "Service Unavailable"

        mock_response_success = MagicMock(spec=httpx.Response)
        mock_response_success.status_code = 200

        with patch.object(
            httpx.AsyncClient,
            "request",
            side_effect=[mock_response_fail, mock_response_success],
        ) as mock_request:
            await transport.connect()
            await transport.send(message)

            # Verify request was called twice
            assert mock_request.call_count == 2

            await transport.disconnect()

    @pytest.mark.asyncio
    async def test_send_retry_max_exceeded(self):
        """Test send with retry logic (max retries exceeded)."""
        transport = HTTPTransport(max_retries=2)
        message = HttpMessage(method="GET", url="/test")

        # All requests fail with 503
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 503
        mock_response.reason_phrase = "Service Unavailable"

        with patch.object(
            httpx.AsyncClient, "request", return_value=mock_response
        ) as mock_request:
            await transport.connect()

            with pytest.raises(HTTPServerError, match="Server error"):
                await transport.send(message)

            # Verify request was called 3 times (initial + 2 retries)
            assert mock_request.call_count == 3

            await transport.disconnect()

    @pytest.mark.asyncio
    async def test_send_network_error_retry(self):
        """Test send with network error retry."""
        transport = HTTPTransport(max_retries=2)
        message = HttpMessage(method="GET", url="/test")

        # First request fails with ConnectError, second succeeds
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200

        with patch.object(
            httpx.AsyncClient,
            "request",
            side_effect=[httpx.ConnectError("Connection error"), mock_response],
        ) as mock_request:
            await transport.connect()
            await transport.send(message)

            # Verify request was called twice
            assert mock_request.call_count == 2

            await transport.disconnect()

    @pytest.mark.asyncio
    async def test_receive_not_connected(self):
        """Test receive when not connected raises ConnectionError."""
        transport = HTTPTransport()

        with pytest.raises(ConnectionError, match="Transport not connected"):
            async for _ in transport.receive():
                pass

    @pytest.mark.asyncio
    async def test_receive_no_message_type(self):
        """Test receive with no message type set."""
        transport = HTTPTransport()
        await transport.connect()

        with pytest.raises(HTTPTransportError, match="No message type has been set"):
            async for _ in transport.receive():
                pass

        await transport.disconnect()

    @pytest.mark.asyncio
    async def test_receive_basic(self):
        """Test basic receive functionality."""
        transport = HTTPTransport()
        transport._message_type = HttpMessage

        # Mock _get_next_response to return a valid response
        mock_response = json.dumps(
            {
                "headers": {"Content-Type": "application/json"},
                "payload": {
                    "method": "GET",
                    "url": "/test",
                    "data": {"result": "success"},
                },
            }
        ).encode("utf-8")

        with patch.object(
            HTTPTransport, "_get_next_response", return_value=mock_response
        ):
            await transport.connect()

            messages = []
            async for message in transport.receive():
                messages.append(message)

            assert len(messages) == 1
            assert isinstance(messages[0], HttpMessage)
            assert messages[0].headers == {"Content-Type": "application/json"}
            assert messages[0].payload["method"] == "GET"
            assert messages[0].payload["url"] == "/test"
            assert messages[0].payload["data"] == {"result": "success"}

            await transport.disconnect()

    @pytest.mark.asyncio
    async def test_stream_response(self):
        """Test stream_response method."""
        transport = HTTPTransport()
        message = HttpMessage(method="GET", url="/test")

        # Mock the AsyncClient.stream method
        mock_response = AsyncMock()
        mock_response.__aenter__.return_value = mock_response
        mock_response.status_code = 200

        # Create an async iterator for aiter_bytes
        async def mock_aiter_bytes():
            for chunk in [b"chunk1", b"chunk2", b"chunk3"]:
                yield chunk

        mock_response.aiter_bytes = mock_aiter_bytes

        with patch.object(
            httpx.AsyncClient, "stream", return_value=mock_response
        ) as mock_stream:
            await transport.connect()

            chunks = []
            async for chunk in transport.stream_response(message):
                chunks.append(chunk)

            # Verify stream was called with correct parameters
            mock_stream.assert_called_once()
            call_args = mock_stream.call_args
            assert call_args.kwargs["method"] == "GET"  # method
            assert call_args.kwargs["url"] == "/test"  # url
            # No need to check for stream parameter as it's not passed anymore

            # Verify chunks were yielded correctly
            assert chunks == [b"chunk1", b"chunk2", b"chunk3"]

            await transport.disconnect()

    @pytest.mark.asyncio
    async def test_stream_response_error(self):
        """Test stream_response with error response."""
        transport = HTTPTransport()
        message = HttpMessage(method="GET", url="/test")

        # Mock the AsyncClient.stream method
        mock_response = AsyncMock()
        mock_response.__aenter__.return_value = mock_response
        mock_response.status_code = 404
        mock_response.reason_phrase = "Not Found"

        # Create an async iterator for aiter_bytes (though it shouldn't be called)
        async def mock_aiter_bytes():
            for chunk in [b"chunk1"]:
                yield chunk

        mock_response.aiter_bytes = mock_aiter_bytes

        with patch.object(httpx.AsyncClient, "stream", return_value=mock_response):
            await transport.connect()
            with pytest.raises(HTTPTransportError, match="HTTP request failed"):
                async for _ in transport.stream_response(message):
                    pass
                    pass

            await transport.disconnect()

    def test_extract_headers(self):
        """Test _extract_headers method."""
        transport = HTTPTransport(headers={"User-Agent": "pynector/1.0"})
        message = HttpMessage(
            headers={"Content-Type": "application/json", "X-Test": "test"}
        )

        headers = transport._extract_headers(message)
        assert headers["User-Agent"] == "pynector/1.0"
        assert headers["Content-Type"] == "application/json"
        assert headers["X-Test"] == "test"

    def test_prepare_request(self):
        """Test _prepare_request method."""
        transport = HTTPTransport()
        message = HttpMessage(
            method="POST", url="/test", params={"q": "test"}, json_data={"data": "test"}
        )

        method, url, request_kwargs = transport._prepare_request(message)
        assert method == "POST"
        assert url == "/test"
        assert request_kwargs["params"] == {"q": "test"}
        assert request_kwargs["json"] == {"data": "test"}

    def test_handle_error_response(self):
        """Test _handle_error_response method."""
        transport = HTTPTransport()

        # Test various status codes
        status_codes_and_errors = [
            (401, HTTPClientError),
            (403, HTTPClientError),
            (404, HTTPClientError),
            (408, HTTPClientError),
            (429, HTTPClientError),
            (400, HTTPClientError),
            (500, HTTPServerError),
        ]

        for status_code, error_class in status_codes_and_errors:
            mock_response = MagicMock(spec=httpx.Response)
            mock_response.status_code = status_code
            mock_response.reason_phrase = "Test"

            with pytest.raises(error_class):
                transport._handle_error_response(mock_response)


@pytest.mark.asyncio
class TestHTTPTransportIntegration:
    """Integration tests for the HTTPTransport class using MockHTTPServer."""

    @pytest.mark.skip(reason="Mock server issues")
    async def test_http_transport_end_to_end(self):
        """Test HTTPTransport end-to-end with mock server."""
        # Set up mock server
        async with MockHTTPServer() as server:
            server.add_route("/test", {"data": "test"}, status_code=200)

            # Create transport
            transport = HTTPTransport(base_url=server.url)
            message = HttpMessage(method="GET", url="/test")

            # Send and receive
            async with transport:
                await transport.send(message)

                # Verify response
                received = False
                async for response in transport.receive():
                    received = True
                    payload = response.get_payload()
                    assert payload["status_code"] == 200
                    assert "data" in payload["data"]

                assert received, "No response was received"

    @pytest.mark.skip(reason="Mock server issues")
    async def test_http_transport_streaming(self):
        """Test HTTPTransport streaming with mock server."""
        # Set up mock server
        async with MockHTTPServer() as server:
            server.add_streaming_route("/stream", ["chunk1", "chunk2", "chunk3"])

            # Create transport
            transport = HTTPTransport(base_url=server.url)
            message = HttpMessage(method="GET", url="/stream")

            # Stream response
            chunks = []
            async with transport:
                async for chunk in transport.stream_response(message):
                    chunks.append(chunk)

            # Verify chunks
            assert len(chunks) == 3
            assert b"".join(chunks).decode("utf-8") == "chunk1chunk2chunk3"

    @pytest.mark.skip(reason="Mock server issues")
    async def test_http_transport_error_handling(self):
        """Test HTTPTransport error handling with mock server."""
        # Set up mock server
        async with MockHTTPServer() as server:
            server.add_route("/error", {"error": "Not found"}, status_code=404)

            # Create transport
            transport = HTTPTransport(base_url=server.url)
            message = HttpMessage(method="GET", url="/error")

            # Send and expect error
            async with transport:
                with pytest.raises(HTTPClientError) as excinfo:
                    await transport.send(message)

                assert excinfo.value.status_code == 404
