"""
Tests for the HTTP transport factory.

This module contains tests for the HTTPTransportFactory class, which creates
and configures HTTPTransport instances.
"""

from pynector.transport.http.factory import HTTPTransportFactory
from pynector.transport.http.message import HttpMessage
from pynector.transport.http.transport import HTTPTransport


class TestHTTPTransportFactory:
    """Tests for the HTTPTransportFactory class."""

    def test_init_default_values(self):
        """Test HTTPTransportFactory initialization with default values."""
        factory = HTTPTransportFactory(
            base_url="https://example.com", message_type=HttpMessage
        )
        assert factory.base_url == "https://example.com"
        assert factory.message_type == HttpMessage
        assert factory.default_headers == {}
        assert factory.default_timeout == 30.0
        assert factory.default_max_retries == 3
        assert factory.default_retry_backoff_factor == 0.5
        assert factory.default_retry_status_codes == {429, 500, 502, 503, 504}
        assert factory.default_follow_redirects is True
        assert factory.default_verify_ssl is True
        assert factory.default_http2 is False

    def test_init_custom_values(self):
        """Test HTTPTransportFactory initialization with custom values."""
        factory = HTTPTransportFactory(
            base_url="https://example.com",
            message_type=HttpMessage,
            default_headers={"User-Agent": "pynector/1.0"},
            default_timeout=5.0,
            default_max_retries=2,
            default_retry_backoff_factor=1.0,
            default_retry_status_codes={500, 502},
            default_follow_redirects=False,
            default_verify_ssl=False,
            default_http2=True,
        )
        assert factory.base_url == "https://example.com"
        assert factory.message_type == HttpMessage
        assert factory.default_headers == {"User-Agent": "pynector/1.0"}
        assert factory.default_timeout == 5.0
        assert factory.default_max_retries == 2
        assert factory.default_retry_backoff_factor == 1.0
        assert factory.default_retry_status_codes == {500, 502}
        assert factory.default_follow_redirects is False
        assert factory.default_verify_ssl is False
        assert factory.default_http2 is True

    def test_create_transport_default(self):
        """Test create_transport with default options."""
        factory = HTTPTransportFactory(
            base_url="https://example.com", message_type=HttpMessage
        )
        transport = factory.create_transport()

        assert isinstance(transport, HTTPTransport)
        assert transport.base_url == "https://example.com"
        assert transport._message_type == HttpMessage
        assert transport.headers == {}
        assert transport.timeout == 30.0
        assert transport.max_retries == 3
        assert transport.retry_backoff_factor == 0.5
        assert transport.retry_status_codes == {429, 500, 502, 503, 504}
        assert transport.follow_redirects is True
        assert transport.verify_ssl is True
        assert transport.http2 is False

    def test_create_transport_custom(self):
        """Test create_transport with custom options."""
        factory = HTTPTransportFactory(
            base_url="https://example.com",
            message_type=HttpMessage,
            default_headers={"User-Agent": "pynector/1.0"},
        )
        transport = factory.create_transport(
            headers={"X-Test": "test"},
            timeout=5.0,
            max_retries=2,
            retry_backoff_factor=1.0,
            retry_status_codes={500, 502},
            follow_redirects=False,
            verify_ssl=False,
            http2=True,
        )

        assert isinstance(transport, HTTPTransport)
        assert transport.base_url == "https://example.com"
        assert transport._message_type == HttpMessage
        assert transport.headers == {"User-Agent": "pynector/1.0", "X-Test": "test"}
        assert transport.timeout == 5.0
        assert transport.max_retries == 2
        assert transport.retry_backoff_factor == 1.0
        assert transport.retry_status_codes == {500, 502}
        assert transport.follow_redirects is False
        assert transport.verify_ssl is False
        assert transport.http2 is True

    def test_create_transport_merge_headers(self):
        """Test that headers are correctly merged."""
        factory = HTTPTransportFactory(
            base_url="https://example.com",
            message_type=HttpMessage,
            default_headers={
                "User-Agent": "pynector/1.0",
                "Accept": "application/json",
            },
        )
        transport = factory.create_transport(
            headers={"X-Test": "test", "Accept": "text/plain"}
        )

        assert transport.headers == {
            "User-Agent": "pynector/1.0",
            "Accept": "text/plain",  # Should override the default
            "X-Test": "test",
        }
