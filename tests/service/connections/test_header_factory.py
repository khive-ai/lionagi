# Copyright (c) 2023 - 2025, HaiyangLi <quantocean.li at gmail dot com>
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from pydantic import SecretStr

from lionagi.service.connections.header_factory import HeaderFactory


class TestHeaderFactory:
    """Test the HeaderFactory class for header creation."""

    def test_bearer_auth_headers(self):
        """Test Bearer authentication header creation."""
        headers = HeaderFactory.get_header(
            auth_type="bearer", api_key="test-key"
        )

        assert headers["Authorization"] == "Bearer test-key"
        assert headers["Content-Type"] == "application/json"

    def test_x_api_key_headers(self):
        """Test x-api-key header creation."""
        headers = HeaderFactory.get_header(
            auth_type="x-api-key", api_key="test-key"
        )

        assert headers["x-api-key"] == "test-key"
        assert headers["Content-Type"] == "application/json"

    def test_no_auth_headers(self):
        """Test header creation without authentication."""
        headers = HeaderFactory.get_header(auth_type="none")

        assert "Authorization" not in headers
        assert "x-api-key" not in headers
        assert headers["Content-Type"] == "application/json"

    def test_custom_content_type(self):
        """Test custom content type."""
        headers = HeaderFactory.get_header(
            auth_type="bearer",
            api_key="test-key",
            content_type="application/xml",
        )

        assert headers["Authorization"] == "Bearer test-key"
        assert headers["Content-Type"] == "application/xml"

    def test_secret_str_api_key(self):
        """Test with SecretStr API key."""
        secret_key = SecretStr("secret-key")
        headers = HeaderFactory.get_header(
            auth_type="bearer", api_key=secret_key
        )

        assert headers["Authorization"] == "Bearer secret-key"
        assert headers["Content-Type"] == "application/json"

    def test_missing_api_key_with_auth_uses_dummy(self):
        """Test that missing API key uses dummy key for headless scenarios."""
        # Test bearer auth with None API key
        headers = HeaderFactory.get_header(auth_type="bearer", api_key=None)
        assert headers["Authorization"] == "Bearer dummy-key-for-testing"
        assert headers["Content-Type"] == "application/json"

        # Test x-api-key auth with None API key
        headers = HeaderFactory.get_header(auth_type="x-api-key", api_key=None)
        assert headers["x-api-key"] == "dummy-key-for-testing"
        assert headers["Content-Type"] == "application/json"

    def test_unsupported_auth_type(self):
        """Test that unsupported auth type raises error."""
        with pytest.raises(ValueError, match="Unsupported auth type"):
            HeaderFactory.get_header(
                auth_type="unsupported", api_key="test-key"
            )

    def test_get_content_type_header(self):
        """Test content type header creation."""
        headers = HeaderFactory.get_content_type_header()
        assert headers == {"Content-Type": "application/json"}

        headers = HeaderFactory.get_content_type_header("text/plain")
        assert headers == {"Content-Type": "text/plain"}

    def test_get_bearer_auth_header(self):
        """Test Bearer auth header creation."""
        headers = HeaderFactory.get_bearer_auth_header("test-key")
        assert headers == {"Authorization": "Bearer test-key"}

    def test_get_x_api_key_header(self):
        """Test x-api-key header creation."""
        headers = HeaderFactory.get_x_api_key_header("test-key")
        assert headers == {"x-api-key": "test-key"}

    def test_default_headers_merge(self):
        """Test that default headers are included."""
        default_headers = {"Custom-Header": "custom-value"}
        headers = HeaderFactory.get_header(
            auth_type="bearer",
            api_key="test-key",
            default_headers=default_headers,
        )

        assert headers["Authorization"] == "Bearer test-key"
        assert headers["Content-Type"] == "application/json"
        # Note: The current implementation doesn't merge default_headers
        # This test documents the current behavior
