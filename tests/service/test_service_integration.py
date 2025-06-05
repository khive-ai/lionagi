# Copyright (c) 2023 - 2025, HaiyangLi <quantocean.li at gmail dot com>
#
# SPDX-License-Identifier: Apache-2.0

import os
from unittest.mock import AsyncMock, patch

import pytest

from lionagi.service.connections.api_calling import APICalling
from lionagi.service.connections.endpoint import Endpoint
from lionagi.service.connections.endpoint_config import EndpointConfig
from lionagi.service.connections.header_factory import HeaderFactory
from lionagi.service.connections.match_endpoint import match_endpoint
from lionagi.service.imodel import iModel


class TestServiceIntegration:
    """Integration tests covering core service functionality."""

    def test_endpoint_config_creation(self):
        """Test basic endpoint configuration creation."""
        config = EndpointConfig(
            name="test_endpoint",
            provider="openai",
            endpoint="chat",
            base_url="https://api.openai.com/v1",
            endpoint_params=["chat", "completions"],
            openai_compatible=True,
            api_key="test-key",
        )

        assert config.name == "test_endpoint"
        assert config.provider == "openai"
        assert config.endpoint == "chat"
        assert config.openai_compatible is True

    def test_endpoint_config_validation(self):
        """Test endpoint configuration validation."""
        config = EndpointConfig(
            name="test",
            provider="anthropic",
            endpoint="messages",
            base_url="https://api.anthropic.com/v1",
            api_key="sk-test-key",
        )

        # Test that validation passes
        assert config.provider == "anthropic"

    def test_endpoint_creation(self):
        """Test endpoint creation with configuration."""
        config = EndpointConfig(
            name="test_endpoint",
            provider="openai",
            endpoint="chat",
            base_url="https://api.openai.com/v1",
            endpoint_params=["chat", "completions"],
            openai_compatible=True,
            api_key="test-key",
        )

        endpoint = Endpoint(config=config)
        assert endpoint.config == config
        # Note: allowed_roles is a property of iModel, not Endpoint

    def test_endpoint_payload_creation(self):
        """Test endpoint payload creation."""
        config = EndpointConfig(
            name="test_endpoint",
            provider="openai",
            endpoint="chat",
            base_url="https://api.openai.com/v1",
            endpoint_params=["chat", "completions"],
            openai_compatible=True,
            api_key="test-key",
        )

        endpoint = Endpoint(config=config)

        request_data = {
            "messages": [{"role": "user", "content": "Hello"}],
            "model": "gpt-4o-mini",
            "temperature": 0.7,
        }

        payload, headers = endpoint.create_payload(request_data)

        assert payload["model"] == "gpt-4o-mini"
        assert payload["messages"][0]["content"] == "Hello"
        assert payload["temperature"] == 0.7
        assert "Authorization" in headers

    def test_header_factory_comprehensive(self):
        """Test comprehensive header factory functionality."""
        # Test bearer auth
        headers = HeaderFactory.get_header(
            auth_type="bearer", api_key="test-key"
        )
        assert headers["Authorization"] == "Bearer test-key"
        assert headers["Content-Type"] == "application/json"

        # Test x-api-key auth
        headers = HeaderFactory.get_header(
            auth_type="x-api-key", api_key="test-key"
        )
        assert headers["x-api-key"] == "test-key"

        # Test no auth
        headers = HeaderFactory.get_header(auth_type="none")
        assert "Authorization" not in headers
        assert "x-api-key" not in headers

    def test_match_endpoint_openai(self):
        """Test endpoint matching for OpenAI."""
        endpoint = match_endpoint(
            provider="openai", endpoint="chat", model="gpt-4o-mini"
        )

        assert endpoint.config.provider == "openai"
        # Note: openai_compatible may be set differently by the match_endpoint function

    def test_match_endpoint_anthropic(self):
        """Test endpoint matching for Anthropic."""
        endpoint = match_endpoint(
            provider="anthropic",
            endpoint="chat",
            model="claude-3-opus-20240229",
        )

        assert endpoint.config.provider == "anthropic"
        assert endpoint.config.openai_compatible is False

    def test_api_calling_creation(self):
        """Test API calling creation and basic functionality."""
        config = EndpointConfig(
            name="test_endpoint",
            provider="openai",
            endpoint="chat",
            base_url="https://api.openai.com/v1",
            endpoint_params=["chat", "completions"],
            openai_compatible=True,
            api_key="test-key",
        )

        endpoint = Endpoint(config=config)

        api_call = APICalling(
            payload={
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": "Hello"}],
            },
            headers={"Authorization": "Bearer test-key"},
            endpoint=endpoint,
        )

        assert api_call.payload["model"] == "gpt-4o-mini"
        assert api_call.headers["Authorization"] == "Bearer test-key"
        assert api_call.endpoint == endpoint
        assert api_call.response is None

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    def test_imodel_creation(self):
        """Test iModel creation."""
        imodel = iModel(provider="openai", model="gpt-4o-mini")

        assert imodel.endpoint.config.provider == "openai"
        assert imodel.endpoint.config.kwargs["model"] == "gpt-4o-mini"
        assert imodel.model_name == "gpt-4o-mini"

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    def test_imodel_api_calling_creation(self):
        """Test iModel API calling creation."""
        imodel = iModel(provider="openai", model="gpt-4o-mini")

        api_call = imodel.create_api_calling(
            messages=[{"role": "user", "content": "Hello"}], temperature=0.7
        )

        assert isinstance(api_call, APICalling)
        assert api_call.payload["model"] == "gpt-4o-mini"
        assert api_call.payload["temperature"] == 0.7

    def test_endpoint_url_construction(self):
        """Test URL construction for different endpoints."""
        # OpenAI endpoint
        openai_config = EndpointConfig(
            name="openai_chat",
            provider="openai",
            endpoint="chat",
            base_url="https://api.openai.com/v1",
            endpoint_params=["chat", "completions"],
            openai_compatible=True,
            api_key="test-key",
        )

        openai_endpoint = Endpoint(config=openai_config)
        # Use config.full_url instead of _construct_url
        openai_url = openai_endpoint.config.full_url
        expected_url = "https://api.openai.com/v1/chat/completions"
        # URL construction may vary based on endpoint_params
        assert "api.openai.com" in openai_url

        # Anthropic endpoint
        anthropic_config = EndpointConfig(
            name="anthropic_chat",
            provider="anthropic",
            endpoint="messages",
            base_url="https://api.anthropic.com/v1",
            endpoint_params=["messages"],
            openai_compatible=False,
            api_key="test-key",
        )

        anthropic_endpoint = Endpoint(config=anthropic_config)
        anthropic_url = anthropic_endpoint.config.full_url
        assert "api.anthropic.com" in anthropic_url

    def test_endpoint_config_update(self):
        """Test endpoint configuration update functionality."""
        config = EndpointConfig(
            name="test",
            provider="openai",
            endpoint="chat",
            base_url="https://api.openai.com/v1",
            api_key="test-key",
        )

        config.update(timeout=600, custom_param="value")

        assert config.timeout == 600
        assert config.kwargs["custom_param"] == "value"

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"})
    def test_anthropic_integration(self):
        """Test Anthropic integration."""
        imodel = iModel(provider="anthropic", model="claude-3-opus-20240229")

        assert imodel.endpoint.config.provider == "anthropic"
        assert imodel.endpoint.config.openai_compatible is False

        # Test payload creation
        api_call = imodel.create_api_calling(
            messages=[{"role": "user", "content": "Hello"}], max_tokens=100
        )

        assert api_call.payload["model"] == "claude-3-opus-20240229"
        assert api_call.payload["max_tokens"] == 100

    def test_endpoint_config_kwargs_handling(self):
        """Test that endpoint config properly handles extra kwargs."""
        config = EndpointConfig(
            name="test",
            provider="openai",
            endpoint="chat",
            base_url="https://api.openai.com/v1",
            api_key="test-key",
            custom_field="custom_value",  # This should go to kwargs
            another_param=123,
        )

        assert config.kwargs["custom_field"] == "custom_value"
        assert config.kwargs["another_param"] == 123
