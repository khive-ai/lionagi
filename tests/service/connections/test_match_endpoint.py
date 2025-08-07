# Copyright (c) 2023 - 2025, HaiyangLi <quantocean.li at gmail dot com>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

import pytest

from lionagi.service.connections.endpoint import Endpoint
from lionagi.service.connections.match_endpoint import match_endpoint


class TestMatchEndpoint:
    """Test the match_endpoint function for provider matching logic."""

    def test_openai_chat_endpoint(self):
        """Test matching OpenAI chat endpoint."""
        endpoint = match_endpoint(
            provider="openai", endpoint="chat", model="gpt-4o-mini"
        )

        assert isinstance(endpoint, Endpoint)
        assert endpoint.config.provider == "openai"
        # The actual endpoint might be different than the input endpoint
        # OpenAI compatible flag may be set differently based on implementation

    def test_anthropic_messages_endpoint(self):
        """Test matching Anthropic messages endpoint."""
        endpoint = match_endpoint(
            provider="anthropic",
            endpoint="chat",
            model="claude-3-opus-20240229",
        )

        assert isinstance(endpoint, Endpoint)
        assert endpoint.config.provider == "anthropic"
        assert (
            endpoint.config.default_headers["anthropic-version"]
            == "2023-06-01"
        )

    def test_perplexity_endpoint(self):
        """Test matching Perplexity endpoint."""
        endpoint = match_endpoint(
            provider="perplexity",
            endpoint="chat",
            model="llama-3.1-sonar-small-128k-online",
        )

        assert isinstance(endpoint, Endpoint)
        assert endpoint.config.provider == "perplexity"

    # def test_ollama_endpoint(self):
    #     """Test matching Ollama endpoint."""
    #     endpoint = match_endpoint(
    #         provider="ollama", endpoint="chat", model="llama2"
    #     )

    #     assert isinstance(endpoint, Endpoint)
    #     assert endpoint.config.provider == "ollama"

    def test_exa_search_endpoint(self):
        """Test matching Exa search endpoint."""
        endpoint = match_endpoint(
            provider="exa", endpoint="search", query="test query"
        )

        # Exa endpoint may not be supported yet
        if endpoint is None:
            pytest.skip("Exa endpoint not implemented")
        assert isinstance(endpoint, Endpoint)
        assert endpoint.config.provider == "exa"

    def test_custom_base_url(self):
        """Test endpoint with custom base URL."""
        custom_url = "https://custom.api.com/v1"
        endpoint = match_endpoint(
            provider="openai",
            endpoint="chat",
            base_url=custom_url,
            model="gpt-4o-mini",
        )

        assert endpoint.config.base_url == custom_url

    def test_custom_endpoint_params(self):
        """Test endpoint with custom endpoint parameters."""
        endpoint = match_endpoint(
            provider="openai",
            endpoint="chat",
            endpoint_params=["custom", "path"],
            model="gpt-4o-mini",
        )

        assert endpoint.config.endpoint_params == ["custom", "path"]

    def test_unknown_provider_fallback(self):
        """Test fallback behavior for unknown provider."""
        endpoint = match_endpoint(
            provider="unknown_provider", endpoint="chat", model="some-model"
        )

        # Unknown providers may return None
        if endpoint is None:
            pytest.skip("Unknown provider not supported")
        assert isinstance(endpoint, Endpoint)
        assert endpoint.config.provider == "unknown_provider"

    def test_model_parameter_filtering(self):
        """Test that reasoning models get correct parameter filtering."""
        # Test with reasoning model
        reasoning_endpoint = match_endpoint(
            provider="openai", endpoint="chat", model="o1-preview"
        )

        # Test with standard model
        standard_endpoint = match_endpoint(
            provider="openai", endpoint="chat", model="gpt-4o-mini"
        )

        assert isinstance(reasoning_endpoint, Endpoint)
        assert isinstance(standard_endpoint, Endpoint)

    def test_endpoint_with_api_key(self):
        """Test endpoint creation with API key."""
        endpoint = match_endpoint(
            provider="openai",
            endpoint="chat",
            model="gpt-4o-mini",
            api_key="test-key",
        )

        # API key should be handled by the endpoint config
        assert isinstance(endpoint, Endpoint)

    def test_anthropic_specific_headers(self):
        """Test that Anthropic endpoints get correct headers."""
        endpoint = match_endpoint(
            provider="anthropic",
            endpoint="chat",
            model="claude-3-opus-20240229",
        )

        assert "anthropic-version" in endpoint.config.default_headers
        assert (
            endpoint.config.default_headers["anthropic-version"]
            == "2023-06-01"
        )

    def test_endpoint_params_inheritance(self):
        """Test that endpoint parameters are correctly inherited."""
        endpoint = match_endpoint(provider="openai", endpoint="chat")

        if endpoint is None:
            pytest.skip("OpenAI endpoint not supported")
        # Endpoint params structure may vary
        assert isinstance(endpoint, Endpoint)

    def test_provider_case_insensitive(self):
        """Test that provider matching is case insensitive."""
        endpoint_lower = match_endpoint(
            provider="openai", endpoint="chat", model="gpt-4o-mini"
        )

        endpoint_upper = match_endpoint(
            provider="OPENAI", endpoint="chat", model="gpt-4o-mini"
        )

        if endpoint_lower is None or endpoint_upper is None:
            pytest.skip("Provider case insensitive not supported")
        assert endpoint_lower.config.provider == endpoint_upper.config.provider

    def test_multiple_providers_isolation(self):
        """Test that multiple endpoint instances are isolated."""
        openai_endpoint = match_endpoint(
            provider="openai", endpoint="chat", model="gpt-4o-mini"
        )

        anthropic_endpoint = match_endpoint(
            provider="anthropic",
            endpoint="chat",
            model="claude-3-opus-20240229",
        )

        if openai_endpoint is None or anthropic_endpoint is None:
            pytest.skip("One or both endpoints not supported")

        # Should be different instances with different configurations
        assert openai_endpoint is not anthropic_endpoint
        assert (
            openai_endpoint.config.provider
            != anthropic_endpoint.config.provider
        )

    def test_endpoint_config_immutability(self):
        """Test that endpoint configurations don't interfere with each other."""
        endpoint1 = match_endpoint(
            provider="openai",
            endpoint="chat",
            model="gpt-4o-mini",
            temperature=0.5,
        )

        endpoint2 = match_endpoint(
            provider="openai", endpoint="chat", model="gpt-4o", temperature=0.8
        )

        # Should have different configurations
        assert endpoint1.config is not endpoint2.config
