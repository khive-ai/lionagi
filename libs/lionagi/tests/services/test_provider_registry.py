# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""
Tests for the hybrid provider registry architecture.

Tests provider detection, registry resolution, adapter interfaces, and the new
parse_provider_prefix() function that replaced the old detect_provider logic.
"""

from typing import Any
from unittest.mock import Mock

import msgspec
import pytest
from pydantic import BaseModel

from lionagi.services.core import Service
from lionagi.services.endpoint import RequestModel
from lionagi.services.provider_detection import parse_provider_prefix
from lionagi.services.providers.provider_registry import (
    ProviderAdapter,
    ProviderRegistry,
    ProviderResolution,
    get_provider_registry,
    register_builtin_adapters,
)


class TestProviderPrefixParsing:
    """Tests for the new parse_provider_prefix() function that replaced detect_provider."""

    def test_parse_provider_prefix_basic_cases(self):
        """Test parse_provider_prefix with basic valid cases."""
        # Test with provider prefix
        provider, model = parse_provider_prefix("openai/gpt-4")
        assert provider == "openai"
        assert model == "gpt-4"

        # Test with no prefix
        provider, model = parse_provider_prefix("gpt-4")
        assert provider is None
        assert model == "gpt-4"

        # Test with None input
        provider, model = parse_provider_prefix(None)
        assert provider is None
        assert model is None

        # Test with empty string
        provider, model = parse_provider_prefix("")
        assert provider is None
        assert model == ""

    def test_parse_provider_prefix_edge_cases(self):
        """Test parse_provider_prefix with edge cases and special characters."""
        # Test with multiple slashes (only first slash is significant)
        provider, model = parse_provider_prefix("openai/model/with/slashes")
        assert provider == "openai"
        assert model == "model/with/slashes"

        # Test with empty provider part
        provider, model = parse_provider_prefix("/gpt-4")
        assert provider is None  # Empty string becomes None
        assert model == "gpt-4"

        # Test with empty model part
        provider, model = parse_provider_prefix("openai/")
        assert provider == "openai"
        assert model is None  # Empty string becomes None

        # Test with just slash
        provider, model = parse_provider_prefix("/")
        assert provider is None
        assert model is None

    def test_parse_provider_prefix_real_world_models(self):
        """Test parse_provider_prefix with real-world model names."""
        test_cases = [
            ("openai/gpt-4o-mini", "openai", "gpt-4o-mini"),
            (
                "anthropic/claude-3-sonnet-20240229",
                "anthropic",
                "claude-3-sonnet-20240229",
            ),
            ("groq/llama-3.1-70b-versatile", "groq", "llama-3.1-70b-versatile"),
            ("mistral/mistral-7b-instruct", "mistral", "mistral-7b-instruct"),
            ("gpt-4o-mini", None, "gpt-4o-mini"),  # No prefix
            ("claude-3-haiku", None, "claude-3-haiku"),  # No prefix
        ]

        for model_string, expected_provider, expected_model in test_cases:
            provider, model = parse_provider_prefix(model_string)
            assert provider == expected_provider, f"Failed for {model_string}"
            assert model == expected_model, f"Failed for {model_string}"


class TestProviderResolution:
    """Tests for ProviderResolution msgspec struct."""

    def test_provider_resolution_msgspec_compliance(self):
        """ProviderResolution msgspec compliance and serialization."""
        assert issubclass(
            ProviderResolution, msgspec.Struct
        ), "ProviderResolution must inherit from msgspec.Struct"

        # Create ProviderResolution with all fields
        resolution = ProviderResolution(
            provider="openai",
            model="gpt-4",
            base_url="https://api.openai.com/v1",
            adapter_name="openai",
        )

        # Test msgspec serialization/deserialization roundtrip
        encoded = msgspec.json.encode(resolution)
        decoded = msgspec.json.decode(encoded, type=ProviderResolution)

        assert decoded.provider == "openai"
        assert decoded.model == "gpt-4"
        assert decoded.base_url == "https://api.openai.com/v1"
        assert decoded.adapter_name == "openai"

    def test_provider_resolution_optional_fields(self):
        """ProviderResolution handles optional fields correctly."""
        # Create with minimal required fields
        resolution = ProviderResolution(
            provider="openai",
            adapter_name="openai",
        )

        assert resolution.provider == "openai"
        assert resolution.model is None
        assert resolution.base_url is None
        assert resolution.adapter_name == "openai"

        # Test serialization preserves None values
        encoded = msgspec.json.encode(resolution)
        decoded = msgspec.json.decode(encoded, type=ProviderResolution)

        assert decoded.model is None
        assert decoded.base_url is None


# Test fixtures for provider adapters


class MockAdapter(ProviderAdapter):
    """Mock adapter for testing."""

    def __init__(self, name: str, supports_all: bool = True, has_config: bool = False):
        self.name = name
        self.default_base_url = f"https://api.{name}.com/v1"
        self.request_model = RequestModel
        self.requires = {f"net.out:api.{name}.com"}
        self._supports_all = supports_all
        self.ConfigModel = MockConfig if has_config else None

    def supports(self, *, provider: str | None, model: str | None, base_url: str | None) -> bool:
        if not self._supports_all:
            return False
        # Simple logic: support if provider matches name or no provider specified
        return provider is None or provider == self.name

    def create_service(self, *, base_url: str | None, **kwargs: Any) -> Service:
        mock_service = Mock(spec=Service)
        mock_service.name = f"{self.name}_service"
        mock_service.requires = self.requires
        return mock_service

    def required_rights(self, *, base_url: str | None, **kwargs: Any) -> set[str]:
        if base_url:
            from urllib.parse import urlparse

            host = urlparse(base_url).netloc
            return {f"net.out:{host}"}
        return self.requires.copy()


class MockConfig(BaseModel):
    """Mock Pydantic config model for testing."""

    api_key: str
    timeout: float = 30.0
    retries: int = 3


class TestProviderRegistry:
    """Tests for ProviderRegistry core functionality."""

    @pytest.fixture
    def registry(self):
        """Create a fresh registry for each test."""
        return ProviderRegistry()

    @pytest.fixture
    def mock_adapters(self):
        """Create mock adapters for testing."""
        return {
            "openai": MockAdapter("openai"),
            "anthropic": MockAdapter("anthropic"),
            "generic": MockAdapter("generic"),
            "strict": MockAdapter("strict", supports_all=False),
            "config_adapter": MockAdapter("config_adapter", has_config=True),
        }

    def test_registry_registration(self, registry, mock_adapters):
        """Test adapter registration and duplicate prevention."""
        openai_adapter = mock_adapters["openai"]

        # Test successful registration
        registry.register(openai_adapter)
        assert "openai" in registry.known_adapters()

        # Test duplicate registration prevention
        with pytest.raises(ValueError, match="already registered"):
            registry.register(openai_adapter)

    def test_registry_register_many(self, registry, mock_adapters):
        """Test bulk adapter registration."""
        adapters = [mock_adapters["openai"], mock_adapters["anthropic"]]

        registry.register_many(adapters)

        known = registry.known_adapters()
        assert "openai" in known
        assert "anthropic" in known
        assert len(known) == 2

    def test_resolve_direct_provider_match(self, registry, mock_adapters):
        """Test resolution with direct provider name match."""
        registry.register(mock_adapters["openai"])

        resolution, adapter = registry.resolve(
            provider="openai",
            model="gpt-4",
            base_url=None,
        )

        assert resolution.provider == "openai"
        assert resolution.model == "gpt-4"
        assert resolution.adapter_name == "openai"
        assert adapter.name == "openai"

    def test_resolve_provider_prefix_parsing(self, registry, mock_adapters):
        """Test resolution with provider prefix in model name."""
        registry.register(mock_adapters["openai"])

        # Test with model prefix
        resolution, adapter = registry.resolve(
            provider=None,
            model="openai/gpt-4",
            base_url=None,
        )

        assert resolution.provider == "openai"
        assert resolution.model == "openai/gpt-4"
        assert adapter.name == "openai"

    def test_resolve_provider_mismatch_error(self, registry, mock_adapters):
        """Test error when provider and model prefix conflict."""
        registry.register(mock_adapters["openai"])

        # Should raise error for conflicting provider and model prefix
        with pytest.raises(ValueError, match="Provider mismatch"):
            registry.resolve(
                provider="anthropic",
                model="openai/gpt-4",
                base_url=None,
            )

    def test_resolve_unique_adapter_match(self, registry, mock_adapters):
        """Test resolution when only one adapter supports the request."""
        registry.register(mock_adapters["openai"])
        registry.register(mock_adapters["strict"])  # doesn't support anything

        resolution, adapter = registry.resolve(
            provider=None,
            model="gpt-4",
            base_url=None,
        )

        assert resolution.provider == "openai"
        assert adapter.name == "openai"

    def test_resolve_ambiguous_adapters_error(self, registry, mock_adapters):
        """Test error when multiple adapters support the request."""
        registry.register(mock_adapters["openai"])
        registry.register(mock_adapters["anthropic"])

        # Both adapters support any provider, should be ambiguous
        with pytest.raises(ValueError, match="Ambiguous adapters"):
            registry.resolve(
                provider=None,
                model="gpt-4",
                base_url=None,
            )

    def test_resolve_generic_fallback(self, registry, mock_adapters):
        """Test fallback to generic adapter with base_url."""
        registry.register(mock_adapters["generic"])

        resolution, adapter = registry.resolve(
            provider=None,
            model="unknown-model",
            base_url="https://custom.api.com/v1",
        )

        assert resolution.provider == "generic"
        assert adapter.name == "generic"

    def test_resolve_no_provider_no_base_url_error(self, registry, mock_adapters):
        """Test error when no provider or base_url is specified and no adapter supports the request."""
        registry.register(mock_adapters["strict"])  # doesn't support anything

        with pytest.raises(ValueError, match="Provider must be specified"):
            registry.resolve(
                provider=None,
                model="some-model",
                base_url=None,
            )

    def test_resolve_no_matching_adapter_error(self, registry, mock_adapters):
        """Test error when no adapter matches the request."""
        registry.register(mock_adapters["strict"])  # doesn't support anything

        with pytest.raises(ValueError, match="No adapter found"):
            registry.resolve(
                provider="unknown",
                model="unknown-model",
                base_url=None,
            )

    def test_create_service_basic(self, registry, mock_adapters):
        """Test service creation without config validation."""
        registry.register(mock_adapters["openai"])

        service, resolution, rights = registry.create_service(
            provider="openai",
            model="gpt-4",
            base_url=None,
            api_key="test-key",
        )

        assert service.name == "openai_service"
        assert resolution.provider == "openai"
        assert "net.out:api.openai.com" in rights

    def test_create_service_with_pydantic_validation(self, registry, mock_adapters):
        """Test service creation with Pydantic config validation."""
        registry.register(mock_adapters["config_adapter"])

        # Test valid config
        service, resolution, rights = registry.create_service(
            provider="config_adapter",
            model="test-model",
            base_url=None,
            api_key="valid-key",
            timeout=60.0,
        )

        assert service.name == "config_adapter_service"
        assert resolution.provider == "config_adapter"

    def test_create_service_pydantic_validation_error(self, registry, mock_adapters):
        """Test service creation with Pydantic validation failure."""
        registry.register(mock_adapters["config_adapter"])

        # Test invalid config (missing required api_key)
        with pytest.raises(ValueError, match="Invalid provider configuration"):
            registry.create_service(
                provider="config_adapter",
                model="test-model",
                base_url=None,
                timeout=60.0,  # api_key is required but missing
            )

    def test_create_service_rights_assignment(self, registry, mock_adapters):
        """Test that required rights are properly assigned to service."""
        adapter = mock_adapters["openai"]
        registry.register(adapter)

        service, resolution, rights = registry.create_service(
            provider="openai",
            model="gpt-4",
            base_url="https://custom.openai.com/v1",
        )

        # Service should have rights assigned
        assert hasattr(service, "requires")
        assert "net.out:custom.openai.com" in rights


class TestProviderRegistryIntegration:
    """Integration tests for the provider registry system."""

    def test_builtin_adapter_registration(self):
        """Test that builtin adapters can be registered."""
        # This tests the actual builtin registration function
        register_builtin_adapters()

        registry = get_provider_registry()
        adapters = registry.known_adapters()

        # Should have at least the core adapters
        assert "openai" in adapters or "generic" in adapters

    def test_singleton_registry_behavior(self):
        """Test that get_provider_registry returns the same instance."""
        registry1 = get_provider_registry()
        registry2 = get_provider_registry()

        assert registry1 is registry2

    def test_entry_points_loading(self):
        """Test entry point loading mechanism (without actual entry points)."""
        registry = ProviderRegistry()

        # Loading non-existent entry points should return 0
        count = registry.load_entry_points("non.existent.group")
        assert count == 0

    def test_real_world_resolution_scenarios(self):
        """Test resolution scenarios that mirror real-world usage."""
        registry = ProviderRegistry()
        mock_openai = MockAdapter("openai")
        mock_generic = MockAdapter("generic")
        registry.register(mock_openai)
        registry.register(mock_generic)

        # Scenario 1: Explicit provider
        resolution, adapter = registry.resolve(
            provider="openai",
            model="gpt-4",
            base_url=None,
        )
        assert adapter.name == "openai"

        # Scenario 2: Provider prefix in model
        resolution, adapter = registry.resolve(
            provider=None,
            model="openai/gpt-4o-mini",
            base_url=None,
        )
        assert adapter.name == "openai"

        # Scenario 3: Custom API with base_url (should use generic)
        resolution, adapter = registry.resolve(
            provider=None,
            model="custom-model",
            base_url="https://my-custom-api.com/v1",
        )
        assert adapter.name == "generic"

    def test_dynamic_rights_computation(self):
        """Test dynamic rights computation based on base_url."""
        registry = ProviderRegistry()
        adapter = MockAdapter("test")
        registry.register(adapter)

        # Test rights computation with custom base_url
        service, resolution, rights = registry.create_service(
            provider="test",
            model="test-model",
            base_url="https://custom-api.example.com/v1",
        )

        # Should compute rights from the custom base_url
        assert "net.out:custom-api.example.com" in rights


class TestBackwardsCompatibility:
    """Tests to ensure the new architecture maintains backwards compatibility."""

    def test_replaces_detect_provider_functionality(self):
        """Test that parse_provider_prefix provides equivalent functionality to detect_provider."""
        # The old detect_provider would have returned provider names
        # parse_provider_prefix returns (provider, model) tuple

        # Test cases that would have worked with detect_provider
        test_cases = [
            "openai/gpt-4",
            "anthropic/claude-3-sonnet",
            "groq/llama-3.1-70b",
            "plain-model-name",
            None,
        ]

        for model_name in test_cases:
            # Should not raise exceptions
            provider, model = parse_provider_prefix(model_name)

            # Behavior should be predictable
            if model_name and "/" in model_name:
                assert provider is not None
                assert model is not None
            elif model_name:
                assert provider is None
                assert model == model_name
            else:
                assert provider is None
                assert model is None

    def test_integration_with_existing_tests(self):
        """Test that the new registry works with existing test patterns."""
        # This verifies that removing detect_provider patches from tests
        # doesn't break the functionality they were testing

        registry = ProviderRegistry()
        mock_adapter = MockAdapter("openai")
        registry.register(mock_adapter)

        # The kind of resolution that integration tests would do
        service, resolution, rights = registry.create_service(
            provider="openai",
            model="gpt-4",
            base_url=None,
        )

        # Should work without needing to mock detect_provider
        assert service is not None
        assert resolution.provider == "openai"
        assert rights is not None
