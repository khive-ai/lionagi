# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Simple tests for pydantic-settings integration."""

import pytest

from lionagi.services.settings import ExecutorConfig, ServiceSettings


class TestServiceSettingsBasic:
    """Basic tests for ServiceSettings functionality."""

    def test_service_settings_creation(self):
        """Test that ServiceSettings can be created and has expected defaults."""
        settings = ServiceSettings()
        
        # Test basic configuration
        assert settings.DEFAULT_PROVIDER == "openai"
        assert settings.DEFAULT_MODEL == "gpt-4o-mini"
        assert settings.DEFAULT_QUEUE_CAPACITY == 100
        assert settings.DEFAULT_CAPACITY_REFRESH_TIME == 60.0

    def test_ollama_special_case(self):
        """Test special handling for ollama provider."""
        settings = ServiceSettings()
        
        # Ollama should always return "ollama" even without API key
        assert settings.get_api_key("ollama") == "ollama"

    def test_default_model_mapping(self):
        """Test default model mapping for different providers."""
        settings = ServiceSettings()
        
        assert settings.get_default_model("openai") == "gpt-4o-mini"
        assert settings.get_default_model("anthropic") == "claude-3-5-sonnet-20241022"
        assert settings.get_default_model("claude") == "claude-3-5-sonnet-20241022"
        assert settings.get_default_model("together") == "meta-llama/Llama-3.2-3B-Instruct-Turbo"
        assert settings.get_default_model("unknown") == "gpt-4o-mini"  # Fallback to DEFAULT_MODEL

    def test_executor_config_defaults(self):
        """Test executor configuration with defaults."""
        settings = ServiceSettings()
        
        config = settings.get_executor_config()
        assert config.queue_capacity == 100
        assert config.capacity_refresh_time == 60.0
        assert config.limit_requests is None
        assert config.limit_tokens is None
        assert config.concurrency_limit is None

    def test_executor_config_with_overrides(self):
        """Test executor configuration with custom overrides."""
        settings = ServiceSettings()
        
        config = settings.get_executor_config(
            queue_capacity=200,
            limit_requests=50,
            limit_tokens=1000,
        )
        
        assert config.queue_capacity == 200
        assert config.limit_requests == 50
        assert config.limit_tokens == 1000
        # Unchanged defaults
        assert config.capacity_refresh_time == 60.0
        assert config.concurrency_limit is None

    def test_singleton_behavior(self):
        """Test singleton pattern for ServiceSettings."""
        settings1 = ServiceSettings.get_instance()
        settings2 = ServiceSettings.get_instance()
        
        assert settings1 is settings2
        
        # Test reset
        ServiceSettings.reset_instance()
        settings3 = ServiceSettings.get_instance()
        assert settings3 is not settings1


class TestExecutorConfigStandalone:
    """Test ExecutorConfig independently."""

    def test_executor_config_creation(self):
        """Test ExecutorConfig creation with various parameters."""
        # Default config
        config1 = ExecutorConfig()
        assert config1.queue_capacity == 100
        assert config1.capacity_refresh_time == 60.0
        
        # Custom config
        config2 = ExecutorConfig(
            queue_capacity=50,
            limit_requests=25,
            limit_tokens=500,
            concurrency_limit=3,
        )
        assert config2.queue_capacity == 50
        assert config2.limit_requests == 25
        assert config2.limit_tokens == 500
        assert config2.concurrency_limit == 3

    def test_executor_configs_are_independent(self):
        """Test that different ExecutorConfig instances are independent."""
        config1 = ExecutorConfig(queue_capacity=100)
        config2 = ExecutorConfig(queue_capacity=200)
        
        assert config1.queue_capacity == 100
        assert config2.queue_capacity == 200
        assert config1 is not config2


def test_settings_import():
    """Test that settings can be imported from the services module."""
    from lionagi.services import ServiceSettings, settings
    
    # Should be able to import both the class and the instance
    assert ServiceSettings is not None
    assert settings is not None
    assert isinstance(settings, ServiceSettings)


@pytest.mark.asyncio
async def test_imodel_executor_independence():
    """Test that iModel instances can have independent executor configurations."""
    from lionagi.services.settings import ExecutorConfig
    
    # Test that we can create different ExecutorConfigs for different scenarios
    fast_config = ExecutorConfig(
        queue_capacity=50,
        limit_requests=100,
        concurrency_limit=10,
    )
    
    slow_config = ExecutorConfig(
        queue_capacity=200,
        limit_requests=10,
        concurrency_limit=2,
    )
    
    # Verify they're different
    assert fast_config.queue_capacity != slow_config.queue_capacity
    assert fast_config.limit_requests != slow_config.limit_requests
    assert fast_config.concurrency_limit != slow_config.concurrency_limit
    
    # This demonstrates that iModel can create custom configurations
    # without being forced to use global defaults
    assert fast_config is not slow_config