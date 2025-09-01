# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for pydantic-settings integration in services layer."""

import os
from unittest.mock import patch

import pytest

from lionagi.services.settings import ExecutorConfig, ServiceSettings


class TestServiceSettings:
    """Test ServiceSettings configuration management."""

    def test_settings_singleton_pattern(self):
        """Test that ServiceSettings follows singleton pattern."""
        settings1 = ServiceSettings.get_instance()
        settings2 = ServiceSettings.get_instance()
        
        # Should be the same instance
        assert settings1 is settings2
        
        # Reset for clean test
        ServiceSettings.reset_instance()
        
        settings3 = ServiceSettings.get_instance()
        assert settings3 is not settings1  # New instance after reset

    def test_api_key_detection(self):
        """Test API key detection from environment variables."""
        settings = ServiceSettings()
        
        # Test special ollama case
        assert settings.get_api_key("ollama") == "ollama"
        
        # Test with mocked environment variables
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-openai-key"}):
            settings_fresh = ServiceSettings()
            assert settings_fresh.get_api_key("openai") == "test-openai-key"
        
        # Test provider name variations
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-anthropic-key"}):
            settings_fresh = ServiceSettings()
            assert settings_fresh.get_api_key("anthropic") == "test-anthropic-key"
            assert settings_fresh.get_api_key("claude") == "test-anthropic-key"

    def test_default_model_selection(self):
        """Test default model selection for providers."""
        settings = ServiceSettings()
        
        assert settings.get_default_model("openai") == settings.OPENAI_DEFAULT_MODEL
        assert settings.get_default_model("anthropic") == settings.ANTHROPIC_DEFAULT_MODEL
        assert settings.get_default_model("claude") == settings.ANTHROPIC_DEFAULT_MODEL
        assert settings.get_default_model("unknown_provider") == settings.DEFAULT_MODEL

    def test_executor_config_generation(self):
        """Test executor configuration generation with overrides."""
        settings = ServiceSettings()
        
        # Default configuration
        config = settings.get_executor_config()
        assert config.queue_capacity == settings.DEFAULT_QUEUE_CAPACITY
        assert config.capacity_refresh_time == settings.DEFAULT_CAPACITY_REFRESH_TIME
        
        # Configuration with overrides
        config_custom = settings.get_executor_config(
            queue_capacity=50,
            limit_requests=100,
        )
        assert config_custom.queue_capacity == 50
        assert config_custom.limit_requests == 100
        assert config_custom.capacity_refresh_time == settings.DEFAULT_CAPACITY_REFRESH_TIME

    def test_environment_variable_loading(self):
        """Test environment variable loading with different prefixes."""
        # Test both with and without LIONAGI_ prefix
        with patch.dict(os.environ, {
            "LIONAGI_DEFAULT_PROVIDER": "custom_provider",
            "DEFAULT_MODEL": "custom-model",  # Should be ignored due to prefix
        }):
            settings = ServiceSettings()
            assert settings.DEFAULT_PROVIDER == "custom_provider"
            assert settings.DEFAULT_MODEL == "gpt-4o-mini"  # Should use class default


class TestExecutorConfig:
    """Test ExecutorConfig model."""

    def test_executor_config_defaults(self):
        """Test ExecutorConfig with default values."""
        config = ExecutorConfig()
        
        assert config.queue_capacity == 100
        assert config.capacity_refresh_time == 60.0
        assert config.interval is None
        assert config.limit_requests is None
        assert config.limit_tokens is None
        assert config.concurrency_limit is None

    def test_executor_config_custom_values(self):
        """Test ExecutorConfig with custom values."""
        config = ExecutorConfig(
            queue_capacity=200,
            limit_requests=50,
            limit_tokens=1000,
            concurrency_limit=5,
        )
        
        assert config.queue_capacity == 200
        assert config.limit_requests == 50
        assert config.limit_tokens == 1000
        assert config.concurrency_limit == 5


@pytest.mark.asyncio
async def test_imodel_settings_integration():
    """Test that iModel integrates properly with settings."""
    from lionagi.services.imodel import iModel
    
    # Mock environment for testing
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key-integration"}):
        # Test that iModel can be created without explicit api_key (should use settings)
        model = iModel(
            provider="openai",
            model="gpt-4o-mini",
            # No api_key provided - should use settings detection
            queue_capacity=50,  # Custom executor config
            limit_requests=25,
        )
        
        # Verify executor config is customized per instance
        assert model.executor.config.queue_capacity == 50
        assert model.executor.config.limit_requests == 25
        
        # Verify provider detection worked
        assert model.provider == "openai"
        assert model.model == "gpt-4o-mini"


@pytest.mark.asyncio 
async def test_multiple_imodel_instances_independent():
    """Test that multiple iModel instances have independent configurations."""
    from lionagi.services.imodel import iModel
    
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
        # Create two iModels with different executor configs
        model1 = iModel(
            provider="openai",
            queue_capacity=100,
            limit_requests=50,
        )
        
        model2 = iModel(
            provider="openai", 
            queue_capacity=200,
            limit_requests=100,
        )
        
        # Verify they have independent configurations
        assert model1.executor.config.queue_capacity == 100
        assert model1.executor.config.limit_requests == 50
        
        assert model2.executor.config.queue_capacity == 200
        assert model2.executor.config.limit_requests == 100
        
        # Different instances should have different executors
        assert model1.executor is not model2.executor