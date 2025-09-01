# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Environment settings management using pydantic-settings.

This module provides structured environment variable management for the lionagi
services layer, following the v0 pattern but adapted for v1 architecture.
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar

from pydantic import BaseModel, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class ExecutorConfig(BaseModel):
    """Configuration for the RateLimitedExecutor."""
    
    queue_capacity: int = 100
    capacity_refresh_time: float = 60.0
    interval: float | None = None
    limit_requests: int | None = None
    limit_tokens: int | None = None
    concurrency_limit: int | None = None


class ServiceSettings(BaseSettings, frozen=True):
    """Service-layer settings with environment variable support.
    
    Provides structured access to API keys, default models, and service
    configuration options. Follows v0 patterns but adapted for v1.
    """
    
    model_config = SettingsConfigDict(
        env_file=(".env", ".env.local", ".secrets.env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )
    
    # API Keys - using SecretStr to avoid accidental logging
    OPENAI_API_KEY: SecretStr | None = None
    ANTHROPIC_API_KEY: SecretStr | None = None
    CLAUDE_API_KEY: SecretStr | None = None  # Alternative name for Anthropic
    TOGETHER_API_KEY: SecretStr | None = None
    OPENROUTER_API_KEY: SecretStr | None = None
    GROQ_API_KEY: SecretStr | None = None
    FIREWORKS_API_KEY: SecretStr | None = None
    PERPLEXITY_API_KEY: SecretStr | None = None
    COHERE_API_KEY: SecretStr | None = None
    OLLAMA_API_KEY: SecretStr | None = None
    NVIDIA_NIM_API_KEY: SecretStr | None = None
    
    # Default Models and Providers
    DEFAULT_PROVIDER: str = "openai"
    DEFAULT_MODEL: str = "gpt-4o-mini"
    DEFAULT_EMBEDDING_PROVIDER: str = "openai"
    DEFAULT_EMBEDDING_MODEL: str = "text-embedding-3-small"
    
    # Provider-specific defaults
    OPENAI_DEFAULT_MODEL: str = "gpt-4o-mini"
    ANTHROPIC_DEFAULT_MODEL: str = "claude-3-5-sonnet-20241022"
    TOGETHER_DEFAULT_MODEL: str = "meta-llama/Llama-3.2-3B-Instruct-Turbo"
    GROQ_DEFAULT_MODEL: str = "llama3-8b-8192"
    NVIDIA_NIM_DEFAULT_MODEL: str = "meta/llama-3.2-1b-instruct"
    
    # Executor defaults
    DEFAULT_QUEUE_CAPACITY: int = 100
    DEFAULT_CAPACITY_REFRESH_TIME: float = 60.0
    DEFAULT_LIMIT_REQUESTS: int | None = None
    DEFAULT_LIMIT_TOKENS: int | None = None
    DEFAULT_CONCURRENCY_LIMIT: int | None = None
    
    # Service behavior
    ENABLE_POLICY_BY_DEFAULT: bool = True
    ENABLE_METRICS_BY_DEFAULT: bool = True
    ENABLE_REDACTION_BY_DEFAULT: bool = True
    
    # Timeout defaults
    DEFAULT_TIMEOUT_S: float = 30.0
    DEFAULT_STREAM_TIMEOUT_S: float = 60.0
    
    # Class variable for singleton pattern
    _instance: ClassVar[ServiceSettings | None] = None
    
    def get_api_key(self, provider: str) -> str | None:
        """Get API key for a specific provider.
        
        Args:
            provider: Provider name (e.g., "openai", "anthropic")
            
        Returns:
            API key string if available, None otherwise
            
        Note:
            Special handling for ollama - returns "ollama" if key is not set
        """
        # Handle provider name variations
        provider_lower = provider.lower()
        
        # Map provider names to environment variable names
        env_var_mapping = {
            "openai": ["OPENAI_API_KEY"],
            "anthropic": ["ANTHROPIC_API_KEY", "CLAUDE_API_KEY"],
            "claude": ["CLAUDE_API_KEY", "ANTHROPIC_API_KEY"],
            "together": ["TOGETHER_API_KEY"],
            "openrouter": ["OPENROUTER_API_KEY"],
            "groq": ["GROQ_API_KEY"],
            "fireworks": ["FIREWORKS_API_KEY"],
            "perplexity": ["PERPLEXITY_API_KEY"],
            "cohere": ["COHERE_API_KEY"],
            "ollama": ["OLLAMA_API_KEY"],
            "nvidia_nim": ["NVIDIA_NIM_API_KEY"],
            "nvidia": ["NVIDIA_NIM_API_KEY"],
        }
        
        # Try provider-specific mappings first
        for env_var_name in env_var_mapping.get(provider_lower, []):
            secret_field = getattr(self, env_var_name, None)
            if secret_field is not None:
                if isinstance(secret_field, SecretStr):
                    return secret_field.get_secret_value()
                return str(secret_field)
        
        # Special case for ollama - return "ollama" even if no key is set
        if provider_lower == "ollama":
            return "ollama"
        
        # Try generic pattern as fallback
        generic_attr = f"{provider_lower.upper()}_API_KEY"
        if hasattr(self, generic_attr):
            secret_field = getattr(self, generic_attr)
            if secret_field is not None:
                if isinstance(secret_field, SecretStr):
                    return secret_field.get_secret_value()
                return str(secret_field)
        
        return None
    
    def get_default_model(self, provider: str) -> str:
        """Get default model for a specific provider.
        
        Args:
            provider: Provider name
            
        Returns:
            Default model name for the provider
        """
        provider_lower = provider.lower()
        
        model_mapping = {
            "openai": self.OPENAI_DEFAULT_MODEL,
            "anthropic": self.ANTHROPIC_DEFAULT_MODEL,
            "claude": self.ANTHROPIC_DEFAULT_MODEL,
            "together": self.TOGETHER_DEFAULT_MODEL,
            "groq": self.GROQ_DEFAULT_MODEL,
            "nvidia_nim": self.NVIDIA_NIM_DEFAULT_MODEL,
            "nvidia": self.NVIDIA_NIM_DEFAULT_MODEL,
        }
        
        return model_mapping.get(provider_lower, self.DEFAULT_MODEL)
    
    def get_executor_config(self, **overrides) -> ExecutorConfig:
        """Get executor configuration with optional overrides.
        
        Args:
            **overrides: Configuration values to override defaults
            
        Returns:
            ExecutorConfig instance with merged settings
        """
        config_data = {
            "queue_capacity": self.DEFAULT_QUEUE_CAPACITY,
            "capacity_refresh_time": self.DEFAULT_CAPACITY_REFRESH_TIME,
            "limit_requests": self.DEFAULT_LIMIT_REQUESTS,
            "limit_tokens": self.DEFAULT_LIMIT_TOKENS,
            "concurrency_limit": self.DEFAULT_CONCURRENCY_LIMIT,
        }
        
        # Apply overrides
        config_data.update(overrides)
        
        return ExecutorConfig(**config_data)
    
    @classmethod
    def get_instance(cls) -> ServiceSettings:
        """Get singleton instance of ServiceSettings.
        
        Returns:
            The singleton ServiceSettings instance
        """
        if cls._instance is None:
            cls._instance = cls()
            logger.debug("Created new ServiceSettings singleton instance")
        return cls._instance
    
    @classmethod 
    def reset_instance(cls) -> None:
        """Reset singleton instance (mainly for testing)."""
        cls._instance = None


# Create and store singleton instance
settings = ServiceSettings.get_instance()