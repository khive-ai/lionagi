# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Explicit provider configuration system - no pattern matching."""

from __future__ import annotations

import msgspec
from typing import Any, Literal

# Provider type definitions for type safety
ProviderType = Literal[
    "openai",
    "anthropic", 
    "google",
    "cohere",
    "together",
    "openrouter",
    "groq",
    "fireworks",
    "ollama",
    "perplexity",
    "claude_code"
]

AuthType = Literal["bearer", "x-api-key", "none"]


@msgspec.Struct
class ProviderConfig:
    """Explicit provider configuration - no detection required."""
    
    name: ProviderType
    base_url: str
    auth_type: AuthType = "bearer"
    auth_prefix: str = "Bearer"
    supports_streaming: bool = True
    supports_functions: bool = True
    max_tokens_field: str = "max_tokens"
    temperature_field: str = "temperature" 
    custom_headers: dict[str, str] | None = None
    default_model: str | None = None


def _create_openai_config(**overrides: Any) -> ProviderConfig:
    """Create OpenAI configuration with defaults."""
    return ProviderConfig(
        name="openai",
        base_url="https://api.openai.com/v1",
        auth_type="bearer",
        supports_streaming=True,
        supports_functions=True,
        default_model="gpt-4o-mini",
        **overrides
    )


def _create_anthropic_config(**overrides: Any) -> ProviderConfig:
    """Create Anthropic configuration with defaults.""" 
    return ProviderConfig(
        name="anthropic",
        base_url="https://api.anthropic.com/v1",
        auth_type="x-api-key",
        auth_prefix="",
        supports_streaming=True,
        supports_functions=True,
        max_tokens_field="max_tokens",
        default_model="claude-3-5-sonnet-20241022",
        **overrides
    )


def _create_google_config(**overrides: Any) -> ProviderConfig:
    """Create Google configuration with defaults."""
    return ProviderConfig(
        name="google", 
        base_url="https://generativelanguage.googleapis.com/v1",
        auth_type="bearer",
        supports_streaming=True,
        supports_functions=False,
        default_model="gemini-1.5-flash",
        **overrides
    )


def _create_cohere_config(**overrides: Any) -> ProviderConfig:
    """Create Cohere configuration with defaults."""
    return ProviderConfig(
        name="cohere",
        base_url="https://api.cohere.ai/v1", 
        auth_type="bearer",
        supports_streaming=True,
        supports_functions=False,
        default_model="command-r-plus",
        **overrides
    )


def _create_together_config(**overrides: Any) -> ProviderConfig:
    """Create Together configuration with defaults."""
    return ProviderConfig(
        name="together",
        base_url="https://api.together.xyz/v1",
        auth_type="bearer", 
        supports_streaming=True,
        supports_functions=True,
        default_model="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
        **overrides
    )


def _create_openrouter_config(**overrides: Any) -> ProviderConfig:
    """Create OpenRouter configuration with defaults."""
    return ProviderConfig(
        name="openrouter",
        base_url="https://openrouter.ai/api/v1",
        auth_type="bearer",
        supports_streaming=True, 
        supports_functions=True,
        custom_headers={"HTTP-Referer": "https://github.com/khive-ai/lionagi"},
        default_model="google/gemini-2.5-flash",
        **overrides
    )


def _create_groq_config(**overrides: Any) -> ProviderConfig:
    """Create Groq configuration with defaults."""
    return ProviderConfig(
        name="groq",
        base_url="https://api.groq.com/openai/v1",
        auth_type="bearer",
        supports_streaming=True,
        supports_functions=True,
        default_model="llama-3.3-70b-versatile",
        **overrides
    )


def _create_fireworks_config(**overrides: Any) -> ProviderConfig:
    """Create Fireworks configuration with defaults."""
    return ProviderConfig(
        name="fireworks", 
        base_url="https://api.fireworks.ai/inference/v1",
        auth_type="bearer",
        supports_streaming=True,
        supports_functions=True,
        default_model="accounts/fireworks/models/llama-v3p1-70b-instruct",
        **overrides
    )


def _create_ollama_config(**overrides: Any) -> ProviderConfig:
    """Create Ollama configuration with defaults."""
    return ProviderConfig(
        name="ollama",
        base_url="http://localhost:11434/v1", 
        auth_type="none",
        auth_prefix="",
        supports_streaming=True,
        supports_functions=True,
        default_model="llama3.2:latest",
        **overrides
    )


def _create_perplexity_config(**overrides: Any) -> ProviderConfig:
    """Create Perplexity configuration with defaults."""
    return ProviderConfig(
        name="perplexity",
        base_url="https://api.perplexity.ai",
        auth_type="bearer",
        supports_streaming=True,
        supports_functions=False,
        default_model="sonar",
        **overrides
    )


def _create_claude_code_config(**overrides: Any) -> ProviderConfig:
    """Create Claude Code configuration with defaults."""
    return ProviderConfig(
        name="claude_code",
        base_url="https://api.claude.ai/api",
        auth_type="bearer", 
        supports_streaming=True,
        supports_functions=True,
        default_model="claude-3-5-sonnet-20241022",
        **overrides
    )


# Explicit provider configuration registry - no pattern matching
PROVIDER_CONFIGS: dict[ProviderType, ProviderConfig] = {
    "openai": _create_openai_config(),
    "anthropic": _create_anthropic_config(),
    "google": _create_google_config(), 
    "cohere": _create_cohere_config(),
    "together": _create_together_config(),
    "openrouter": _create_openrouter_config(),
    "groq": _create_groq_config(),
    "fireworks": _create_fireworks_config(),
    "ollama": _create_ollama_config(),
    "perplexity": _create_perplexity_config(),
    "claude_code": _create_claude_code_config(),
}


def get_provider_config(
    provider: ProviderType, 
    base_url: str | None = None,
    **overrides: Any
) -> ProviderConfig:
    """Get provider configuration with explicit provider selection.
    
    Args:
        provider: Provider name (must be explicit, no detection)
        base_url: Override base URL 
        **overrides: Override any config fields
        
    Returns:
        ProviderConfig with applied overrides
        
    Raises:
        ValueError: If provider is not supported
    """
    if provider not in PROVIDER_CONFIGS:
        raise ValueError(f"Unsupported provider: {provider}. Supported: {list(PROVIDER_CONFIGS.keys())}")
        
    config = PROVIDER_CONFIGS[provider]
    
    # Apply overrides by creating new config
    overrides_dict = {}
    if base_url:
        overrides_dict["base_url"] = base_url
    overrides_dict.update(overrides)
    
    if not overrides_dict:
        return config
        
    # Create new config with overrides
    config_dict = {
        "name": config.name,
        "base_url": config.base_url,
        "auth_type": config.auth_type,
        "auth_prefix": config.auth_prefix,
        "supports_streaming": config.supports_streaming,
        "supports_functions": config.supports_functions,
        "max_tokens_field": config.max_tokens_field,
        "temperature_field": config.temperature_field,
        "custom_headers": config.custom_headers,
        "default_model": config.default_model,
    }
    config_dict.update(overrides_dict)
    
    return ProviderConfig(**config_dict)


def get_capability_requirements(provider: ProviderType, endpoint: str = "chat") -> set[str]:
    """Get capability requirements for a provider/endpoint combination.
    
    Args:
        provider: Provider name
        endpoint: Endpoint type
        
    Returns:
        Set of required capabilities
    """
    config = PROVIDER_CONFIGS.get(provider)
    if not config:
        return {f"net.out:api.{provider}.com"}
        
    # Extract host from base_url
    from urllib.parse import urlparse
    parsed = urlparse(config.base_url)
    host = parsed.netloc
    
    return {f"net.out:{host}"}


def list_supported_providers() -> list[ProviderType]:
    """Get list of all supported provider names."""
    return list(PROVIDER_CONFIGS.keys())


def parse_provider_from_model(model: str) -> tuple[ProviderType | None, str]:
    """Parse provider from model name using explicit prefix format.
    
    Supports provider-prefixed models like "openai/gpt-4" or "anthropic/claude-3-5-sonnet".
    
    Args:
        model: Model name, potentially with provider prefix
        
    Returns:
        Tuple of (provider, clean_model_name) or (None, original_model)
    """
    if not model or "/" not in model:
        return None, model
        
    parts = model.split("/", 1)  # Split only on first slash
    if len(parts) != 2:
        return None, model
        
    potential_provider, clean_model = parts
    potential_provider = potential_provider.lower().strip()
    
    # Check if it's a supported provider
    if potential_provider in PROVIDER_CONFIGS:
        return potential_provider, clean_model  # type: ignore[return-value]
    
    return None, model


def validate_provider(provider: str) -> ProviderType:
    """Validate and return provider as ProviderType.
    
    Args:
        provider: Provider name to validate
        
    Returns:
        Validated ProviderType
        
    Raises:
        ValueError: If provider is not supported
    """
    if provider not in PROVIDER_CONFIGS:
        supported = list(PROVIDER_CONFIGS.keys())
        raise ValueError(f"Unsupported provider '{provider}'. Supported: {supported}")
    return provider  # type: ignore[return-value]


def resolve_provider_and_model(
    provider: str | None = None, 
    model: str | None = None
) -> tuple[ProviderType, str]:
    """Resolve provider and model from explicit or prefixed syntax.
    
    Supports both syntaxes:
    1. Explicit: provider="openai", model="gpt-4"
    2. Prefixed: model="openai/gpt-4" (provider can be None)
    
    Args:
        provider: Explicit provider name (optional)
        model: Model name, potentially with provider prefix
        
    Returns:
        Tuple of (validated_provider, clean_model_name)
        
    Raises:
        ValueError: If provider cannot be determined or is unsupported
    """
    if not model:
        raise ValueError("Model must be specified")
        
    # Try parsing provider from model first
    parsed_provider, clean_model = parse_provider_from_model(model)
    
    # If both explicit and parsed provider exist, they must match
    if provider and parsed_provider:
        if provider.lower() != parsed_provider:
            raise ValueError(
                f"Provider mismatch: explicit '{provider}' vs model prefix '{parsed_provider}'"
            )
        return validate_provider(provider), clean_model
    
    # Use explicit provider if available
    if provider:
        return validate_provider(provider), model
        
    # Use parsed provider if available  
    if parsed_provider:
        return parsed_provider, clean_model
        
    # If no provider specified and model has no prefix, require explicit provider
    raise ValueError(
        f"Provider must be specified explicitly or as model prefix. "
        f"Model '{model}' has no provider prefix. "
        f"Supported providers: {list(PROVIDER_CONFIGS.keys())}"
    )