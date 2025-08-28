# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Provider intelligence and auto-detection - ported from v0 sophistication."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

# Provider patterns for model name detection
PROVIDER_PATTERNS = {
    "openai": [
        r"^gpt-[345]",
        r"^gpt-4o",
        r"^text-",
        r"^davinci",
        r"^curie",
        r"^babbage",
        r"^ada",
        r"^code-",
        r"^text-embedding-",
        r"^whisper-",
        r"^tts-",
        r"^dall-e",
    ],
    "anthropic": [
        r"^claude-",
        r"^claude\b",
        r"^sonnet",
        r"^haiku",
        r"^opus",
    ],
    "google": [
        r"^gemini",
        r"^palm",
        r"^bison",
        r"^chat-bison",
        r"^text-bison",
        r"^embedding-gecko",
    ],
    "cohere": [
        r"^command",
        r"^embed-",
        r"^rerank-",
        r"^generate",
    ],
    "together": [
        r"^togethercomputer/",
        r"^mistralai/",
        r"^meta-llama/",
        r"^teknium/",
        r"^NousResearch/",
        r"^WizardLM/",
    ],
    "huggingface": [
        r"^microsoft/",
        r"^facebook/",
        r"^google/",
        r"^EleutherAI/",
        r"^bigscience/",
    ],
    "openrouter": [
        r"/",  # Catch-all for models with slashes
    ],
    "ollama": [
        r"^llama",
        r"^mistral",
        r"^codellama",
        r"^vicuna",
        r"^alpaca",
        r"^wizard",
        r"^orca",
    ],
    "groq": [
        r"^llama2-",
        r"^mixtral-",
        r"^gemma-",
    ],
    "fireworks": [
        r"^accounts/fireworks/",
    ],
}


@dataclass
class ProviderConfig:
    """Configuration for a specific provider."""

    name: str
    base_url: str
    auth_header: str = "Authorization"
    auth_prefix: str = "Bearer"
    supports_streaming: bool = True
    supports_functions: bool = True
    max_tokens_field: str = "max_tokens"
    temperature_field: str = "temperature"
    custom_headers: dict[str, str] | None = None


# Provider configurations
PROVIDER_CONFIGS = {
    "openai": ProviderConfig(
        name="openai",
        base_url="https://api.openai.com/v1",
        supports_streaming=True,
        supports_functions=True,
    ),
    "anthropic": ProviderConfig(
        name="anthropic",
        base_url="https://api.anthropic.com/v1",
        auth_header="x-api-key",
        auth_prefix="",
        supports_streaming=True,
        supports_functions=True,
        max_tokens_field="max_tokens",
    ),
    "google": ProviderConfig(
        name="google",
        base_url="https://generativelanguage.googleapis.com/v1",
        auth_header="Authorization",
        auth_prefix="Bearer",
        supports_streaming=True,
        supports_functions=False,
    ),
    "cohere": ProviderConfig(
        name="cohere",
        base_url="https://api.cohere.ai/v1",
        supports_streaming=True,
        supports_functions=False,
        max_tokens_field="max_tokens",
    ),
    "together": ProviderConfig(
        name="together",
        base_url="https://api.together.xyz/v1",
        supports_streaming=True,
        supports_functions=True,
    ),
    "openrouter": ProviderConfig(
        name="openrouter",
        base_url="https://openrouter.ai/api/v1",
        supports_streaming=True,
        supports_functions=True,
        custom_headers={"HTTP-Referer": "https://github.com/khive-ai/lionagi"},
    ),
    "groq": ProviderConfig(
        name="groq",
        base_url="https://api.groq.com/openai/v1",
        supports_streaming=True,
        supports_functions=True,
    ),
    "fireworks": ProviderConfig(
        name="fireworks",
        base_url="https://api.fireworks.ai/inference/v1",
        supports_streaming=True,
        supports_functions=True,
    ),
    "ollama": ProviderConfig(
        name="ollama",
        base_url="http://localhost:11434/v1",
        auth_header="Authorization",
        auth_prefix="Bearer",
        supports_streaming=True,
        supports_functions=True,
    ),
    "claude_code": ProviderConfig(
        name="claude_code",
        base_url="https://api.claude.ai/api",
        auth_header="Authorization",
        auth_prefix="Bearer",
        supports_streaming=True,
        supports_functions=True,
    ),
}


def detect_provider_from_model(model: str) -> str | None:
    """Detect provider from model name using pattern matching.

    Args:
        model: Model name to analyze

    Returns:
        Provider name or None if not detected
    """
    if not model:
        return None

    # Handle explicit provider prefixes (e.g., "openai/gpt-4")
    if "/" in model:
        parts = model.split("/")
        if len(parts) >= 2:
            potential_provider = parts[0].lower()
            if potential_provider in PROVIDER_CONFIGS:
                return potential_provider

    # Pattern matching
    model_lower = model.lower()

    for provider, patterns in PROVIDER_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, model_lower):
                return provider

    # Default fallback - if model contains slashes, assume OpenRouter
    if "/" in model:
        return "openrouter"

    return None


def infer_provider_config(
    provider: str, base_url: str | None = None, **overrides: Any
) -> ProviderConfig:
    """Get provider configuration with optional overrides.

    Args:
        provider: Provider name
        base_url: Override base URL
        **overrides: Override any config fields

    Returns:
        ProviderConfig with applied overrides
    """
    if provider not in PROVIDER_CONFIGS:
        # Create generic config for unknown providers
        config = ProviderConfig(
            name=provider,
            base_url=base_url or f"https://api.{provider}.com/v1",
        )
    else:
        config = PROVIDER_CONFIGS[provider]

    # Apply overrides
    config_dict = {
        "name": config.name,
        "base_url": base_url or config.base_url,
        "auth_header": config.auth_header,
        "auth_prefix": config.auth_prefix,
        "supports_streaming": config.supports_streaming,
        "supports_functions": config.supports_functions,
        "max_tokens_field": config.max_tokens_field,
        "temperature_field": config.temperature_field,
        "custom_headers": config.custom_headers,
    }

    # Apply any overrides
    config_dict.update(overrides)

    return ProviderConfig(**config_dict)


def get_model_info(model: str) -> dict[str, Any]:
    """Get comprehensive information about a model.

    Args:
        model: Model name

    Returns:
        Dict with model information
    """
    provider = detect_provider_from_model(model)

    info = {
        "model": model,
        "provider": provider,
        "original_name": model,
    }

    if provider:
        config = PROVIDER_CONFIGS.get(provider)
        if config:
            info.update(
                {
                    "base_url": config.base_url,
                    "supports_streaming": config.supports_streaming,
                    "supports_functions": config.supports_functions,
                    "max_tokens_field": config.max_tokens_field,
                    "temperature_field": config.temperature_field,
                }
            )

    # Extract clean model name if it has provider prefix
    if "/" in model:
        parts = model.split("/", 1)
        if len(parts) >= 2:
            info["clean_model"] = parts[1]
    else:
        info["clean_model"] = model

    return info


def normalize_model_name(model: str, provider: str | None = None) -> str:
    """Normalize model name for API calls.

    Some providers need model names adjusted for their API format.

    Args:
        model: Original model name
        provider: Provider name (auto-detected if None)

    Returns:
        Normalized model name for API
    """
    if provider is None:
        provider = detect_provider_from_model(model)

    # Remove provider prefix for API calls
    if "/" in model:
        parts = model.split("/", 1)
        if len(parts) >= 2:
            model = parts[1]

    # Provider-specific normalization
    if provider == "anthropic":
        # Anthropic uses specific model names
        anthropic_aliases = {
            "claude": "claude-3-haiku-20240307",
            "sonnet": "claude-3-sonnet-20240229",
            "opus": "claude-3-opus-20240229",
            "haiku": "claude-3-haiku-20240307",
        }
        return anthropic_aliases.get(model.lower(), model)

    elif provider == "google":
        # Google uses specific prefixes
        if not model.startswith(("gemini-", "palm-", "bison-")):
            return f"gemini-{model}"

    return model


def get_capability_requirements(provider: str, endpoint: str = "chat") -> set[str]:
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


# Model validation
def validate_model_provider_compatibility(model: str, provider: str) -> bool:
    """Check if a model is compatible with the specified provider.

    Args:
        model: Model name
        provider: Provider name

    Returns:
        True if compatible, False otherwise
    """
    detected = detect_provider_from_model(model)

    # If we can't detect provider from model, assume compatibility
    if detected is None:
        return True

    return detected == provider


def suggest_alternative_models(model: str, target_provider: str) -> list[str]:
    """Suggest alternative models for a different provider.

    Args:
        model: Original model name
        target_provider: Target provider

    Returns:
        List of suggested alternative models
    """
    suggestions = []
    model_lower = model.lower()

    # Simple mapping of model capabilities to alternatives
    if "gpt-4" in model_lower:
        if target_provider == "anthropic":
            suggestions.extend(["claude-3-opus-20240229", "claude-3-sonnet-20240229"])
        elif target_provider == "google":
            suggestions.extend(["gemini-pro", "gemini-pro-vision"])
    elif "gpt-3.5" in model_lower:
        if target_provider == "anthropic":
            suggestions.extend(["claude-3-haiku-20240307"])
        elif target_provider == "google":
            suggestions.extend(["gemini-pro"])
    elif "claude" in model_lower:
        if target_provider == "openai":
            suggestions.extend(["gpt-4", "gpt-4-turbo"])
        elif target_provider == "google":
            suggestions.extend(["gemini-pro"])

    return suggestions
