# Copyright (c) 2023 - 2025, HaiyangLi <quantocean.li at gmail dot com>
#
# SPDX-License-Identifier: Apache-2.0

from datetime import timezone

# Default service configuration for OpenAI
DEFAULT_OPENAI_CONFIG = {
    "provider": "openai",
    "model": "gpt-4o",
    "api_key": "OPENAI_API_KEY",  # Will be loaded from environment variable
}

# Default service configuration for Anthropic
DEFAULT_ANTHROPIC_CONFIG = {
    "provider": "anthropic",
    "model": "claude-3-opus-20240229",
    "api_key": "ANTHROPIC_API_KEY",  # Will be loaded from environment variable
}

# Logging configuration
LOG_CONFIG = {
    "persist_dir": "./data/logs",
    "subfolder": None,
    "capacity": 50,
    "extension": ".json",
    "use_timestamp": True,
    "hash_digits": 5,
    "file_prefix": "log",
    "auto_save_on_exit": True,
    "clear_after_dump": True,
}

# Session persistence configuration
SESSION_CONFIG = {
    "persist_dir": "./data/sessions",
    "extension": ".json",
    "use_timestamp": True,
}

# Tool configuration
TOOL_CONFIG = {
    "schema_generation": True,  # Automatically generate schemas for tools
}


class Settings:
    """Global settings for the lionagi package."""

    class Config:
        """General configuration settings."""
        TIMEZONE: timezone = timezone.utc
        LOG: dict = LOG_CONFIG

    class Service:
        """Service configuration settings."""
        OPENAI: dict = DEFAULT_OPENAI_CONFIG
        ANTHROPIC: dict = DEFAULT_ANTHROPIC_CONFIG
        DEFAULT_PROVIDER: str = "openai"

    class Session:
        """Session configuration settings."""
        PERSISTENCE: dict = SESSION_CONFIG

    class Tool:
        """Tool configuration settings."""
        CONFIG: dict = TOOL_CONFIG
