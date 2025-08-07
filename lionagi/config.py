# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
#
# SPDX-License-Identifier: Apache-2.0

from datetime import timezone
from typing import Any, ClassVar

from pydantic import BaseModel, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class CacheConfig(BaseModel):
    """Configuration for aiocache."""

    ttl: int = 300
    key: str | None = None
    namespace: str | None = None
    key_builder: Any = None
    skip_cache_func: Any = lambda _: False
    serializer: dict[str, Any] | None = None
    plugins: Any = None
    alias: str | None = None
    noself: Any = lambda _: False

    def as_kwargs(self) -> dict[str, Any]:
        """Convert config to kwargs dict for @cached decorator.

        Removes unserialisable callables that aiocache can't pickle.
        """
        raw = self.model_dump(exclude_none=True)
        # Remove all unserialisable callables
        unserialisable_keys = (
            "key_builder",
            "skip_cache_func",
            "noself",
            "serializer",
            "plugins",
        )
        for key in unserialisable_keys:
            raw.pop(key, None)
        return raw


def _get_log_config(prefix: str) -> dict[str, Any]:
    return {
        "persist_dir": f"./data/logs/{prefix}",
        "subfolder": None,
        "capacity": 50,
        "extension": ".json",
        "use_timestamp": True,
        "hash_digits": 5,
        "file_prefix": "action",
        "auto_save_on_exit": True,
        "clear_after_dump": True,
    }


class AppSettings(BaseSettings, frozen=True):
    """Application settings with environment variable support."""

    model_config = SettingsConfigDict(
        env_file=(".env", ".env.local", ".secrets.env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    aiocache_config: CacheConfig = Field(
        default_factory=CacheConfig, description="Cache settings for aiocache"
    )

    HOOK_LOG_CONFIG: dict = Field(
        default_factory=lambda: _get_log_config("hook"),
        description="Default configuration for hook logs",
    )

    ACTION_LOG_CONFIG: dict = Field(
        default_factory=lambda: _get_log_config("action"),
        description="Default configuration for action logs",
    )
    API_LOG_CONFIG: dict = Field(
        default_factory=lambda: _get_log_config("api"),
        description="Default configuration for API logs",
    )
    TIMEZONE: timezone = timezone.utc

    # secrets
    OPENAI_API_KEY: SecretStr | None = None
    OPENROUTER_API_KEY: SecretStr | None = None
    OLLAMA_API_KEY: SecretStr | None = None
    EXA_API_KEY: SecretStr | None = None
    PERPLEXITY_API_KEY: SecretStr | None = None
    GROQ_API_KEY: SecretStr | None = None
    ANTHROPIC_API_KEY: SecretStr | None = None

    # defaults models
    LIONAGI_EMBEDDING_PROVIDER: str = "openai"
    LIONAGI_EMBEDDING_MODEL: str = "text-embedding-3-small"

    LIONAGI_CHAT_PROVIDER: str = "openai"
    LIONAGI_CHAT_MODEL: str = "gpt-4.1-nano"

    # default storage
    LIONAGI_AUTO_STORE_EVENT: bool = False
    LIONAGI_STORAGE_PROVIDER: str = "async_qdrant"

    LIONAGI_AUTO_EMBED_LOG: bool = False

    # specific storage
    LIONAGI_QDRANT_URL: str = "http://localhost:6333"
    LIONAGI_DEFAULT_QDRANT_COLLECTION: str = "event_logs"

    # Class variable to store the singleton instance
    _instance: ClassVar[Any] = None

    def get_secret(self, key_name: str) -> str:
        """
        Get the secret value for a given key name.

        Args:
            key_name: The name of the secret key (e.g., "OPENAI_API_KEY")

        Returns:
            The secret value as a string

        Raises:
            AttributeError: If the key doesn't exist
            ValueError: If the key exists but is None
        """
        if not hasattr(self, key_name):
            if "ollama" in key_name.lower():
                return "ollama"
            raise AttributeError(
                f"Secret key '{key_name}' not found in settings"
            )

        secret = getattr(self, key_name)
        if secret is None:
            # Special case for Ollama - return "ollama" even if key exists but is None
            if "ollama" in key_name.lower():
                return "ollama"
            raise ValueError(f"Secret key '{key_name}' is not set")

        if isinstance(secret, SecretStr):
            return secret.get_secret_value()

        return str(secret)


# Create a singleton instance
settings = AppSettings()
# Store the instance in the class variable for singleton pattern
AppSettings._instance = settings
