from typing import Any, ClassVar

from pydantic import BaseModel, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = ("settings",)


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


class AppSettings(BaseSettings, frozen=True):
    """Application settings with environment variable support."""

    model_config = SettingsConfigDict(
        env_file=(".env", ".env.local", ".secrets.env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="allow",
    )

    aiocache_config: CacheConfig = Field(
        default_factory=CacheConfig, description="Cache settings for aiocache"
    )

    OPENAI_API_KEY: SecretStr | None = None
    OPENROUTER_API_KEY: SecretStr | None = None
    EXA_API_KEY: SecretStr | None = None
    PERPLEXITY_API_KEY: SecretStr | None = None
    ANTHROPIC_API_KEY: SecretStr | None = None
    GROQ_API_KEY: SecretStr | None = None

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
            raise AttributeError(
                f"Secret key '{key_name}' not found in settings"
            )

        secret = getattr(self, key_name)
        if secret is None:
            raise ValueError(f"Secret key '{key_name}' is not set")

        if isinstance(secret, SecretStr):
            return secret.get_secret_value()

        return str(secret)


# Create a singleton instance
settings = AppSettings()
# Store the instance in the class variable for singleton pattern
AppSettings._instance = settings
