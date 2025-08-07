# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
#
# SPDX-License-Identifier: Apache-2.0

from pydantic import BaseModel

from lionagi.config import settings
from lionagi.service.connections.endpoint import Endpoint
from lionagi.service.connections.endpoint_config import EndpointConfig

__all__ = (
    "OpenaiChatEndpoint",
    "OpenaiResponseEndpoint",
    "OpenrouterChatEndpoint",
    "GroqChatEndpoint",
    "OpenaiEmbedEndpoint",
)


REASONING_MODELS = (
    "o1",
    "o1-2024-12-17",
    "o1-preview-2024-09-12",
    "o1-pro",
    "o1-pro-2025-03-19",
    "o3-pro",
    "o3-pro-2025-06-10",
    "o3",
    "o3-2025-04-16",
    "o4-mini",
    "o4-mini-2025-04-16",
    "o3-mini",
    "o3-mini-2025-01-31",
    "o1-mini",
    "o1-mini-2024-09-12",
)

REASONING_NOT_SUPPORT_PARAMS = (
    "temperature",
    "top_p",
    "logit_bias",
    "logprobs",
    "top_logprobs",
)


class OpenaiChatEndpoint(Endpoint):
    def __init__(self, config=None, **kwargs):
        from ...third_party.openai_models import CreateChatCompletionRequest

        # Handle headless scenarios - if api_key is explicitly None, use no auth
        if "api_key" in kwargs and kwargs["api_key"] is None:
            api_key = None
            auth_type = "none"
        else:
            # Use provided api_key, fall back to settings, then dummy key
            api_key = kwargs.get("api_key")
            if api_key is None:
                api_key = settings.OPENAI_API_KEY or "dummy-key-for-testing"
            auth_type = "bearer"

        _config = {
            "name": "openai_chat",
            "provider": "openai",
            "base_url": "https://api.openai.com/v1",
            "endpoint": "chat/completions",
            "kwargs": {"model": "gpt-4.1-nano"},
            "api_key": api_key,
            "auth_type": auth_type,
            "content_type": "application/json",
            "method": "POST",
            "requires_tokens": True,
            "openai_compatible": True,
            "context_window": 128_000,  # Default context window for OpenAI models
            "request_options": CreateChatCompletionRequest,
        }

        config = config or {}
        if isinstance(config, EndpointConfig):
            config = config.model_dump()
        _config.update(config)
        config = EndpointConfig(**_config)

        super().__init__(config, **kwargs)

    def create_payload(
        self,
        request: dict | BaseModel,
        extra_headers: dict | None = None,
        **kwargs,
    ):
        """Override to handle model-specific parameter filtering."""
        payload, headers = super().create_payload(
            request, extra_headers, **kwargs
        )

        # Handle reasoning models
        model = payload.get("model")
        if model in REASONING_MODELS:
            # Remove unsupported parameters for reasoning models
            for param in REASONING_NOT_SUPPORT_PARAMS:
                payload.pop(param, None)

            # Convert system role to developer role for reasoning models
            if "messages" in payload and payload["messages"]:
                if payload["messages"][0].get("role") == "system":
                    payload["messages"][0]["role"] = "developer"
        else:
            # Remove reasoning_effort for non-reasoning models
            payload.pop("reasoning_effort", None)

        return (payload, headers)


class OpenaiResponseEndpoint(Endpoint):
    def __init__(self, config=None, **kwargs):
        from ...third_party.openai_models import CreateResponse

        _config = dict(
            name="openai_response",
            provider="openai",
            base_url="https://api.openai.com/v1",
            endpoint="chat/completions",  # OpenAI responses API uses same endpoint
            kwargs={"model": "gpt-4.1-nano"},
            api_key=settings.OPENAI_API_KEY or "dummy-key-for-testing",
            auth_type="bearer",
            content_type="application/json",
            method="POST",
            requires_tokens=True,
            request_options=CreateResponse,
        )
        config = config or {}
        if isinstance(config, EndpointConfig):
            config = config.model_dump(exclude_none=True)
        _config.update(config)
        config = EndpointConfig(**_config)

        super().__init__(config, **kwargs)


class OpenrouterChatEndpoint(Endpoint):
    def __init__(self, config=None, **kwargs):
        _config = dict(
            name="openrouter_chat",
            provider="openrouter",
            base_url="https://openrouter.ai/api/v1",
            endpoint="chat/completions",
            kwargs={"model": "google/gemini-2.5-flash"},
            api_key=settings.OPENROUTER_API_KEY or "dummy-key-for-testing",
            auth_type="bearer",
            content_type="application/json",
            method="POST",
        )
        config = config or {}
        if isinstance(config, EndpointConfig):
            config = config.model_dump(exclude_none=True)
        _config.update(config)
        config = EndpointConfig(**_config)

        super().__init__(config, **kwargs)


class GroqChatEndpoint(Endpoint):
    def __init__(self, config=None, **kwargs):
        _config = {
            "name": "groq_chat",
            "provider": "groq",
            "base_url": "https://api.groq.com/openai/v1",
            "endpoint": "chat/completions",
            "api_key": settings.GROQ_API_KEY or "dummy-key-for-testing",
            "auth_type": "bearer",
            "content_type": "application/json",
            "method": "POST",
            "context_window": 128_000,  # Groq context window
        }

        config = config or {}
        if isinstance(config, EndpointConfig):
            config = config.model_dump()
        _config.update(config)
        config = EndpointConfig(**_config)

        super().__init__(config, **kwargs)


class OpenaiEmbedEndpoint(Endpoint):
    def __init__(self, config=None, **kwargs):
        from ...third_party.openai_models import CreateEmbeddingRequest

        _config = {
            "name": "openai_embed",
            "provider": "openai",
            "base_url": "https://api.openai.com/v1",
            "endpoint": "embeddings",
            "kwargs": {"model": "text-embedding-3-small"},
            "api_key": settings.OPENAI_API_KEY or "dummy-key-for-testing",
            "auth_type": "bearer",
            "content_type": "application/json",
            "method": "POST",
            "request_options": CreateEmbeddingRequest,
        }

        config = config or {}
        if isinstance(config, EndpointConfig):
            config = config.model_dump()
        _config.update(config)
        config = EndpointConfig(**_config)

        super().__init__(config, **kwargs)
