# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
#
# SPDX-License-Identifier: Apache-2.0

"""
OpenAI and OpenAI-compatible endpoint configurations.

This module provides endpoint configurations for:
- OpenAI (chat, response, embedding)
- OpenRouter (OpenAI-compatible)
- Groq (OpenAI-compatible)

Each provider has a helper function (_get_*_config) that creates
configurations with sensible defaults that can be overridden.
"""

from pydantic import BaseModel

from lionagi.config import settings
from lionagi.service.connections.endpoint import Endpoint
from lionagi.service.connections.endpoint_config import EndpointConfig
from lionagi.service.third_party.openai_model_names import (
    REASONING_MODELS,
    is_reasoning_model,
)

__all__ = (
    "OpenaiChatEndpoint",
    "OpenaiResponseEndpoint",
    "OpenaiEmbedEndpoint",
    "OpenrouterChatEndpoint",
    "GroqChatEndpoint",
    "OPENAI_CHAT_ENDPOINT_CONFIG",
    "OPENAI_RESPONSE_ENDPOINT_CONFIG",
    "OPENAI_EMBEDDING_ENDPOINT_CONFIG",
    "OPENROUTER_CHAT_ENDPOINT_CONFIG",
    "OPENROUTER_GEMINI_ENDPOINT_CONFIG",
    "GROQ_CHAT_ENDPOINT_CONFIG",
    "REASONING_MODELS",
    "REASONING_NOT_SUPPORT_PARAMS",
)


def _get_openai_config(**kwargs):
    """Create OpenAI endpoint configuration with defaults."""
    config = dict(
        name="openai_chat",
        provider="openai",
        base_url="https://api.openai.com/v1",
        endpoint="chat/completions",
        kwargs={"model": settings.OPENAI_DEFAULT_MODEL},
        api_key=settings.OPENAI_API_KEY or "dummy-key-for-testing",
        auth_type="bearer",
        content_type="application/json",
        method="POST",
        requires_tokens=True,
        # NOTE: OpenAI models have incorrect role literals, only use for param validation
        # request_options=CreateChatCompletionRequest,
    )
    config.update(kwargs)
    return EndpointConfig(**config)


def _get_openrouter_config(**kwargs):
    """Create OpenRouter endpoint configuration with defaults."""
    config = dict(
        name="openrouter_chat",
        provider="openrouter",
        base_url="https://openrouter.ai/api/v1",
        endpoint="chat/completions",
        kwargs={"model": "google/gemini-2.5-flash"},
        api_key=settings.OPENROUTER_API_KEY or "dummy-key-for-testing",
        auth_type="bearer",
        content_type="application/json",
        method="POST",
        # NOTE: OpenRouter uses OpenAI-compatible format
        # request_options=CreateChatCompletionRequest,
    )
    config.update(kwargs)
    return EndpointConfig(**config)


def _get_groq_config(**kwargs):
    """Create Groq endpoint configuration with defaults."""
    config = dict(
        name="groq_chat",
        provider="groq",
        base_url="https://api.groq.com/openai/v1",
        endpoint="chat/completions",
        kwargs={"model": "llama-3.3-70b-versatile"},
        api_key=settings.GROQ_API_KEY or "dummy-key-for-testing",
        auth_type="bearer",
        content_type="application/json",
        method="POST",
    )
    config.update(kwargs)
    return EndpointConfig(**config)


# OpenAI endpoints
OPENAI_CHAT_ENDPOINT_CONFIG = _get_openai_config()

OPENAI_RESPONSE_ENDPOINT_CONFIG = _get_openai_config(
    name="openai_response",
    endpoint="responses",
)

OPENAI_EMBEDDING_ENDPOINT_CONFIG = _get_openai_config(
    name="openai_embed",
    endpoint="embeddings",
    kwargs={"model": "text-embedding-3-small"},
)

# OpenRouter endpoints
OPENROUTER_CHAT_ENDPOINT_CONFIG = _get_openrouter_config()

OPENROUTER_GEMINI_ENDPOINT_CONFIG = _get_openrouter_config(
    name="openrouter_gemini",
    kwargs={"model": "google/gemini-2.5-flash"},
)

# Groq endpoints
GROQ_CHAT_ENDPOINT_CONFIG = _get_groq_config()

REASONING_NOT_SUPPORT_PARAMS = (
    "temperature",
    "top_p",
    "logit_bias",
    "logprobs",
    "top_logprobs",
)


class OpenaiChatEndpoint(Endpoint):
    def __init__(self, config=None, **kwargs):
        config = config or _get_openai_config()
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
        if (
            model
            and is_reasoning_model(model)
            and not model.startswith("gpt-5")
        ):
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
        config = config or _get_openai_config(
            name="openai_response",
            endpoint="responses",
        )
        super().__init__(config, **kwargs)


class OpenrouterChatEndpoint(Endpoint):
    def __init__(self, config=None, **kwargs):
        config = config or _get_openrouter_config()
        super().__init__(config, **kwargs)


class GroqChatEndpoint(Endpoint):
    def __init__(self, config=None, **kwargs):
        config = config or _get_groq_config()
        super().__init__(config, **kwargs)


class OpenaiEmbedEndpoint(Endpoint):
    def __init__(self, config=None, **kwargs):
        config = config or _get_openai_config(
            name="openai_embed",
            endpoint="embeddings",
            kwargs={"model": "text-embedding-3-small"},
        )
        super().__init__(config, **kwargs)
